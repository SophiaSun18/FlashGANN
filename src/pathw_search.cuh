#pragma once

#include "pathw_utils.cuh"

/**
 * @brief PathWeaver-style beam search kernel for the PathW mode, launched by IndexGraph::search_pathw.
 *
 * Block blockIdx.x serves query blockIdx.x with one warp. Iterations before prune_iteration_limit
 * use sign-bit pruning, later ones exact distances and the visited hash.
 *
 * Shared memory order: query (dim floats), beam (padded_beam_size) plus candidate IDs, their
 * distances, visited hash, parent list, query sign words, expander count.
 *
 * @tparam T vector element type
 * @param K results per query
 * @param nq number of queries
 * @param dim vector dimension
 * @param beam_sz beam size and iteration limit
 * @param bitlen visited hash bit length
 * @param npoints number of data rows
 * @param d_queries nq x dim queries
 * @param d_data row-major data
 * @param d_sign_bit per-node neighbor sign words, or nullptr to disable pruning
 * @param d_results nq x K output IDs
 * @param d_result_dists nq x K output distances
 * @param d_iters nq outputs of expanded node counts
 * @param entry_point seed node
 * @param gg device graph
 * @param neighbor_keep_ratio fraction of neighbors kept by pruning; pruning is off unless in (0, 1)
 * @param iteration_prune_ratio fraction of the iteration limit that uses pruning
 * @param use_ip true for negated inner product, false for squared L2
 */
template <typename T>
static __global__ __launch_bounds__(PATHW_BLOCK_SIZE)
void PathWBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    const uint32_t* __restrict__ d_sign_bit,
    vidType* __restrict__ d_results,
    float* __restrict__ d_result_dists,
    uint32_t* d_iters,
    vidType entry_point, GraphGPU gg,
    float neighbor_keep_ratio, float iteration_prune_ratio, bool use_ip)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq || beam_sz <= 0) return;

    const uint32_t padded_beam_size = effective_sort_beam_size(beam_sz);
    const uint32_t graph_degree = gg.get_max_degree();
    const uint32_t candidate_buffer_size = graph_degree;
    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    const uint32_t packed_dim = (dim + 31) / 32;
    const uint32_t sign_bit_vector_size = graph_degree * packed_dim;
    const int iteration_limit = beam_sz;
    const float prune_iteration_limit = static_cast<float>(iteration_limit) * iteration_prune_ratio;

    const int tid = threadIdx.x;
    uint32_t expanded = 0;

    // [1] carve shared memory
    extern __shared__ __align__(16) uint32_t smem[];
    DATA_T* query_buffer = reinterpret_cast<DATA_T*>(smem);
    INDEX_T* result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + ((dim + 3) & ~3));
    DISTANCE_T* result_distances_buffer = reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size);
    INDEX_T* visited_hash_buffer = reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size);
    INDEX_T* parent_list_buffer = visited_hash_buffer + hashtable_getsize(bitlen);
    uint32_t* query_sign_bits_buffer = reinterpret_cast<uint32_t*>(parent_list_buffer + SEARCH_WIDTH);
    uint32_t* num_expanders = query_sign_bits_buffer + static_cast<size_t>(SEARCH_WIDTH) * packed_dim;
    if (tid == 0) num_expanders[0] = 0;

    // [2] clear the hash, load the query, clear the beam and candidates
    hashtable_init(visited_hash_buffer, bitlen);

    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = d_queries[static_cast<size_t>(query_id) * dim + i];
    }
    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices_buffer[i] = MAX_INDEX;
        result_distances_buffer[i] = FLT_MAX;
    }
    __syncthreads();

    // [3] as upstream, evaluate the seed's neighbors without pruning, then drop the seed
    const bool valid_entry = entry_point != MAX_INDEX && entry_point < npoints;
    if (tid == 0) {
        result_indices_buffer[0] = valid_entry ? entry_point : MAX_INDEX;
        parent_list_buffer[0] = 0;
    }
    __syncthreads();
    pathw_compute_candidates_plain<T, TEAM_SIZE>(
        dim, parent_list_buffer, result_indices_buffer,
        result_indices_buffer + padded_beam_size, result_distances_buffer + padded_beam_size,
        query_buffer, d_data, visited_hash_buffer, bitlen, gg, 
        graph_degree, candidate_buffer_size, use_ip);
    __syncthreads();
    if (tid == 0) result_indices_buffer[0] = MAX_INDEX;
    __syncthreads();

    // [4] main loop, at most beam_sz iterations
    for (int iter = 0; iter < iteration_limit; ++iter) {
        // [5] clear the hash every SMALL_HASH_RESET_INTERVAL iterations
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(visited_hash_buffer, bitlen);
        }
        __syncthreads();

        // [6] merge candidates into the beam, clear slots past beam_sz
        dispatch_beam_management(result_indices_buffer, result_distances_buffer, nullptr,
                                 candidate_buffer_size, padded_beam_size, iter == 0);
        __syncthreads();

        for (uint32_t i = beam_sz + tid; i < padded_beam_size; i += blockDim.x) {
            result_indices_buffer[i] = MAX_INDEX;
            result_distances_buffer[i] = FLT_MAX;
        }
        __syncthreads();

        // [7] stop after the last merge
        if (iter + 1 == iteration_limit) {
            break;
        }

        // [8] pick the next unexpanded beam node
        if (tid < WARP_SIZE) {
            const uint32_t picked = pick_expanders(
                SEARCH_WIDTH, parent_list_buffer, beam_sz, result_indices_buffer);
            if (tid == 0) {
                num_expanders[0] = picked;
            }
        }
        __syncthreads();

        // [9] after a clear, reinsert the beam into the hash
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            const unsigned first_tid = ((blockDim.x <= WARP_SIZE) ? 0 : WARP_SIZE);
            hashtable_restore(
                visited_hash_buffer,
                bitlen,
                result_indices_buffer,
                beam_sz,
                first_tid);
        }
        __syncthreads();

        // [10] stop when no node is left to expand
        if (num_expanders[0] == 0) {
            break;
        }
        expanded += num_expanders[0];

        // [11] expand: sign-bit pruning before prune_iteration_limit, else exact with the hash
        if (d_sign_bit != nullptr && neighbor_keep_ratio > 0.0f &&
            neighbor_keep_ratio < 1.0f && iter < prune_iteration_limit) {
            pathw_compute_candidates_with_signbit_pruning<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size,
                query_buffer, d_data, d_sign_bit, query_sign_bits_buffer,
                gg, graph_degree, candidate_buffer_size, sign_bit_vector_size, packed_dim,
                neighbor_keep_ratio, beam_sz, use_ip);
        } else {
            pathw_compute_candidates_plain<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size, query_buffer, d_data,
                visited_hash_buffer, bitlen, gg, graph_degree,
                candidate_buffer_size, use_ip);
        }
        __syncthreads();
    }

    // [12] as upstream, merge once more and write the top K, duplicate IDs included
    dispatch_beam_management(result_indices_buffer, result_distances_buffer, nullptr,
                             candidate_buffer_size, padded_beam_size, false);
    __syncthreads();
    for (uint32_t i = tid; i < static_cast<uint32_t>(K); i += blockDim.x) {
        d_results[static_cast<size_t>(query_id) * K + i] = result_indices_buffer[i] & ~INDEX_MSB_1_MASK;
        d_result_dists[static_cast<size_t>(query_id) * K + i] = result_distances_buffer[i];
    }
    if (tid == 0) d_iters[query_id] = expanded;
}

