#pragma once

#include "pathw_utils.cuh"

template <typename T, int VECTOR_DIM = 0, int BEAM_SIZE = 0>
static __global__ __launch_bounds__(PATHW_BLOCK_SIZE)
void PathWBeamSearch(
    int K, int nq, int runtime_dim, int runtime_beam, int bitlen, size_t npoints,
    const T* __restrict__ d_queries,
    const T* __restrict__ d_data,
    const uint32_t* __restrict__ d_sign_bit,
    vidType* __restrict__ d_results,
    uint32_t* d_iters,
    vidType entry_point, GraphGPU gg,
    float neighbor_keep_ratio, float iteration_prune_ratio, bool use_ip)
{
    const int dim = VECTOR_DIM == 0 ? runtime_dim : VECTOR_DIM;
    const int beam_sz = BEAM_SIZE == 0 ? runtime_beam : BEAM_SIZE;
    const int query_id = blockIdx.x;
    if (query_id >= nq || beam_sz <= 0) return;

    const uint32_t logical_beam_size = static_cast<uint32_t>(beam_sz);
    const uint32_t padded_beam_size = effective_sort_beam_size(logical_beam_size);
    const uint32_t graph_degree = gg.get_max_degree();
    const uint32_t candidate_work_count = static_cast<uint32_t>(SEARCH_WIDTH) * graph_degree;
    const uint32_t candidate_buffer_size = round_up_power2_u32(candidate_work_count);
    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    const uint32_t packed_dim = (dim + 31) / 32;
    const uint32_t sign_bit_vector_size = graph_degree * packed_dim;
    const int iteration_limit = beam_sz;
    const float prune_iteration_limit = static_cast<float>(iteration_limit) * iteration_prune_ratio;

    const int tid = threadIdx.x;
    uint32_t expanded = 0;

    // initialize the shared memory
    extern __shared__ uint32_t smem[];
    DATA_T* query_buffer = reinterpret_cast<DATA_T*>(smem);
    INDEX_T* result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + dim);
    DISTANCE_T* result_distances_buffer = reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size);
    INDEX_T* visited_hash_buffer = reinterpret_cast<INDEX_T*>(result_distances_buffer + result_buffer_size);
    INDEX_T* parent_list_buffer = visited_hash_buffer + hashtable_getsize(bitlen);
    uint32_t* query_sign_bits_buffer = reinterpret_cast<uint32_t*>(parent_list_buffer + SEARCH_WIDTH);
    uint32_t* num_expanders = query_sign_bits_buffer + static_cast<size_t>(SEARCH_WIDTH) * packed_dim;
    if (tid == 0) num_expanders[0] = 0;

    // initialize the hash table
    hashtable_init(visited_hash_buffer, bitlen);

    // load query and initialize result buffer
    for (uint32_t i = tid; i < dim; i += blockDim.x) {
        query_buffer[i] = d_queries[static_cast<size_t>(query_id) * dim + i];
    }
    for (uint32_t i = tid; i < result_buffer_size; i += blockDim.x) {
        result_indices_buffer[i] = MAX_INDEX;
        result_distances_buffer[i] = FLT_MAX;
    }
    __syncthreads();

    // Upstream starts from the seed's neighbors, evaluated without pruning.
    const bool valid_entry = entry_point != MAX_INDEX && entry_point < npoints;
    if (tid == 0) {
        result_indices_buffer[0] = valid_entry ? entry_point : MAX_INDEX;
        parent_list_buffer[0] = 0;
    }
    __syncthreads();
    pathw_compute_candidates_plain<T, TEAM_SIZE>(
        dim, parent_list_buffer, result_indices_buffer,
        result_indices_buffer + padded_beam_size, result_distances_buffer + padded_beam_size,
        query_buffer, d_data, visited_hash_buffer, bitlen, gg, valid_entry ? 1u : 0u,
        graph_degree, candidate_work_count, candidate_buffer_size, use_ip);
    __syncthreads();
    if (tid == 0) result_indices_buffer[0] = MAX_INDEX;
    __syncthreads();

    // start the main search loop
    for (int iter = 0; iter < iteration_limit; ++iter) {
        // hash table periodical reset
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(visited_hash_buffer, bitlen);
        }
        __syncthreads();

        // sort the candidates and select the top-k
        dispatch_beam_management(result_indices_buffer, result_distances_buffer, nullptr,
                                 candidate_buffer_size, padded_beam_size, iter == 0);
        __syncthreads();

        for (uint32_t i = logical_beam_size + tid; i < padded_beam_size; i += blockDim.x) {
            result_indices_buffer[i] = MAX_INDEX;
            result_distances_buffer[i] = FLT_MAX;
        }
        __syncthreads();

        // check termination condition
        if (iter + 1 == iteration_limit) {
            break;
        }

        // pick the next expander
        if (tid < WARP_SIZE) {
            const uint32_t picked = pick_expanders(
                SEARCH_WIDTH, parent_list_buffer, logical_beam_size, result_indices_buffer);
            if (tid == 0) {
                num_expanders[0] = picked;
            }
        }
        __syncthreads();

        // restore the hash table
        if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            const unsigned first_tid = ((blockDim.x <= WARP_SIZE) ? 0 : WARP_SIZE);
            hashtable_restore(
                visited_hash_buffer,
                bitlen,
                result_indices_buffer,
                logical_beam_size,
                first_tid);
        }
        __syncthreads();

        if (num_expanders[0] == 0) {
            break;
        }
        expanded += num_expanders[0];

        if (d_sign_bit != nullptr && neighbor_keep_ratio > 0.0f &&
            neighbor_keep_ratio < 1.0f && iter < prune_iteration_limit) {
            pathw_compute_candidates_with_signbit_pruning<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size,
                query_buffer, d_data, d_sign_bit, query_sign_bits_buffer,
                gg, num_expanders[0], graph_degree, candidate_work_count,
                candidate_buffer_size, sign_bit_vector_size, packed_dim,
                neighbor_keep_ratio, logical_beam_size, use_ip);
        } else {
            pathw_compute_candidates_plain<T, TEAM_SIZE>(
                dim, parent_list_buffer, result_indices_buffer,
                result_indices_buffer + padded_beam_size,
                result_distances_buffer + padded_beam_size, query_buffer, d_data,
                visited_hash_buffer, bitlen, gg, num_expanders[0], graph_degree,
                candidate_work_count, candidate_buffer_size,
                use_ip);
        }
        __syncthreads();
    }

    // Reproduce upstream's final merge and direct output, including duplicate IDs.
    dispatch_beam_management(result_indices_buffer, result_distances_buffer, nullptr,
                             candidate_buffer_size, padded_beam_size, false);
    __syncthreads();
    for (uint32_t i = tid; i < static_cast<uint32_t>(K); i += blockDim.x)
        d_results[static_cast<size_t>(query_id) * K + i] = result_indices_buffer[i] & ~INDEX_MSB_1_MASK;
    if (tid == 0) d_iters[query_id] = expanded;
}

