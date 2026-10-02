#pragma once

#include "rabitq_utils.cuh"

/**
 * @brief Estimate-only quantized beam search, launched by gpu_search_rabitq.
 *
 * Block b serves query b. The beam ranks children by estimate; the closest K exactly scored
 * parents form the result pool. Dynamic shared memory, in order: TOP_K + CANDIDATE ids, their
 * distances, HASH_TABLE, PARENT_NODE_LIST, PARENT_DISTANCE_LIST, QUERY_BUFFER, RESULT_POOL ids
 * and distances, LUT_BUFFER, SIGN_LUT_BUFFER (turbop), then a transient region for query
 * transforms or radix scratch.
 *
 * @tparam CODEBITS quantizer code width
 * @tparam turbop whether the index carries a TurboQuant sketch
 * @param K results per query, also the result pool capacity
 * @param nq number of queries; blocks past nq exit
 * @param dim raw dimension
 * @param beam_sz beam size
 * @param bitlen visited hash table bit length
 * @param max_degree graph degree
 * @param npoints number of indexed points; neighbor ids at or above it are skipped
 * @param d_queries queries, nq x dim floats
 * @param d_qg_data index rows, each holding the raw vector, codes, signs, factors and neighbor ids
 * @param d_qg_signs sign vector of the index rotation
 * @param d_qg_sketch TurboQuant sketch, three padded_dim vectors, read only when turbop
 * @param d_qg_levels TurboQuant MSE-stage levels, read only when turbop
 * @param d_results result ids, nq x K, padded with MAX_INDEX
 * @param d_result_dists result distances, nq x K, padded with FLT_MAX
 * @param d_iters receives the iteration count per query
 * @param entry_point start node
 * @param row_offset row stride of d_qg_data in floats
 * @param neighbor_offset offset of the neighbor ids within a row
 * @param code_offset offset of the neighbor codes within a row
 * @param sign_offset offset of the neighbor sign codes within a row
 * @param factor_offset offset of the neighbor factors within a row
 * @param use_ip whether distances are inner product rather than L2
 */
template <int CODEBITS, bool turbop>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree, size_t npoints,
    const float* __restrict__ d_queries, const float* __restrict__ d_qg_data, const float* __restrict__ d_qg_signs,
    const float* __restrict__ d_qg_sketch, const float* __restrict__ d_qg_levels,
    vidType* __restrict__ d_results, float* __restrict__ d_result_dists,
    uint32_t* __restrict__ d_iters, vidType entry_point,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t sign_offset,
    size_t factor_offset, bool use_ip)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + static_cast<size_t>(query_id) * dim;

    const size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const uint32_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(SEARCH_WIDTH * max_degree));
    const uint32_t padded_beam_size = effective_sort_beam_size(beam_sz);
    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;

    // [1] lay out dynamic shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = reinterpret_cast<INDEX_T*>(shared_memory);
    INDEX_T* TOP_K_INDEX = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX = ALL_INDEX + padded_beam_size;
    DISTANCE_T* ALL_DISTANCE = reinterpret_cast<DISTANCE_T*>(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = reinterpret_cast<INDEX_T*>(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_NODE_LIST = HASH_TABLE + hashtable_getsize(bitlen);
    DISTANCE_T* PARENT_DISTANCE_LIST = reinterpret_cast<DISTANCE_T*>(PARENT_NODE_LIST + SEARCH_WIDTH);
    DATA_T* QUERY_BUFFER = reinterpret_cast<DATA_T*>(PARENT_DISTANCE_LIST + SEARCH_WIDTH);
    INDEX_T* RESULT_POOL_INDEX = reinterpret_cast<INDEX_T*>(QUERY_BUFFER + dim);
    DISTANCE_T* RESULT_POOL_DISTANCE = reinterpret_cast<DISTANCE_T*>(RESULT_POOL_INDEX + K);

    char* tail_base = reinterpret_cast<char*>(RESULT_POOL_DISTANCE + K);
    uint8_t* LUT_BUFFER = reinterpret_cast<uint8_t*>(allocate_shared_tail_array<uint4>(tail_base, quant_lutbytes(padded_dim, CODEBITS) >> 4));
    uint8_t* SIGN_LUT_BUFFER = nullptr;
    if constexpr (turbop) {
        SIGN_LUT_BUFFER = reinterpret_cast<uint8_t*>(allocate_shared_tail_array<uint4>(tail_base, quant_lutbytes(padded_dim, 1) >> 4));
    }

    char* transient = reinterpret_cast<char*>(allocate_shared_tail_array<uint4>(tail_base, 0));
    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(transient);
    float* SKETCH_QUERY_BUFFER = turbop ? ROTATED_QUERY_BUFFER + padded_dim : nullptr;
    void* CANDIDATE_RADIX_SCRATCH = nullptr;
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && candidate_buffer_size > 256) {
        using CandidateRadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
        char* radix_base = transient;
        CANDIDATE_RADIX_SCRATCH = allocate_shared_tail_array<typename CandidateRadixSort::TempStorage>(radix_base, 1);
    }

    __shared__ QueryFactors qf;
    __shared__ QueryFactors qb;
    __shared__ uint32_t keep_expanding;
    __shared__ uint32_t result_pool_size;
    __shared__ uint32_t result_pool_worst_idx;
    __shared__ DISTANCE_T result_pool_worst_dist;

    // [2] initialize the query factors and the result pool state
    if (tidx() == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        qb.rotated_query = SKETCH_QUERY_BUFFER;
        qb.lut = SIGN_LUT_BUFFER;
        qb.low_val = FLT_MAX;
        qb.high_val = -FLT_MAX;
        qb.width = 0.0f;
        qb.sum_q = 0;
        result_pool_size = 0;
        result_pool_worst_idx = 0;
        result_pool_worst_dist = -FLT_MAX;
    }

    // [3] clear the visited table, copy the query, and clear the beam and candidate buffers
    hashtable_init(HASH_TABLE, bitlen);
    for (int i = tidx(); i < dim; i += blockDim.x) {
        QUERY_BUFFER[i] = query[i];
    }
    for (int i = tidx(); i < static_cast<int>(padded_beam_size); i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tidx(); i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    __syncthreads();

    // [4] warp 0 seeds candidate slot 0 with the entry point and its exact distance
    if (warpidx() == 0) {
        DISTANCE_T entry_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (laneidx() == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
        }
    }

    // [5] prepare the query LUTs; the sketch takes the rotated query
    if constexpr (turbop) {
        turboq_prepare_lut_gpu<CODEBITS>(QUERY_BUFFER, qf, d_qg_signs, d_qg_levels, dim, padded_dim);
        __syncthreads();
        sketch_apply(qf.rotated_query, qb.rotated_query, d_qg_sketch, d_qg_sketch + padded_dim,
                     d_qg_sketch + 2 * padded_dim, padded_dim, padded_dim);
        lut_build<1>(qb, nullptr, padded_dim);
    } else if constexpr (CODEBITS > 1) {
        turboq_prepare_lut_gpu<CODEBITS>(QUERY_BUFFER, qf, d_qg_signs, nullptr, dim, padded_dim);
    } else {
        query_prepare_lut_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    }
    __syncthreads();

    constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
    constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
    const int group_lane = laneidx() & (lanes_per_neighbor - 1);
    const int warp_group = laneidx() / lanes_per_neighbor;
    const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));
    const int sign_bytes = static_cast<int>(quant_bytes(padded_dim, 1));

    // [6] iterate until no parent is left or MAX_ITERATIONS is reached
    int iter = 0;
    for (; iter < MAX_ITERATIONS; iter++) {
        // [7] sort and merge the estimated candidates into the beam
        dispatch_beam_management(
            ALL_INDEX, ALL_DISTANCE, CANDIDATE_RADIX_SCRATCH, candidate_buffer_size, padded_beam_size, (iter == 0));
        __syncthreads();

        // [8] clear the beam tail past beam_sz and the candidate buffer
        for (uint32_t i = beam_sz + tidx(); i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        for (int i = tidx(); i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }

        // [9] warp 0 picks parents; lane 0 drops any already expanded through another beam entry
        if (warpidx() == 0) {
            keep_expanding = pickparents(SEARCH_WIDTH, beam_sz, TOP_K_INDEX, TOP_K_DISTANCE,
                                         PARENT_NODE_LIST, PARENT_DISTANCE_LIST);
            __syncwarp();
            if (laneidx() == 0) {
                for (uint32_t p = 0; p < SEARCH_WIDTH; ++p) {
                    if (p >= keep_expanding || !hashtable_insert(HASH_TABLE, bitlen, PARENT_NODE_LIST[p])) {
                        PARENT_NODE_LIST[p] = MAX_INDEX;
                    }
                }
            }
        }
        __syncthreads();

        if (!keep_expanding) {
            break;
        }

        // [10] exact parent distances, and the unexpanded children of every parent
        for (int p = warpidx(); p < SEARCH_WIDTH; p += WARPS_PER_BLOCK) {
            const INDEX_T parent_node = PARENT_NODE_LIST[p];
            if (parent_node == MAX_INDEX) continue;
            DISTANCE_T exact_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(parent_node) * row_offset, use_ip);
            if (laneidx() == 0) {
                PARENT_DISTANCE_LIST[p] = exact_dist;
            }
        }
        for (int p = 0; p < SEARCH_WIDTH; ++p) {
            const INDEX_T parent_node = PARENT_NODE_LIST[p];
            if (parent_node == MAX_INDEX) continue;
            const vidType* parent_neighbors = reinterpret_cast<const vidType*>(
                d_qg_data + static_cast<size_t>(parent_node) * row_offset + neighbor_offset);
            for (int i = tidx(); i < max_degree; i += blockDim.x) {
                INDEX_T child_id = parent_neighbors[i];
                if (child_id >= npoints || hashtable_contains(HASH_TABLE, bitlen, child_id)) child_id = MAX_INDEX;
                CANDIDATE_INDEX[p * max_degree + i] = child_id;
            }
        }
        __syncthreads();

        // [11] warp 0 enters the expanded parents into the result pool with their exact distances
        if (warpidx() == 0) {
            for (int p = 0; p < SEARCH_WIDTH; ++p) {
                result_pool_push_unsorted(
                    RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, &result_pool_size,
                    &result_pool_worst_idx, &result_pool_worst_dist, K,
                    PARENT_NODE_LIST[p], PARENT_DISTANCE_LIST[p]);
            }
        }

        // [12] lane groups estimate every child against its parent's exact distance
        for (int p = 0; p < SEARCH_WIDTH; ++p) {
            if (PARENT_NODE_LIST[p] == MAX_INDEX) continue;
            const float* parent_row = d_qg_data + static_cast<size_t>(PARENT_NODE_LIST[p]) * row_offset;
            const uint8_t* parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);
            const uint8_t* parent_sign_base = reinterpret_cast<const uint8_t*>(parent_row + sign_offset);
            const float* parent_factors = parent_row + factor_offset;
            const DISTANCE_T parent_distance = PARENT_DISTANCE_LIST[p];
            INDEX_T* child_index = CANDIDATE_INDEX + p * max_degree;
            DISTANCE_T* child_distance = CANDIDATE_DISTANCE + p * max_degree;

            for (int base = warpidx() * groups_per_warp; base < max_degree; base += WARPS_PER_BLOCK * groups_per_warp) {
                const int i = base + warp_group;
                if (i < max_degree && child_index[i] != MAX_INDEX) {
                    DISTANCE_T est_dist;
                    if constexpr (turbop) {
                        est_dist = turbop_scan<lanes_per_neighbor, CODEBITS>(
                            qf, qb, parent_code_base, parent_sign_base, i, parent_factors,
                            max_degree, parent_distance, padded_dim, bytes_per_neighbor, sign_bytes);
                    } else {
                        est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor, CODEBITS>(
                            qf, parent_code_base, i, parent_factors + i, parent_factors + max_degree + i,
                            parent_factors + 2 * max_degree + i, parent_distance, padded_dim, bytes_per_neighbor);
                    }
                    if (group_lane == 0) {
                        const bool valid = isfinite(est_dist) && est_dist < FLT_MAX;
                        child_distance[i] = valid ? est_dist : FLT_MAX;
                        if (!valid) child_index[i] = MAX_INDEX;
                    }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    // [13] sort the result pool, nearest first, so its last entry is the worst
    result_pool_sort(RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size);
    __syncthreads();
    if (tidx() == 0 && result_pool_size > 0) {
        result_pool_worst_idx = result_pool_size - 1;
        result_pool_worst_dist = RESULT_POOL_DISTANCE[result_pool_size - 1];
    }
    __syncthreads();

    // [14] top up a short result pool with exact distances of its members' unvisited neighbors, nearest member first
    const uint32_t base_pool_size = result_pool_size;
    for (uint32_t m = 0; m < base_pool_size && result_pool_size < static_cast<uint32_t>(K); ++m) {
        const vidType* member_neighbors = reinterpret_cast<const vidType*>(
            d_qg_data + static_cast<size_t>(RESULT_POOL_INDEX[m]) * row_offset + neighbor_offset);
        for (int i = warpidx(); i < max_degree; i += WARPS_PER_BLOCK) {
            INDEX_T child_id = member_neighbors[i];
            uint32_t inserted = 0;
            if (laneidx() == 0 && child_id < npoints) inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
            __syncwarp();
            inserted = SHFL(inserted, 0);
            DISTANCE_T child_dist = FLT_MAX;
            if (inserted) {
                child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
            }
            if (laneidx() == 0) {
                CANDIDATE_INDEX[i] = inserted ? child_id : MAX_INDEX;
                CANDIDATE_DISTANCE[i] = child_dist;
            }
        }
        __syncthreads();
        if (warpidx() == 0) {
            for (int i = 0; i < max_degree; ++i) {
                result_pool_push_unsorted(
                    RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, &result_pool_size,
                    &result_pool_worst_idx, &result_pool_worst_dist, K,
                    CANDIDATE_INDEX[i], CANDIDATE_DISTANCE[i]);
            }
        }
        __syncthreads();
    }

    // [15] re-sort a topped-up pool, then thread i writes result slot i, padding the rest
    if (result_pool_size > base_pool_size) {
        result_pool_sort(RESULT_POOL_INDEX, RESULT_POOL_DISTANCE, result_pool_size);
        __syncthreads();
    }
    for (int i = tidx(); i < K; i += blockDim.x) {
        const bool filled = i < static_cast<int>(result_pool_size);
        d_results[query_id * K + i] = filled ? RESULT_POOL_INDEX[i] : MAX_INDEX;
        d_result_dists[query_id * K + i] = filled ? RESULT_POOL_DISTANCE[i] : FLT_MAX;
    }
    if (tidx() == 0) d_iters[query_id] = static_cast<uint32_t>(iter);
}
