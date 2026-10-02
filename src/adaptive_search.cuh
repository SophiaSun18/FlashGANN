#pragma once

#include "adaptive_search_utils.cuh"

/**
 * @brief FlashGANN beam search with exact beam distances and estimate-pruned expansion, launched by gpu_search_adaptive.
 *
 * Block b serves query b; theta 1 scans a parent block-wide, larger theta one per warp.
 * Dynamic shared memory, in order: TOP_K + CANDIDATE ids, their distances, HASH_TABLE,
 * PARENT_NODE_LIST, PARENT_DISTANCE_LIST, QUERY_BUFFER, LUT_BUFFER, SIGN_LUT_BUFFER (turbop),
 * then a transient region for query transforms or radix scratch.
 *
 * @tparam CODEBITS quantizer code width
 * @tparam turbop whether the index carries a TurboQuant sketch
 * @param K results per query
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
 * @param max_iter_by_beam iteration cap, set by the driver to (beam_sz * 11 + 9) / 10
 * @param phase2_rho rho of phase 2, from compute_max_rho_bound
 * @param use_ip whether distances are inner product rather than L2
 */
template <int CODEBITS, bool turbop>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedPrunedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree, size_t npoints,
    const float* __restrict__ d_queries, const float* __restrict__ d_qg_data, const float* __restrict__ d_qg_signs,
    const float* __restrict__ d_qg_sketch, const float* __restrict__ d_qg_levels,
    vidType* __restrict__ d_results, float* __restrict__ d_result_dists,
    uint32_t* __restrict__ d_iters, vidType entry_point,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t sign_offset,
    size_t factor_offset, int max_iter_by_beam, float phase2_rho, bool use_ip)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const uint32_t candidate_buffer_size = round_up_power2_u32(candidate_buffer_capacity(static_cast<uint32_t>(max_degree)));
    uint32_t candidate_collect_capacity = static_cast<uint32_t>(BUFFER_BOUND);
    if (candidate_collect_capacity > candidate_buffer_size) candidate_collect_capacity = candidate_buffer_size;
    const uint32_t padded_beam_size = effective_sort_beam_size(beam_sz);

    // [1] lay out dynamic shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = reinterpret_cast<INDEX_T*>(shared_memory);
    INDEX_T* TOP_K_INDEX = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX = ALL_INDEX + padded_beam_size;

    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    DISTANCE_T* ALL_DISTANCE = reinterpret_cast<DISTANCE_T*>(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = reinterpret_cast<INDEX_T*>(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_NODE_LIST = HASH_TABLE + hashtable_getsize(bitlen);
    DISTANCE_T* PARENT_DISTANCE_LIST = reinterpret_cast<DISTANCE_T*>(PARENT_NODE_LIST + SEARCH_WIDTH);
    DATA_T* QUERY_BUFFER = reinterpret_cast<DATA_T*>(PARENT_DISTANCE_LIST + SEARCH_WIDTH);

    char* tail_base = reinterpret_cast<char*>(QUERY_BUFFER + dim);
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
    __shared__ GPUAdaptiveSearchState adaptive_state;
    __shared__ uint32_t compact_candidate_count;
    __shared__ DISTANCE_T current_kth_cutoff;
    __shared__ uint32_t shared_kth_near_count;
    __shared__ uint32_t warp_stat_counts[2 * WARPS_PER_BLOCK + 1];

    // [2] initialize the query factors and the policy state
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
        adaptive_state.adaptive_spec_degree = static_cast<int>(PHASE1_THETA);
        adaptive_state.adaptive_rho = static_cast<float>(PHASE1_RHO);
        adaptive_state.policy_iters = 0;
        adaptive_state.top1_node = MAX_INDEX;
        adaptive_state.top1_stall_iters = 0;
        adaptive_state.warmup_done = false;
        adaptive_state.entry_distance = FLT_MAX;
        adaptive_state.last_expander_distance = FLT_MAX;
        adaptive_state.current_expander_distance = FLT_MAX;
        compact_candidate_count = 0;
    }

    // [3] clear the visited table, copy the query, and clear the beam, candidate and parent buffers
    hashtable_init(HASH_TABLE, bitlen);
    for (int i = tidx(); i < dim; i += blockDim.x) {
        QUERY_BUFFER[i] = query[i];
    }
    for (int i = tidx(); i < static_cast<int>(padded_beam_size); i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tidx(); i < candidate_buffer_size; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tidx(); i < SEARCH_WIDTH; i += blockDim.x) {
        PARENT_NODE_LIST[i] = MAX_INDEX;
        PARENT_DISTANCE_LIST[i] = FLT_MAX;
    }
    __syncthreads();

    // [4] warp 0 seeds candidate slot 0 with the entry point and its exact distance
    if (warpidx() == 0) {
        DISTANCE_T entry_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (laneidx() == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
            adaptive_state.entry_distance = entry_dist;
            hashtable_insert(HASH_TABLE, bitlen, entry_point);
        }
    }

    // [5] prepare the query LUTs; the sketch takes the rotated query, the space the builder encodes in
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

    // [6] iterate until no parent is left, or max_iter_by_beam or MAX_ITERATIONS is reached
    int iter = 0;
    for (; iter < MAX_ITERATIONS; iter++) {

        if (iter >= max_iter_by_beam) {
            break;
        }

        // [7] sort and merge the candidates into the beam
        dispatch_beam_management(
            ALL_INDEX, ALL_DISTANCE, CANDIDATE_RADIX_SCRATCH, candidate_buffer_size, padded_beam_size, (iter == 0));
        __syncthreads();

        // [8] clear the beam tail past beam_sz and the candidate buffer
        for (uint32_t i = beam_sz + tidx(); i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        for (int i = tidx(); i < candidate_buffer_size; i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }

        // [9] warp 0 updates the policy and picks parents; the other warps periodically reset the visited table
        if (warpidx() == 0) {
            if (laneidx() == 0) {
                compact_candidate_count = 0;
                current_kth_cutoff = get_current_kth_cutoff(K, beam_sz, TOP_K_INDEX, TOP_K_DISTANCE);
                INDEX_T current_top1 = MAX_INDEX;
                const int head_stall_iters = update_head_stall_counter(TOP_K_INDEX, adaptive_state, &current_top1);
                const bool phase_changed = adaptive_state.warmup_done || head_stall_iters >= static_cast<int>(TOP1_WARMUP_STALL_ITERS);
                DISTANCE_T last_expander_distance = FLT_MAX;
                DISTANCE_T current_expander_distance = FLT_MAX;
                const float distance_reduction_rate = update_expander_progress(adaptive_state, &last_expander_distance, &current_expander_distance);

                if (phase_changed) {
                    adaptive_state = apply_stage2_state(current_top1, head_stall_iters, last_expander_distance,
                        current_expander_distance, adaptive_state, phase2_rho);
                } else {
                    const bool adaptive_should_check = (iter == 0) || ((iter % static_cast<int>(CHECK_INTERVAL)) == 0);
                    if (adaptive_should_check) {
                        adaptive_state = update_adaptive_state(current_top1, head_stall_iters, last_expander_distance,
                            current_expander_distance, distance_reduction_rate, adaptive_state, phase2_rho);
                    } else {
                        adaptive_state.top1_node = current_top1;
                        adaptive_state.top1_stall_iters = head_stall_iters;
                        adaptive_state.last_expander_distance = last_expander_distance;
                        adaptive_state.current_expander_distance = current_expander_distance;
                    }
                }
            }
            __syncwarp();
            keep_expanding = pickparents(adaptive_state.adaptive_spec_degree, beam_sz, TOP_K_INDEX,
                                          TOP_K_DISTANCE, PARENT_NODE_LIST, PARENT_DISTANCE_LIST);
            __syncwarp();
            if (laneidx() == 0 && keep_expanding) {
                record_selected_expander_dist(keep_expanding, PARENT_DISTANCE_LIST, &adaptive_state);
            }
        } else if ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0) {
            hashtable_init(HASH_TABLE, bitlen, WARP_SIZE);
            namedsync(1, BLOCK_SIZE - WARP_SIZE);
            hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz, WARP_SIZE);
        }
        __syncthreads();

        if (!keep_expanding) {
            break;
        }

        // [10] expand: block-wide for one parent, one warp per parent otherwise, one or two neighbors per lane
        const bool use_local_gate = adaptive_state.adaptive_spec_degree > 1;
        if (!use_local_gate) {
            collect_phase1_candidates_block_scan<CODEBITS, turbop>(
                padded_dim, dim, max_degree, npoints, &qf, &qb, d_qg_data, QUERY_BUFFER,
                row_offset, neighbor_offset, code_offset, sign_offset, factor_offset,
                HASH_TABLE, bitlen, PARENT_NODE_LIST, PARENT_DISTANCE_LIST,
                candidate_buffer_size, &adaptive_state, current_kth_cutoff,
                CANDIDATE_INDEX, CANDIDATE_DISTANCE, CANDIDATE_RADIX_SCRATCH, &shared_kth_near_count,
                warp_stat_counts, use_ip);
        } else {
            const uint32_t parent_work_count = (static_cast<uint32_t>(max_degree) < candidate_buffer_size) ? static_cast<uint32_t>(max_degree) : candidate_buffer_size;

            if (parent_work_count <= WARP_SIZE) {
                collect_phase2_degree32_candidates_warp_local<CODEBITS, turbop>(
                    padded_dim, max_degree, npoints, &qf, &qb, d_qg_data,
                    row_offset, neighbor_offset, code_offset, sign_offset, factor_offset,
                    HASH_TABLE, bitlen, PARENT_NODE_LIST, PARENT_DISTANCE_LIST,
                    keep_expanding, parent_work_count, &adaptive_state, current_kth_cutoff,
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, &compact_candidate_count,
                    candidate_collect_capacity, warp_stat_counts);
            } else {
                collect_phase2_degree64_candidates_warp_local<CODEBITS, turbop>(
                    padded_dim, max_degree, npoints, &qf, &qb, d_qg_data,
                    row_offset, neighbor_offset, code_offset, sign_offset, factor_offset,
                    HASH_TABLE, bitlen, PARENT_NODE_LIST, PARENT_DISTANCE_LIST,
                    keep_expanding, parent_work_count, &adaptive_state, current_kth_cutoff,
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, &compact_candidate_count,
                    candidate_collect_capacity, warp_stat_counts);
            }

            if (tidx() == 0 && compact_candidate_count > candidate_collect_capacity) {
                compact_candidate_count = candidate_collect_capacity;
            }
            __syncthreads();

            // [11] one warp per compacted child computes its exact distance
            for (int i = warpidx(); i < static_cast<int>(compact_candidate_count); i += WARPS_PER_BLOCK) {
                const INDEX_T child_id = CANDIDATE_INDEX[i];
                DISTANCE_T child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                if (laneidx() == 0) {
                    CANDIDATE_DISTANCE[i] = child_dist;
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // [12] merge the final candidates into the beam
    dispatch_beam_management(
        ALL_INDEX, ALL_DISTANCE, CANDIDATE_RADIX_SCRATCH, candidate_buffer_size, padded_beam_size, false);
    __syncthreads();

    // [13] clear the visited table and reuse it to write the first K distinct beam ids, padding the rest
    hashtable_init(HASH_TABLE, bitlen);
    __syncthreads();

    if (tidx() == 0) {
        int output_count = 0;
        for (int i = 0; i < beam_sz && output_count < K; ++i) {
            const INDEX_T raw_result = TOP_K_INDEX[i];
            if (raw_result == MAX_INDEX) {
                continue;
            }
            const INDEX_T result = raw_result & 0x7fffffffu;
            if (hashtable_insert(HASH_TABLE, bitlen, result)) {
                d_results[query_id * K + output_count] = result;
                d_result_dists[query_id * K + output_count] = TOP_K_DISTANCE[i];
                ++output_count;
            }
        }
        for (int i = output_count; i < K; ++i) {
            d_results[query_id * K + i] = MAX_INDEX;
            d_result_dists[query_id * K + i] = FLT_MAX;
        }
        d_iters[query_id] = static_cast<uint32_t>(iter);
    }

}
