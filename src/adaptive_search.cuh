#pragma once

#include "adaptive_search_utils.cuh"

template <int CODEBITS>
static __device__ __forceinline__ void collect_phase2_degree32_candidates_warp_local(
    size_t padded_dim, int max_degree, size_t npoints,
    const QueryFactors* __restrict__ qf,
    const float* __restrict__ d_qg_data,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t factor_offset,
    INDEX_T* __restrict__ hash_table, int bitlen,
    const INDEX_T* __restrict__ parent_node_list,
    const DISTANCE_T* __restrict__ parent_distance_list,
    uint32_t keep_expanding, uint32_t parent_work_count,
    const GPUAdaptiveSearchState* __restrict__ adaptive_state,
    DISTANCE_T current_kth_cutoff,
    INDEX_T* __restrict__ candidate_index,
    DISTANCE_T* __restrict__ candidate_distance,
    uint32_t* __restrict__ compact_candidate_count,
    uint32_t candidate_collect_capacity,
    uint32_t* __restrict__ warp_stat_counts)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));

    for (int task_base = 0; task_base < static_cast<int>(keep_expanding); task_base += WARPS_PER_BLOCK) {
        const int task = task_base + warp_id;
        const bool task_in_range = task < static_cast<int>(keep_expanding);
        const uint32_t parent_idx = task_in_range ? static_cast<uint32_t>(task) : 0u;
        const INDEX_T parent_node = task_in_range ? parent_node_list[parent_idx] : MAX_INDEX;
        const bool task_active = task_in_range && parent_node != MAX_INDEX;
        const uint32_t neighbor_idx = static_cast<uint32_t>(lane_id);

        INDEX_T child_id = MAX_INDEX;
        float triple_x = FLT_MAX;
        float factor_dq = 0.0f;
        float factor_vq = 0.0f;
        const uint8_t* parent_code_base = nullptr;
        bool valid_candidate = false;

        if (task_active && neighbor_idx < parent_work_count) {
            const float* parent_row = d_qg_data + static_cast<size_t>(parent_node) * row_offset;
            const vidType* parent_neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);
            const float* parent_factors = parent_row + factor_offset;
            parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);

            child_id = parent_neighbors[neighbor_idx];
            valid_candidate = child_id != MAX_INDEX && child_id < npoints && !hashtable_contains(hash_table, bitlen, child_id);
            if (valid_candidate) {
                triple_x = parent_factors[neighbor_idx];
                factor_dq = parent_factors[max_degree + neighbor_idx];
                factor_vq = parent_factors[2 * max_degree + neighbor_idx];
            }
        }

        const int keep_count = task_active ? keep_count_for_expander(parent_work_count, *adaptive_state) : 0;
        const uint32_t raw_valid_mask = __ballot_sync(FULL_MASK, valid_candidate);
        const uint32_t raw_valid_count = __popc(raw_valid_mask);
        const bool keep_all_valid = keep_count >= static_cast<int>(raw_valid_count);

        DISTANCE_T est_dist = FLT_MAX;
        if (!keep_all_valid && valid_candidate) {
            est_dist = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                *qf, parent_code_base, static_cast<int>(neighbor_idx), triple_x, factor_dq, factor_vq,
                parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            valid_candidate = isfinite(est_dist);
        }

        int effective_keep_count = keep_count;
        if (!keep_all_valid) {
            const DISTANCE_T kth_cutoff = current_kth_cutoff;
            const bool kth_cutoff_valid = isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
            const uint32_t near_cutoff_mask =
                __ballot_sync(FULL_MASK, valid_candidate && kth_cutoff_valid && est_dist <= kth_cutoff);
            effective_keep_count = keep_count_for_expander(
                parent_work_count, *adaptive_state, kth_cutoff_valid, __popc(near_cutoff_mask));
        }

        const bool keep_lane = keep_all_valid
            ? valid_candidate
            : warp_keep_topk_smallest_f32(est_dist, valid_candidate, effective_keep_count);
        uint32_t inserted = 0;
        if (keep_lane) {
            inserted = hashtable_insert(hash_table, bitlen, child_id);
        }

        const uint32_t accepted_mask = __ballot_sync(FULL_MASK, inserted != 0);
        const uint32_t accepted_count = __popc(accepted_mask);
        const uint32_t base_slot = allocate_warp_compact_slots(
            accepted_count, compact_candidate_count, warp_stat_counts, warp_stat_counts + WARPS_PER_BLOCK);
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const uint32_t lane_rank = __popc(accepted_mask & lane_mask);
        if (inserted) {
            const uint32_t slot = base_slot + lane_rank;
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child_id;
                candidate_distance[slot] = FLT_MAX;
            }
        }
    }
}

template <int CODEBITS>
static __device__ __noinline__ void collect_phase2_degree64_candidates_warp_local(
    size_t padded_dim, int max_degree, size_t npoints,
    const QueryFactors* __restrict__ qf,
    const float* __restrict__ d_qg_data,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t factor_offset,
    INDEX_T* __restrict__ hash_table, int bitlen,
    const INDEX_T* __restrict__ parent_node_list,
    const DISTANCE_T* __restrict__ parent_distance_list,
    uint32_t keep_expanding, uint32_t parent_work_count,
    const GPUAdaptiveSearchState* __restrict__ adaptive_state,
    DISTANCE_T current_kth_cutoff,
    INDEX_T* __restrict__ candidate_index,
    DISTANCE_T* __restrict__ candidate_distance,
    uint32_t* __restrict__ compact_candidate_count,
    uint32_t candidate_collect_capacity,
    uint32_t* __restrict__ warp_stat_counts)
{
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));

    for (int task_base = 0; task_base < static_cast<int>(keep_expanding); task_base += WARPS_PER_BLOCK) {
        const uint32_t parent_idx = static_cast<uint32_t>(task_base + warp_id);
        const INDEX_T parent_node = (parent_idx < keep_expanding) ? parent_node_list[parent_idx] : MAX_INDEX;
        const bool task_active = parent_idx < keep_expanding && parent_node != MAX_INDEX;
        const float* parent_row = task_active
            ? d_qg_data + static_cast<size_t>(parent_node) * row_offset
            : nullptr;
        const vidType* parent_neighbors = task_active ? reinterpret_cast<const vidType*>(parent_row + neighbor_offset) : nullptr;
        const float* parent_factors = task_active ? parent_row + factor_offset : nullptr;
        const uint8_t* parent_code_base = task_active ? reinterpret_cast<const uint8_t*>(parent_row + code_offset) : nullptr;

        const uint32_t neighbor_idx0 = static_cast<uint32_t>(lane_id);
        const uint32_t neighbor_idx1 = static_cast<uint32_t>(lane_id + WARP_SIZE);
        const bool active0 = task_active && neighbor_idx0 < parent_work_count;
        const bool active1 = task_active && neighbor_idx1 < parent_work_count;

        INDEX_T child0 = MAX_INDEX;
        INDEX_T child1 = MAX_INDEX;
        bool valid0 = false;
        bool valid1 = false;
        if (active0) {
            child0 = parent_neighbors[neighbor_idx0];
            valid0 = child0 != MAX_INDEX && child0 < npoints && !hashtable_contains(hash_table, bitlen, child0);
        }
        if (active1) {
            child1 = parent_neighbors[neighbor_idx1];
            valid1 = child1 != MAX_INDEX && child1 < npoints && !hashtable_contains(hash_table, bitlen, child1);
        }

        const int keep_count = task_active ? keep_count_for_expander(parent_work_count, *adaptive_state) : 0;
        const uint32_t raw_valid_count =
            __popc(__ballot_sync(FULL_MASK, valid0)) +
            __popc(__ballot_sync(FULL_MASK, valid1));
        const bool keep_all_valid = keep_count >= static_cast<int>(raw_valid_count);

        DISTANCE_T est0 = FLT_MAX;
        DISTANCE_T est1 = FLT_MAX;
        if (!keep_all_valid && valid0) {
            est0 = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                *qf, parent_code_base, static_cast<int>(neighbor_idx0),
                parent_factors[neighbor_idx0],
                parent_factors[max_degree + neighbor_idx0],
                parent_factors[2 * max_degree + neighbor_idx0],
                parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            valid0 = isfinite(est0);
        }
        if (!keep_all_valid && valid1) {
            est1 = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                *qf, parent_code_base, static_cast<int>(neighbor_idx1),
                parent_factors[neighbor_idx1],
                parent_factors[max_degree + neighbor_idx1],
                parent_factors[2 * max_degree + neighbor_idx1],
                parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            valid1 = isfinite(est1);
        }

        int effective_keep_count = keep_count;
        if (!keep_all_valid) {
            const DISTANCE_T kth_cutoff = current_kth_cutoff;
            const bool kth_cutoff_valid = isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
            const uint32_t near_count =
                __popc(__ballot_sync(FULL_MASK, valid0 && kth_cutoff_valid && est0 <= kth_cutoff)) +
                __popc(__ballot_sync(FULL_MASK, valid1 && kth_cutoff_valid && est1 <= kth_cutoff));
            effective_keep_count = keep_count_for_expander(parent_work_count, *adaptive_state, kth_cutoff_valid, near_count);
        }

        const bool keep0 = keep_all_valid ? valid0 : warp_keep_topk_smallest_pair_f32(
            est0, valid0, static_cast<int>(neighbor_idx0), est0, valid0, est1, valid1, effective_keep_count);
        const bool keep1 = keep_all_valid ? valid1 : warp_keep_topk_smallest_pair_f32(
            est1, valid1, static_cast<int>(neighbor_idx1), est0, valid0, est1, valid1, effective_keep_count);

        uint32_t inserted0 = 0;
        uint32_t inserted1 = 0;
        if (keep0) inserted0 = hashtable_insert(hash_table, bitlen, child0);
        if (keep1) inserted1 = hashtable_insert(hash_table, bitlen, child1);

        const uint32_t accepted_mask0 = __ballot_sync(FULL_MASK, inserted0 != 0);
        const uint32_t accepted_count0 = __popc(accepted_mask0);
        const uint32_t base_slot0 = allocate_warp_compact_slots(
            accepted_count0, compact_candidate_count, warp_stat_counts, warp_stat_counts + WARPS_PER_BLOCK);
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        const uint32_t lane_rank0 = __popc(accepted_mask0 & lane_mask);
        if (inserted0) {
            const uint32_t slot = base_slot0 + lane_rank0;
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child0;
                candidate_distance[slot] = FLT_MAX;
            }
        }

        const uint32_t accepted_mask1 = __ballot_sync(FULL_MASK, inserted1 != 0);
        const uint32_t accepted_count1 = __popc(accepted_mask1);
        const uint32_t base_slot1 = allocate_warp_compact_slots(
            accepted_count1, compact_candidate_count, warp_stat_counts, warp_stat_counts + WARPS_PER_BLOCK);
        const uint32_t lane_rank1 = __popc(accepted_mask1 & lane_mask);
        if (inserted1) {
            const uint32_t slot = base_slot1 + lane_rank1;
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child1;
                candidate_distance[slot] = FLT_MAX;
            }
        }
    }
}

/** SHAME(TALLFUNC) SHAME(MANYARG) */
template <int CODEBITS, bool turboq>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void QuantizedPrunedBeamSearch(
    int K, int nq, int dim, int beam_sz, int bitlen, int max_degree, size_t npoints,
    const float* __restrict__ d_queries, const float* __restrict__ d_qg_data, const float* __restrict__ d_qg_signs,
    const float* __restrict__ d_qg_levels,
    vidType* __restrict__ d_results, float* __restrict__ d_result_dists,
    vidType entry_point, size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t factor_offset,
    int max_iter_by_beam, float phase2_rho, bool use_ip)
{
    const int query_id = blockIdx.x;
    if (query_id >= nq) return;
    const float* query = d_queries + query_id * dim;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const uint32_t candidate_buffer_size = round_up_power2_u32(candidate_buffer_capacity(static_cast<uint32_t>(max_degree)));
    uint32_t candidate_collect_capacity = static_cast<uint32_t>(BUFFER_BOUND);
    if (candidate_collect_capacity > candidate_buffer_size) candidate_collect_capacity = candidate_buffer_size;
    const uint32_t padded_beam_size = effective_sort_beam_size(beam_sz);

    // initialize the shared memory
    extern __shared__ char shared_memory[];
    INDEX_T* ALL_INDEX = reinterpret_cast<INDEX_T*>(shared_memory);
    INDEX_T* TOP_K_INDEX = ALL_INDEX;
    INDEX_T* CANDIDATE_INDEX = ALL_INDEX + padded_beam_size;

    const uint32_t result_buffer_size = padded_beam_size + candidate_buffer_size;
    DISTANCE_T* ALL_DISTANCE = reinterpret_cast<DISTANCE_T*>(ALL_INDEX + result_buffer_size);
    DISTANCE_T* TOP_K_DISTANCE = ALL_DISTANCE;
    DISTANCE_T* CANDIDATE_DISTANCE = ALL_DISTANCE + padded_beam_size;

    INDEX_T* HASH_TABLE = reinterpret_cast<INDEX_T*>(ALL_DISTANCE + result_buffer_size);
    INDEX_T* PARENT_LIST = HASH_TABLE + hashtable_getsize(bitlen);
    INDEX_T* PARENT_NODE_LIST = PARENT_LIST + SEARCH_WIDTH;
    DISTANCE_T* PARENT_DISTANCE_LIST = reinterpret_cast<DISTANCE_T*>(PARENT_NODE_LIST + SEARCH_WIDTH);
    DATA_T* QUERY_BUFFER = reinterpret_cast<DATA_T*>(PARENT_DISTANCE_LIST + SEARCH_WIDTH);

    float* ROTATED_QUERY_BUFFER = reinterpret_cast<float*>(QUERY_BUFFER + dim);
    uint8_t* LUT_BUFFER = reinterpret_cast<uint8_t*>(ROTATED_QUERY_BUFFER + padded_dim);
    char* query_factor_base = reinterpret_cast<char*>(LUT_BUFFER + quant_lutbytes(padded_dim, CODEBITS));
    float* low_val = reinterpret_cast<float*>(query_factor_base);
    float* high_val = low_val + 1;
    float* width = high_val + 1;
    int32_t* sum_q = reinterpret_cast<int32_t*>(width + 1);
    char* tail_base = reinterpret_cast<char*>(sum_q + 1);
    // reserve optional scratch space for large-beam top-k merge
    INDEX_T* MERGED_TOPK_INDEX = nullptr;
    DISTANCE_T* MERGED_TOPK_DISTANCE = nullptr;
    if (topk_external_merge_scratch_needed(padded_beam_size)) {
        MERGED_TOPK_INDEX = allocate_shared_tail_array<INDEX_T>(tail_base, padded_beam_size);
        MERGED_TOPK_DISTANCE = allocate_shared_tail_array<DISTANCE_T>(tail_base, padded_beam_size);
    }

    __shared__ QueryFactors qf;
    __shared__ uint32_t keep_expanding;
    __shared__ GPUAdaptiveSearchState adaptive_state;
    __shared__ uint32_t compact_candidate_count;
    __shared__ DISTANCE_T current_kth_cutoff;
    __shared__ uint32_t shared_kth_near_count;
    __shared__ uint32_t warp_stat_counts[2 * WARPS_PER_BLOCK + 1];

    // initialize the query and search state
    if (tid == 0) {
        qf.rotated_query = ROTATED_QUERY_BUFFER;
        qf.quantized_query = nullptr;
        qf.lut = LUT_BUFFER;
        qf.low_val = FLT_MAX;
        qf.high_val = -FLT_MAX;
        qf.width = 0.0f;
        qf.sum_q = 0;
        adaptive_state.adaptive_spec_degree = static_cast<int>(PHASE1_THETA);
        adaptive_state.adaptive_rho = initial_rho_for_bound(phase2_rho);
        adaptive_state.policy_iters = 0;
        adaptive_state.top1_node = MAX_INDEX;
        adaptive_state.top1_stall_iters = 0;
        adaptive_state.warmup_done = false;
        adaptive_state.entry_distance = FLT_MAX;
        adaptive_state.last_expander_distance = FLT_MAX;
        adaptive_state.current_expander_distance = FLT_MAX;
        compact_candidate_count = 0;
    }

    // initialize the hash table
    hashtable_init(HASH_TABLE, bitlen);

    // initialize shared memory buffers for query, top-k, and candidate lists
    for (int i = tid; i < dim; i += blockDim.x) {
        QUERY_BUFFER[i] = query[i];
    }
    for (int i = tid; i < static_cast<int>(padded_beam_size); i += blockDim.x) {
        TOP_K_INDEX[i] = MAX_INDEX;
        TOP_K_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < candidate_buffer_size; i += blockDim.x) {
        CANDIDATE_INDEX[i] = MAX_INDEX;
        CANDIDATE_DISTANCE[i] = FLT_MAX;
    }
    for (int i = tid; i < SEARCH_WIDTH; i += blockDim.x) {
        PARENT_NODE_LIST[i] = MAX_INDEX;
        PARENT_DISTANCE_LIST[i] = FLT_MAX;
    }
    __syncthreads();

    // only the first slot in candidate list is actually initialized with valid entry point
    if (warp_id == 0) {
        DISTANCE_T entry_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(entry_point) * row_offset, use_ip);
        if (lane_id == 0) {
            CANDIDATE_INDEX[0] = entry_point;
            CANDIDATE_DISTANCE[0] = entry_dist;
            adaptive_state.entry_distance = entry_dist;
            hashtable_insert(HASH_TABLE, bitlen, entry_point);
        }
    }
    __syncthreads();

    // query preparation, quantize the query and prepare the scan LUT for fast neighbor estimation
    if constexpr (turboq) {
        turboq_prepare_lut_gpu<CODEBITS>(QUERY_BUFFER, qf, d_qg_signs, d_qg_levels, dim, padded_dim);
    } else {
        query_prepare_lut_gpu(QUERY_BUFFER, qf, d_qg_signs, dim, padded_dim);
    }
    __syncthreads();

    // loop end condition: either entire topK expanded, or reach max iteration
    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {

        if (iter >= max_iter_by_beam) {
            break;
        }

        // periodically reset the hash table
        const bool reset_hash = ((iter + 1) % SMALL_HASH_RESET_INTERVAL == 0);

        // sort and merge existing candidates into the beam
        const uint32_t merge_candidate_count = candidate_buffer_size;
        dispatch_topk_candidate_sort_and_merge(
            ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
            merge_candidate_count, padded_beam_size, (iter == 0));
        __syncthreads();

        // clear the beam stall part and candidate buffer to avoid stale entries
        for (uint32_t i = beam_sz + tid; i < padded_beam_size; i += blockDim.x) {
            TOP_K_INDEX[i] = MAX_INDEX;
            TOP_K_DISTANCE[i] = FLT_MAX;
        }
        for (int i = tid; i < candidate_buffer_size; i += blockDim.x) {
            CANDIDATE_INDEX[i] = MAX_INDEX;
            CANDIDATE_DISTANCE[i] = FLT_MAX;
        }
        for (int i = tid; i < SEARCH_WIDTH; i += blockDim.x) {
            PARENT_NODE_LIST[i] = MAX_INDEX;
            PARENT_DISTANCE_LIST[i] = FLT_MAX;
        }
        if (reset_hash) {
            hashtable_init(HASH_TABLE, bitlen);
            __syncthreads();
            hashtable_restore(HASH_TABLE, bitlen, TOP_K_INDEX, beam_sz);
        }
        __syncthreads();
        if (tid == 0) {
            compact_candidate_count = 0;
            current_kth_cutoff = get_current_kth_cutoff(K, beam_sz, TOP_K_INDEX, TOP_K_DISTANCE);
        }
        __syncthreads();

        // update adaptive parameters before expander selection
        if (tid == 0) {
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
        __syncthreads();

        // select expanders from the current beam based on the adaptive speculation degree
        if (warp_id == 0) {
            keep_expanding = pick_expanders(adaptive_state.adaptive_spec_degree, PARENT_LIST, beam_sz, TOP_K_INDEX);
        }
        __syncthreads();

        if (!keep_expanding) {
            break;
        }

        // collect the expanders into the parent list along with their exact distance
        for (int p = tid; p < SEARCH_WIDTH; p += blockDim.x) {
            if (p < static_cast<int>(keep_expanding)) {
                const uint32_t parent_pos = PARENT_LIST[p];
                const INDEX_T parent_node = TOP_K_INDEX[parent_pos] & 0x7fffffffu;
                PARENT_NODE_LIST[p] = parent_node;
                PARENT_DISTANCE_LIST[p] = TOP_K_DISTANCE[parent_pos];
            }
        }
        __syncthreads();

        if (tid == 0) {
            record_selected_expander_dist(keep_expanding, PARENT_DISTANCE_LIST, &adaptive_state);
        }
        __syncthreads();

        // use shared-memory pruning for one expander, or warp-local pruning for multiple expanders
        const bool use_local_gate = adaptive_state.adaptive_spec_degree > 1;
        if (!use_local_gate) {
            const uint32_t candidate_work_count = (static_cast<uint32_t>(max_degree) < candidate_buffer_size) ? static_cast<uint32_t>(max_degree) : candidate_buffer_size;
            const INDEX_T parent_node = PARENT_NODE_LIST[0];
            const bool parent_valid = parent_node != MAX_INDEX;
            const float* parent_row = parent_valid ? d_qg_data + static_cast<size_t>(parent_node) * row_offset : nullptr;
            const vidType* parent_neighbors = parent_valid ? reinterpret_cast<const vidType*>(parent_row + neighbor_offset) : nullptr;

            // gather valid neighbors into shared memory
            for (int i = tid; i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
                INDEX_T child_id = MAX_INDEX;
                if (i < static_cast<int>(candidate_work_count) && parent_valid) {
                    child_id = parent_neighbors[i];
                    if (child_id >= npoints || hashtable_contains(HASH_TABLE, bitlen, child_id)) {
                        child_id = MAX_INDEX;
                    }
                }
                CANDIDATE_INDEX[i] = child_id;
                CANDIDATE_DISTANCE[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0.0f;
            }
            __syncthreads();

            // skip estimation when all valid neighbors fit within the keep budget
            const int keep_count = keep_count_for_expander(candidate_work_count, adaptive_state);
            const uint32_t valid_candidate_count = count_valid_candidates_warp_reduced(
                CANDIDATE_INDEX, CANDIDATE_DISTANCE, static_cast<int>(candidate_work_count), MAX_INDEX, warp_stat_counts);
            const bool keep_all_valid = keep_count >= static_cast<int>(valid_candidate_count);
            int effective_keep_count = keep_count;

            // estimate and prune only when valid candidates exceed the keep budget
            if (!keep_all_valid) {
                const float* parent_factors = parent_row + factor_offset;
                const uint8_t* parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);
                const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));
                constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
                constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
                const int group_lane = lane_id & (lanes_per_neighbor - 1);
                const int warp_group = lane_id / lanes_per_neighbor;

                for (int base = warp_id * groups_per_warp; base < static_cast<int>(candidate_work_count); base += WARPS_PER_BLOCK * groups_per_warp) {
                    const int i = base + warp_group;
                    bool valid_candidate = i < static_cast<int>(candidate_work_count) && CANDIDATE_INDEX[i] != MAX_INDEX;
                    DISTANCE_T est_dist = FLT_MAX;
                    if (valid_candidate) {
                        const float* triple_x = parent_factors + i;
                        const float* factor_dq = parent_factors + max_degree + i;
                        const float* factor_vq = parent_factors + 2 * max_degree + i;
                        est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor, CODEBITS>(
                            qf, parent_code_base, i, triple_x, factor_dq, factor_vq,
                            PARENT_DISTANCE_LIST[0], padded_dim, bytes_per_neighbor);
                        valid_candidate = isfinite(est_dist);
                        if (group_lane == 0) {
                            CANDIDATE_DISTANCE[i] = valid_candidate ? est_dist : FLT_MAX;
                            if (!valid_candidate) {
                                CANDIDATE_INDEX[i] = MAX_INDEX;
                            }
                        }
                    }
                }
                __syncthreads();

                // tighten the keep count using the current kth-result cutoff when available
                const DISTANCE_T kth_cutoff = current_kth_cutoff;
                const bool kth_cutoff_valid = isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
                if (tid == 0) {
                    shared_kth_near_count = 0;
                }
                __syncthreads();

                if (kth_cutoff_valid) {
                    uint32_t thread_near_count = 0;
                    for (int i = tid; i < static_cast<int>(candidate_work_count); i += blockDim.x) {
                        const DISTANCE_T dist = CANDIDATE_DISTANCE[i];
                        if (CANDIDATE_INDEX[i] != MAX_INDEX && isfinite(dist) && dist <= kth_cutoff) {
                            thread_near_count++;
                        }
                    }
                    const uint32_t block_near_count = block_reduce_sum_u32(thread_near_count, warp_stat_counts);
                    if (tid == 0) shared_kth_near_count = block_near_count;
                }
                __syncthreads();

                effective_keep_count = keep_count_for_expander(candidate_work_count, adaptive_state, kth_cutoff_valid, shared_kth_near_count);

                // sort estimated candidates and invalidate entries past the keep budget
                dispatch_candidate_bitonic_sort(CANDIDATE_INDEX, CANDIDATE_DISTANCE, candidate_buffer_size);
                __syncthreads();

                for (int i = tid; i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
                    if (i >= effective_keep_count) {
                        CANDIDATE_INDEX[i] = MAX_INDEX;
                        CANDIDATE_DISTANCE[i] = FLT_MAX;
                    }
                }
                __syncthreads();
            }

            // insert kept children and compute their exact distances for the next merge
            const int admit_scan_count = keep_all_valid ? static_cast<int>(candidate_work_count) : effective_keep_count;
            for (int i = warp_id; i < admit_scan_count; i += WARPS_PER_BLOCK) {
                INDEX_T child_id = MAX_INDEX;
                uint32_t inserted = 0;
                if (lane_id == 0) {
                    child_id = CANDIDATE_INDEX[i];
                    if (child_id != MAX_INDEX && CANDIDATE_DISTANCE[i] < FLT_MAX) {
                        inserted = hashtable_insert(HASH_TABLE, bitlen, child_id);
                    }
                    if (!inserted) {
                        child_id = MAX_INDEX;
                        CANDIDATE_INDEX[i] = MAX_INDEX;
                        CANDIDATE_DISTANCE[i] = FLT_MAX;
                    }
                }
                child_id = SHFL(child_id, 0);
                inserted = SHFL(inserted, 0);

                DISTANCE_T child_dist = FLT_MAX;
                if (inserted) {
                    child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                }
                if (lane_id == 0 && inserted) {
                    CANDIDATE_DISTANCE[i] = child_dist;
	                }
	            }
        } else {
            const uint32_t parent_work_count = (static_cast<uint32_t>(max_degree) < candidate_buffer_size) ? static_cast<uint32_t>(max_degree) : candidate_buffer_size;

            if (parent_work_count <= WARP_SIZE) {
                collect_phase2_degree32_candidates_warp_local<CODEBITS>(
                    padded_dim, max_degree, npoints, &qf, d_qg_data,
                    row_offset, neighbor_offset, code_offset, factor_offset,
                    HASH_TABLE, bitlen, PARENT_NODE_LIST, PARENT_DISTANCE_LIST,
                    keep_expanding, parent_work_count, &adaptive_state, current_kth_cutoff,
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, &compact_candidate_count,
                    candidate_collect_capacity, warp_stat_counts);
            } else {
                collect_phase2_degree64_candidates_warp_local<CODEBITS>(
                    padded_dim, max_degree, npoints, &qf, d_qg_data,
                    row_offset, neighbor_offset, code_offset, factor_offset,
                    HASH_TABLE, bitlen, PARENT_NODE_LIST, PARENT_DISTANCE_LIST,
                    keep_expanding, parent_work_count, &adaptive_state, current_kth_cutoff,
                    CANDIDATE_INDEX, CANDIDATE_DISTANCE, &compact_candidate_count,
                    candidate_collect_capacity, warp_stat_counts);
            }
            __syncthreads();

            if (tid == 0 && compact_candidate_count > candidate_collect_capacity) {
                compact_candidate_count = candidate_collect_capacity;
            }
            __syncthreads();

            // compute exact distances for compacted speculative candidates
            const uint32_t compact_count = compact_candidate_count;
            for (int i = warp_id; i < static_cast<int>(compact_count); i += WARPS_PER_BLOCK) {
                const INDEX_T child_id = CANDIDATE_INDEX[i];
                DISTANCE_T child_dist = FLT_MAX;
                child_dist = warp_distance(dim, QUERY_BUFFER, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
                if (lane_id == 0) {
                    CANDIDATE_DISTANCE[i] = child_dist;
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // merge the final candidate batch into the beam
    dispatch_topk_candidate_sort_and_merge(
        ALL_INDEX, ALL_DISTANCE, MERGED_TOPK_INDEX, MERGED_TOPK_DISTANCE,
        candidate_buffer_size, padded_beam_size, false);
    __syncthreads();

    hashtable_init(HASH_TABLE, bitlen);
    __syncthreads();

    // write top-k results
    if (tid == 0) {
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
    }

}
