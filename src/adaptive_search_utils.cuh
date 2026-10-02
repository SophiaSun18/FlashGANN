#pragma once

#include <cuda_runtime.h>

#include "include/utils.cuh"
#include "include/hash_table.cuh"
#include "include/distance.cuh"
#include "include/beam_management.cuh"
#include "include/quant.cuh"
#include "adaptive_search_config.cuh"

/*-------------------------------------------- common --------------------------------------------*/
/**
 * @brief Barrier over a subset of the block's warps, on a named hardware barrier instead of barrier 0.
 * @param id named barrier id in 1 .. 15, since 0 is the one __syncthreads uses
 * @param threads participating threads, a multiple of the warp size
 */
static __device__ __forceinline__ void namedsync(int id, int threads) {
    asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(threads) : "memory");
}

__host__ __device__ inline int clamp_int(int value, int lo, int hi) {
    if (value < lo)
        return lo;
    if (value > hi)
        return hi;
    return value;
}

__host__ __device__ inline float clamp_float(float value, float lo, float hi) {
    if (value < lo)
        return lo;
    if (value > hi)
        return hi;
    return value;
}

__host__ __device__ inline uint32_t keep_count_per_parent(uint32_t max_degree, float rho) {
    float keep_fraction = 1.0f - rho;
    if (keep_fraction < 0.0f)
        keep_fraction = 0.0f;
    if (keep_fraction > 1.0f)
        keep_fraction = 1.0f;

    uint32_t keep_count = static_cast<uint32_t>(ceilf(static_cast<float>(max_degree) * keep_fraction));
    if (keep_count < 1)
        keep_count = 1;
    if (keep_count > max_degree)
        keep_count = max_degree;
    return keep_count;
}

/*-------------------------------------------- warp and block helpers --------------------------------------------*/
static __device__ __forceinline__ uint32_t block_reduce_sum_u32(uint32_t thread_count, uint32_t *warp_counts) { // SHAME(WIDEFUNC)
    const int lane_id = tidx() & (WARP_SIZE - 1);
    const int warp_id = tidx() / WARP_SIZE;

    uint32_t sum = thread_count;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if (lane_id == 0)
        warp_counts[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t total = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(FULL_MASK, total, offset);
        }
        if (lane_id == 0)
            warp_counts[0] = total;
    }
    __syncthreads();
    return warp_counts[0];
}

static __device__ __forceinline__ uint32_t allocate_warp_compact_slots(
    uint32_t accepted_count, uint32_t *compact_count, uint32_t *warp_counts, uint32_t *warp_bases) {
    const int lane_id = tidx() & (WARP_SIZE - 1);
    const int warp_id = tidx() / WARP_SIZE;

    if (lane_id == 0)
        warp_counts[warp_id] = accepted_count;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t count = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        uint32_t inclusive = count;
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            const uint32_t other = __shfl_up_sync(FULL_MASK, inclusive, offset);
            if (lane_id >= offset)
                inclusive += other;
        }
        if (lane_id < WARPS_PER_BLOCK) warp_bases[lane_id] = inclusive - count;
        const uint32_t round_total = __shfl_sync(FULL_MASK, inclusive, WARPS_PER_BLOCK - 1);
        if (lane_id == 0) {
            const uint32_t block_base = *compact_count;
            *compact_count = block_base + round_total;
            warp_bases[WARPS_PER_BLOCK] = block_base;
        }
    }
    __syncthreads();
    return warp_bases[WARPS_PER_BLOCK] + warp_bases[warp_id];
}

template <typename INDEX_T, typename DISTANCE_T>
static __device__ __forceinline__ uint32_t count_valid_candidates_warp_reduced(
    const INDEX_T *candidate_index, const DISTANCE_T *candidate_distance,
    int candidate_count, INDEX_T invalid_index, uint32_t *warp_counts) {
    const int lane_id = tidx() & (WARP_SIZE - 1);
    const int warp_id = tidx() / WARP_SIZE;

    if (lane_id == 0)
        warp_counts[warp_id] = 0;
    __syncthreads();

    uint32_t warp_valid_count = 0;
    for (int base = warp_id * WARP_SIZE; base < candidate_count; base += WARPS_PER_BLOCK * WARP_SIZE) {
        const int i = base + lane_id;
        const bool valid = i < candidate_count && candidate_index[i] != invalid_index && candidate_distance[i] < FLT_MAX;
        warp_valid_count += __popc(__ballot_sync(FULL_MASK, valid));
    }

    if (lane_id == 0)
        warp_counts[warp_id] = warp_valid_count;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t total = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(FULL_MASK, total, offset);
        }
        if (lane_id == 0)
            warp_counts[0] = total;
    }
    __syncthreads();

    return warp_counts[0];
}

/**
 * @brief Whether this lane holds one of the keep_k smallest valid values, lower lane first on ties.
 *
 * @param value this lane's value
 * @param valid whether this lane takes part
 * @param keep_k number of values to keep
 * @return whether this lane is kept
 */
static __device__ __forceinline__ bool warp_keep_topk_smallest_f32(float value, bool valid, int keep_k) {
    const int lane_id = tidx() & (WARP_SIZE - 1);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // [1] order-preserving key; + 0.0f folds -0.0f into +0.0f, invalid lanes take the largest key
    const uint32_t bits = __float_as_uint(value + 0.0f);
    const uint32_t flip = (bits & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    uint32_t key = valid ? (bits ^ flip) : 0xffffffffu;
    const int picks = min(keep_k, __popc(__ballot_sync(FULL_MASK, valid)));

    // [2] each round keeps the lowest lane that holds the smallest remaining key
    bool kept = false;
    for (int round = 0; round < picks; ++round) {
        const uint32_t smallest = __reduce_min_sync(FULL_MASK, key);
        const uint32_t owners = __ballot_sync(FULL_MASK, key == smallest);
        const bool win = lane_id == __ffs(owners) - 1;
        kept = kept || win;
        key = win ? 0xffffffffu : key;
    }
    return kept;
#else
    const int valid_int = valid ? 1 : 0;
    const float my_value = valid ? value : FLT_MAX;
    int rank = 0;

#pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, my_value, src);
        const int other_valid = __shfl_sync(FULL_MASK, valid_int, src);
        if (other_valid && (other_value < my_value || (other_value == my_value && src < lane_id))) {
            rank++;
        }
    }

    return valid && keep_k > 0 && rank < keep_k;
#endif
}

/**
 * @brief Whether one of this lane's two values is among the keep_k smallest of the warp's 64.
 *
 * Lane l holds positions l and l + 32; lower position first on ties.
 *
 * @param value the value asked about, value0 or value1
 * @param valid whether that value takes part
 * @param position its position, lane or lane + 32
 * @param value0 this lane's value at position lane
 * @param valid0 whether value0 takes part
 * @param value1 this lane's value at position lane + 32
 * @param valid1 whether value1 takes part
 * @param keep_k number of values to keep
 * @return whether the value is kept
 */
static __device__ __forceinline__ bool warp_keep_topk_smallest_pair_f32(
    float value, bool valid, int position,
    float value0, bool valid0, float value1, bool valid1, int keep_k) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int lane_id = tidx() & (WARP_SIZE - 1);
    // [1] order-preserving keys as in warp_keep_topk_smallest_f32, invalid values take the top key
    const uint32_t bits0 = __float_as_uint(value0 + 0.0f);
    const uint32_t bits1 = __float_as_uint(value1 + 0.0f);
    const uint32_t flip0 = (bits0 & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    const uint32_t flip1 = (bits1 & 0x80000000u) ? 0xffffffffu : 0x80000000u;
    uint32_t key0 = valid0 ? (bits0 ^ flip0) : 0xffffffffu;
    uint32_t key1 = valid1 ? (bits1 ^ flip1) : 0xffffffffu;
    const int valid_count =
        __popc(__ballot_sync(FULL_MASK, valid0)) + __popc(__ballot_sync(FULL_MASK, valid1));
    const int picks = min(keep_k, valid_count);

    // [2] each round keeps the lowest position that holds the smallest remaining key
    bool kept0 = false;
    bool kept1 = false;
    for (int round = 0; round < picks; ++round) {
        const uint32_t smallest = __reduce_min_sync(FULL_MASK, min(key0, key1));
        const uint32_t owners0 = __ballot_sync(FULL_MASK, key0 == smallest);
        const uint32_t owners1 = __ballot_sync(FULL_MASK, key1 == smallest);
        const bool first = owners0 != 0;
        const bool win0 = first && lane_id == __ffs(owners0) - 1;
        const bool win1 = !first && lane_id == __ffs(owners1) - 1;
        kept0 = kept0 || win0;
        kept1 = kept1 || win1;
        key0 = win0 ? 0xffffffffu : key0;
        key1 = win1 ? 0xffffffffu : key1;
    }
    return valid && (position < WARP_SIZE ? kept0 : kept1);
#else
    const float my_value = valid ? value : FLT_MAX;
    const int valid0_int = valid0 ? 1 : 0;
    const int valid1_int = valid1 ? 1 : 0;
    int rank = 0;

#pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other0 = __shfl_sync(FULL_MASK, value0, src);
        const int other0_valid = __shfl_sync(FULL_MASK, valid0_int, src);
        const int other0_pos = src;
        if (other0_valid && (other0 < my_value || (other0 == my_value && other0_pos < position))) {
            rank++;
        }

        const float other1 = __shfl_sync(FULL_MASK, value1, src);
        const int other1_valid = __shfl_sync(FULL_MASK, valid1_int, src);
        const int other1_pos = src + WARP_SIZE;
        if (other1_valid && (other1 < my_value || (other1 == my_value && other1_pos < position))) {
            rank++;
        }
    }

    return valid && keep_k > 0 && rank < keep_k;
#endif
}

/*-------------------------------------------- rabitq --------------------------------------------*/
template <int CODEBITS = 1>
static __device__ inline DISTANCE_T scan_one_neighbor_lane_seq_lut_gpu(
    const QueryFactors &qf, const uint8_t *parent_code_base, int neighbor_idx,
    float triple_x, float factor_dq, float factor_vq,
    float exact_dist, int padded_dim, int bytes_per_neighbor) { // SHAME(MANYARG)
    if (triple_x == FLT_MAX)
        return exact_dist;

    const float result_float = lut_reduce<1, CODEBITS>(
        qf, parent_code_base, neighbor_idx, padded_dim, bytes_per_neighbor);

    DISTANCE_T est_dist = triple_x + exact_dist;
    est_dist += factor_dq * qf.width * result_float;
    est_dist += factor_vq * qf.low_val;
    return est_dist;
}

struct GPUAdaptiveSearchState {
    int adaptive_spec_degree;               // Current speculation degree.
    float adaptive_rho;                     // Current rho value.
    int policy_iters;                       // Number of policy updates.
    INDEX_T top1_node;                      // Last top-1 node.
    int top1_stall_iters;                   // Top-1 stall checks.
    bool warmup_done;                       // Whether AP has switched to phase 2.
    DISTANCE_T entry_distance;              // Entry-point distance.
    DISTANCE_T last_expander_distance;      // Previous expander distance.
    DISTANCE_T current_expander_distance;   // Current expander distance.
};

static __host__ __forceinline__ float compute_max_rho_bound(uint32_t degree) {
    if (degree <= static_cast<uint32_t>(MIN_PHASE2_KEEP)) return RHO_MIN;
    return 1.0f - static_cast<float>(MIN_PHASE2_KEEP) / static_cast<float>(degree);
}

static __device__ __forceinline__ DISTANCE_T get_current_kth_cutoff(
    int K, int beam_sz, const INDEX_T* topk_index, const DISTANCE_T* topk_distance) {
    if (topk_index[beam_sz - 1] == MAX_INDEX) return FLT_MAX;
    const uint32_t kth_pos = ((K < beam_sz) ? static_cast<uint32_t>(K) : static_cast<uint32_t>(beam_sz)) - 1;
    return topk_distance[kth_pos];
}

static __device__ __forceinline__ int update_head_stall_counter(
    const INDEX_T* topk_index, const GPUAdaptiveSearchState& previous_state, INDEX_T* current_top1) {
    *current_top1 = topk_index[0] & 0x7fffffffu;
    if (previous_state.policy_iters > 0 && *current_top1 == previous_state.top1_node) {
        return previous_state.top1_stall_iters + 1;
    }
    return 0;
}

static __device__ __forceinline__ void record_selected_expander_dist(
    uint32_t keep_expanding, const DISTANCE_T* parent_distance_list, GPUAdaptiveSearchState* state) {
    const DISTANCE_T selected_expander_distance = parent_distance_list[keep_expanding - 1];
    const bool have_previous_expander =
        isfinite(state->current_expander_distance) && state->current_expander_distance < FLT_MAX;
    state->last_expander_distance = have_previous_expander
        ? state->current_expander_distance : state->entry_distance;
    state->current_expander_distance = selected_expander_distance;
}

static __device__ __forceinline__ float update_expander_progress(
    const GPUAdaptiveSearchState& previous_state,
    DISTANCE_T* last_expander_distance, DISTANCE_T* current_expander_distance) {
    const DISTANCE_T entry_distance = previous_state.entry_distance;
    const bool have_last_expander =
        isfinite(previous_state.last_expander_distance) && previous_state.last_expander_distance < FLT_MAX;
    const bool have_current_expander =
        isfinite(previous_state.current_expander_distance) && previous_state.current_expander_distance < FLT_MAX;
    *last_expander_distance = have_last_expander ? previous_state.last_expander_distance : entry_distance;
    *current_expander_distance = have_current_expander
        ? previous_state.current_expander_distance : *last_expander_distance;

    const float denom = fabsf(static_cast<float>(entry_distance));
    if (!isfinite(*last_expander_distance) || !isfinite(*current_expander_distance) ||
        !isfinite(entry_distance) || denom <= 1.0e-20f) {
        return 0.0f;
    }

    const float progress = static_cast<float>(*last_expander_distance - *current_expander_distance) / denom;
    if (!isfinite(progress) || progress <= 0.0f) return 0.0f;
    return progress < 1.0f ? progress : 1.0f;
}

static __device__ __forceinline__ GPUAdaptiveSearchState apply_stage2_state(
    INDEX_T current_top1, int head_stall_iters, DISTANCE_T last_expander_distance, DISTANCE_T current_expander_distance,
    const GPUAdaptiveSearchState& previous_state, float phase2_rho) {
    GPUAdaptiveSearchState state = previous_state;

    state.top1_node = current_top1;
    state.top1_stall_iters = head_stall_iters;
    state.warmup_done = true;
    state.last_expander_distance = last_expander_distance;
    state.current_expander_distance = current_expander_distance;

    state.adaptive_spec_degree = static_cast<int>(PHASE2_THETA);
    state.adaptive_rho = phase2_rho;
    state.policy_iters = previous_state.policy_iters + 1;
    return state;
}

static __device__ __forceinline__ GPUAdaptiveSearchState update_adaptive_state(
    INDEX_T current_top1, int head_stall_iters, DISTANCE_T last_expander_distance,
    DISTANCE_T current_expander_distance, float distance_reduction_rate,
    const GPUAdaptiveSearchState& previous_state, float phase2_rho) {
    GPUAdaptiveSearchState state = previous_state;

    state.top1_node = current_top1;
    state.top1_stall_iters = head_stall_iters;
    state.warmup_done = false;
    state.last_expander_distance = last_expander_distance;
    state.current_expander_distance = current_expander_distance;

    const float tune_span = phase2_rho > PHASE1_RHO
        ? phase2_rho - PHASE1_RHO
        : RHO_MAX - RHO_MIN;
    const float base_prune = previous_state.policy_iters == 0
        ? PHASE1_RHO
        : previous_state.adaptive_rho;
    const float rho_delta = tune_span * distance_reduction_rate;
    const float next_prune_cap = phase2_rho > PHASE1_RHO
        ? phase2_rho
        : RHO_MAX;
    const float next_prune = clamp_float(base_prune + rho_delta, RHO_MIN, next_prune_cap);

    state.adaptive_spec_degree = static_cast<int>(PHASE1_THETA);
    state.adaptive_rho = next_prune;
    state.policy_iters = previous_state.policy_iters + 1;
    return state;
}

static __device__ __forceinline__ int keep_count_for_expander(
    uint32_t active_neighbors, const GPUAdaptiveSearchState& state,
    bool kth_cutoff_valid = false, uint32_t kth_near_count = 0) {
    const int active_count = static_cast<int>(active_neighbors);

    const int rho_keep = static_cast<int>(keep_count_per_parent(active_neighbors, state.adaptive_rho));
    const uint32_t spec_degree = static_cast<uint32_t>(state.adaptive_spec_degree);
    uint32_t candidate_bound = static_cast<uint32_t>(BUFFER_BOUND);
    if (active_neighbors <= WARP_SIZE && candidate_bound > WARP_SIZE) {
        candidate_bound = WARP_SIZE;
    }
    const uint32_t per_parent_bound = candidate_bound / spec_degree;
    const int budget_keep = clamp_int(static_cast<int>(per_parent_bound), 1, active_count);
    int max_keep = (rho_keep < budget_keep) ? rho_keep : budget_keep;
    if (kth_cutoff_valid) {
        max_keep = clamp_int(static_cast<int>(kth_near_count), 1, max_keep);
    }
    return clamp_int(max_keep, 1, active_count);
}

static __host__ __device__ inline uint32_t candidate_buffer_capacity(uint32_t max_degree) {
    const uint32_t max_capacity = static_cast<uint32_t>(THETA_MAX) * max_degree;
    uint32_t capacity = static_cast<uint32_t>(BUFFER_BOUND);
    if (max_degree <= WARP_SIZE && capacity > WARP_SIZE) {
        capacity = WARP_SIZE;
    }
    if (capacity > max_capacity) capacity = max_capacity;
    return capacity;
}

static __host__ inline uint32_t calculate_shared_mem_size(int dim, int beam_sz, int max_deg, int bitlen,
                                                          int bits, QuantType quant) { // SHAME(MANYARG)
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const size_t candidate_buffer_size = round_up_power2_u32(candidate_buffer_capacity(static_cast<uint32_t>(max_deg)));
    const uint32_t padded_beam_size = effective_sort_beam_size(static_cast<uint32_t>(beam_sz));
    const size_t result_buffer_size = static_cast<size_t>(padded_beam_size) + candidate_buffer_size;
    size_t size = 0;
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);       // visited-node hash table
    size += result_buffer_size * sizeof(INDEX_T);              // TOP_K_INDEX + CANDIDATE_INDEX
    size += result_buffer_size * sizeof(DISTANCE_T);           // TOP_K_DISTANCE + CANDIDATE_DISTANCE
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_LIST: beam positions/ranks selected by pick_expanders
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_NODE_LIST: graph node ids for selected parents
    size += SEARCH_WIDTH * sizeof(DISTANCE_T);                 // PARENT_DISTANCE_LIST: exact distances for selected parents
    size += dim * sizeof(DATA_T);                              // QUERY_BUFFER
    const bool prod = quant == QUANT_TBQ;
    const int stage_bits = prod ? quant_stage(bits) : bits;
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint4)));
    size += quant_lutbytes(padded_dim, stage_bits) * sizeof(uint8_t); // LUT_BUFFER
    if (prod) {
        size += quant_lutbytes(padded_dim, 1) * sizeof(uint8_t); // SIGN_LUT_BUFFER
    }
    // one region for buffers never live together: rotated (+ sketch) query, radix scratch
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint4)));
    size_t transient = padded_dim * sizeof(float) * (prod ? 2 : 1); // ROTATED_QUERY_BUFFER (+ SKETCH)
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && candidate_buffer_size > 256) {
        const size_t radix = align_up_uintptr(size, candidate_radix_sort_scratch_alignment()) - size
            + candidate_radix_sort_scratch_bytes();             // candidate radix sort scratch
        transient = transient > radix ? transient : radix;
    }
    size += transient;
    return static_cast<uint32_t>(size);
}


/**
 * @brief Single-expander expansion, the whole block working on one parent's neighbors.
 *
 * Gathers the unvisited neighbors into the candidate buffer, estimates and prunes them only when
 * they exceed the keep budget, then inserts the kept children and computes their exact distances.
 *
 * @tparam CODEBITS quantizer code width
 * @tparam turbop whether the index carries a TurboQuant sketch
 */
template <int CODEBITS, bool turbop>
static __device__ __forceinline__ void collect_phase1_candidates_block_scan(
    size_t padded_dim, int dim, int max_degree, size_t npoints,
    const QueryFactors* __restrict__ qf, const QueryFactors* __restrict__ qb,
    const float* __restrict__ d_qg_data, const DATA_T* __restrict__ query,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t sign_offset,
    size_t factor_offset,
    INDEX_T* __restrict__ hash_table, int bitlen,
    const INDEX_T* __restrict__ parent_node_list,
    const DISTANCE_T* __restrict__ parent_distance_list,
    uint32_t candidate_buffer_size,
    const GPUAdaptiveSearchState* __restrict__ adaptive_state,
    DISTANCE_T current_kth_cutoff,
    INDEX_T* __restrict__ candidate_index,
    DISTANCE_T* __restrict__ candidate_distance,
    void* candidate_radix_scratch,
    uint32_t* __restrict__ kth_near_count,
    uint32_t* __restrict__ warp_stat_counts,
    bool use_ip)
{
    const uint32_t candidate_work_count = (static_cast<uint32_t>(max_degree) < candidate_buffer_size) ? static_cast<uint32_t>(max_degree) : candidate_buffer_size;
    const INDEX_T parent_node = parent_node_list[0];
    const bool parent_valid = parent_node != MAX_INDEX;
    const float* parent_row = parent_valid ? d_qg_data + static_cast<size_t>(parent_node) * row_offset : nullptr;
    const vidType* parent_neighbors = parent_valid ? reinterpret_cast<const vidType*>(parent_row + neighbor_offset) : nullptr;

    // gather valid neighbors into shared memory
    for (int i = tidx(); i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
        INDEX_T child_id = MAX_INDEX;
        if (i < static_cast<int>(candidate_work_count) && parent_valid) {
            child_id = parent_neighbors[i];
            if (child_id >= npoints || hashtable_contains(hash_table, bitlen, child_id)) {
                child_id = MAX_INDEX;
            }
        }
        candidate_index[i] = child_id;
        candidate_distance[i] = (child_id == MAX_INDEX) ? FLT_MAX : 0.0f;
    }
    __syncthreads();

    // skip estimation when all valid neighbors fit within the keep budget
    const int keep_count = keep_count_for_expander(candidate_work_count, *adaptive_state);
    const uint32_t valid_candidate_count = count_valid_candidates_warp_reduced(
        candidate_index, candidate_distance, static_cast<int>(candidate_work_count), MAX_INDEX, warp_stat_counts);
    const bool keep_all_valid = keep_count >= static_cast<int>(valid_candidate_count);
    int effective_keep_count = keep_count;

    // estimate and prune only when valid candidates exceed the keep budget
    if (!keep_all_valid) {
        const float* parent_factors = parent_row + factor_offset;
        const uint8_t* parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);
        const uint8_t* parent_sign_base = reinterpret_cast<const uint8_t*>(parent_row + sign_offset);
        const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));
        const int sign_bytes = static_cast<int>(quant_bytes(padded_dim, 1));
        constexpr int lanes_per_neighbor = GPU_RABITQ_FASTSCAN_SUBWARP_LANES;
        constexpr int groups_per_warp = WARP_SIZE / lanes_per_neighbor;
        const int group_lane = laneidx() & (lanes_per_neighbor - 1);
        const int warp_group = laneidx() / lanes_per_neighbor;

        for (int base = warpidx() * groups_per_warp; base < static_cast<int>(candidate_work_count); base += WARPS_PER_BLOCK * groups_per_warp) {
            const int i = base + warp_group;
            bool valid_candidate = i < static_cast<int>(candidate_work_count) && candidate_index[i] != MAX_INDEX;
            DISTANCE_T est_dist = FLT_MAX;
            if (valid_candidate) {
                if constexpr (turbop) {
                    est_dist = turbop_scan<lanes_per_neighbor, CODEBITS>(
                        *qf, *qb, parent_code_base, parent_sign_base, i, parent_factors,
                        max_degree, parent_distance_list[0], padded_dim,
                        bytes_per_neighbor, sign_bytes);
                } else {
                    const float* triple_x = parent_factors + i;
                    const float* factor_dq = parent_factors + max_degree + i;
                    const float* factor_vq = parent_factors + 2 * max_degree + i;
                    est_dist = scan_one_neighbor_lanes_gpu<lanes_per_neighbor, CODEBITS>(
                        *qf, parent_code_base, i, triple_x, factor_dq, factor_vq,
                        parent_distance_list[0], padded_dim, bytes_per_neighbor);
                }
                valid_candidate = isfinite(est_dist);
                if (group_lane == 0) {
                    candidate_distance[i] = valid_candidate ? est_dist : FLT_MAX;
                    if (!valid_candidate) {
                        candidate_index[i] = MAX_INDEX;
                    }
                }
            }
        }
        __syncthreads();

        // tighten the keep count using the current kth-result cutoff when available
        const DISTANCE_T kth_cutoff = current_kth_cutoff;
        const bool kth_cutoff_valid = isfinite(static_cast<float>(kth_cutoff)) && kth_cutoff < FLT_MAX;
        if (tidx() == 0) {
            *kth_near_count = 0;
        }
        __syncthreads();

        if (kth_cutoff_valid) {
            uint32_t thread_near_count = 0;
            for (int i = tidx(); i < static_cast<int>(candidate_work_count); i += blockDim.x) {
                const DISTANCE_T dist = candidate_distance[i];
                if (candidate_index[i] != MAX_INDEX && isfinite(dist) && dist <= kth_cutoff) {
                    thread_near_count++;
                }
            }
            const uint32_t block_near_count = block_reduce_sum_u32(thread_near_count, warp_stat_counts);
            if (tidx() == 0) *kth_near_count = block_near_count;
        }
        __syncthreads();

        effective_keep_count = keep_count_for_expander(candidate_work_count, *adaptive_state, kth_cutoff_valid, *kth_near_count);

        // sort estimated candidates and invalidate entries past the keep budget
        dispatch_candidate_sort(
            candidate_index, candidate_distance, candidate_buffer_size,
            candidate_radix_scratch);
        __syncthreads();

        for (int i = tidx(); i < static_cast<int>(candidate_buffer_size); i += blockDim.x) {
            if (i >= effective_keep_count) {
                candidate_index[i] = MAX_INDEX;
                candidate_distance[i] = FLT_MAX;
            }
        }
        __syncthreads();
    }

    // insert kept children and compute their exact distances for the next merge
    const int admit_scan_count = keep_all_valid ? static_cast<int>(candidate_work_count) : effective_keep_count;
    for (int i = warpidx(); i < admit_scan_count; i += WARPS_PER_BLOCK) {
        INDEX_T child_id = MAX_INDEX;
        uint32_t inserted = 0;
        if (laneidx() == 0) {
            child_id = candidate_index[i];
            if (child_id != MAX_INDEX && candidate_distance[i] < FLT_MAX) {
                inserted = hashtable_insert(hash_table, bitlen, child_id);
            }
            if (!inserted) {
                child_id = MAX_INDEX;
                candidate_index[i] = MAX_INDEX;
                candidate_distance[i] = FLT_MAX;
            }
        }
        // reconverge after lane 0's insert so the broadcasts compile to native shuffles
        __syncwarp();
        child_id = SHFL(child_id, 0);
        inserted = SHFL(inserted, 0);

        DISTANCE_T child_dist = FLT_MAX;
        if (inserted) {
            child_dist = warp_distance(dim, query, d_qg_data + static_cast<size_t>(child_id) * row_offset, use_ip);
        }
        if (laneidx() == 0 && inserted) {
            candidate_distance[i] = child_dist;
        }
    }
}

/**
 * @brief Speculative phase-2 expansion for parents of at most 32 neighbors.
 *
 * Warp w of a wave takes parent task_base + w, lane l its neighbor l. Every warp of a wave
 * checks and prunes before any inserts, then compacts inserted children into the candidate buffer.
 *
 * @tparam CODEBITS quantizer code width
 * @tparam turbop whether the index carries a TurboQuant sketch
 * SHAME(TALLFUNC) SHAME(WIDEFUNC) SHAME(MANYARG)
 */
template <int CODEBITS, bool turbop>
static __device__ __forceinline__ void collect_phase2_degree32_candidates_warp_local(
    size_t padded_dim, int max_degree, size_t npoints,
    const QueryFactors* __restrict__ qf, const QueryFactors* __restrict__ qb,
    const float* __restrict__ d_qg_data,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t sign_offset,
    size_t factor_offset,
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
    const int tid = tidx();
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));
    const int sign_bytes = static_cast<int>(quant_bytes(padded_dim, 1));

    for (int task_base = 0; task_base < static_cast<int>(keep_expanding); task_base += WARPS_PER_BLOCK) {
        // [1] check each lane's neighbor of this warp's parent against the visited set
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
        const uint8_t* parent_sign_base = nullptr;
        const float* parent_factor_base = nullptr;
        bool valid_candidate = false;

        if (task_active && neighbor_idx < parent_work_count) {
            const float* parent_row = d_qg_data + static_cast<size_t>(parent_node) * row_offset;
            const vidType* parent_neighbors = reinterpret_cast<const vidType*>(parent_row + neighbor_offset);
            parent_factor_base = parent_row + factor_offset;
            parent_code_base = reinterpret_cast<const uint8_t*>(parent_row + code_offset);
            parent_sign_base = reinterpret_cast<const uint8_t*>(parent_row + sign_offset);

            child_id = parent_neighbors[neighbor_idx];
            valid_candidate = child_id != MAX_INDEX && child_id < npoints && !hashtable_contains(hash_table, bitlen, child_id);
            if (valid_candidate) {
                triple_x = parent_factor_base[neighbor_idx];
                factor_dq = parent_factor_base[max_degree + neighbor_idx];
                factor_vq = parent_factor_base[2 * max_degree + neighbor_idx];
            }
        }

        // [2] estimate only when the valid lanes exceed the keep budget
        const int keep_count = task_active ? keep_count_for_expander(parent_work_count, *adaptive_state) : 0;
        const uint32_t raw_valid_count = __popc(__ballot_sync(FULL_MASK, valid_candidate));
        const bool keep_all_valid = keep_count >= static_cast<int>(raw_valid_count);

        DISTANCE_T est_dist = FLT_MAX;
        if (!keep_all_valid && valid_candidate) {
            if constexpr (turbop) {
                est_dist = turbop_scan<1, CODEBITS>(
                    *qf, *qb, parent_code_base, parent_sign_base, static_cast<int>(neighbor_idx),
                    parent_factor_base, max_degree, parent_distance_list[parent_idx],
                    padded_dim, bytes_per_neighbor, sign_bytes);
            } else {
                est_dist = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                    *qf, parent_code_base, static_cast<int>(neighbor_idx), triple_x, factor_dq, factor_vq,
                    parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            }
            valid_candidate = isfinite(est_dist);
        }

        // [3] tighten the budget by the kth-result cutoff and keep the smallest estimates
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

        // [4] every warp finishes this wave's visited check before any warp inserts
        __syncthreads();
        uint32_t inserted = 0;
        if (keep_lane) {
            inserted = hashtable_insert(hash_table, bitlen, child_id);
        }

        // [5] compact accepted lanes into shared candidate slots
        const uint32_t accepted_mask = __ballot_sync(FULL_MASK, inserted != 0);
        const uint32_t base_slot = allocate_warp_compact_slots(
            __popc(accepted_mask), compact_candidate_count, warp_stat_counts, warp_stat_counts + WARPS_PER_BLOCK);
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        if (inserted) {
            const uint32_t slot = base_slot + __popc(accepted_mask & lane_mask);
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child_id;
                candidate_distance[slot] = FLT_MAX;
            }
        }
        // [6] every warp finishes this wave's inserts before the next wave checks
        __syncthreads();
    }
}

/**
 * @brief Speculative phase-2 expansion for parents of 33 to 64 neighbors, ranked together.
 *
 * Warp w of a wave takes parent task_base + w, lane l its neighbors l and l + 32 in registers.
 * All 64 are checked and estimated before one selection keeps the smallest across both halves.
 * Every warp of a wave checks and prunes before any inserts.
 *
 * @tparam CODEBITS quantizer code width
 * @tparam turbop whether the index carries a TurboQuant sketch
 * SHAME(TALLFUNC) SHAME(WIDEFUNC) SHAME(MANYARG)
 */
template <int CODEBITS, bool turbop>
static __device__ __forceinline__ void collect_phase2_degree64_candidates_warp_local(
    size_t padded_dim, int max_degree, size_t npoints,
    const QueryFactors* __restrict__ qf, const QueryFactors* __restrict__ qb,
    const float* __restrict__ d_qg_data,
    size_t row_offset, size_t neighbor_offset, size_t code_offset, size_t sign_offset,
    size_t factor_offset,
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
    const int tid = tidx();
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int bytes_per_neighbor = static_cast<int>(quant_bytes(padded_dim, CODEBITS));
    const int sign_bytes = static_cast<int>(quant_bytes(padded_dim, 1));

    for (int task_base = 0; task_base < static_cast<int>(keep_expanding); task_base += WARPS_PER_BLOCK) {
        // [1] check both of each lane's neighbors of this warp's parent against the visited set
        const uint32_t parent_idx = static_cast<uint32_t>(task_base + warp_id);
        const INDEX_T parent_node = (parent_idx < keep_expanding) ? parent_node_list[parent_idx] : MAX_INDEX;
        const bool task_active = parent_idx < keep_expanding && parent_node != MAX_INDEX;
        const float* parent_row = task_active ? d_qg_data + static_cast<size_t>(parent_node) * row_offset : nullptr;
        const vidType* parent_neighbors = task_active ? reinterpret_cast<const vidType*>(parent_row + neighbor_offset) : nullptr;
        const float* parent_factors = task_active ? parent_row + factor_offset : nullptr;
        const uint8_t* parent_code_base = task_active ? reinterpret_cast<const uint8_t*>(parent_row + code_offset) : nullptr;
        const uint8_t* parent_sign_base = task_active ? reinterpret_cast<const uint8_t*>(parent_row + sign_offset) : nullptr;

        const uint32_t neighbor_idx0 = static_cast<uint32_t>(lane_id);
        const uint32_t neighbor_idx1 = static_cast<uint32_t>(lane_id + WARP_SIZE);
        INDEX_T child0 = MAX_INDEX;
        INDEX_T child1 = MAX_INDEX;
        bool valid0 = false;
        bool valid1 = false;
        if (task_active && neighbor_idx0 < parent_work_count) {
            child0 = parent_neighbors[neighbor_idx0];
            valid0 = child0 != MAX_INDEX && child0 < npoints && !hashtable_contains(hash_table, bitlen, child0);
        }
        if (task_active && neighbor_idx1 < parent_work_count) {
            child1 = parent_neighbors[neighbor_idx1];
            valid1 = child1 != MAX_INDEX && child1 < npoints && !hashtable_contains(hash_table, bitlen, child1);
        }

        // [2] estimate both only when the valid neighbors exceed the keep budget
        const int keep_count = task_active ? keep_count_for_expander(parent_work_count, *adaptive_state) : 0;
        const uint32_t raw_valid_count =
            __popc(__ballot_sync(FULL_MASK, valid0)) + __popc(__ballot_sync(FULL_MASK, valid1));
        const bool keep_all_valid = keep_count >= static_cast<int>(raw_valid_count);

        DISTANCE_T est0 = FLT_MAX;
        DISTANCE_T est1 = FLT_MAX;
        if (!keep_all_valid && valid0) {
            if constexpr (turbop) {
                est0 = turbop_scan<1, CODEBITS>(
                    *qf, *qb, parent_code_base, parent_sign_base, static_cast<int>(neighbor_idx0),
                    parent_factors, max_degree, parent_distance_list[parent_idx],
                    padded_dim, bytes_per_neighbor, sign_bytes);
            } else {
                est0 = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                    *qf, parent_code_base, static_cast<int>(neighbor_idx0),
                    parent_factors[neighbor_idx0],
                    parent_factors[max_degree + neighbor_idx0],
                    parent_factors[2 * max_degree + neighbor_idx0],
                    parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            }
            valid0 = isfinite(est0);
        }
        if (!keep_all_valid && valid1) {
            if constexpr (turbop) {
                est1 = turbop_scan<1, CODEBITS>(
                    *qf, *qb, parent_code_base, parent_sign_base, static_cast<int>(neighbor_idx1),
                    parent_factors, max_degree, parent_distance_list[parent_idx],
                    padded_dim, bytes_per_neighbor, sign_bytes);
            } else {
                est1 = scan_one_neighbor_lane_seq_lut_gpu<CODEBITS>(
                    *qf, parent_code_base, static_cast<int>(neighbor_idx1),
                    parent_factors[neighbor_idx1],
                    parent_factors[max_degree + neighbor_idx1],
                    parent_factors[2 * max_degree + neighbor_idx1],
                    parent_distance_list[parent_idx], padded_dim, bytes_per_neighbor);
            }
            valid1 = isfinite(est1);
        }

        // [3] tighten the budget by the kth-result cutoff and keep the smallest of all 64 estimates
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

        // [4] every warp finishes this wave's visited check before any warp inserts
        __syncthreads();
        uint32_t inserted0 = 0;
        uint32_t inserted1 = 0;
        if (keep0) inserted0 = hashtable_insert(hash_table, bitlen, child0);
        if (keep1) inserted1 = hashtable_insert(hash_table, bitlen, child1);

        // [5] compact both halves through one slot allocation, first half ahead of the second
        const uint32_t accepted_mask0 = __ballot_sync(FULL_MASK, inserted0 != 0);
        const uint32_t accepted_mask1 = __ballot_sync(FULL_MASK, inserted1 != 0);
        const uint32_t base_slot = allocate_warp_compact_slots(
            __popc(accepted_mask0) + __popc(accepted_mask1), compact_candidate_count,
            warp_stat_counts, warp_stat_counts + WARPS_PER_BLOCK);
        const uint32_t lane_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1u);
        if (inserted0) {
            const uint32_t slot = base_slot + __popc(accepted_mask0 & lane_mask);
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child0;
                candidate_distance[slot] = FLT_MAX;
            }
        }
        if (inserted1) {
            const uint32_t slot = base_slot + __popc(accepted_mask0) + __popc(accepted_mask1 & lane_mask);
            if (slot < candidate_collect_capacity) {
                candidate_index[slot] = child1;
                candidate_distance[slot] = FLT_MAX;
            }
        }
        // [6] every warp finishes this wave's inserts before the next wave checks
        __syncthreads();
    }
}
