#pragma once

#include <cuda_runtime.h>

#include "include/utils.cuh"
#include "adaptive_search_config.cuh"

struct GPUAdaptiveSearchState {
    int adaptive_spec_degree;               // Current speculation degree.
    float adaptive_rho;             // Current rho value.
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
    const uint32_t per_parent_bound = static_cast<uint32_t>(BUFFER_BOUND) / spec_degree;
    const int budget_keep = clamp_int(static_cast<int>(per_parent_bound), 1, active_count);
    int max_keep = (rho_keep < budget_keep) ? rho_keep : budget_keep;
    if (kth_cutoff_valid) {
        max_keep = clamp_int(static_cast<int>(kth_near_count), 1, max_keep);
    }
    return clamp_int(max_keep, 1, active_count);
}

template <typename T>
static __device__ __forceinline__ T* allocate_shared_tail_array(char*& tail_base, uint32_t count) {
    tail_base = reinterpret_cast<char*>(align_up_uintptr(reinterpret_cast<uintptr_t>(tail_base), alignof(T)));
    T* ptr = reinterpret_cast<T*>(tail_base);
    tail_base = reinterpret_cast<char*>(ptr + count);
    return ptr;
}

static __host__ __device__ inline uint32_t candidate_buffer_capacity(uint32_t max_degree) {
    const uint32_t max_capacity = static_cast<uint32_t>(THETA_MAX) * max_degree;
    uint32_t capacity = static_cast<uint32_t>(BUFFER_BOUND);
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
