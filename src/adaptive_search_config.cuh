#pragma once

/**
 * @brief Internal top-k size, printed by gpu_search_adaptive as internal_topk; no kernel reads it.
 */
#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

/**
 * @brief Parent slots of QuantizedPrunedBeamSearch, the length of PARENT_LIST, PARENT_NODE_LIST and PARENT_DISTANCE_LIST.
 *
 * pickparents fills up to theta slots, so it must be at least PHASE1_THETA and PHASE2_THETA.
 * rabitq_utils.cuh defines its own value before including this file.
 */
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH (WARPS_PER_BLOCK)
#endif

/**
 * @brief Hard cap on search-loop iterations in QuantizedPrunedBeamSearch and QuantizedBeamSearch.
 */
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS (1 << 10)
#endif

/**
 * @brief Lanes cooperating on one neighbor estimate in the block-wide scans, one of 2, 4, 8, 16 or 32.
 *
 * Used by collect_phase1_candidates_block_scan and QuantizedBeamSearch; the phase-2 paths use one lane per neighbor.
 */
#ifndef GPU_RABITQ_FASTSCAN_SUBWARP_LANES
#define GPU_RABITQ_FASTSCAN_SUBWARP_LANES 8
#endif

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 2 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 4 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 8 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 16 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 2, 4, 8, 16, or 32"
#endif

/**
 * @brief Bounds of theta, the number of parents expanded per iteration.
 *
 * THETA_MAX also caps candidate_buffer_capacity at THETA_MAX * max_degree.
 */
#ifndef THETA_MIN
#define THETA_MIN 1
#endif

#ifndef THETA_MAX
#define THETA_MAX WARPS_PER_BLOCK
#endif

/**
 * @brief Bounds of rho, the fraction of each parent's neighbors pruned, applied by update_adaptive_state.
 *
 * RHO_MAX caps rho only when phase2_rho is not above PHASE1_RHO; compute_max_rho_bound returns RHO_MIN for degrees up to MIN_PHASE2_KEEP.
 */
#ifndef RHO_MIN
#define RHO_MIN 0.0f
#endif

#ifndef RHO_MAX
#define RHO_MAX 0.99f
#endif

/**
 * @brief Neighbors per parent that phase 2 still keeps, from which compute_max_rho_bound derives phase2_rho.
 */
#ifndef MIN_PHASE2_KEEP
#define MIN_PHASE2_KEEP 4
#endif

/**
 * @brief Iterations between phase-1 policy updates by update_adaptive_state; the phase switch is checked every iteration.
 */
#ifndef CHECK_INTERVAL
#define CHECK_INTERVAL 1
#endif

/**
 * @brief Candidate budget of one iteration, shared by the parents.
 *
 * candidate_buffer_capacity caps the candidate buffer at it (at WARP_SIZE for degrees up to 32),
 * keep_count_for_expander splits it across theta parents, and the kernel caps compacted phase-2 children at it.
 */
#ifndef BUFFER_BOUND
#define BUFFER_BOUND 64
#endif

/**
 * @brief Phase policy: phase 1 expands PHASE1_THETA parents and raises rho from PHASE1_RHO with expander progress.
 *
 * After TOP1_WARMUP_STALL_ITERS consecutive iterations with an unchanged beam head, phase 2 expands
 * PHASE2_THETA parents at the phase2_rho kernel argument.
 */
#ifndef PHASE1_THETA
#define PHASE1_THETA THETA_MIN
#endif

#ifndef PHASE1_RHO
#define PHASE1_RHO 0.25f
#endif

#ifndef PHASE2_THETA
#define PHASE2_THETA THETA_MAX
#endif

#ifndef TOP1_WARMUP_STALL_ITERS
#define TOP1_WARMUP_STALL_ITERS 4
#endif

/**
 * @brief Compile-time range checks of the knobs above.
 */
static_assert(SEARCH_WIDTH >= 1, "Adaptive AP requires SEARCH_WIDTH >= 1");
static_assert(THETA_MIN >= 1, "THETA_MIN must be >= 1");
static_assert(BUFFER_BOUND > 0, "BUFFER_BOUND must be > 0");
static_assert(THETA_MAX >= THETA_MIN, "THETA_MAX must be >= THETA_MIN");
static_assert(CHECK_INTERVAL >= 1, "CHECK_INTERVAL must be >= 1");
static_assert(RHO_MIN >= 0.0f && RHO_MIN < 1.0f, "RHO_MIN must be in [0, 1)");
static_assert(RHO_MAX >= RHO_MIN && RHO_MAX < 1.0f, "RHO_MAX must be in [RHO_MIN, 1)");
static_assert(TOP1_WARMUP_STALL_ITERS >= 1, "TOP1_WARMUP_STALL_ITERS must be >= 1");
static_assert(PHASE1_THETA >= THETA_MIN && PHASE1_THETA <= THETA_MAX, "PHASE1_THETA must be within theta bounds");
static_assert(PHASE2_THETA >= THETA_MIN && PHASE2_THETA <= THETA_MAX, "PHASE2_THETA must be within theta bounds");
static_assert(PHASE1_RHO >= RHO_MIN && PHASE1_RHO <= RHO_MAX, "PHASE1_RHO must be within rho bounds");
static_assert(MIN_PHASE2_KEEP > 0, "MIN_PHASE2_KEEP must be > 0");
