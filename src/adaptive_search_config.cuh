#pragma once

#ifndef INTERNAL_TOPK
#define INTERNAL_TOPK 128
#endif

#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH (WARPS_PER_BLOCK)
#endif

#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS (1 << 10)
#endif

// RaBitQ LUT scan layout. This controls how many lanes cooperate on one neighbor estimate.
#ifndef GPU_RABITQ_FASTSCAN_SUBWARP_LANES
#define GPU_RABITQ_FASTSCAN_SUBWARP_LANES 8
#endif

#if GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 2 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 4 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 8 && GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 16 && \
    GPU_RABITQ_FASTSCAN_SUBWARP_LANES != 32
#error "GPU_RABITQ_FASTSCAN_SUBWARP_LANES must be one of 2, 4, 8, 16, or 32"
#endif

// Hard theta/rho bounds.
#ifndef THETA_MIN
#define THETA_MIN 1
#endif

#ifndef THETA_MAX
#define THETA_MAX WARPS_PER_BLOCK
#endif

#ifndef RHO_MIN
#define RHO_MIN 0.0f
#endif

#ifndef RHO_MAX
#define RHO_MAX 0.99f
#endif

#ifndef MIN_PHASE2_KEEP
#define MIN_PHASE2_KEEP 4
#endif

// Recompute the AP policy every N iterations.
#ifndef CHECK_INTERVAL
#define CHECK_INTERVAL 1
#endif

// Candidate-buffer budget used by the AP controller.
#ifndef BUFFER_BOUND
#define BUFFER_BOUND 64
#endif

// Phase policy. Stage I uses progress-based rho tuning from PHASE1_RHO;
// Stage II jumps to PHASE2_THETA/PHASE2_RHO after head-stall detection.
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
