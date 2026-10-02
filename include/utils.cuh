#pragma once

#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "common.hpp"

/*-------------------------------------------- cuda --------------------------------------------*/
/**
 * @brief Active-lane mask for full-warp shuffles and ballots.
 */
#define FULL_MASK 0xffffffff

/**
 * @brief Threads per search block; overridable at compile time.
 */
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

/**
 * @brief Lanes per warp and warps per search block.
 */
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

/**
 * @brief Sentinel node index for empty beam, candidate, and hash-table slots.
 *
 * Its most significant bit is set, so the expanded-flag checks in pick_expanders and
 * pickparents never select it.
 */
#define MAX_INDEX UINT_MAX

/**
 * @brief Full-warp shuffle-down and broadcast shorthands used by the warp reductions.
 */
#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)

/**
 * @brief Maximum resident warps per SM for the compiled architecture.
 *
 * 48 for sm_86 to sm_89 and sm_120 and above, 64 otherwise.
 */
#if defined(__CUDA_ARCH__) && ((__CUDA_ARCH__ >= 860 && __CUDA_ARCH__ < 900) || __CUDA_ARCH__ >= 1200)
#define GPU_MAX_WARPS_PER_SM 48
#else
#define GPU_MAX_WARPS_PER_SM 64
#endif

/**
 * @brief Launch bounds for the search kernels: block_size threads, enough blocks per SM to fill
 * GPU_MAX_WARPS_PER_SM, which caps registers per thread.
 *
 * GPU_MIN_BLOCKS_PER_SM_FOR ignores its argument and divides by WARPS_PER_BLOCK.
 */
#define GPU_MIN_BLOCKS_PER_SM_FOR(block_size) (GPU_MAX_WARPS_PER_SM / WARPS_PER_BLOCK)
#define GPU_LAUNCH_BOUNDS(block_size) __launch_bounds__((block_size), GPU_MIN_BLOCKS_PER_SM_FOR(block_size))

/**
 * @brief Host check for a CUDA runtime call: prints the error with file and line, then exits.
 */
#define CUDA_SAFE_CALL(call)                                                        \
    {                                                                               \
        cudaError err = call;                                                       \
        if (cudaSuccess != err) {                                                   \
            fprintf(stderr, "error %d: Cuda error in file '%s' in line %i : %s.\n", \
                    err, __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    }

/*-------------------------------------------- types --------------------------------------------*/
/**
 * @brief Vector element, node index, and distance types shared by all kernels.
 */
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

/*-------------------------------------------- thread index --------------------------------------------*/
/**
 * @brief Thread index, read afresh on every call.
 *
 * A plain threadIdx.x read is merged into one value that stays live across the whole search
 * loop and is spilled under the register cap; the volatile read keeps each use local.
 *
 * @return threadIdx.x
 */
static __device__ __forceinline__ unsigned tidx() {
    unsigned t;
    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(t));
    return t;
}

/**
 * @brief Warp index within the block, from a fresh thread-index read.
 * @return threadIdx.x / WARP_SIZE
 */
static __device__ __forceinline__ int warpidx() {
    return static_cast<int>(tidx() / WARP_SIZE);
}

/**
 * @brief Lane index within the warp, from a fresh thread-index read.
 * @return threadIdx.x % WARP_SIZE
 */
static __device__ __forceinline__ int laneidx() {
    return static_cast<int>(tidx() % WARP_SIZE);
}

/*-------------------------------------------- common --------------------------------------------*/
/**
 * @brief Smallest power of two not below x; sizes the candidate buffers and padded beams.
 * @param x value to round
 * @return 1 for x <= 1, otherwise the power of two
 */
__host__ __device__ inline uint32_t round_up_power2_u32(uint32_t x) {
    if (x <= 1)
        return 1;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

/**
 * @brief Whether x is a nonzero power of two; checks the block bitonic candidate sort size.
 * @param x value to test
 * @return true for a power of two
 */
__host__ __device__ inline bool is_power2_u32(uint32_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

/**
 * @brief Round value up to a multiple of alignment; lays out shared-memory regions on host and device.
 * @param value address or byte offset
 * @param alignment power-of-two alignment
 * @return the aligned value
 */
__host__ __device__ inline uintptr_t align_up_uintptr(uintptr_t value, uintptr_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Carve an aligned T[count] from the dynamic shared-memory tail and advance tail_base past it.
 *
 * The adaptive and RaBitQ kernels place their LUTs, transient query buffers, and radix scratch
 * this way. count 0 only aligns tail_base.
 *
 * @tparam T element type, whose alignment is applied
 * @param tail_base next free byte, updated to the end of the array
 * @param count number of elements
 * @return start of the array
 */
template <typename T>
static __device__ __forceinline__ T* allocate_shared_tail_array(char*& tail_base, uint32_t count) {
    tail_base = reinterpret_cast<char*>(align_up_uintptr(reinterpret_cast<uintptr_t>(tail_base), alignof(T)));
    T* ptr = reinterpret_cast<T*>(tail_base);
    tail_base = reinterpret_cast<char*>(ptr + count);
    return ptr;
}

/*-------------------------------------------- expander selection --------------------------------------------*/
/**
 * @brief Pick up to N unexpanded beam positions in rank order and flag them expanded; serves PathW.
 *
 * Only warp 0 calls it: lane l scans positions l, l + 32, and so on. An entry is unexpanded while
 * its most significant bit is clear; picking sets that bit in pri_queue.
 *
 * @param N maximum number of expanders
 * @param output_list receives the picked beam positions, not node indices
 * @param M beam length
 * @param pri_queue beam node indices, flag bit in the most significant bit
 * @return number of positions picked, at most N
 */
static __device__ uint32_t pick_expanders(int N, INDEX_T *output_list, int M, INDEX_T *pri_queue) {
    // [1] round the beam length up to whole 32-wide chunks
    uint32_t num_exp = 0;
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const uint32_t lane_id = tidx() & (WARP_SIZE - 1);
    uint32_t itopk_max = M;
    if (itopk_max % 32) {
        itopk_max += 32 - (itopk_max % 32);
    }
    for (uint32_t j = tidx(); j < itopk_max; j += 32) {
        // [2] each lane checks one beam position for a clear flag bit
        INDEX_T index;
        int new_parent = 0;
        if (j < M) {
            index = pri_queue[j];
            if ((index & index_msb_1_mask) == 0) {
                new_parent = 1;
            }
        }
        const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);

        // [3] fresh lanes take consecutive output slots in rank order and set the flag bit
        if (new_parent) {
            const auto i = __popc(ballot_mask & ((1u << lane_id) - 1u)) + num_exp;
            if (i < N) {
                output_list[i] = j;
                pri_queue[j] |= index_msb_1_mask;
            }
        }

        // [4] stop once N positions are picked
        num_exp += __popc(ballot_mask);
        if (num_exp > static_cast<uint32_t>(N)) {
            num_exp = static_cast<uint32_t>(N);
        }
        if (num_exp >= N) {
            break;
        }
    }
    return num_exp;
}

/**
 * @brief Pick up to limit unexpanded beam entries as parents in rank order; serves the adaptive
 * and RaBitQ kernels.
 *
 * Only warp 0 calls it: lane l checks positions l, l + 32, and so on.
 *
 * @param limit maximum number of parents
 * @param beamsize beam length
 * @param beam beam node indices; the most significant bit flags expanded and is set on picks
 * @param beamdist beam distances
 * @param nodes receives the picked node indices
 * @param dists receives the picked distances
 * @return number of parents picked, at most limit
 */
static __device__ __forceinline__ uint32_t pickparents(
    int limit, int beamsize, INDEX_T *beam, const DISTANCE_T *beamdist, INDEX_T *nodes, DISTANCE_T *dists) {
    constexpr INDEX_T EXPANDED = 0x80000000;
    const uint32_t lane = tidx() & (WARP_SIZE - 1);
    const uint32_t below = (1u << lane) - 1u;
    uint32_t picked = 0;
    for (int base = 0; base < beamsize; base += WARP_SIZE) {
        // [1] each lane checks one beam position of this 32-wide chunk
        const int j = base + static_cast<int>(lane);
        INDEX_T node = MAX_INDEX;
        if (j < beamsize) node = beam[j];
        const bool fresh = j < beamsize && (node & EXPANDED) == 0;
        const uint32_t mask = __ballot_sync(FULL_MASK, fresh);

        // [2] fresh lanes take consecutive parent slots in rank order
        if (fresh) {
            const uint32_t slot = picked + __popc(mask & below);
            if (slot < static_cast<uint32_t>(limit)) {
                beam[j] = node | EXPANDED;
                nodes[slot] = node;
                dists[slot] = beamdist[j];
            }
        }
        picked += __popc(mask);
        if (picked >= static_cast<uint32_t>(limit)) return static_cast<uint32_t>(limit);
    }
    return picked;
}
