#pragma once

#include <cuda_runtime.h>

#include "include/utils.cuh"
#include "include/hash_table.cuh"
#include "include/distance.cuh"
#include "include/beam_management.cuh"
#include "include/quant.cuh"

/**
 * @brief Parents expanded per iteration of QuantizedBeamSearch, also the parent list length.
 *
 * Defined before adaptive_search_config.cuh so its default does not apply; the other knobs come from that file.
 */
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH 1
#endif

/**
 * @brief Iteration cap of QuantizedBeamSearch, matching the 8192-parent budget of QuantizedPrunedBeamSearch.
 */
#ifndef MAX_ITERATIONS
#define MAX_ITERATIONS (1 << 13)
#endif

#include "adaptive_search_config.cuh"

/*-------------------------------------------- result pool --------------------------------------------*/
/**
 * @brief Find the largest distance of the unsorted result pool, the replacement slot of result_pool_push_unsorted.
 *
 * Lane l scans positions l, l + 32, and so on; a shuffle reduction keeps the lowest position on
 * ties. An empty pool yields position 0 and -FLT_MAX. All 32 lanes of the warp must call it.
 *
 * @param pool_distance pool distances
 * @param pool_size entries in the pool
 * @param worst_idx receives the position of the largest distance
 * @param worst_dist receives the largest distance
 */
static __device__ inline void result_pool_recompute_worst(
    const DISTANCE_T* pool_distance, uint32_t pool_size, uint32_t* worst_idx, DISTANCE_T* worst_dist) {
    // [1] each lane keeps the largest distance of its strided positions
    const uint32_t lane = static_cast<uint32_t>(laneidx());
    DISTANCE_T best = -FLT_MAX;
    uint32_t where = MAX_INDEX;
    for (uint32_t i = lane; i < pool_size; i += WARP_SIZE) {
        const DISTANCE_T dist = pool_distance[i];
        if (where == MAX_INDEX || dist > best) {
            best = dist;
            where = i;
        }
    }

    // [2] reduce to the largest distance, lowest position first
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        const DISTANCE_T other = SHFL_DOWN(best, offset);
        const uint32_t otherwhere = SHFL_DOWN(where, offset);
        const bool better = where == MAX_INDEX || other > best || (other == best && otherwhere < where);
        if (otherwhere != MAX_INDEX && better) {
            best = other;
            where = otherwhere;
        }
    }
    if (lane == 0) {
        *worst_idx = where == MAX_INDEX ? 0 : where;
        *worst_dist = where == MAX_INDEX ? -FLT_MAX : best;
    }
    __syncwarp();
}

/**
 * @brief Insert a node into the unsorted result pool of QuantizedBeamSearch, keeping the closest capacity entries.
 *
 * Lane 0 writes the entry; a replacement rescans the pool warp-wide. MAX_INDEX and non-finite
 * distances are ignored. The node must be new to the pool, as the never-reset expanded-node hash
 * guarantees. All 32 lanes of the warp must call it with the same node.
 *
 * @param pool_index pool node ids
 * @param pool_distance pool distances
 * @param pool_size entries in the pool, updated
 * @param worst_idx position of the largest distance, updated
 * @param worst_dist largest distance, updated
 * @param capacity pool capacity
 * @param node node id
 * @param distance exact distance of the node
 */
static __device__ inline void result_pool_push_unsorted(
    INDEX_T* pool_index, DISTANCE_T* pool_distance, uint32_t* pool_size, uint32_t* worst_idx,
    DISTANCE_T* worst_dist, int capacity, INDEX_T node, DISTANCE_T distance) {
    if (node == MAX_INDEX || !isfinite(distance)) return;

    // [1] append while the pool has room
    const uint32_t size = *pool_size;
    if (size < static_cast<uint32_t>(capacity)) {
        if (laneidx() == 0) {
            pool_index[size] = node;
            pool_distance[size] = distance;
            *pool_size = size + 1;
            if (size == 0 || distance > *worst_dist) {
                *worst_idx = size;
                *worst_dist = distance;
            }
        }
        __syncwarp();
        return;
    }

    // [2] otherwise replace the worst entry when closer, then find the new worst
    if (distance >= *worst_dist) return;
    if (laneidx() == 0) {
        pool_index[*worst_idx] = node;
        pool_distance[*worst_idx] = distance;
    }
    __syncwarp();
    result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
}

/**
 * @brief Sort the result pool by ascending distance before QuantizedBeamSearch writes it out.
 *
 * Pools up to 256 entries use the warp-bitonic candidate_by_bitonic_sort, so only warp 0 works;
 * larger pools fall back to an insertion sort on thread 0. Every thread of the block must call
 * it; callers add a barrier after.
 *
 * @param pool_index pool node ids
 * @param pool_distance pool distances
 * @param pool_size entries in the pool
 */
static __device__ inline void result_pool_sort(
    INDEX_T* pool_index, DISTANCE_T* pool_distance, uint32_t pool_size) {
    // [1] warp-bitonic width by pool size
    if (pool_size <= WARP_SIZE) {
        candidate_by_bitonic_sort<1, 0>(pool_index, pool_distance, pool_size);
    } else if (pool_size <= 2 * WARP_SIZE) {
        candidate_by_bitonic_sort<2, 0>(pool_index, pool_distance, pool_size);
    } else if (pool_size <= 4 * WARP_SIZE) {
        candidate_by_bitonic_sort<4, 0>(pool_index, pool_distance, pool_size);
    } else if (pool_size <= 8 * WARP_SIZE) {
        candidate_by_bitonic_sort<8, 0>(pool_index, pool_distance, pool_size);
    } else if (tidx() == 0) {
        // [2] insertion sort for pools above 256
        for (uint32_t i = 1; i < pool_size; ++i) {
            const INDEX_T key_index = pool_index[i];
            const DISTANCE_T key_dist = pool_distance[i];
            int j = static_cast<int>(i) - 1;
            while (j >= 0 && pool_distance[j] > key_dist) {
                pool_distance[j + 1] = pool_distance[j];
                pool_index[j + 1] = pool_index[j];
                --j;
            }
            pool_distance[j + 1] = key_dist;
            pool_index[j + 1] = key_index;
        }
    }
}

/*-------------------------------------------- sizing --------------------------------------------*/
/**
 * @brief Bytes of dynamic shared memory of QuantizedBeamSearch, used by gpu_search_rabitq for the launch.
 *
 * Mirrors the kernel layout, in order: TOP_K_INDEX + CANDIDATE_INDEX, TOP_K_DISTANCE + CANDIDATE_DISTANCE,
 * visited hash table, PARENT_NODE_LIST, PARENT_DISTANCE_LIST, QUERY_BUFFER, RESULT_POOL_INDEX,
 * RESULT_POOL_DISTANCE, then uint4-aligned LUT_BUFFER, SIGN_LUT_BUFFER (TurboQuant only), and one
 * uint4-aligned region holding the larger of the rotated (+ sketch) query and the candidate radix scratch.
 *
 * @param dim raw dimension
 * @param beam_sz beam size
 * @param result_k result pool capacity
 * @param max_deg graph degree
 * @param bitlen visited table bit length
 * @param bits total bits per dimension
 * @param quant quantizer family
 * @return bytes of dynamic shared memory
 */
static __host__ inline uint32_t calculate_shared_mem_size(int dim, int beam_sz, int result_k, int max_deg,
                                                          int bitlen, int bits, QuantType quant) {
    size_t padded_dim = 1ULL << static_cast<size_t>(ceilf(log2f(dim)));
    const size_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(SEARCH_WIDTH * max_deg));
    const uint32_t padded_beam_size = effective_sort_beam_size(static_cast<uint32_t>(beam_sz));
    const size_t result_buffer_size = static_cast<size_t>(padded_beam_size) + candidate_buffer_size;
    size_t size = 0;
    size += result_buffer_size * sizeof(INDEX_T);
    size += result_buffer_size * sizeof(DISTANCE_T);
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);
    size += SEARCH_WIDTH * sizeof(INDEX_T);
    size += SEARCH_WIDTH * sizeof(DISTANCE_T);
    size += dim * sizeof(DATA_T);
    size += result_k * sizeof(INDEX_T);
    size += result_k * sizeof(DISTANCE_T);
    const bool prod = quant == QUANT_TBQ;
    const int stage_bits = prod ? quant_stage(bits) : bits;
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint4)));
    size += quant_lutbytes(padded_dim, stage_bits);
    if (prod) {
        size += quant_lutbytes(padded_dim, 1);
    }
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint4)));
    size_t transient = padded_dim * sizeof(float) * (prod ? 2 : 1);
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && candidate_buffer_size > 256) {
        const size_t radix = align_up_uintptr(size, candidate_radix_sort_scratch_alignment()) - size
            + candidate_radix_sort_scratch_bytes();
        transient = transient > radix ? transient : radix;
    }
    size += transient;
    return static_cast<uint32_t>(size);
}
