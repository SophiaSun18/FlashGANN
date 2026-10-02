#pragma once

#include <cuda_runtime.h>

#include "include/utils.cuh"

// every iteration expands SEARCH_WIDTH parents; the remaining knobs come from the adaptive search config
#ifndef SEARCH_WIDTH
#define SEARCH_WIDTH 1
#endif

#include "adaptive_search_config.cuh"

/**
 * @brief Recompute the largest distance of the unsorted result pool.
 * @param pool_distance pool distances
 * @param pool_size entries in the pool
 * @param worst_idx receives the position of the largest distance
 * @param worst_dist receives the largest distance
 */
static __device__ inline void result_pool_recompute_worst(
    const DISTANCE_T* pool_distance, uint32_t pool_size, uint32_t* worst_idx, DISTANCE_T* worst_dist) {
    if (pool_size == 0) {
        *worst_idx = 0;
        *worst_dist = -FLT_MAX;
        return;
    }
    uint32_t local_worst_idx = 0;
    DISTANCE_T local_worst_dist = pool_distance[0];
    for (uint32_t i = 1; i < pool_size; ++i) {
        if (pool_distance[i] > local_worst_dist) {
            local_worst_dist = pool_distance[i];
            local_worst_idx = i;
        }
    }
    *worst_idx = local_worst_idx;
    *worst_dist = local_worst_dist;
}

/**
 * @brief Insert a node into the unsorted result pool, keeping the closest capacity entries.
 * @param pool_index pool node ids
 * @param pool_distance pool distances
 * @param pool_size entries in the pool
 * @param worst_idx position of the largest distance
 * @param worst_dist largest distance
 * @param capacity pool capacity
 * @param node node id
 * @param distance exact distance of the node
 */
static __device__ inline void result_pool_push_unique_unsorted(
    INDEX_T* pool_index, DISTANCE_T* pool_distance, uint32_t* pool_size, uint32_t* worst_idx,
    DISTANCE_T* worst_dist, int capacity, INDEX_T node, DISTANCE_T distance) {
    if (node == MAX_INDEX || !isfinite(distance)) return;

    uint32_t size = *pool_size;
    for (uint32_t i = 0; i < size; ++i) {
        if (pool_index[i] == node) {
            if (distance < pool_distance[i]) {
                pool_distance[i] = distance;
                result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
            }
            return;
        }
    }

    if (size < static_cast<uint32_t>(capacity)) {
        pool_index[size] = node;
        pool_distance[size] = distance;
        *pool_size = size + 1;
        if (size == 0 || distance > *worst_dist) {
            *worst_idx = size;
            *worst_dist = distance;
        }
        return;
    }

    if (distance >= *worst_dist) return;

    pool_index[*worst_idx] = node;
    pool_distance[*worst_idx] = distance;
    result_pool_recompute_worst(pool_distance, size, worst_idx, worst_dist);
}

/**
 * @brief Insertion-sort the result pool by ascending distance.
 * @param pool_index pool node ids
 * @param pool_distance pool distances
 * @param pool_size entries in the pool
 */
static __device__ inline void result_pool_sort_small(
    INDEX_T* pool_index, DISTANCE_T* pool_distance, uint32_t pool_size) {
    for (uint32_t i = 1; i < pool_size; ++i) {
        INDEX_T key_index = pool_index[i];
        DISTANCE_T key_dist = pool_distance[i];
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

/**
 * @brief Dynamic shared memory of the estimate-only beam search.
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
    size += result_buffer_size * sizeof(INDEX_T);              // TOP_K_INDEX + CANDIDATE_INDEX
    size += result_buffer_size * sizeof(DISTANCE_T);           // TOP_K_DISTANCE + CANDIDATE_DISTANCE
    size += hashtable_getsize(bitlen) * sizeof(INDEX_T);       // visited-node hash table
    size += SEARCH_WIDTH * sizeof(INDEX_T);                    // PARENT_NODE_LIST
    size += SEARCH_WIDTH * sizeof(DISTANCE_T);                 // PARENT_DISTANCE_LIST: exact parent distances
    size += dim * sizeof(DATA_T);                              // QUERY_BUFFER
    size += result_k * sizeof(INDEX_T);                        // RESULT_POOL_INDEX
    size += result_k * sizeof(DISTANCE_T);                     // RESULT_POOL_DISTANCE
    const bool prod = quant == QUANT_TBQ;
    const int stage_bits = prod ? quant_stage(bits) : bits;
    size = static_cast<size_t>(align_up_uintptr(size, alignof(uint4)));
    size += quant_lutbytes(padded_dim, stage_bits);            // LUT_BUFFER
    if (prod) {
        size += quant_lutbytes(padded_dim, 1);                 // SIGN_LUT_BUFFER
    }
    // one region for buffers never live together: rotated (+ sketch) query, radix scratch
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
