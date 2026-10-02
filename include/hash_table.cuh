#pragma once

#include "utils.cuh"

/*-------------------------------------------- hash table --------------------------------------------*/
/**
 * @brief Minimum log2 capacity of the visited hash in the fallback path of
 * hash_bitlen_for_search_workload; the calibrated path ignores it.
 */
#ifndef GPU_HASH_BASE_BITLEN
#define GPU_HASH_BASE_BITLEN 8
#endif

/**
 * @brief Search iterations between visited-hash resets in the adaptive and PathW kernels; also
 * the reset_interval the launchers pass to hash_bitlen_for_search_workload.
 */
#ifndef SMALL_HASH_RESET_INTERVAL
#define SMALL_HASH_RESET_INTERVAL 16
#endif

/**
 * @brief Slot count of a visited hash table with 2^bitlen slots; sizes its shared-memory region.
 * @param bitlen log2 of the slot count
 * @return 1 << bitlen
 */
__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
    return 1 << bitlen;
}

/**
 * @brief Mark every slot of the visited hash empty (MAX_INDEX), using threads from FIRST_TID up.
 *
 * Threads below FIRST_TID return at once, so the adaptive kernel can reset with warps 1 and up
 * while warp 0 picks parents. No barrier inside.
 *
 * @param table hash slots
 * @param bitlen log2 of the slot count
 * @param FIRST_TID first participating thread
 */
__device__ inline void hashtable_init(INDEX_T *const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
    if (tidx() < FIRST_TID)
        return;
    for (uint32_t i = tidx() - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
        table[i] = MAX_INDEX;
    }
}

/**
 * @brief Insert key into the visited hash; records a node as visited.
 *
 * Open addressing with linear probing; atomicCAS on an empty slot makes concurrent inserts of
 * the same key report exactly one success.
 *
 * @param table hash slots
 * @param bitlen log2 of the slot count
 * @param key node index
 * @return 1 if key was newly inserted, 0 if already present or the table is full
 */
__device__ inline uint32_t hashtable_insert(INDEX_T *const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    INDEX_T index = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = atomicCAS(&table[index], ~static_cast<INDEX_T>(0), key);
        if (old == ~static_cast<INDEX_T>(0)) {
            return 1;
        } else if (old == key) {
            return 0;
        }
        index = (index + stride) & bit_mask;
    }
    return 0;
}

/**
 * @brief Read-only lookup of key in the visited hash; filters already visited neighbors.
 *
 * Probes the same sequence as hashtable_insert without atomics and stops at the first empty slot.
 *
 * @param table hash slots
 * @param bitlen log2 of the slot count
 * @param key node index
 * @return 1 if key is present, otherwise 0
 */
__device__ inline uint32_t hashtable_contains(INDEX_T *const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    INDEX_T index = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    for (unsigned i = 0; i < size; i++) {
        const INDEX_T old = table[index];
        if (old == key)
            return 1;
        if (old == ~static_cast<INDEX_T>(0))
            return 0;
        index = (index + stride) & bit_mask;
    }
    return 0;
}

/**
 * @brief Reinsert the beam nodes into the visited hash after a reset, using threads from first_tid up.
 *
 * MAX_INDEX entries are skipped and the expanded flag (most significant bit) is cleared before insert.
 *
 * @param table hash slots
 * @param BITLEN log2 of the slot count
 * @param itopk_indices beam node indices
 * @param itopk_size beam length
 * @param first_tid first participating thread
 */
__device__ inline void hashtable_restore(
    INDEX_T *const table,
    const unsigned BITLEN,
    const INDEX_T *itopk_indices,
    const uint32_t itopk_size,
    const uint32_t first_tid = 0) {
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    if (tidx() < first_tid)
        return;
    for (unsigned i = tidx() - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
        const auto raw_key = itopk_indices[i];
        if (raw_key == MAX_INDEX) {
            continue;
        }
        auto key = raw_key & ~index_msb_1_mask;
        hashtable_insert(table, BITLEN, key);
    }
}

/**
 * @brief Log2 capacity of the visited hash for one search shape; the launchers call it on the host.
 * @param beam_size padded beam size
 * @param candidate_buffer_size candidate buffer capacity
 * @param reset_interval iterations between hash resets
 * @return log2 of the slot count
 */
__host__ __device__ inline uint32_t hash_bitlen_for_search_workload(
    uint32_t beam_size, uint32_t candidate_buffer_size, uint32_t reset_interval) {
    // [1] calibrated shape (beam 32 to 1024, 32 candidates, reset every 16 iterations): 1024 or 2048 slots
    if (beam_size >= 32u && beam_size <= 1024u &&
        candidate_buffer_size == 32u && reset_interval == 16u) {
        return beam_size <= 512u ? 10u : 11u;
    }

    // [2] other shapes: fit beam_size + candidate_buffer_size * reset_interval keys at load factor at most 3/4
    const uint64_t required64 = static_cast<uint64_t>(beam_size) + static_cast<uint64_t>(candidate_buffer_size) * reset_interval;
    const uint32_t required = required64 > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(required64);
    uint32_t target_capacity = (required * 4u + 2u) / 3u;
    uint32_t capacity = 1u << GPU_HASH_BASE_BITLEN;
    while (capacity < target_capacity) {
        capacity <<= 1;
    }

    // [3] convert the power-of-two capacity to its bit length
    uint32_t bitlen = 0;
    while ((1u << bitlen) < capacity) {
        ++bitlen;
    }
    return bitlen;
}
