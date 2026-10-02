#pragma once

#include "utils.cuh"

/*-------------------------------------------- hash table --------------------------------------------*/
// Floor for the visited-hash capacity. The workload-derived size in
// hash_bitlen_for_search_workload() is what normally decides; this only sets the minimum.
#ifndef GPU_HASH_BASE_BITLEN
#define GPU_HASH_BASE_BITLEN 8
#endif

#ifndef SMALL_HASH_RESET_INTERVAL
#define SMALL_HASH_RESET_INTERVAL 16
#endif

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
    return 1 << bitlen;
}

__device__ inline void hashtable_init(INDEX_T *const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
    if (tidx() < FIRST_TID)
        return;
    for (uint32_t i = tidx() - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
        table[i] = MAX_INDEX;
    }
}

__device__ inline uint32_t hashtable_insert(INDEX_T *const table, const unsigned bitlen, const INDEX_T key) {
    // Open addressing is used for collision resolution
    const uint32_t size = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    // Linear probing
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

__device__ inline uint32_t hashtable_contains(INDEX_T *const table, const unsigned bitlen, const INDEX_T key) {
    const uint32_t size = hashtable_getsize(bitlen);
    const uint32_t bit_mask = size - 1;

    INDEX_T index = (key ^ (key >> bitlen)) & bit_mask;
    constexpr uint32_t stride = 1;

    // only check if the slot is filled, no actual insertion
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
        auto key = raw_key & ~index_msb_1_mask; // clear most significant bit
        hashtable_insert(table, BITLEN, key);
    }
}

__host__ __device__ inline uint32_t hash_bitlen_for_search_workload(
    uint32_t beam_size, uint32_t candidate_buffer_size, uint32_t reset_interval) {
    // calibrated for beams 32–1024, candidate capacity 32, and a reset every 16 iterations.
    if (beam_size >= 32u && beam_size <= 1024u &&
        candidate_buffer_size == 32u && reset_interval == 16u) {
        return beam_size <= 512u ? 10u : 11u;  // 1024 or 2048 slots
    }

    // uncalibrated shape: fall back to the worst-case bound
    const uint64_t required64 = static_cast<uint64_t>(beam_size) + static_cast<uint64_t>(candidate_buffer_size) * reset_interval;
    const uint32_t required = required64 > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(required64);
    uint32_t target_capacity = (required * 4u + 2u) / 3u;
    uint32_t capacity = 1u << GPU_HASH_BASE_BITLEN;
    while (capacity < target_capacity) {
        capacity <<= 1;
    }
    uint32_t bitlen = 0;
    while ((1u << bitlen) < capacity) {
        ++bitlen;
    }
    return bitlen;
}
