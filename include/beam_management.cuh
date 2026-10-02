#pragma once

#include "utils.cuh"

/*-------------------------------------------- beam sizing --------------------------------------------*/
#define GPU_FAST_BITONIC_TOPK_CAP 128
#ifndef GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT
#define GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT 0
#endif

__host__ __device__ inline uint32_t effective_sort_beam_size(uint32_t beam_size) {
    if (beam_size == 0)
        return 0;
    const uint32_t rounded = round_up_power2_u32(beam_size);
    if (rounded <= GPU_FAST_BITONIC_TOPK_CAP) {
        return rounded < WARP_SIZE ? WARP_SIZE : rounded;
    }
    return beam_size;
}

__host__ __device__ inline uint32_t candidate_radix_sort_scratch_bytes() {
    using RadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
    return static_cast<uint32_t>(sizeof(typename RadixSort::TempStorage));
}

__host__ __device__ inline uint32_t candidate_radix_sort_scratch_alignment() {
    using RadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
    return static_cast<uint32_t>(alignof(typename RadixSort::TempStorage));
}

/*-------------------------------------------- sort --------------------------------------------*/
template <class K, class V>
__device__ inline void swap_if_needed(K &k0, V &v0, const unsigned lane_offset, const bool asc) {
    auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
    auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
    if ((k0 != k1) && ((k0 < k1) != asc)) {
        k0 = k1;
        v0 = v1;
    }
}

template <class K, class V>
__device__ inline void swap_if_needed(K &k0, V &v0, K &k1, V &v1, const bool asc) {
    if ((k0 != k1) && ((k0 < k1) != asc)) {
        const auto tmp_k = k0;
        k0 = k1;
        k1 = tmp_k;
        const auto tmp_v = v0;
        v0 = v1;
        v1 = tmp_v;
    }
}

template <class K, class V, unsigned _N, unsigned warp_size>
__device__ inline void warp_merge_core(K k[2], V v[2], const std::uint32_t range, const bool asc) {
    if (_N == 1) {
        const auto lane_id = tidx() % warp_size;
        if (range == 1) {
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
            swap_if_needed<float, uint32_t>(k[0], v[0], c, p);
        }
    } else if (_N == 2) {
        constexpr unsigned N = 2;
        const auto lane_id = tidx() % warp_size;

        if (range == 1) {
            const auto p = ((lane_id & 1) == 0);
            swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                swap_if_needed<float, uint32_t>(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        swap_if_needed<float, uint32_t>(k[0], v[0], k[1], v[1], p);
    } else {
        constexpr unsigned N = _N;
        const auto lane_id = tidx() % warp_size;
        if (range == 1) {
            for (std::uint32_t b = 2; b <= N; b <<= 1) {
                for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
#pragma unroll
                    for (std::uint32_t i = 0; i < N; i++) {
                        std::uint32_t j = i ^ c;
                        if (i >= j)
                            continue;
                        const auto line_id = i + (N * lane_id);
                        const auto p = static_cast<bool>(line_id & b) == static_cast<bool>(line_id & c);
                        swap_if_needed(k[i], v[i], k[j], v[j], p);
                    }
                }
            }
            return;
        }

        const std::uint32_t b = range;
        for (std::uint32_t c = b / 2; c >= 1; c >>= 1) {
            const auto p = static_cast<bool>(lane_id & b) == static_cast<bool>(lane_id & c);
#pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                swap_if_needed(k[i], v[i], c, p);
            }
        }
        const auto p = ((lane_id & b) == 0);
        for (std::uint32_t c = N / 2; c >= 1; c >>= 1) {
#pragma unroll
            for (std::uint32_t i = 0; i < N; i++) {
                std::uint32_t j = i ^ c;
                if (i >= j)
                    continue;
                swap_if_needed(k[i], v[i], k[j], v[j], p);
            }
        }
    }
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
    warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
    for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
        warp_merge<K, V, N, warp_size>(k, v, range, asc);
    }
}

/*-------------------------------------------- sort and merge primitives --------------------------------------------*/
template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
    const unsigned lane_id = tidx() % 32;
    const unsigned warp_id = tidx() / 32;

    // sort 1
    if (warp_id > 0)
        return;
    if (CANDIDATE_BUFFER_SIZE > N_1 * WARP_SIZE) {
        printf("CANDIDATE_BUFFER_SIZE must be <= %u\n", N_1 * WARP_SIZE);
        assert(false);
    }
    // constexpr unsigned N_1 = 2;
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE) {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        } else {
            key_1[i] = FLT_MAX;
            val_1[i] = MAX_INDEX;
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE) {
            candidate_distances[j] = key_1[i];
            candidate_indices[j] = val_1[i];
        }
    }
}

__device__ inline void candidate_by_block_bitonic_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t candidate_buffer_size) {
    if (!is_power2_u32(candidate_buffer_size)) {
        printf("block candidate sort requires power-of-two buffer size, got %u\n",
               candidate_buffer_size);
        assert(false);
    }
    for (uint32_t k = 2; k <= candidate_buffer_size; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            for (uint32_t i = tidx(); i < candidate_buffer_size; i += blockDim.x) {
                const uint32_t ixj = i ^ j;
                if (ixj > i && ixj < candidate_buffer_size) {
                    const bool up = ((i & k) == 0);
                    const DISTANCE_T di = candidate_distances[i];
                    const DISTANCE_T dj = candidate_distances[ixj];
                    if ((di > dj) == up) {
                        candidate_distances[i] = dj;
                        candidate_distances[ixj] = di;
                        const INDEX_T ti = candidate_indices[i];
                        candidate_indices[i] = candidate_indices[ixj];
                        candidate_indices[ixj] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <unsigned ITEMS_PER_THREAD>
__device__ __noinline__ void candidate_by_radix_sort_impl(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t candidate_buffer_size,
    void *radix_scratch) {
    using RadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, ITEMS_PER_THREAD, INDEX_T>;
    typename RadixSort::TempStorage &temp_storage =
        *reinterpret_cast<typename RadixSort::TempStorage *>(radix_scratch);
    DISTANCE_T keys[ITEMS_PER_THREAD];
    INDEX_T values[ITEMS_PER_THREAD];
    for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
        const uint32_t pos = tidx() * ITEMS_PER_THREAD + i;
        if (pos < candidate_buffer_size) {
            keys[i] = candidate_distances[pos];
            values[i] = candidate_indices[pos];
        } else {
            keys[i] = FLT_MAX;
            values[i] = MAX_INDEX;
        }
    }
    RadixSort(temp_storage).Sort(keys, values);
    for (unsigned i = 0; i < ITEMS_PER_THREAD; ++i) {
        const uint32_t pos = tidx() * ITEMS_PER_THREAD + i;
        if (pos < candidate_buffer_size) {
            candidate_distances[pos] = keys[i];
            candidate_indices[pos] = values[i];
        }
    }
}

__device__ __noinline__ void candidate_by_radix_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t candidate_buffer_size,
    void *radix_scratch) {
    const uint32_t items_per_thread =
        (candidate_buffer_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (items_per_thread <= 1) {
        candidate_by_radix_sort_impl<1>(candidate_indices, candidate_distances,
                                        candidate_buffer_size, radix_scratch);
    } else if (items_per_thread <= 2) {
        candidate_by_radix_sort_impl<2>(candidate_indices, candidate_distances,
                                        candidate_buffer_size, radix_scratch);
    } else if (items_per_thread <= 4) {
        candidate_by_radix_sort_impl<4>(candidate_indices, candidate_distances,
                                        candidate_buffer_size, radix_scratch);
    } else {
        assert(items_per_thread <= 8);
        candidate_by_radix_sort_impl<8>(candidate_indices, candidate_distances,
                                        candidate_buffer_size, radix_scratch);
    }
}

template <unsigned N_1, unsigned N_2>
__device__ __forceinline__ void topk_candidate_bitonic_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
    const unsigned lane_id = tidx() % 32;
    const unsigned warp_id = tidx() / 32;

    // Sort part 1
    if (warp_id > 0)
        return;
    assert(CANDIDATE_BUFFER_SIZE <= N_1 * WARP_SIZE);

    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];
    auto candidate_distances = result_distances_ptr + internal_topk;
    auto candidate_indices = result_indices_ptr + internal_topk;
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = lane_id + (32 * i);
        if (j < CANDIDATE_BUFFER_SIZE) {
            key_1[i] = candidate_distances[j];
            val_1[i] = candidate_indices[j];
        } else {
            key_1[i] = FLT_MAX;
            val_1[i] = MAX_INDEX;
        }
    }
    /* Sort */
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE && j < internal_topk) {
            candidate_distances[j] = key_1[i];
            candidate_indices[j] = val_1[i];
        }
    }
    __syncwarp();

    // Sort part 2
    // constexpr unsigned N_2 = 4;
    DISTANCE_T key_2[N_2];
    INDEX_T val_2[N_2];
    if (first) {
        /* Load itopk results */
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = lane_id + (32 * i);
            if (j < internal_topk) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = FLT_MAX;
                val_2[i] = MAX_INDEX;
            }
        }
        /* Warp Sort */
        warp_sort<float, uint32_t, N_2>(key_2, val_2);
    } else {
        /* Load itopk results */
        for (unsigned i = 0; i < N_2; i++) {
            unsigned j = (N_2 * lane_id) + i;
            if (j < internal_topk) {
                key_2[i] = result_distances_ptr[j];
                val_2[i] = result_indices_ptr[j];
            } else {
                key_2[i] = FLT_MAX;
                val_2[i] = MAX_INDEX;
            }
        }
    }

    /* Merge candidates */
    for (unsigned i = 0; i < N_2; i++) {
        unsigned j = (N_2 * lane_id) + i; // [0:MAX_ITOPK-1]
        unsigned k = N_2 * WARP_SIZE - 1 - j;
        if (k >= internal_topk || k >= CANDIDATE_BUFFER_SIZE)
            continue;
        auto candidate_key = candidate_distances[k];
        if (key_2[i] > candidate_key) {
            key_2[i] = candidate_key;
            val_2[i] = candidate_indices[k];
        }
    }
    /* Warp Merge */
    warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

    /* Store new itopk results */
    for (unsigned i = 0; i < N_2; i++) {
        unsigned j = (N_2 * lane_id) + i;
        if (j < internal_topk) {
            result_distances_ptr[j] = key_2[i];
            result_indices_ptr[j] = val_2[i];
        }
    }
}

/**
 * @brief Beam entries among the first `diag` outputs of the stable merge, beam first on ties.
 *
 * A lower-bound search on a monotone predicate, so every diagonal gets the same canonical
 * split even through runs of equal distances.
 *
 * @param beam sorted beam distances
 * @param topk beam length
 * @param cand sorted candidate distances
 * @param count candidate run length
 * @param diag output position
 * @return beam entries before that position
 */
static __device__ __forceinline__ int mergesplit(
    const DISTANCE_T *beam, int topk, const DISTANCE_T *cand, int count, int diag) {
    int low = max(0, diag - count);
    int high = min(diag, topk);
    while (low < high) {
        const int mid = (low + high) >> 1;
        if (beam[mid] <= cand[diag - 1 - mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}

/**
 * @brief Merge the sorted candidate run into the sorted beam in place, one output per thread.
 *
 * Output o comes from beam position at most o, so windows of blockDim outputs are merged from
 * the end of the beam backwards: a window reads only positions below the windows already
 * written, and writes only positions the remaining windows never read. One barrier per window
 * replaces the scratch copy of the beam.
 *
 * @param indices beam indices, followed by the candidate indices
 * @param distances beam distances, followed by the candidate distances
 * @param candidates candidate run length
 * @param topk beam length
 */
static __device__ inline void mergepath(
    INDEX_T *indices, DISTANCE_T *distances, uint32_t candidates, uint32_t topk) {
    const INDEX_T *candindex = indices + topk;
    const DISTANCE_T *canddist = distances + topk;
    const int total = static_cast<int>(topk);
    const int count = static_cast<int>(candidates);
    const int wide = static_cast<int>(blockDim.x);
    for (int base = ((total - 1) / wide) * wide; base >= 0; base -= wide) {
        // [1] pick this thread's output while the positions it depends on are still unwritten
        const int out = base + static_cast<int>(tidx());
        INDEX_T keepindex = MAX_INDEX;
        DISTANCE_T keepdist = FLT_MAX;
        if (out < total) {
            const int top = mergesplit(distances, total, canddist, count, out);
            const int cand = out - top;
            const bool fromtop = cand >= count || (top < total && distances[top] <= canddist[cand]);
            keepdist = fromtop ? distances[top] : canddist[cand];
            keepindex = fromtop ? indices[top] : candindex[cand];
        }
        __syncthreads();

        // [2] the next window reads only below base, so this write overlaps its reads safely
        if (out < total) {
            distances[out] = keepdist;
            indices[out] = keepindex;
        }
    }
    __syncthreads();
}

// Sort candidates with the known warp-bitonic capacity N_1, then merge into the sorted beam.
// Uses the in-place block merge-path; selected for beams above 256.
template <unsigned N_1>
__device__ __noinline__ void dispatch_topk_candidate_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk) {
    auto candidate_indices = result_indices_ptr + internal_topk;
    auto candidate_distances = result_distances_ptr + internal_topk;

    candidate_by_bitonic_sort<N_1, 0>(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
    __syncthreads();

    mergepath(result_indices_ptr, result_distances_ptr, CANDIDATE_BUFFER_SIZE, internal_topk);
}

/*-------------------------------------------- runtime dispatchers --------------------------------------------*/
// Sort candidates only: warp bitonic through 256, block radix above 256.
// The forced-block flag overrides both routes with block bitonic if enabled.
__device__ inline void dispatch_candidate_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t candidate_buffer_size,
    void *candidate_radix_scratch) {
    const uint32_t warp_items_per_thread =
        (candidate_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
    if (GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT) {
        candidate_by_block_bitonic_sort(candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 1) {
        candidate_by_bitonic_sort<1, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 2) {
        candidate_by_bitonic_sort<2, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 4) {
        candidate_by_bitonic_sort<4, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 8) {
        candidate_by_bitonic_sort<8, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else {
        candidate_by_radix_sort(candidate_indices, candidate_distances,
                                candidate_buffer_size, candidate_radix_scratch);
    }
}

// Dispatch candidate sorting at runtime, then merge into the sorted beam in place.
// General fallback for candidates above 256 or forced block sorting, regardless of beam size.
static __device__ inline void dispatch_topk_candidate_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    void *candidate_radix_scratch,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk) {
    auto candidate_indices = result_indices_ptr + internal_topk;
    auto candidate_distances = result_distances_ptr + internal_topk;

    dispatch_candidate_sort(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE,
                                    candidate_radix_scratch);
    __syncthreads();

    mergepath(result_indices_ptr, result_distances_ptr, CANDIDATE_BUFFER_SIZE, internal_topk);
}

// For a known warp-bitonic candidate capacity N_1, select beam capacity 32/64/128/256.
// Larger beams use specialized candidate sorting followed by block merge-path.
template <unsigned N_1>
__device__ inline void dispatch_bitonic_beam_width(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
    if (internal_topk <= 32) {
        topk_candidate_bitonic_sort_and_merge<N_1, 1>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else if (internal_topk <= 64) {
        topk_candidate_bitonic_sort_and_merge<N_1, 2>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else if (internal_topk <= 128) {
        topk_candidate_bitonic_sort_and_merge<N_1, 4>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else {
        dispatch_topk_candidate_sort_and_merge<N_1>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk);
    }
}

// Top-level route: small candidates select a specialized beam-width path.
// Large candidates or forced block sorting use the general sort-and-merge fallback.
__device__ inline void dispatch_beam_management(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    void *candidate_radix_scratch,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
    const uint32_t items_per_thread = (candidate_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
    if (GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT) {
        dispatch_topk_candidate_sort_and_merge(
            result_indices_ptr, result_distances_ptr,
            candidate_radix_scratch,
            candidate_buffer_size, internal_topk);
    } else if (items_per_thread <= 1) {
        dispatch_bitonic_beam_width<1>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else if (items_per_thread <= 2) {
        dispatch_bitonic_beam_width<2>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else if (items_per_thread <= 4) {
        dispatch_bitonic_beam_width<4>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else if (items_per_thread <= 8) {
        dispatch_bitonic_beam_width<8>(
            result_indices_ptr, result_distances_ptr,
            candidate_buffer_size, internal_topk, first);
    } else {
        dispatch_topk_candidate_sort_and_merge(
            result_indices_ptr, result_distances_ptr,
            candidate_radix_scratch, candidate_buffer_size, internal_topk);
    }
}
