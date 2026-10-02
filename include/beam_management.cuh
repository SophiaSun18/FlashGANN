#pragma once

#include "utils.cuh"

/*-------------------------------------------- beam sizing --------------------------------------------*/
/**
 * @brief Largest beam merged by the warp-bitonic route of dispatch_bitonic_beam_width; also the
 * padding limit of effective_sort_beam_size.
 */
#define GPU_FAST_BITONIC_TOPK_CAP 128

/**
 * @brief When 1, every candidate sort uses candidate_by_block_bitonic_sort and the beam merge uses
 * the general merge-path route; the kernels then allocate no radix scratch.
 */
#ifndef GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT
#define GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT 0
#endif

/**
 * @brief Padded beam length that the kernels allocate and pass to dispatch_beam_management.
 *
 * Beams up to GPU_FAST_BITONIC_TOPK_CAP round up to a power of two, at least WARP_SIZE, to fit
 * the warp-bitonic merge; larger beams stay as given.
 *
 * @param beam_size requested beam length
 * @return the padded length, 0 for 0
 */
__host__ __device__ inline uint32_t effective_sort_beam_size(uint32_t beam_size) {
    if (beam_size == 0)
        return 0;
    const uint32_t rounded = round_up_power2_u32(beam_size);
    if (rounded <= GPU_FAST_BITONIC_TOPK_CAP) {
        return rounded < WARP_SIZE ? WARP_SIZE : rounded;
    }
    return beam_size;
}

/**
 * @brief Bytes of the shared-memory scratch for candidate_by_radix_sort, sized for 8 items per thread.
 *
 * Shared-memory sizing adds it when candidates exceed 256 and GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT is 0.
 *
 * @return sizeof the cub::BlockRadixSort TempStorage
 */
__host__ __device__ inline uint32_t candidate_radix_sort_scratch_bytes() {
    using RadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
    return static_cast<uint32_t>(sizeof(typename RadixSort::TempStorage));
}

/**
 * @brief Alignment of the scratch sized by candidate_radix_sort_scratch_bytes.
 * @return alignof the cub::BlockRadixSort TempStorage
 */
__host__ __device__ inline uint32_t candidate_radix_sort_scratch_alignment() {
    using RadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
    return static_cast<uint32_t>(alignof(typename RadixSort::TempStorage));
}

/*-------------------------------------------- sort --------------------------------------------*/
/**
 * @brief Cross-lane compare-exchange of one bitonic step: trade (k0, v0) with lane ^ lane_offset.
 * @tparam K key type
 * @tparam V value type
 * @param k0 this lane's key
 * @param v0 this lane's value
 * @param lane_offset XOR distance to the partner lane
 * @param asc true keeps the smaller key, false the larger
 */
template <class K, class V>
__device__ inline void swap_if_needed(K &k0, V &v0, const unsigned lane_offset, const bool asc) {
    auto k1 = __shfl_xor_sync(~0u, k0, lane_offset);
    auto v1 = __shfl_xor_sync(~0u, v0, lane_offset);
    if ((k0 != k1) && ((k0 < k1) != asc)) {
        k0 = k1;
        v0 = v1;
    }
}

/**
 * @brief In-register compare-exchange of one bitonic step between two pairs held by one lane.
 * @tparam K key type
 * @tparam V value type
 * @param k0 first key
 * @param v0 first value
 * @param k1 second key
 * @param v1 second value
 * @param asc true leaves the smaller key in k0, false the larger
 */
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

/**
 * @brief One bitonic merge stage over runs of range lanes, _N keys per lane.
 *
 * Lane l holds positions _N * l onward. range 1 sorts within each lane; larger ranges exchange
 * across lanes, then within registers. Directions come from lane and position bits, so asc is
 * unused. The _N 1 and 2 branches fix K to float and V to uint32_t.
 *
 * @tparam K key type
 * @tparam V value type
 * @tparam _N keys per lane
 * @tparam warp_size lanes per warp
 * @param k keys
 * @param v values
 * @param range lanes per merged run
 * @param asc unused
 */
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

/**
 * @brief One bitonic merge stage of a warp-held sequence; forwards to warp_merge_core.
 * @tparam K key type
 * @tparam V value type
 * @tparam N keys per lane
 * @tparam warp_size lanes per warp
 * @param k keys
 * @param v values
 * @param range lanes per merged run
 * @param asc passed through, unused by warp_merge_core
 */
template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_merge(K k[N], V v[N], unsigned range, const bool asc = true) {
    warp_merge_core<K, V, N, warp_size>(k, v, range, asc);
}

/**
 * @brief Ascending bitonic sort of N * warp_size pairs held by one warp, N per lane.
 *
 * Runs warp_merge for range 1, 2, 4, up to warp_size. On return lane l holds sorted positions
 * N * l to N * l + N - 1. All 32 lanes of the warp must call it.
 *
 * @tparam K key type
 * @tparam V value type
 * @tparam N keys per lane
 * @tparam warp_size lanes per warp
 * @param k keys
 * @param v values
 * @param asc passed through, unused by warp_merge_core
 */
template <class K, class V, unsigned N, unsigned warp_size = 32>
__device__ void warp_sort(K k[N], V v[N], const bool asc = true) {
    for (std::uint32_t range = 1; range <= warp_size; range <<= 1) {
        warp_merge<K, V, N, warp_size>(k, v, range, asc);
    }
}

/*-------------------------------------------- sort and merge primitives --------------------------------------------*/
/**
 * @brief Sort the candidate buffer ascending by distance in place with one warp-bitonic sort.
 *
 * Only warp 0 works; other warps return at once. Requires CANDIDATE_BUFFER_SIZE <= N_1 * 32.
 *
 * @tparam N_1 candidates per lane
 * @tparam N_2 unused
 * @param candidate_indices candidate node indices
 * @param candidate_distances candidate distances, the sort key
 * @param CANDIDATE_BUFFER_SIZE candidate count
 */
template <unsigned N_1, unsigned N_2>
__device__ void candidate_by_bitonic_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t CANDIDATE_BUFFER_SIZE) {
    const unsigned lane_id = tidx() % 32;
    const unsigned warp_id = tidx() / 32;

    // [1] keep warp 0 and check the capacity
    if (warp_id > 0)
        return;
    if (CANDIDATE_BUFFER_SIZE > N_1 * WARP_SIZE) {
        printf("CANDIDATE_BUFFER_SIZE must be <= %u\n", N_1 * WARP_SIZE);
        assert(false);
    }
    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];

    // [2] load candidates strided by lane, padding with FLT_MAX
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
    // [3] warp sort, then store lane l's keys at positions N_1 * l onward
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE) {
            candidate_distances[j] = key_1[i];
            candidate_indices[j] = val_1[i];
        }
    }
}

/**
 * @brief Sort the candidate buffer ascending by distance in place with a block-wide bitonic network.
 *
 * Used when GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT is 1. Each thread takes compare pairs i, i +
 * blockDim.x, and so on, with one barrier per step, so every thread of the block must call it.
 * Requires a power-of-two candidate_buffer_size.
 *
 * @param candidate_indices candidate node indices
 * @param candidate_distances candidate distances, the sort key
 * @param candidate_buffer_size candidate count
 */
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

/**
 * @brief Sort the candidate buffer ascending by distance in place with cub::BlockRadixSort.
 *
 * Thread t holds positions ITEMS_PER_THREAD * t onward, padding with FLT_MAX. Every thread of
 * the BLOCK_SIZE block must call it. Requires candidate_buffer_size <= BLOCK_SIZE * ITEMS_PER_THREAD.
 *
 * @tparam ITEMS_PER_THREAD candidates per thread
 * @param candidate_indices candidate node indices
 * @param candidate_distances candidate distances, the sort key
 * @param candidate_buffer_size candidate count
 * @param radix_scratch shared-memory TempStorage
 */
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

/**
 * @brief Block radix sort of the candidate buffer, choosing 1, 2, 4, or 8 items per thread from
 * candidate_buffer_size; serves candidate buffers above 256.
 *
 * Asserts candidate_buffer_size <= 8 * BLOCK_SIZE. Every thread of the block must call it.
 *
 * @param candidate_indices candidate node indices
 * @param candidate_distances candidate distances, the sort key
 * @param candidate_buffer_size candidate count
 * @param radix_scratch shared-memory scratch sized by candidate_radix_sort_scratch_bytes
 */
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

/**
 * @brief Sort the candidates and merge them into the beam with warp-bitonic steps; serves beams up
 * to GPU_FAST_BITONIC_TOPK_CAP.
 *
 * Only warp 0 works; other warps return at once. The beam occupies the first internal_topk
 * slots, the candidates follow. The beam is taken as sorted unless first is set.
 *
 * @tparam N_1 candidates per lane, CANDIDATE_BUFFER_SIZE <= N_1 * 32
 * @tparam N_2 beam entries per lane, internal_topk <= N_2 * 32
 * @param result_indices_ptr beam indices, followed by the candidate indices
 * @param result_distances_ptr beam distances, followed by the candidate distances
 * @param CANDIDATE_BUFFER_SIZE candidate count
 * @param internal_topk beam length
 * @param first true to sort the beam before merging
 */
template <unsigned N_1, unsigned N_2>
__device__ __forceinline__ void topk_candidate_bitonic_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
    const unsigned lane_id = tidx() % 32;
    const unsigned warp_id = tidx() / 32;

    // [1] keep warp 0 and load candidates strided by lane, padding with FLT_MAX
    if (warp_id > 0)
        return;
    assert(CANDIDATE_BUFFER_SIZE <= N_1 * WARP_SIZE);

    DISTANCE_T key_1[N_1];
    INDEX_T val_1[N_1];
    auto candidate_distances = result_distances_ptr + internal_topk;
    auto candidate_indices = result_indices_ptr + internal_topk;
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
    // [2] sort the candidates and store back the first internal_topk of them
    warp_sort<float, uint32_t, N_1>(key_1, val_1);
    for (unsigned i = 0; i < N_1; i++) {
        unsigned j = (N_1 * lane_id) + i;
        if (j < CANDIDATE_BUFFER_SIZE && j < internal_topk) {
            candidate_distances[j] = key_1[i];
            candidate_indices[j] = val_1[i];
        }
    }
    __syncwarp();

    // [3] load the beam N_2 per lane: strided then sorted when first, otherwise in sorted order
    DISTANCE_T key_2[N_2];
    INDEX_T val_2[N_2];
    if (first) {
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
        warp_sort<float, uint32_t, N_2>(key_2, val_2);
    } else {
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

    // [4] keep the smaller of beam position j and reversed candidate position, giving a bitonic sequence
    for (unsigned i = 0; i < N_2; i++) {
        unsigned j = (N_2 * lane_id) + i;
        unsigned k = N_2 * WARP_SIZE - 1 - j;
        if (k >= internal_topk || k >= CANDIDATE_BUFFER_SIZE)
            continue;
        auto candidate_key = candidate_distances[k];
        if (key_2[i] > candidate_key) {
            key_2[i] = candidate_key;
            val_2[i] = candidate_indices[k];
        }
    }
    // [5] bitonic-merge into ascending order and store the new beam
    warp_merge<float, uint32_t, N_2>(key_2, val_2, 32);

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
 * the end of the beam backwards: each window reads only positions below those already written.
 * One barrier per window replaces a scratch copy of the beam.
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

/**
 * @brief Warp-bitonic sort of the candidates, then mergepath into the sorted beam; serves beams
 * above GPU_FAST_BITONIC_TOPK_CAP with candidates up to N_1 * 32.
 *
 * Every thread of the block must call it.
 *
 * @tparam N_1 candidates per lane
 * @param result_indices_ptr beam indices, followed by the candidate indices
 * @param result_distances_ptr beam distances, followed by the candidate distances
 * @param CANDIDATE_BUFFER_SIZE candidate count
 * @param internal_topk beam length
 */
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
/**
 * @brief Sort the candidate buffer in place, picking the sort from candidate_buffer_size; also
 * used alone by the adaptive kernel before its keep-budget cut.
 *
 * Warp bitonic up to 256 candidates, block radix above 256; GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT
 * forces block bitonic. Every thread of the block must call it.
 *
 * @param candidate_indices candidate node indices
 * @param candidate_distances candidate distances, the sort key
 * @param candidate_buffer_size candidate count
 * @param candidate_radix_scratch radix scratch, may be nullptr when the radix route is not taken
 */
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

/**
 * @brief Sort the candidates with dispatch_candidate_sort, then mergepath them into the sorted beam;
 * the route for candidates above 256 or GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT, at any beam size.
 *
 * Every thread of the block must call it.
 *
 * @param result_indices_ptr beam indices, followed by the candidate indices
 * @param result_distances_ptr beam distances, followed by the candidate distances
 * @param candidate_radix_scratch radix scratch for dispatch_candidate_sort
 * @param CANDIDATE_BUFFER_SIZE candidate count
 * @param internal_topk beam length
 */
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

/**
 * @brief For candidates up to N_1 * 32, choose the beam merge by internal_topk.
 *
 * Beams up to 32, 64, and 128 use topk_candidate_bitonic_sort_and_merge with 1, 2, and 4 entries
 * per lane; larger beams use the warp-sort plus mergepath route.
 *
 * @tparam N_1 candidates per lane
 * @param result_indices_ptr beam indices, followed by the candidate indices
 * @param result_distances_ptr beam distances, followed by the candidate distances
 * @param candidate_buffer_size candidate count
 * @param internal_topk beam length
 * @param first true to sort the beam before merging, warp-bitonic routes only
 */
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

/**
 * @brief Sort the candidates and merge them into the beam in place; the per-iteration beam update
 * of the adaptive, PathW, and RaBitQ kernels.
 *
 * Candidates up to 256 go to dispatch_bitonic_beam_width; larger ones, or
 * GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT, go to the general sort-and-mergepath route. Every thread
 * of the block must call it; callers add a barrier after.
 *
 * @param result_indices_ptr beam indices, followed by the candidate indices
 * @param result_distances_ptr beam distances, followed by the candidate distances
 * @param candidate_radix_scratch radix scratch, may be nullptr for candidates up to 256
 * @param candidate_buffer_size candidate count
 * @param internal_topk padded beam length
 * @param first true on the first iteration, so the warp-bitonic routes sort the beam
 */
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
