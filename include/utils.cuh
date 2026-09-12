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
#include "quant.hpp"

#define FULL_MASK 0xffffffff

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

#define GPU_HASH_BASE_BITLEN 11
#define GPU_FAST_BITONIC_TOPK_CAP 128
#define GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT 0

#ifndef GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES
#define GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES 32
#endif

#define MAX_INDEX UINT_MAX

#define SHFL_DOWN(val, offset) __shfl_down_sync(FULL_MASK, val, offset)
#define SHFL(val, lane) __shfl_sync(FULL_MASK, val, lane)

/*-------------------------------------------- cuda --------------------------------------------*/
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
typedef float DATA_T;
typedef uint32_t INDEX_T;
typedef float DISTANCE_T;

struct QueryFactors {
    float *rotated_query = nullptr;
    uint8_t *quantized_query = nullptr;
    uint8_t *lut = nullptr;
    float low_val = 0.0f;
    float high_val = 0.0f;
    float width = 0.0f;
    int32_t sum_q = 0;
};

/*-------------------------------------------- common --------------------------------------------*/
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

__host__ __device__ inline bool is_power2_u32(uint32_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

__host__ __device__ inline uintptr_t align_up_uintptr(uintptr_t value, uintptr_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

__host__ __device__ inline int clamp_int(int value, int lo, int hi) {
    if (value < lo)
        return lo;
    if (value > hi)
        return hi;
    return value;
}

__host__ __device__ inline float clamp_float(float value, float lo, float hi) {
    if (value < lo)
        return lo;
    if (value > hi)
        return hi;
    return value;
}

__host__ __device__ inline uint32_t keep_count_per_parent(uint32_t max_degree, float rho) {
    float keep_fraction = 1.0f - rho;
    if (keep_fraction < 0.0f)
        keep_fraction = 0.0f;
    if (keep_fraction > 1.0f)
        keep_fraction = 1.0f;

    uint32_t keep_count = static_cast<uint32_t>(ceilf(static_cast<float>(max_degree) * keep_fraction));
    if (keep_count < 1)
        keep_count = 1;
    if (keep_count > max_degree)
        keep_count = max_degree;
    return keep_count;
}

__host__ __device__ inline uint32_t effective_sort_beam_size(uint32_t beam_size) {
    if (beam_size == 0)
        return 0;
    const uint32_t rounded = round_up_power2_u32(beam_size);
    if (rounded <= GPU_FAST_BITONIC_TOPK_CAP) {
        return rounded < WARP_SIZE ? WARP_SIZE : rounded;
    }
    return beam_size;
}

__host__ __device__ inline bool topk_external_merge_scratch_needed(uint32_t internal_topk) {
    return GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT ||
#if defined(SEARCH_WIDTH) && SEARCH_WIDTH > 4
           true ||
#endif
           ((internal_topk + WARP_SIZE - 1) / WARP_SIZE) > 4;
}

__host__ __device__ inline uint32_t hash_bitlen_for_search_workload(
    uint32_t beam_size, uint32_t candidate_buffer_size, uint32_t reset_interval) {
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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890) && (__CUDA_ARCH__ < 900)
#define GPU_MAX_WARPS_PER_SM 48
#else
#define GPU_MAX_WARPS_PER_SM 64
#endif

#define GPU_MIN_BLOCKS_PER_SM_FOR(block_size) (GPU_MAX_WARPS_PER_SM / WARPS_PER_BLOCK)
#define GPU_LAUNCH_BOUNDS(block_size) __launch_bounds__((block_size), GPU_MIN_BLOCKS_PER_SM_FOR(block_size))

/*-------------------------------------------- warp and block helpers --------------------------------------------*/
static __device__ __forceinline__ uint32_t block_reduce_sum_u32(uint32_t thread_count, uint32_t *warp_counts) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;

    uint32_t sum = thread_count;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    if (lane_id == 0)
        warp_counts[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t total = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(FULL_MASK, total, offset);
        }
        if (lane_id == 0)
            warp_counts[0] = total;
    }
    __syncthreads();
    return warp_counts[0];
}

static __device__ __forceinline__ uint32_t allocate_warp_compact_slots(
    uint32_t accepted_count, uint32_t *compact_count, uint32_t *warp_counts, uint32_t *warp_bases) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0)
        warp_counts[warp_id] = accepted_count;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t count = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        uint32_t inclusive = count;
        for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
            const uint32_t other = __shfl_up_sync(FULL_MASK, inclusive, offset);
            if (lane_id >= offset)
                inclusive += other;
        }
        if (lane_id < WARPS_PER_BLOCK) warp_bases[lane_id] = inclusive - count;
        const uint32_t round_total = __shfl_sync(FULL_MASK, inclusive, WARPS_PER_BLOCK - 1);
        if (lane_id == 0) {
            const uint32_t block_base = *compact_count;
            *compact_count = block_base + round_total;
            warp_bases[WARPS_PER_BLOCK] = block_base;
        }
    }
    __syncthreads();
    return warp_bases[WARPS_PER_BLOCK] + warp_bases[warp_id];
}

template <typename INDEX_T, typename DISTANCE_T>
static __device__ __forceinline__ uint32_t count_valid_candidates_warp_reduced(
    const INDEX_T *candidate_index, const DISTANCE_T *candidate_distance,
    int candidate_count, INDEX_T invalid_index, uint32_t *warp_counts) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0)
        warp_counts[warp_id] = 0;
    __syncthreads();

    uint32_t warp_valid_count = 0;
    for (int base = warp_id * WARP_SIZE; base < candidate_count; base += WARPS_PER_BLOCK * WARP_SIZE) {
        const int i = base + lane_id;
        const bool valid = i < candidate_count && candidate_index[i] != invalid_index && candidate_distance[i] < FLT_MAX;
        warp_valid_count += __popc(__ballot_sync(FULL_MASK, valid));
    }

    if (lane_id == 0)
        warp_counts[warp_id] = warp_valid_count;
    __syncthreads();

    if (warp_id == 0) {
        uint32_t total = (lane_id < WARPS_PER_BLOCK) ? warp_counts[lane_id] : 0u;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(FULL_MASK, total, offset);
        }
        if (lane_id == 0)
            warp_counts[0] = total;
    }
    __syncthreads();

    return warp_counts[0];
}

static __device__ __forceinline__ bool warp_keep_topk_smallest_f32(float value, bool valid, int keep_k) {
    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int valid_int = valid ? 1 : 0;
    const float my_value = valid ? value : FLT_MAX;
    int rank = 0;

#pragma unroll
    for (int src = 0; src < WARP_SIZE; ++src) {
        const float other_value = __shfl_sync(FULL_MASK, my_value, src);
        const int other_valid = __shfl_sync(FULL_MASK, valid_int, src);
        if (other_valid && (other_value < my_value || (other_value == my_value && src < lane_id))) {
            rank++;
        }
    }

    return valid && keep_k > 0 && rank < keep_k;
}

/*-------------------------------------------- distance --------------------------------------------*/
template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T *a, const T *b) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1); // thread index within the warp
    T val = 0.;
    for (int i = thread_lane; i < dim; i += WARP_SIZE)
        val += (a[i] - b[i]) * (a[i] - b[i]);
    T sum = val;
    sum += SHFL_DOWN(sum, 16);
    sum += SHFL_DOWN(sum, 8);
    sum += SHFL_DOWN(sum, 4);
    sum += SHFL_DOWN(sum, 2);
    sum += SHFL_DOWN(sum, 1);
    sum = SHFL(sum, 0);
    return sum;
}

template <typename T = float>
__device__ __forceinline__ T warp_ip_distance(int dim, const T *a, const T *b) {
    int thread_lane = threadIdx.x & (WARP_SIZE - 1);
    T val = 0.;
    for (int i = thread_lane; i < dim; i += WARP_SIZE)
        val += a[i] * b[i];
    T sum = val;
    sum += SHFL_DOWN(sum, 16);
    sum += SHFL_DOWN(sum, 8);
    sum += SHFL_DOWN(sum, 4);
    sum += SHFL_DOWN(sum, 2);
    sum += SHFL_DOWN(sum, 1);
    sum = SHFL(sum, 0);
    return -sum;
}

template <typename T = float>
__device__ __forceinline__ T warp_distance(int dim, const T *a, const T *b, bool use_ip) {
    return use_ip ? warp_ip_distance(dim, a, b) : warp_l2_distance(dim, a, b);
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
        const auto lane_id = threadIdx.x % warp_size;
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
        const auto lane_id = threadIdx.x % warp_size;

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
        const auto lane_id = threadIdx.x % warp_size;
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
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

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
            for (uint32_t i = threadIdx.x; i < candidate_buffer_size; i += blockDim.x) {
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

template <unsigned N_1, unsigned N_2>
__device__ void topk_small_by_bitonic_sort(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
    const unsigned lane_id = threadIdx.x % 32;
    const unsigned warp_id = threadIdx.x / 32;

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
        unsigned k = internal_topk - 1 - j;
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

__device__ inline int merge_path_partition(
    const DISTANCE_T *a,
    int a_count,
    const DISTANCE_T *b,
    int b_count,
    int diag) {
    int low = max(0, diag - b_count);
    int high = min(diag, a_count);

    while (low <= high) {
        const int a_idx = (low + high) >> 1;
        const int b_idx = diag - a_idx;

        const DISTANCE_T a_left = (a_idx > 0) ? a[a_idx - 1] : -FLT_MAX;
        const DISTANCE_T a_right = (a_idx < a_count) ? a[a_idx] : FLT_MAX;
        const DISTANCE_T b_left = (b_idx > 0) ? b[b_idx - 1] : -FLT_MAX;
        const DISTANCE_T b_right = (b_idx < b_count) ? b[b_idx] : FLT_MAX;

        if (a_left > b_right) {
            high = a_idx - 1;
        } else if (b_left > a_right) {
            low = a_idx + 1;
        } else {
            return a_idx;
        }
    }

    return low;
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_candidate_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk) {
    auto candidate_indices = result_indices_ptr + internal_topk;
    auto candidate_distances = result_distances_ptr + internal_topk;

    candidate_by_bitonic_sort<N_1, 0>(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
    __syncthreads();

    constexpr int kItemsPerThread = 4;
    int active_threads = (internal_topk + kItemsPerThread - 1) / kItemsPerThread;
    active_threads = min(static_cast<int>(blockDim.x),
                         ((active_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

    if (threadIdx.x < active_threads) {
        const int out_begin =
            min(internal_topk, (static_cast<int>(threadIdx.x) * internal_topk) / active_threads);
        const int out_end =
            min(internal_topk, (static_cast<int>(threadIdx.x + 1) * internal_topk) / active_threads);

        const int top_begin = merge_path_partition(
            result_distances_ptr, static_cast<int>(internal_topk),
            candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
            out_begin);
        const int top_end = merge_path_partition(
            result_distances_ptr, static_cast<int>(internal_topk),
            candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
            out_end);

        int top_i = top_begin;
        int cand_i = out_begin - top_begin;
        const int cand_end = out_end - top_end;

        for (int out_i = out_begin; out_i < out_end; ++out_i) {
            const bool take_topk =
                (cand_i >= cand_end) ||
                (top_i < top_end &&
                 result_distances_ptr[top_i] <= candidate_distances[cand_i]);

            if (take_topk) {
                merged_topk_dist_shared[out_i] = result_distances_ptr[top_i];
                merged_topk_index_shared[out_i] = result_indices_ptr[top_i];
                ++top_i;
            } else {
                merged_topk_dist_shared[out_i] = candidate_distances[cand_i];
                merged_topk_index_shared[out_i] = candidate_indices[cand_i];
                ++cand_i;
            }
        }
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < internal_topk; i += blockDim.x) {
        result_distances_ptr[i] = merged_topk_dist_shared[i];
        result_indices_ptr[i] = merged_topk_index_shared[i];
    }
    __syncthreads();
}

/*-------------------------------------------- hash table --------------------------------------------*/
#define SMALL_HASH_RESET_INTERVAL 16

__host__ __device__ inline uint32_t hashtable_getsize(const uint32_t bitlen) {
    return 1 << bitlen;
}

__device__ inline void hashtable_init(INDEX_T *const table, const unsigned bitlen, unsigned FIRST_TID = 0) {
    if (threadIdx.x < FIRST_TID)
        return;
    for (uint32_t i = threadIdx.x - FIRST_TID; i < hashtable_getsize(bitlen); i += blockDim.x - FIRST_TID) {
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
    if (threadIdx.x < first_tid)
        return;
    for (unsigned i = threadIdx.x - first_tid; i < itopk_size; i += blockDim.x - first_tid) {
        const auto raw_key = itopk_indices[i];
        if (raw_key == MAX_INDEX) {
            continue;
        }
        auto key = raw_key & ~index_msb_1_mask; // clear most significant bit
        hashtable_insert(table, BITLEN, key);
    }
}

static __device__ uint32_t pick_expanders(int N, INDEX_T *output_list, int M, INDEX_T *pri_queue) {
    uint32_t num_exp = 0; // number of expanders actually seleceted; must be <= N
    constexpr INDEX_T index_msb_1_mask = 0x80000000;
    const uint32_t lane_id = threadIdx.x & (WARP_SIZE - 1);
    uint32_t itopk_max = M;
    if (itopk_max % 32) {
        itopk_max += 32 - (itopk_max % 32);
    } // round up to be multiple of 32
    for (uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
        INDEX_T index;
        int new_parent = 0;
        if (j < M) { // within the priority queue bound
            index = pri_queue[j];
            if ((index & index_msb_1_mask) == 0) { // check if most significant bit is set
                new_parent = 1;
            }
        }
        const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
        if (new_parent) {
            const auto i = __popc(ballot_mask & ((1u << lane_id) - 1u)) + num_exp;
            if (i < N) {
                output_list[i] = j;
                //  set most significant bit as used node
                pri_queue[j] |= index_msb_1_mask;
            }
        }
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

/*-------------------------------------------- rabitq --------------------------------------------*/
static __device__ inline void rotate_vector_gpu(const float *src, float *dst,
                                                const float *signs, int dim, int padded_dim) {

    int tid = threadIdx.x;

    for (size_t i = tid; i < dim; i += blockDim.x) {
        dst[i] = src[i] * signs[i];
    }
    for (int i = dim + tid; i < padded_dim; i += blockDim.x) {
        dst[i] = 0.0f;
    }
    __syncthreads();

    // Fast Walsh-Hadamard Transform (in-place, power-of-2 B, B >= 8)
    for (size_t len = 1; len < padded_dim; len <<= 1) {
        int pairs = padded_dim >> 1;
        for (int pair_idx = tid; pair_idx < pairs; pair_idx += blockDim.x) {
            int block = pair_idx / len;
            int j = pair_idx % len;
            int base = block * (len << 1);

            int idx1 = base + j;
            int idx2 = idx1 + len;

            float u = dst[idx1];
            float v = dst[idx2];
            dst[idx1] = u + v;
            dst[idx2] = u - v;
        }
        __syncthreads();
    }
}

template <int LUT_LAYOUT_LANES = GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES>
static __device__ __forceinline__ uint8_t fastscan_decode_code_for_neighbor_seq_lut_gpu(
    const uint8_t *code_base, int codebook_idx, int neighbor_in_tile = 0) {
    static_assert(LUT_LAYOUT_LANES == 2 || LUT_LAYOUT_LANES == 4 || LUT_LAYOUT_LANES == 8 || LUT_LAYOUT_LANES == 16 || LUT_LAYOUT_LANES == 32,
                  "GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES must be 2, 4, 8, 16, or 32");
    const int pair_idx = codebook_idx >> 1;
    uint8_t packed;
    if constexpr (LUT_LAYOUT_LANES == 32) {
        packed = code_base[pair_idx];
    } else {
        constexpr int neighbors_per_tile = WARP_SIZE / LUT_LAYOUT_LANES;
        packed = code_base[pair_idx * neighbors_per_tile + neighbor_in_tile];
    }
    return (codebook_idx & 1) ? static_cast<uint8_t>(packed >> 4)
                              : static_cast<uint8_t>(packed & 0x0f);
}

static __device__ inline void query_prepare_lut_gpu(const float *query_raw, QueryFactors &scratch,
                                                    const float *signs_ptr, int dim, int padded_dim) {
    constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // rotate the query
    rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
    __syncthreads();

    __shared__ float warp_min[WARPS_PER_BLOCK];
    __shared__ float warp_max[WARPS_PER_BLOCK];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // data range: find the lowest and highest dimension in the rotated query
    for (size_t i = tid; i < padded_dim; i += blockDim.x) {
        float tmp = scratch.rotated_query[i];
        local_min = fminf(local_min, tmp);
        local_max = fmaxf(local_max, tmp);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
        local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
        }

        if (lane_id == 0) {
            scratch.low_val = local_min;
            scratch.high_val = local_max;
            const float query_span = scratch.high_val - scratch.low_val;
            scratch.width = query_span * kQueryLevelsInv;
        }
    }
    __syncthreads();

    // quantize the query and build the LUT
    const float inv_width = 1.0f / scratch.width;
    __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
    int32_t local_sum = 0;
    const int num_codebook = padded_dim >> 2;

    for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
        uint8_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = (cb << 2) + j;
            const float scaled = ((scratch.rotated_query[idx] - scratch.low_val) * inv_width) + 0.5f;
            q[j] = static_cast<uint8_t>(lroundf(scaled));
            local_sum += q[j];
        }

        // Store the 16 subset sums contiguously for this codebook chunk. The scan
        // path later indexes qf.lut[(cb << 4) + code] directly for each neighbor.
        uint8_t *lut_chunk = scratch.lut + (cb << 4);
        lut_chunk[0] = 0;
        for (int mask = 1; mask < 16; ++mask) {
            const int lowbit = mask & -mask;
            const int pos = (lowbit == 8) ? 0 : ((lowbit == 4) ? 1 : ((lowbit == 2) ? 2 : 3));
            lut_chunk[mask] = static_cast<uint8_t>(lut_chunk[mask - lowbit] + q[pos]);
        }
    }

    // compute the sum of quantized query values across all dimensions
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
        if (lane_id == 0) {
            scratch.sum_q = local_sum;
        }
    }
}

/**
 * @brief Build the query lookup table for a TurboQuant level codebook.
 *
 * Groups QUANT_NIBBLE_DIMS / CODEBITS dimensions into one 4-bit scan code, most significant
 * dimension first, matching pack_codes. Entries carry a gain of CODEBITS so that the widest
 * group still fills the uint8 range, and sum_q carries the same gain, which lets the RaBitQ
 * scan path consume the table without change.
 *
 * @tparam CODEBITS bits per dimension
 * @param query_raw raw query vector
 * @param scratch query factors, receives the rotated query, its range, and the table
 * @param signs_ptr sign vector of the index rotation
 * @param levels normalized reconstruction levels, 2^CODEBITS entries spanning [0, 1]
 * @param dim raw dimension
 * @param padded_dim padded dimension
 */
template <int CODEBITS>
static __device__ inline void turboq_prepare_lut_gpu(
    const float *query_raw, QueryFactors &scratch, const float *signs_ptr,
    const float *levels, int dim, int padded_dim) { // SHAME(MANYARG) SHAME(TALLFUNC)
    constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);
    constexpr int GROUP = QUANT_NIBBLE_DIMS / CODEBITS;
    constexpr int LMASK = (1 << CODEBITS) - 1;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // [1] rotate the query with the same sign flip and transform the index used
    rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
    __syncthreads();

    __shared__ float warp_min[WARPS_PER_BLOCK];
    __shared__ float warp_max[WARPS_PER_BLOCK];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // [2] reduce the rotated query range across the block
    for (size_t i = tid; i < padded_dim; i += blockDim.x) {
        float tmp = scratch.rotated_query[i];
        local_min = fminf(local_min, tmp);
        local_max = fmaxf(local_max, tmp);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
        local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
        }

        if (lane_id == 0) {
            scratch.low_val = local_min;
            scratch.high_val = local_max;
            const float query_span = scratch.high_val - scratch.low_val;
            scratch.width = query_span * kQueryLevelsInv;
        }
    }
    __syncthreads();

    // [3] quantize the query and tabulate every 4-bit group code
    const float inv_width = 1.0f / scratch.width;
    __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
    int32_t local_sum = 0;
    const int num_codebook = (padded_dim * CODEBITS) >> 2;

    for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
        float q[GROUP];
#pragma unroll
        for (int j = 0; j < GROUP; ++j) {
            const int idx = cb * GROUP + j;
            const float scaled = ((scratch.rotated_query[idx] - scratch.low_val) * inv_width) + 0.5f;
            const int level = static_cast<int>(lroundf(scaled));
            q[j] = static_cast<float>(level);
            local_sum += level;
        }

        uint8_t *lut_chunk = scratch.lut + (cb << 4);
        for (int code = 0; code < 16; ++code) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                constexpr int base_shift = QUANT_NIBBLE_DIMS;
                const int shift = base_shift - (j + 1) * CODEBITS;
                acc += q[j] * levels[(code >> shift) & LMASK];
            }
            lut_chunk[code] = static_cast<uint8_t>(lroundf(acc * static_cast<float>(CODEBITS)));
        }
    }

    // [4] reduce the gain-scaled query sum used by the scan correction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
        if (lane_id == 0) {
            scratch.sum_q = local_sum * CODEBITS;
        }
    }
}

template <int LANES_PER_NEIGHBOR, int CODEBITS = 1>
static __device__ inline DISTANCE_T scan_one_neighbor_lanes_gpu(
    const QueryFactors &qf, const uint8_t *packed_codes_block, int neighbor_idx,
    const float *triple_x, const float *factor_dq, const float *factor_vq,
    float exact_dist, int padded_dim, int bytes_per_neighbor) { // SHAME(MANYARG)
    static_assert(LANES_PER_NEIGHBOR == 2 || LANES_PER_NEIGHBOR == 4 ||
                      LANES_PER_NEIGHBOR == 8 || LANES_PER_NEIGHBOR == 16 ||
                      LANES_PER_NEIGHBOR == 32,
                  "LANES_PER_NEIGHBOR must be one of 2, 4, 8, 16, or 32");
    const float triple_x_value = triple_x[0];
    if (triple_x_value == FLT_MAX)
        return 0.0f;

    const int lane_id = threadIdx.x & (WARP_SIZE - 1);
    const int group_lane = lane_id & (LANES_PER_NEIGHBOR - 1);
    unsigned group_mask = FULL_MASK;
    if constexpr (LANES_PER_NEIGHBOR != WARP_SIZE) {
        const int group_start = lane_id - group_lane;
        constexpr unsigned group_bits = (1u << LANES_PER_NEIGHBOR) - 1u;
        group_mask = group_bits << group_start;
    }
    const int num_codebook = (padded_dim * CODEBITS) >> 2;
    uint32_t local_raw_sum = 0;

    constexpr int lut_layout_neighbors = WARP_SIZE / GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES;
    const uint8_t *seq_lut_tile = packed_codes_block;
    int seq_lut_neighbor_in_tile = 0;
    if constexpr (GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES == 32) {
        seq_lut_tile = packed_codes_block + neighbor_idx * bytes_per_neighbor;
    } else {
        const int tile_idx = neighbor_idx / lut_layout_neighbors;
        seq_lut_neighbor_in_tile = neighbor_idx & (lut_layout_neighbors - 1);
        seq_lut_tile = packed_codes_block + tile_idx * lut_layout_neighbors * bytes_per_neighbor;
    }

    for (int cb = group_lane; cb < num_codebook; cb += LANES_PER_NEIGHBOR) {
        const uint8_t code = fastscan_decode_code_for_neighbor_seq_lut_gpu(
            seq_lut_tile, cb, seq_lut_neighbor_in_tile);
        local_raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
    }

    for (int offset = LANES_PER_NEIGHBOR / 2; offset > 0; offset >>= 1) {
        local_raw_sum += __shfl_down_sync(group_mask, local_raw_sum, offset, LANES_PER_NEIGHBOR);
    }

    if (group_lane != 0)
        return 0.0f;

    const float result_float = static_cast<float>((static_cast<int32_t>(local_raw_sum) << 1) - qf.sum_q);
    DISTANCE_T est_dist = triple_x_value + exact_dist;
    est_dist += factor_dq[0] * qf.width * result_float;
    est_dist += factor_vq[0] * qf.low_val;
    return est_dist;
}

template <int CODEBITS = 1>
static __device__ inline DISTANCE_T scan_one_neighbor_lane_seq_lut_gpu(
    const QueryFactors &qf, const uint8_t *parent_code_base, int neighbor_idx,
    float triple_x, float factor_dq, float factor_vq,
    float exact_dist, int padded_dim, int bytes_per_neighbor) { // SHAME(MANYARG)
    if (triple_x == FLT_MAX)
        return FLT_MAX;

    constexpr int lut_layout_neighbors = WARP_SIZE / GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES;
    const uint8_t *seq_lut_tile = parent_code_base;
    int seq_lut_neighbor_in_tile = 0;
    if constexpr (GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES == 32) {
        seq_lut_tile = parent_code_base + neighbor_idx * bytes_per_neighbor;
    } else {
        const int tile_idx = neighbor_idx / lut_layout_neighbors;
        seq_lut_neighbor_in_tile = neighbor_idx & (lut_layout_neighbors - 1);
        seq_lut_tile = parent_code_base + tile_idx * lut_layout_neighbors * bytes_per_neighbor;
    }

    const int num_codebook = (padded_dim * CODEBITS) >> 2;
    uint32_t raw_sum = 0;
    for (int cb = 0; cb < num_codebook; ++cb) {
        const uint8_t code = fastscan_decode_code_for_neighbor_seq_lut_gpu(
            seq_lut_tile, cb, seq_lut_neighbor_in_tile);
        raw_sum += static_cast<uint32_t>(qf.lut[(cb << 4) + code]);
    }

    const float result_float = static_cast<float>((static_cast<int32_t>(raw_sum) << 1) - qf.sum_q);
    DISTANCE_T est_dist = triple_x + exact_dist;
    est_dist += factor_dq * qf.width * result_float;
    est_dist += factor_vq * qf.low_val;
    return est_dist;
}

/*-------------------------------------------- runtime dispatchers --------------------------------------------*/
__device__ inline void dispatch_candidate_bitonic_sort(
    INDEX_T *candidate_indices,
    DISTANCE_T *candidate_distances,
    uint32_t candidate_buffer_size) {
    const uint32_t warp_items_per_thread =
        (candidate_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
    if (GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT || warp_items_per_thread > 4) {
        candidate_by_block_bitonic_sort(candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 1) {
        candidate_by_bitonic_sort<1, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else if (warp_items_per_thread <= 2) {
        candidate_by_bitonic_sort<2, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    } else {
        candidate_by_bitonic_sort<4, 0>(
            candidate_indices, candidate_distances, candidate_buffer_size);
    }
}

static __device__ inline void topk_runtime_candidate_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk) {
    auto candidate_indices = result_indices_ptr + internal_topk;
    auto candidate_distances = result_distances_ptr + internal_topk;

    dispatch_candidate_bitonic_sort(candidate_indices, candidate_distances, CANDIDATE_BUFFER_SIZE);
    __syncthreads();

    constexpr int kItemsPerThread = 4;
    int active_threads = (internal_topk + kItemsPerThread - 1) / kItemsPerThread;
    active_threads = min(static_cast<int>(blockDim.x),
                         ((active_threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);

    if (threadIdx.x < active_threads) {
        const int out_begin =
            min(static_cast<int>(internal_topk), (static_cast<int>(threadIdx.x) * static_cast<int>(internal_topk)) / active_threads);
        const int out_end =
            min(static_cast<int>(internal_topk), (static_cast<int>(threadIdx.x + 1) * static_cast<int>(internal_topk)) / active_threads);

        const int top_begin = merge_path_partition(
            result_distances_ptr, static_cast<int>(internal_topk),
            candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
            out_begin);
        const int top_end = merge_path_partition(
            result_distances_ptr, static_cast<int>(internal_topk),
            candidate_distances, static_cast<int>(CANDIDATE_BUFFER_SIZE),
            out_end);

        int top_i = top_begin;
        int cand_i = out_begin - top_begin;
        const int cand_end = out_end - top_end;

        for (int out_i = out_begin; out_i < out_end; ++out_i) {
            const bool take_topk =
                (cand_i >= cand_end) ||
                (top_i < top_end &&
                 result_distances_ptr[top_i] <= candidate_distances[cand_i]);

            if (take_topk) {
                merged_topk_dist_shared[out_i] = result_distances_ptr[top_i];
                merged_topk_index_shared[out_i] = result_indices_ptr[top_i];
                ++top_i;
            } else {
                merged_topk_dist_shared[out_i] = candidate_distances[cand_i];
                merged_topk_index_shared[out_i] = candidate_indices[cand_i];
                ++cand_i;
            }
        }
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < internal_topk; i += blockDim.x) {
        result_distances_ptr[i] = merged_topk_dist_shared[i];
        result_indices_ptr[i] = merged_topk_index_shared[i];
    }
    __syncthreads();
}

template <unsigned N_1, unsigned N_2>
__device__ void topk_candidate_merge_policy(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t CANDIDATE_BUFFER_SIZE,
    uint32_t internal_topk,
    bool first) {
    if constexpr (N_2 == 1 || N_2 == 2 || N_2 == 4) {
        topk_small_by_bitonic_sort<N_1, N_2>(
            result_indices_ptr, result_distances_ptr,
            CANDIDATE_BUFFER_SIZE, internal_topk, first);
    } else {
        topk_candidate_sort_and_merge<N_1, N_2>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            CANDIDATE_BUFFER_SIZE, internal_topk);
    }
}

template <unsigned N_1, unsigned N_2>
__device__ __noinline__ void topk_candidate_merge_width_case(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
    if (internal_topk % WARP_SIZE == 0) {
        topk_candidate_merge_policy<N_1, N_2>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
    } else {
        topk_candidate_sort_and_merge<N_1, N_2>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk);
    }
}

// Keep the full 1..32 specialization table. Collapsing it to fewer runtime cases
// substantially increases ptxas spill loads for the AP kernel.
template <unsigned N_1>
__device__ inline void dispatch_topk_candidate_merge_width(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
    switch ((internal_topk + WARP_SIZE - 1) / WARP_SIZE) {
    case 1:
        topk_candidate_merge_width_case<N_1, 1>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 2:
        topk_candidate_merge_width_case<N_1, 2>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 3:
        topk_candidate_merge_width_case<N_1, 3>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 4:
        topk_candidate_merge_width_case<N_1, 4>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 5:
        topk_candidate_merge_width_case<N_1, 5>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 6:
        topk_candidate_merge_width_case<N_1, 6>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 7:
        topk_candidate_merge_width_case<N_1, 7>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 8:
        topk_candidate_merge_width_case<N_1, 8>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 9:
        topk_candidate_merge_width_case<N_1, 9>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 10:
        topk_candidate_merge_width_case<N_1, 10>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 11:
        topk_candidate_merge_width_case<N_1, 11>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 12:
        topk_candidate_merge_width_case<N_1, 12>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 13:
        topk_candidate_merge_width_case<N_1, 13>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 14:
        topk_candidate_merge_width_case<N_1, 14>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 15:
        topk_candidate_merge_width_case<N_1, 15>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 16:
        topk_candidate_merge_width_case<N_1, 16>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 17:
        topk_candidate_merge_width_case<N_1, 17>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 18:
        topk_candidate_merge_width_case<N_1, 18>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 19:
        topk_candidate_merge_width_case<N_1, 19>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 20:
        topk_candidate_merge_width_case<N_1, 20>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 21:
        topk_candidate_merge_width_case<N_1, 21>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 22:
        topk_candidate_merge_width_case<N_1, 22>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 23:
        topk_candidate_merge_width_case<N_1, 23>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 24:
        topk_candidate_merge_width_case<N_1, 24>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 25:
        topk_candidate_merge_width_case<N_1, 25>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 26:
        topk_candidate_merge_width_case<N_1, 26>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 27:
        topk_candidate_merge_width_case<N_1, 27>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 28:
        topk_candidate_merge_width_case<N_1, 28>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 29:
        topk_candidate_merge_width_case<N_1, 29>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 30:
        topk_candidate_merge_width_case<N_1, 30>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 31:
        topk_candidate_merge_width_case<N_1, 31>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    case 32:
        topk_candidate_merge_width_case<N_1, 32>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    default:
        topk_candidate_merge_width_case<N_1, 4>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
        break;
    }
}

__device__ inline void dispatch_topk_candidate_sort_and_merge(
    INDEX_T *result_indices_ptr,
    DISTANCE_T *result_distances_ptr,
    INDEX_T *merged_topk_index_shared,
    DISTANCE_T *merged_topk_dist_shared,
    uint32_t candidate_buffer_size,
    uint32_t internal_topk,
    bool first) {
    const uint32_t items_per_thread = (candidate_buffer_size + WARP_SIZE - 1) / WARP_SIZE;
    if (GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT || items_per_thread > 4) {
        topk_runtime_candidate_sort_and_merge(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk);
    } else if (items_per_thread <= 1) {
        dispatch_topk_candidate_merge_width<1>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
    } else if (items_per_thread <= 2) {
        dispatch_topk_candidate_merge_width<2>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
    } else {
        dispatch_topk_candidate_merge_width<4>(
            result_indices_ptr, result_distances_ptr,
            merged_topk_index_shared, merged_topk_dist_shared,
            candidate_buffer_size, internal_topk, first);
    }
}
