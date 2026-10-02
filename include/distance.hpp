#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if !defined(__CUDACC__) && (defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__))
#include <immintrin.h>
#endif

#include "metric.hpp"

/**
 * @brief Scalar squared L2 distance, the fallback of compute_distance without AVX2.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return sum of squared coordinate differences
 */
inline float compute_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

/**
 * @brief Scalar negated inner product, the fallback of compute_distance without AVX2.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return minus the dot product, so smaller is closer
 */
inline float compute_ip_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#ifdef __AVX2__
/**
 * @brief Horizontal sum of the eight lanes of an AVX register, taken from DiskANN.
 * @param x register to reduce
 * @return sum of all lanes
 */
static inline float _mm256_reduce_add_ps(__m256 x) {
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

/**
 * @brief AVX2 squared L2 distance over 8-float blocks plus a scalar tail.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return sum of squared coordinate differences
 */
inline float compute_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    float dist = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief AVX2 negated inner product over 8-float blocks plus a scalar tail.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return minus the dot product, so smaller is closer
 */
inline float compute_ip_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j + 1 < niters) {
            _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
    }
    float ip = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        ip += a[i] * b[i];
    }
    return -ip;
}
#endif // __AVX2__

#ifdef __AVX512F__
/**
 * @brief AVX-512 squared L2 distance over 16-float blocks plus a scalar tail, adapted from DiskANN.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return sum of squared coordinate differences
 */
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        __m512 tmp_vec = _mm512_sub_ps(a_vec, b_vec);
        sum = _mm512_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    float dist = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief AVX-512 negated inner product over 16-float blocks plus a scalar tail.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return minus the dot product, so smaller is closer
 */
inline float compute_ip_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        if (j + 1 < niters) {
            _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        sum = _mm512_fmadd_ps(a_vec, b_vec, sum);
    }
    float ip = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        ip += a[i] * b[i];
    }
    return -ip;
}
#endif // __AVX512F__

/**
 * @brief Host distance under a given metric, using the widest SIMD path the build enables.
 *
 * The PathW driver calls it to pick its entry point.
 *
 * @param metric METRIC_L2 for squared L2, METRIC_IP for negated inner product
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return the distance, smaller is closer
 */
inline float compute_distance(MetricType metric, int dim, const float* __restrict__ a, const float* __restrict__ b) {
    if (metric == METRIC_IP) {
#ifdef __AVX512F__
        return compute_ip_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
        return compute_ip_distance_vec256(dim, a, b);
#else
        return compute_ip_distance_scalar(dim, a, b);
#endif
    }
#ifdef __AVX512F__
    return compute_distance_vec512(dim, a, b);
#elif defined(__AVX2__)
    return compute_distance_vec256(dim, a, b);
#else
    return compute_distance_scalar(dim, a, b);
#endif
}

/**
 * @brief Overload of compute_distance under the process-wide g_metric_type, used by compute_rabitq_entry_point.
 * @param dim number of coordinates
 * @param a first vector
 * @param b second vector
 * @return the distance, smaller is closer
 */
inline float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    return compute_distance(g_metric_type, dim, a, b);
}

