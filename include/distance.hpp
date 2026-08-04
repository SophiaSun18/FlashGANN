#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#if !defined(__CUDACC__) && (defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE2__))
#include <immintrin.h>
#endif

#include "metric.hpp"

inline float compute_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float compute_ip_distance_scalar(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return -sum;
}

#ifdef __AVX2__
// from DiskANN
static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

inline float compute_distance_vec256(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    // assume size is divisible by 8
    uint16_t niters = (uint16_t)(dim / 8);
    __m256 sum = _mm256_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[8j:8j+7], b[8j:8j+7]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 8 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 8 * (j + 1)), _MM_HINT_T0);
        }
        __m256 a_vec = _mm256_loadu_ps(a + 8 * j);
        __m256 b_vec = _mm256_loadu_ps(b + 8 * j);
        __m256 tmp_vec = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm256_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 8; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

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
// adapted from DiskANN
inline float compute_distance_vec512(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    uint16_t niters = (uint16_t)(dim / 16);
    __m512 sum = _mm512_setzero_ps();
    for (uint16_t j = 0; j < niters; j++) {
        // scope is a[16j:16j+15], b[16j:16j+15]
        if (j+1 < niters) {
        _mm_prefetch((char *)(a + 16 * (j + 1)), _MM_HINT_T0);
        _mm_prefetch((char *)(b + 16 * (j + 1)), _MM_HINT_T0);
        }
        __m512 a_vec = _mm512_loadu_ps(a + 16 * j);
        __m512 b_vec = _mm512_loadu_ps(b + 16 * j);
        __m512 tmp_vec = _mm512_sub_ps(a_vec, b_vec);
        sum = _mm512_fmadd_ps(tmp_vec, tmp_vec, sum);
    }
    // horizontal add sum
    float dist = _mm512_reduce_add_ps(sum);
    for (int i = static_cast<int>(niters) * 16; i < dim; ++i) {
        const float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

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

inline float compute_distance(int dim, const float* __restrict__ a, const float* __restrict__ b) {
    return compute_distance(g_metric_type, dim, a, b);
}

