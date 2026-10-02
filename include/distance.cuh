#pragma once

#include "utils.cuh"

/*-------------------------------------------- distance --------------------------------------------*/
/**
 * @brief Squared L2 distance between a and b, computed by one full warp.
 *
 * Lane l accumulates dimensions l, l + 32, and so on; a shuffle reduction then broadcasts the
 * sum to every lane. All 32 lanes of the warp must call it.
 *
 * @tparam T element and result type
 * @param dim vector length
 * @param a first vector
 * @param b second vector
 * @return the squared distance, on every lane
 */
template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T *a, const T *b) {
    int thread_lane = tidx() & (WARP_SIZE - 1);
    T val = 0.;
#pragma unroll 4
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

/**
 * @brief Negated inner product of a and b, computed by one full warp, so smaller means closer.
 *
 * Same lane layout and reduction as warp_l2_distance. All 32 lanes of the warp must call it.
 *
 * @tparam T element and result type
 * @param dim vector length
 * @param a first vector
 * @param b second vector
 * @return the negated inner product, on every lane
 */
template <typename T = float>
__device__ __forceinline__ T warp_ip_distance(int dim, const T *a, const T *b) {
    int thread_lane = tidx() & (WARP_SIZE - 1);
    T val = 0.;
#pragma unroll 4
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

/**
 * @brief Exact warp distance under the search metric; the adaptive and RaBitQ kernels use it for
 * entry points, parents, and children.
 * @tparam T element and result type
 * @param dim vector length
 * @param a first vector
 * @param b second vector
 * @param use_ip true for warp_ip_distance, false for warp_l2_distance
 * @return the distance, on every lane
 */
template <typename T = float>
__device__ __forceinline__ T warp_distance(int dim, const T *a, const T *b, bool use_ip) {
    return use_ip ? warp_ip_distance(dim, a, b) : warp_l2_distance(dim, a, b);
}
