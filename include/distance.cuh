#pragma once

#include "utils.cuh"

/*-------------------------------------------- distance --------------------------------------------*/
template <typename T = float>
__device__ __forceinline__ T warp_l2_distance(int dim, const T *a, const T *b) {
    int thread_lane = tidx() & (WARP_SIZE - 1); // thread index within the warp
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

template <typename T = float>
__device__ __forceinline__ T warp_distance(int dim, const T *a, const T *b, bool use_ip) {
    return use_ip ? warp_ip_distance(dim, a, b) : warp_l2_distance(dim, a, b);
}
