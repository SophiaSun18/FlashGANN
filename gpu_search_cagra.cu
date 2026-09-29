#include "include/index.hpp"

#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>

// warmup searches before timing; the first search of a configuration links its JIT kernels
static constexpr int CAGRA_WARMUP = 2;

/**
 * @brief Map a command-line algorithm name onto the cuVS CAGRA search algorithm.
 * @param name auto, single, multi or kernel
 * @return the cuVS search algorithm
 */
static cuvs::neighbors::cagra::search_algo cagra_algo(const std::string& name) {
  using cuvs::neighbors::cagra::search_algo;
  if (name == "auto") return search_algo::AUTO;
  if (name == "single") return search_algo::SINGLE_CTA;
  if (name == "multi") return search_algo::MULTI_CTA;
  if (name == "kernel") return search_algo::MULTI_KERNEL;
  throw std::runtime_error("Unknown CAGRA algorithm: " + name + " (auto|single|multi|kernel)");
}

template <typename T>
void IndexGraph<T>::search_cagra(int nq, const T* queries, int K, vid_t* result_idx, float* result_dist,
                                 int beam_sz, int search_width, int max_iterations,
                                 const std::string& algo, double* elapsed, uint32_t* iters,
                                 int repeat) {
  const auto check = [](cudaError_t err) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  };
  if (K > beam_sz) {
    throw std::runtime_error("GPU CAGRA requires K <= beam_size");
  }
  const bool use_ip = (this->metric == METRIC_IP);
  const auto metric = use_ip ? cuvs::distance::DistanceType::InnerProduct
                             : cuvs::distance::DistanceType::L2Expanded;

  // [1] one set of resources per call on the current device, index built from the external graph
  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  const int64_t npoints = static_cast<int64_t>(this->ntotal);
  const int64_t dim = this->d;
  auto base_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
      this->data.data(), npoints, dim);
  auto graph_view = raft::make_host_matrix_view<const uint32_t, int64_t, raft::row_major>(
      this->edges.data(), npoints, static_cast<int64_t>(this->maxDeg));
  printf("CAGRA index from external graph: n=%ld, dim=%ld, degree=%d, metric=%s\n",
         npoints, dim, this->maxDeg, use_ip ? "InnerProduct" : "L2Expanded");
  cuvs::neighbors::cagra::index<T, uint32_t> index(res, metric, base_view, graph_view);
  raft::resource::sync_stream(res);

  // [2] query upload
  auto d_query = raft::make_device_matrix<T, int64_t>(res, nq, dim);
  auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, nq, K);
  auto distances = raft::make_device_matrix<float, int64_t>(res, nq, K);
  raft::resource::sync_stream(res);
  const auto query_load_start = std::chrono::high_resolution_clock::now();
  check(cudaMemcpyAsync(d_query.data_handle(), queries, static_cast<size_t>(nq) * dim * sizeof(T),
                        cudaMemcpyHostToDevice, stream));
  raft::resource::sync_stream(res);
  const auto query_load_end = std::chrono::high_resolution_clock::now();
  elapsed[QG_TIMER_QUERY_TRANSFER] = std::chrono::duration<double>(query_load_end - query_load_start).count();
  printf("Query H2D time: %.6f ms\n", elapsed[QG_TIMER_QUERY_TRANSFER] * 1000.0);

  // [3] warmup, then timed searches
  cuvs::neighbors::cagra::search_params params;
  params.algo = cagra_algo(algo);
  params.itopk_size = static_cast<size_t>(beam_sz);
  params.search_width = static_cast<size_t>(search_width);
  params.max_iterations = static_cast<size_t>(max_iterations);
  printf("CAGRA params: algo=%s, itopk_size=%d, search_width=%d, max_iterations=%d, warmup=%d\n",
         algo.c_str(), beam_sz, search_width, max_iterations, CAGRA_WARMUP);
  const auto query_view = raft::make_const_mdspan(d_query.view());
  for (int i = 0; i < CAGRA_WARMUP; ++i) {
    cuvs::neighbors::cagra::search(res, params, index, query_view, neighbors.view(), distances.view());
  }
  raft::resource::sync_stream(res);

  printf("\nStarting GPU CAGRA search...\n");
  double total = 0.0;
  for (int launch = 0; launch < repeat; ++launch) {
    const auto start = std::chrono::high_resolution_clock::now();
    cuvs::neighbors::cagra::search(res, params, index, query_view, neighbors.view(), distances.view());
    raft::resource::sync_stream(res);
    const auto end = std::chrono::high_resolution_clock::now();
    total += std::chrono::duration<double>(end - start).count();
  }
  elapsed[QG_TIMER_SEARCH] = total / repeat;

  // [4] result copy; inner product comes back as a similarity, negated so smaller is better
  const size_t result_count = static_cast<size_t>(nq) * K;
  const auto result_copy_start = std::chrono::high_resolution_clock::now();
  check(cudaMemcpyAsync(result_idx, neighbors.data_handle(), result_count * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream));
  check(cudaMemcpyAsync(result_dist, distances.data_handle(), result_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  raft::resource::sync_stream(res);
  const auto result_copy_end = std::chrono::high_resolution_clock::now();
  elapsed[QG_TIMER_RESULT_COPY] = std::chrono::duration<double>(result_copy_end - result_copy_start).count();
  if (use_ip) {
    for (size_t i = 0; i < result_count; ++i) result_dist[i] = -result_dist[i];
  }
  if (iters) std::fill(iters, iters + nq, 0u);
}

template void IndexGraph<float>::search_cagra(int, const float*, int, vid_t*, float*, int, int, int,
                                              const std::string&, double*, uint32_t*, int);
