#include "include/index.hpp"
#include "src/pathw_search.cuh"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

static vid_t compute_gpu_pathw_entry_point(const float* data, size_t npoints, size_t dim, MetricType metric) {
  std::vector<float> centroid(dim, 0.0f);
  const float inv_npoints = 1.0f / static_cast<float>(npoints);
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    for (size_t d = 0; d < dim; ++d) {
      centroid[d] += row[d] * inv_npoints;
    }
  }

  vid_t best = 0;
  float best_dist = FLT_MAX;
  for (size_t i = 0; i < npoints; ++i) {
    const float* row = data + i * dim;
    float dist = compute_distance(metric, static_cast<int>(dim), row, centroid.data());
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<vid_t>(i);
    }
  }
  return best;
}

static bool get_gpu_pathw_seed_override(size_t npoints, vid_t& seed_node) {
  const char* seed_env = std::getenv("PATHW_SEED_NODE");
  if (seed_env == nullptr || seed_env[0] == '\0') {
    return false;
  }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(seed_env, &end, 10);
  if (end == seed_env || *end != '\0' || value >= npoints) {
    throw std::runtime_error("Invalid PATHW_SEED_NODE: " + std::string(seed_env));
  }
  seed_node = static_cast<vid_t>(value);
  return true;
}

template <typename T>
static auto pathw_kernel(int dim, int beam) {
  using Kernel = decltype(&PathWBeamSearch<T, 0, 0>);
  Kernel kernel = PathWBeamSearch<T, 0, 0>;
  if (dim == 128) {
    if (beam == 64) kernel = PathWBeamSearch<T, 128, 64>;
    else if (beam == 128) kernel = PathWBeamSearch<T, 128, 128>;
  } else if (dim == 960) {
    if (beam == 64) kernel = PathWBeamSearch<T, 960, 64>;
    else if (beam == 128) kernel = PathWBeamSearch<T, 960, 128>;
  }
  return kernel;
}

template <typename T>
void IndexGraph<T>::search_pathw(int nq, const T* queries, int K, vid_t* result_idx, int beam_sz,
                                 const char* signbit_file,
                                 float neighbor_keep_ratio,
                                 float iteration_prune_ratio,
                                 double& elapsed, uint32_t* iters, int repeat) {
  auto dim = this->d;
  auto npoints = this->ntotal;
  auto max_degree = this->maxDeg;
  auto beam_size = beam_sz;
  auto padded_beam_size = static_cast<int>(effective_sort_beam_size(static_cast<uint32_t>(beam_size)));
  const bool use_ip = (this->metric == METRIC_IP);
  const int active_nq = nq;
  if (K > beam_size) {
    throw std::runtime_error("GPU PathW requires K <= beam_size");
  }
  if (beam_size <= 0) {
    throw std::runtime_error("GPU PathW requires beam_size > 0");
  }
  const size_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(max_degree));
  auto bitlen = hash_bitlen_for_search_workload(padded_beam_size, candidate_buffer_size, SMALL_HASH_RESET_INTERVAL);

  printf("Beam search PathW on GPU: K=%d, nq=%d, dim=%d, npoints=%ld, beam_size=%d, padded_beam_size=%d\n",
         K, active_nq, dim, npoints, beam_size, padded_beam_size);
  printf("Metric: %s\n", metric_name(this->metric));
  printf("PathW params: neighbor_keep_ratio=%.4f, iteration_prune_ratio=%.4f, iteration_limit=beam_size\n",
         neighbor_keep_ratio, iteration_prune_ratio);
  printf("GPU PathW speculation: search_width=%d, candidate_buffer_size=%zu\n",
         1, candidate_buffer_size);

  size_t sign_n = 0;
  size_t sign_width = 0;
  uint32_t* h_sign_bit_raw = read_bin<uint32_t>(signbit_file, sign_n, sign_width);
  auto signbit_deleter = [sign_width](uint32_t* ptr) {
    release_bin_buffer(ptr, sign_width);
  };
  std::unique_ptr<uint32_t, decltype(signbit_deleter)> h_sign_bit(
      h_sign_bit_raw, signbit_deleter);
  const size_t packed_dim = (static_cast<size_t>(dim) + 31) / 32;
  const size_t expected_width = static_cast<size_t>(max_degree) * packed_dim;
  if (sign_n != npoints || sign_width != expected_width) {
    throw std::runtime_error(
        "PathW signbit shape mismatch: got n=" + std::to_string(sign_n) +
        ", width=" + std::to_string(sign_width) +
        "; expected n=" + std::to_string(npoints) +
        ", width=" + std::to_string(expected_width));
  }

  size_t num_threads = PATHW_BLOCK_SIZE;
  size_t num_blocks = active_nq;
  std::cout << "num_blocks = " << num_blocks << " num_threads = " << num_threads << "\n";
  cudaDeviceProp prop;
  int dev;
  CUDA_SAFE_CALL(cudaGetDevice(&dev));
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
  printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

  T* d_queries = nullptr;
  T* d_data = nullptr;
  T* h_data = this->get_data_ptr();
  vid_t* d_results = nullptr;
  uint32_t* d_iters = nullptr;
  uint32_t* d_sign_bit = nullptr;

  size_t free_mem_bytes = 0;
  size_t total_mem_bytes = 0;
  CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));
  const size_t query_bytes = static_cast<size_t>(active_nq) * dim * sizeof(T);
  const size_t data_bytes = static_cast<size_t>(npoints) * dim * sizeof(T);
  const size_t result_bytes = static_cast<size_t>(active_nq) * K * sizeof(vid_t);
  const size_t sign_bytes = static_cast<size_t>(sign_n) * sign_width * sizeof(uint32_t);
  const size_t iters_bytes = static_cast<size_t>(active_nq) * sizeof(uint32_t);
  const size_t graph_bytes = this->edges.size() * sizeof(vid_t);
  const size_t required_bytes = query_bytes + data_bytes + result_bytes + sign_bytes + graph_bytes + iters_bytes;
  constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
  printf("GPU memory: free=%.2f GiB, total=%.2f GiB, required=%.2f GiB (data=%.2f GiB, signbit=%.2f GiB)\n",
         static_cast<double>(free_mem_bytes) / kBytesPerGiB,
         static_cast<double>(total_mem_bytes) / kBytesPerGiB,
         static_cast<double>(required_bytes) / kBytesPerGiB,
         static_cast<double>(data_bytes) / kBytesPerGiB,
         static_cast<double>(sign_bytes) / kBytesPerGiB);
  if (required_bytes > free_mem_bytes) {
    throw std::runtime_error("PathW GPU upload exceeds currently available GPU memory");
  }

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_queries, query_bytes));
  const auto query_load_start = std::chrono::high_resolution_clock::now();
  CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, query_bytes, cudaMemcpyHostToDevice));
  const auto query_load_end = std::chrono::high_resolution_clock::now();
  printf("Query H2D time: %.6f ms\n", std::chrono::duration<double, std::milli>(query_load_end - query_load_start).count());
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_data, data_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_results, result_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_sign_bit, sign_bytes));
  CUDA_SAFE_CALL(cudaMemcpy(d_sign_bit, h_sign_bit.get(), sign_bytes, cudaMemcpyHostToDevice));

  CUDA_SAFE_CALL(cudaMalloc((void**)&d_iters, iters_bytes));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  GraphGPU gg(npoints, max_degree, this->edges.data());

  const vid_t entry_point = compute_gpu_pathw_entry_point(h_data, npoints, dim, this->metric);
  printf("GPU pathw entry point: %u\n", entry_point);
  vid_t search_entry_point = entry_point;
  if (get_gpu_pathw_seed_override(npoints, search_entry_point)) {
    printf("GPU pathw seed node: %u\n", search_entry_point);
  }

  uint32_t shm_size = pathw_shared(dim, padded_beam_size, max_degree, bitlen);
  auto kernel = pathw_kernel<T>(dim, beam_size);
  int numBlocksPerSM = 0;
  CUDA_SAFE_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shm_size)));
  CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, kernel, num_threads, shm_size));
  printf("Max active blocks per SM = %d\n", numBlocksPerSM);

  printf("\nStarting GPU PathW search...\n");
  elapsed = 0.0;
  for (int launch = 0; launch < repeat; ++launch) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel<<<num_blocks, num_threads, shm_size>>>(
        K, active_nq, dim, beam_size, bitlen, npoints, d_queries, d_data,
        d_sign_bit, d_results, d_iters, search_entry_point, gg,
        neighbor_keep_ratio, iteration_prune_ratio, use_ip);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    elapsed += std::chrono::duration<double>(end - start).count();
  }
  elapsed /= repeat;

  CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, result_bytes, cudaMemcpyDeviceToHost));
  if (iters) CUDA_SAFE_CALL(cudaMemcpy(iters, d_iters, iters_bytes, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaFree(d_queries));
  CUDA_SAFE_CALL(cudaFree(d_data));
  CUDA_SAFE_CALL(cudaFree(d_results));
  CUDA_SAFE_CALL(cudaFree(d_sign_bit));
  CUDA_SAFE_CALL(cudaFree(d_iters));
  gg.release();

}

template class IndexGraph<float>;
