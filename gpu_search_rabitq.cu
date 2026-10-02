#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "include/qg.hpp"
#include "include/distance.hpp"
#include "include/common.hpp"
#include "include/quant.hpp"
#include "src/rabitq_search.cuh"

typedef void (*SearchKernel)(int, int, int, int, int, int, size_t,
                             const float*, const float*, const float*, const float*, const float*,
                             vidType*, float*, uint32_t*, vidType, size_t, size_t, size_t, size_t,
                             size_t, bool);

/**
 * @brief Pick the estimate-only beam search instantiation matching the index quantizer.
 * @param quant quantizer family
 * @param bits total bits per dimension
 * @return kernel pointer, or nullptr when the combination is unsupported
 */
static SearchKernel select_kernel(QuantType quant, int bits) {
    if (quant == QUANT_TBQ) {
        switch (quant_stage(bits)) {
            case 1: return QuantizedBeamSearch<1, true>;
            case 2: return QuantizedBeamSearch<2, true>;
            case 4: return QuantizedBeamSearch<4, true>;
            default: return nullptr;
        }
    }
    switch (bits) {
        case 1: return QuantizedBeamSearch<1, false>;
        case 2: return QuantizedBeamSearch<2, false>;
        case 4: return QuantizedBeamSearch<4, false>;
        default: return nullptr;
    }
}

void QuantizationGraph::gpu_search_rabitq(
    int nq, const float *queries, int K, vidType *result_idx, float *result_dist, uint32_t *iters,
    int repeat, int beam_sz, double *elapsed) {
    int padded_beam_size = static_cast<int>(effective_sort_beam_size(static_cast<uint32_t>(beam_sz)));
    const size_t candidate_work_count = static_cast<size_t>(SEARCH_WIDTH) * this->degree_;
    const size_t candidate_buffer_size = round_up_power2_u32(static_cast<uint32_t>(candidate_work_count));
    const int bitlen = hash_bitlen_for_search_workload(
        static_cast<uint32_t>(padded_beam_size),
        static_cast<uint32_t>(candidate_buffer_size),
        static_cast<uint32_t>(SMALL_HASH_RESET_INTERVAL));

    printf("\nGPU rabitq search: K=%d, nq=%d, dim=%zu, npoints=%zu, beam_size=%d, padded_beam_size=%d\n",
           K, nq, this->dim_, this->num_nodes_, beam_sz, padded_beam_size);
    printf("Metric: %s\n", metric_name(this->get_metric()));
    printf("GPU rabitq search: search_width=%d, lanes_per_neighbor=%d, candidate_work_count=%zu, candidate_buffer_size=%zu, hash_bitlen=%d, hash_size=%u\n",
           SEARCH_WIDTH, GPU_RABITQ_FASTSCAN_SUBWARP_LANES, candidate_work_count, candidate_buffer_size, bitlen,
           hashtable_getsize(bitlen));

    int dev;
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDevice(&dev));
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s, SMs: %d\n", prop.name, prop.multiProcessorCount);

    size_t num_threads = BLOCK_SIZE;
    size_t num_blocks = nq;
    printf("\nnum_blocks = %zu num_threads = %zu\n", num_blocks, num_threads);

    SearchKernel kernel = select_kernel(this->get_quant(), this->get_bits());
    if (kernel == nullptr) {
        throw std::runtime_error("Unsupported quantizer and bit width combination");
    }
    printf("Quantizer: %s, bits=%d\n", quant_name(this->get_quant()), this->get_bits());

    float *d_queries = nullptr;
    float *d_qg_data = nullptr;
    float *d_qg_signs = nullptr;
    float *d_qg_levels = nullptr;
    float *d_qg_sketch = nullptr;
    vidType *d_results = nullptr;
    float *d_result_dists = nullptr;
    uint32_t *d_iters = nullptr;
    size_t free_mem_bytes = 0;
    size_t total_mem_bytes = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));

    const size_t query_bytes = static_cast<size_t>(nq) * this->dim_ * sizeof(float);
    const size_t results_bytes = static_cast<size_t>(nq) * K * sizeof(vidType);
    const size_t result_dists_bytes = static_cast<size_t>(nq) * K * sizeof(float);
    const size_t iters_bytes = static_cast<size_t>(nq) * sizeof(uint32_t);
    const size_t qg_data_bytes = this->get_data_bytes();
    const size_t qg_signs_bytes = this->get_signs_bytes();
    const size_t required_bytes = query_bytes + results_bytes + result_dists_bytes + iters_bytes +
                                  qg_data_bytes + qg_signs_bytes;

    constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
    printf("GPU memory: free=%.2f GiB, total=%.2f GiB, required=%.2f GiB (qg_data=%.2f GiB, qg_signs=%.4f GiB)\n",
           static_cast<double>(free_mem_bytes) / kBytesPerGiB,
           static_cast<double>(total_mem_bytes) / kBytesPerGiB,
           static_cast<double>(required_bytes) / kBytesPerGiB,
           static_cast<double>(qg_data_bytes) / kBytesPerGiB,
           static_cast<double>(qg_signs_bytes) / kBytesPerGiB);
    if (required_bytes > free_mem_bytes) {
        throw std::runtime_error("QG upload exceeds currently available GPU memory");
    }

    CUDA_SAFE_CALL(cudaMalloc((void **)&d_queries, query_bytes));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto query_load_start = std::chrono::high_resolution_clock::now();
    CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, query_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto query_load_end = std::chrono::high_resolution_clock::now();
    elapsed[QG_TIMER_QUERY_TRANSFER] = std::chrono::duration<double>(query_load_end - query_load_start).count();
    printf("Query Load Time: %.6f ms\n", elapsed[QG_TIMER_QUERY_TRANSFER] * 1000.0);
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_qg_data, qg_data_bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_qg_data, this->get_data_ptr(), qg_data_bytes, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_qg_signs, qg_signs_bytes));
    CUDA_SAFE_CALL(cudaMemcpy(d_qg_signs, this->get_signs_ptr(), qg_signs_bytes, cudaMemcpyHostToDevice));
    if (this->get_level_bytes() > 0) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_qg_levels, this->get_level_bytes()));
        CUDA_SAFE_CALL(cudaMemcpy(d_qg_levels, this->get_level_ptr(), this->get_level_bytes(),
                                  cudaMemcpyHostToDevice));
    }
    if (this->get_sketch_bytes() > 0) {
        CUDA_SAFE_CALL(cudaMalloc((void **)&d_qg_sketch, this->get_sketch_bytes()));
        CUDA_SAFE_CALL(cudaMemcpy(d_qg_sketch, this->get_sketch_ptr(), this->get_sketch_bytes(),
                                  cudaMemcpyHostToDevice));
    }
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_results, results_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_result_dists, result_dists_bytes));
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_iters, iters_bytes));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    vidType entry_point = this->get_entry_point();
    if (entry_point == UINT32_MAX) {
        entry_point = compute_rabitq_entry_point(this->get_data_ptr(), this->num_nodes_, this->dim_, this->get_row_offset());
    }
    printf("GPU rabitq search entry point: %u\n", entry_point);

    uint32_t shm_size = calculate_shared_mem_size(
        static_cast<int>(this->dim_), padded_beam_size, K, static_cast<int>(this->degree_), bitlen,
        this->get_bits(), this->get_quant());
    printf("Dynamic shared memory size = %u bytes\n", shm_size);
    CUDA_SAFE_CALL(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shm_size)));

    int numBlocksPerSM = 0;
    CUDA_SAFE_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, reinterpret_cast<const void *>(kernel), static_cast<int>(num_threads), shm_size));
    printf("Max active blocks per SM = %d\n", numBlocksPerSM);

    printf("\nStarting GPU rabitq search...\n");
    double total = 0.0;
    for (int launch = 0; launch < repeat; launch++) {
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<num_blocks, num_threads, shm_size>>>(
            K, nq, static_cast<int>(this->dim_), beam_sz, bitlen, static_cast<int>(this->degree_),
            this->num_nodes_, d_queries, d_qg_data, d_qg_signs, d_qg_sketch, d_qg_levels, d_results,
            d_result_dists, d_iters,
            entry_point, this->get_row_offset(), this->get_neighbor_offset(),
            this->get_code_offset(), this->get_sign_offset(), this->get_factor_offset(),
            this->get_metric() == METRIC_IP);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double>(end - start).count();
    }
    elapsed[QG_TIMER_SEARCH] = total / repeat;

    const auto result_copy_start = std::chrono::high_resolution_clock::now();
    CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, results_bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(result_dist, d_result_dists, result_dists_bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(iters, d_iters, iters_bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    const auto result_copy_end = std::chrono::high_resolution_clock::now();
    elapsed[QG_TIMER_RESULT_COPY] = std::chrono::duration<double>(result_copy_end - result_copy_start).count();
    CUDA_SAFE_CALL(cudaFree(d_queries));
    CUDA_SAFE_CALL(cudaFree(d_qg_data));
    CUDA_SAFE_CALL(cudaFree(d_qg_signs));
    if (d_qg_levels != nullptr) CUDA_SAFE_CALL(cudaFree(d_qg_levels));
    if (d_qg_sketch != nullptr) CUDA_SAFE_CALL(cudaFree(d_qg_sketch));
    CUDA_SAFE_CALL(cudaFree(d_results));
    CUDA_SAFE_CALL(cudaFree(d_result_dists));
    CUDA_SAFE_CALL(cudaFree(d_iters));
}
