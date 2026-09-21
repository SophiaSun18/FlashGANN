#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "include/qg.hpp"
#include "include/distance.hpp"
#include "include/common.hpp"
#include "include/quant.hpp"
#include "src/adaptive_search.cuh"

typedef void (*SearchKernel)(int, int, int, int, int, int, size_t,
                             const float*, const float*, const float*, const float*, const float*,
                             vidType*, uint32_t*, vidType, size_t, size_t, size_t, size_t, size_t,
                             int, float, bool);

/**
 * @brief Pick the beam search instantiation matching the index quantizer.
 *
 * For the TurboQuant path the template width is the MSE stage, one bit narrower than the
 * total, since the remaining bit is spent on the QJL sign block.
 *
 * @param quant quantizer family
 * @param bits total bits per dimension
 * @return kernel pointer, or nullptr when the combination is unsupported
 */
static SearchKernel select_kernel(QuantType quant, int bits) {
    if (quant == QUANT_TBQ) {
        switch (quant_stage(bits)) {
            case 1: return QuantizedPrunedBeamSearch<1, true>;
            case 2: return QuantizedPrunedBeamSearch<2, true>;
            case 4: return QuantizedPrunedBeamSearch<4, true>;
            default: return nullptr;
        }
    }
    switch (bits) {
        case 1: return QuantizedPrunedBeamSearch<1, false>;
        case 2: return QuantizedPrunedBeamSearch<2, false>;
        case 4: return QuantizedPrunedBeamSearch<4, false>;
        default: return nullptr;
    }
}

void QuantizationGraph::gpu_search_adaptive(
    int nq, const float *queries, int K, vidType *result_idx, uint32_t *iters, int repeat,
    int beam_sz, double &elapsed) {
    int padded_beam_size = static_cast<int>(effective_sort_beam_size(static_cast<uint32_t>(beam_sz)));
    const size_t candidate_work_count = static_cast<size_t>(SEARCH_WIDTH) * this->degree_;
    const size_t candidate_buffer_size = round_up_power2_u32(candidate_buffer_capacity(static_cast<uint32_t>(this->degree_)));
    auto bitlen = hash_bitlen_for_search_workload(
        static_cast<uint32_t>(padded_beam_size),
        static_cast<uint32_t>(candidate_buffer_size),
        static_cast<uint32_t>(SMALL_HASH_RESET_INTERVAL));

    printf("\nGPU adaptive search: K=%d, nq=%d, dim=%zu, npoints=%zu, beam_size=%d, padded_beam_size=%d, internal_topk=%d\n",
           K, nq, this->dim_, this->num_nodes_, beam_sz, padded_beam_size, INTERNAL_TOPK);
    printf("Metric: %s\n", metric_name(this->get_metric()));
    printf("GPU adaptive search mode: %s, search_width=%d, candidate_work_count=%zu, candidate_buffer_size=%zu, hash_bitlen=%d, hash_size=%u\n",
           "spec_adaptive", SEARCH_WIDTH, candidate_work_count, candidate_buffer_size, bitlen, hashtable_getsize(bitlen));
    const float phase2_rho = compute_max_rho_bound(static_cast<uint32_t>(this->degree_));

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
    uint32_t *d_iters = nullptr;
    size_t free_mem_bytes = 0;
    size_t total_mem_bytes = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem_bytes, &total_mem_bytes));

    const size_t query_bytes = static_cast<size_t>(nq) * this->dim_ * sizeof(float);
    const size_t results_bytes = static_cast<size_t>(nq) * K * sizeof(vidType);
    const size_t iters_bytes = static_cast<size_t>(nq) * sizeof(uint32_t);
    const size_t qg_data_bytes = this->get_data_bytes();
    const size_t qg_signs_bytes = this->get_signs_bytes();
    const size_t required_bytes = query_bytes + results_bytes + iters_bytes + qg_data_bytes + qg_signs_bytes;

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
    const auto query_load_start = std::chrono::high_resolution_clock::now();
    CUDA_SAFE_CALL(cudaMemcpy(d_queries, queries, query_bytes, cudaMemcpyHostToDevice));
    const auto query_load_end = std::chrono::high_resolution_clock::now();
    printf("Query Load Time: %.6f ms\n", std::chrono::duration<double, std::milli>(query_load_end - query_load_start).count());
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
    CUDA_SAFE_CALL(cudaMalloc((void **)&d_iters, iters_bytes));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    vidType entry_point = this->get_entry_point();
    if (entry_point == UINT32_MAX) {
        entry_point = compute_rabitq_entry_point(this->get_data_ptr(), this->num_nodes_, this->dim_, this->get_row_offset());
    }
    printf("GPU adaptive search entry point: %u\n", entry_point);

    uint32_t shm_size = calculate_shared_mem_size(
        static_cast<int>(this->dim_), padded_beam_size, static_cast<int>(this->degree_), bitlen,
        this->get_bits(), this->get_quant());
    printf("Dynamic shared memory size = %u bytes\n", shm_size);
    CUDA_SAFE_CALL(cudaFuncSetAttribute(reinterpret_cast<const void *>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shm_size)));

    int numBlocksPerSM = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM, reinterpret_cast<const void *>(kernel), static_cast<int>(num_threads), shm_size);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    printf("Max active blocks per SM = %d\n", numBlocksPerSM);

    int max_iter_by_beam = (beam_sz * 11 + 9) / 10;
    printf("\nStarting GPU adaptive search...\n");
    double total = 0.0;
    for (int launch = 0; launch < repeat; launch++) {
        // [1] wall-clock stamps let external samplers isolate the search window
        const auto wallstart = std::chrono::system_clock::now();
        auto start = std::chrono::high_resolution_clock::now();
        kernel<<<num_blocks, num_threads, shm_size>>>(
            K, nq, static_cast<int>(this->dim_), beam_sz, bitlen, static_cast<int>(this->degree_),
            this->num_nodes_, d_queries, d_qg_data, d_qg_signs, d_qg_sketch, d_qg_levels, d_results, d_iters,
            entry_point, this->get_row_offset(), this->get_neighbor_offset(),
            this->get_code_offset(), this->get_sign_offset(), this->get_factor_offset(),
            max_iter_by_beam, phase2_rho, this->get_metric() == METRIC_IP);
        CUDA_SAFE_CALL(cudaGetLastError());
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        const auto wallend = std::chrono::system_clock::now();
        total += std::chrono::duration<double>(end - start).count();
        printf("Launch %d window: start_ms=%lld end_ms=%lld\n", launch,
               static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(wallstart.time_since_epoch()).count()),
               static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(wallend.time_since_epoch()).count()));
    }
    elapsed = total / repeat;

    CUDA_SAFE_CALL(cudaMemcpy(result_idx, d_results, results_bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(iters, d_iters, iters_bytes, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_queries));
    CUDA_SAFE_CALL(cudaFree(d_qg_data));
    CUDA_SAFE_CALL(cudaFree(d_qg_signs));
    if (d_qg_levels != nullptr) CUDA_SAFE_CALL(cudaFree(d_qg_levels));
    if (d_qg_sketch != nullptr) CUDA_SAFE_CALL(cudaFree(d_qg_sketch));
    CUDA_SAFE_CALL(cudaFree(d_results));
    CUDA_SAFE_CALL(cudaFree(d_iters));
}
