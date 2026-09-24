#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "src/adaptive_search_utils.cuh"

constexpr unsigned QUERIES = 4096;
constexpr unsigned DEGREE = 32;
constexpr unsigned RUNS = 5;
constexpr unsigned SEED = 42;
constexpr unsigned BEAMS[] = {64, 128, 256, 512};
constexpr unsigned SPECS[] = {1, 2, 4, 6, 8, 10, 12, 14, 16};

/** @brief One profiled configuration of the beam search sort sequence. */
struct Shape {
    unsigned beam;
    unsigned cand;
    unsigned iter;
    unsigned spec;
};

/** @brief Wall time per launch of the sorting kernel and of its sort-free twin, in ms. */
struct Timing {
    Shape form;
    double base;
    double sort;
};

/**
 * @brief Replay the sort calls of QuantizedPrunedBeamSearch, one block per query.
 *
 * Shared memory holds [beam | candidates] indices, then [beam | candidates] distances,
 * then the optional merge scratch, matching src/adaptive_search.cuh. Each iteration
 * loads one candidate batch per query and merges it into the beam; a single-parent
 * speculation also loads and sorts a second batch. The sort-free twin keeps every load
 * and barrier.
 *
 * @tparam sorted whether the sort calls run
 * @param nq number of queries, one block each
 * @param beam requested beam size
 * @param cand candidate buffer size
 * @param iters iterations per query
 * @param spec speculation degree
 * @param dist candidate distance batches, iters * cand
 * @param index candidate index batches, iters * cand
 */
template <bool sorted>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void flashsort(int nq, unsigned beam, unsigned cand, unsigned iters, unsigned spec,
               const float *__restrict__ dist, const uint32_t *__restrict__ index) { // SHAME(MANYARG) SHAME(WIDEFUNC)
    const int qid = blockIdx.x;
    if (qid >= nq) return;
    const uint32_t tid = threadIdx.x;
    const uint32_t padded = effective_sort_beam_size(beam);

    // [1] carve the beam, candidate and scratch buffers
    extern __shared__ char shared_memory[];
    const uint32_t total = padded + cand;
    INDEX_T *allindex = reinterpret_cast<INDEX_T *>(shared_memory);
    DISTANCE_T *alldist = reinterpret_cast<DISTANCE_T *>(allindex + total);
    INDEX_T *candindex = allindex + padded;
    DISTANCE_T *canddist = alldist + padded;
    char *tail = reinterpret_cast<char *>(alldist + total);
    INDEX_T *mergedindex = nullptr;
    DISTANCE_T *mergeddist = nullptr;
    if (topk_external_merge_scratch_needed(padded, cand)) {
        mergedindex = allocate_shared_tail_array<INDEX_T>(tail, padded);
        mergeddist = allocate_shared_tail_array<DISTANCE_T>(tail, padded);
    }
    void* candidate_radix_scratch = nullptr;
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && cand > 256) {
        using CandidateRadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
        candidate_radix_scratch = allocate_shared_tail_array<typename CandidateRadixSort::TempStorage>(
            tail, 1);
    }
    for (uint32_t i = tid; i < total; i += blockDim.x) {
        allindex[i] = MAX_INDEX;
        alldist[i] = FLT_MAX;
    }
    __syncthreads();

    for (unsigned iter = 0; iter < iters; iter++) {
        // [2] load this iteration's batch, then merge it into the beam
        const size_t merged = ((qid + iter) % iters) * cand;
        for (uint32_t i = tid; i < cand; i += blockDim.x) {
            candindex[i] = index[merged + i];
            canddist[i] = dist[merged + i];
        }
        __syncthreads();
        if (sorted) dispatch_beam_management(
            allindex, alldist, mergedindex, mergeddist, candidate_radix_scratch,
            cand, padded, iter == 0);
        __syncthreads();

        // [3] clear the padded beam tail and the consumed candidates
        for (uint32_t i = beam + tid; i < padded; i += blockDim.x) {
            allindex[i] = MAX_INDEX;
            alldist[i] = FLT_MAX;
        }
        for (uint32_t i = tid; i < cand; i += blockDim.x) {
            candindex[i] = MAX_INDEX;
            canddist[i] = FLT_MAX;
        }
        __syncthreads();

        // [4] a single parent sorts its estimated children before admission
        if (spec > 1) continue;
        const size_t pruned = ((qid + iter + 1) % iters) * cand;
        for (uint32_t i = tid; i < cand; i += blockDim.x) {
            candindex[i] = index[pruned + i];
            canddist[i] = dist[pruned + i];
        }
        __syncthreads();
        if (sorted) dispatch_candidate_sort(candindex, canddist, cand, candidate_radix_scratch);
        __syncthreads();
    }
}

/**
 * @brief Average wall time of one launch after a warm-up launch.
 * @tparam LAUNCH callable that launches and synchronizes one kernel run
 * @param launch the launch callable
 * @return milliseconds per launch
 */
template <class LAUNCH>
static double elapsed(LAUNCH &&launch) {
    launch();
    const auto start = std::chrono::high_resolution_clock::now();
    for (unsigned run = 0; run < RUNS; run++) launch();
    const auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / RUNS;
}

/**
 * @brief Time the sorting kernel and its sort-free twin for one shape. SHAME(WIDEFUNC)
 * @param form profiled configuration
 * @param dist device candidate distance batches
 * @param index device candidate index batches
 * @return both timings
 */
static Timing flashrun(Shape form, const float *dist, const uint32_t *index) {
    const uint32_t padded = effective_sort_beam_size(form.beam);
    size_t size = static_cast<size_t>(padded + form.cand) * (sizeof(INDEX_T) + sizeof(DISTANCE_T));
    if (topk_external_merge_scratch_needed(padded, form.cand)) {
        size = align_up_uintptr(size, alignof(INDEX_T)) + padded * sizeof(INDEX_T);
        size = align_up_uintptr(size, alignof(DISTANCE_T)) + padded * sizeof(DISTANCE_T);
    }
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && form.cand > 256) {
        size = align_up_uintptr(size, candidate_radix_sort_scratch_alignment()) + candidate_radix_sort_scratch_bytes();
    }
    const int bytes = static_cast<int>(size);
    CUDA_SAFE_CALL(cudaFuncSetAttribute(reinterpret_cast<const void *>(flashsort<true>),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    CUDA_SAFE_CALL(cudaFuncSetAttribute(reinterpret_cast<const void *>(flashsort<false>),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));

    Timing timing{form, 0.0, 0.0};
    timing.base = elapsed([&] {
        flashsort<false><<<QUERIES, BLOCK_SIZE, size>>>(QUERIES, form.beam, form.cand, form.iter, form.spec, dist, index);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    });
    timing.sort = elapsed([&] {
        flashsort<true><<<QUERIES, BLOCK_SIZE, size>>>(QUERIES, form.beam, form.cand, form.iter, form.spec, dist, index);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    });
    return timing;
}

/**
 * @brief Print one row per shape with the isolated sort cost. SHAME(WIDEFUNC)
 * @param rows timings of every shape
 */
static void report(const std::vector<Timing> &rows) {
    printf("block=%d max_theta=%d queries=%u runs=%u\n", BLOCK_SIZE, WARPS_PER_BLOCK, QUERIES, RUNS);
    printf("%6s %5s %6s %5s %5s %10s %10s %14s %7s\n",
           "theta", "beam", "padded", "cand", "iters", "base_ms", "sort_ms", "sort_us/q/iter", "share");
    for (const Timing &row : rows) {
        const double cost = (row.sort - row.base) * 1e3 / (static_cast<double>(QUERIES) * row.form.iter);
        const double share = (row.sort - row.base) / row.sort;
        printf("%6u %5u %6u %5u %5u %10.2f %10.2f %14.4f %6.1f%%\n",
               row.form.spec, row.form.beam, effective_sort_beam_size(row.form.beam), row.form.cand,
               row.form.iter, row.base, row.sort, cost, share * 100.0);
    }
}

int main() {
    // [1] candidate buffer and iteration budget as QuantizedPrunedBeamSearch derives them
    const unsigned cand = round_up_power2_u32(candidate_buffer_capacity(DEGREE));
    const unsigned maxiter = (BEAMS[3] * 11 + 9) / 10;

    // [2] upload enough random batches for the longest iteration budget
    const size_t count = static_cast<size_t>(maxiter) * cand;
    std::mt19937 rng(SEED);
    std::uniform_real_distribution<float> fdist(0.0f, 1e6f);
    std::vector<float> hdist(count);
    std::vector<uint32_t> hindex(count);
    for (size_t i = 0; i < count; i++) {
        hdist[i] = fdist(rng);
        hindex[i] = static_cast<uint32_t>(i);
    }
    float *dist = nullptr;
    uint32_t *index = nullptr;
    CUDA_SAFE_CALL(cudaMalloc((void **)&dist, count * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void **)&index, count * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMemcpy(dist, hdist.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(index, hindex.data(), count * sizeof(uint32_t),
                              cudaMemcpyHostToDevice));

    // [3] sweep every speculation degree this block admits over every beam
    std::vector<Timing> rows;
    for (unsigned spec : SPECS) {
        if (spec > WARPS_PER_BLOCK) continue;
        for (unsigned beam : BEAMS) {
            rows.push_back(flashrun(Shape{beam, cand, (beam * 11 + 9) / 10, spec}, dist, index));
        }
    }
    report(rows);

    CUDA_SAFE_CALL(cudaFree(dist));
    CUDA_SAFE_CALL(cudaFree(index));
    return 0;
}
