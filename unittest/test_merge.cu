#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "src/adaptive_search_utils.cuh"

constexpr unsigned TRIALS = 64;
constexpr unsigned SEED = 7;
constexpr unsigned LEVELS = 16;
constexpr unsigned CANDS[] = {32, 64, 128, 256, 512};
constexpr unsigned BEAMS[] = {32, 64, 100, 128, 192, 256, 300, 384, 512, 640, 768, 1000, 1024, 1500, 2048, 3000};

/**
 * @brief Merge one candidate batch into one beam per block through dispatch_beam_management.
 *
 * Block b owns span b of every buffer. Shared memory holds [beam | candidates] indices, then
 * [beam | candidates] distances, then the optional candidate radix scratch, matching
 * src/adaptive_search.cuh.
 *
 * @param topk beam length
 * @param cand candidate batch length
 * @param index beam then candidate indices, TRIALS spans of topk + cand
 * @param dist beam then candidate distances, same layout
 * @param outindex merged beam indices, TRIALS spans of topk
 * @param outdist merged beam distances, same layout
 * SHAME(WIDEFUNC)
 */
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void mergecheck(unsigned topk, unsigned cand, const uint32_t *__restrict__ index, const float *__restrict__ dist,
                uint32_t *__restrict__ outindex, float *__restrict__ outdist) {
    const uint32_t total = topk + cand;
    const size_t span = static_cast<size_t>(blockIdx.x) * total;

    // [1] carve the buffers and load this block's beam and candidates
    extern __shared__ char shared_memory[];
    INDEX_T *allindex = reinterpret_cast<INDEX_T *>(shared_memory);
    DISTANCE_T *alldist = reinterpret_cast<DISTANCE_T *>(allindex + total);
    char *tail = reinterpret_cast<char *>(alldist + total);
    void *scratch = nullptr;
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && cand > 256) {
        using CandidateRadixSort = cub::BlockRadixSort<DISTANCE_T, BLOCK_SIZE, 8, INDEX_T>;
        scratch = allocate_shared_tail_array<typename CandidateRadixSort::TempStorage>(tail, 1);
    }
    for (uint32_t i = threadIdx.x; i < total; i += blockDim.x) {
        allindex[i] = index[span + i];
        alldist[i] = dist[span + i];
    }
    __syncthreads();

    // [2] merge, then write the beam back
    dispatch_beam_management(allindex, alldist, scratch, cand, topk, false);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < topk; i += blockDim.x) {
        outindex[static_cast<size_t>(blockIdx.x) * topk + i] = allindex[i];
        outdist[static_cast<size_t>(blockIdx.x) * topk + i] = alldist[i];
    }
}

/**
 * @brief Fill one trial: a sorted beam with an empty tail and an unsorted, partly empty batch.
 * @param rng random source
 * @param topk beam length
 * @param cand candidate batch length
 * @param tied draw distances from LEVELS values instead of keeping them distinct
 * @param index span to fill with indices
 * @param dist span to fill with distances
 */
static void filltrial(std::mt19937 &rng, unsigned topk, unsigned cand, bool tied, uint32_t *index, float *dist) {
    const unsigned total = topk + cand;
    std::vector<unsigned> order(total);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng);
    const unsigned beamfill = std::uniform_int_distribution<unsigned>(0, topk)(rng);
    const unsigned candfill = std::uniform_int_distribution<unsigned>(0, cand)(rng);
    auto distance = [&](unsigned id) {
        return 0.5f * static_cast<float>(tied ? 1 + id % LEVELS : id + 1);
    };

    // [1] a sorted beam whose indices stay distinct, some carrying the expanded flag
    std::vector<std::pair<float, uint32_t>> beam(topk, {FLT_MAX, MAX_INDEX});
    for (unsigned i = 0; i < beamfill; ++i) {
        const uint32_t expanded = i % 3 == 0 ? 0x80000000u : 0u;
        beam[i] = {distance(order[i]), order[i] | expanded};
    }
    std::sort(beam.begin(), beam.end());
    for (unsigned i = 0; i < topk; ++i) {
        index[i] = beam[i].second;
        dist[i] = beam[i].first;
    }

    // [2] the batch stays unsorted, with its empty slots scattered through it
    for (unsigned i = 0; i < cand; ++i) index[topk + i] = i < candfill ? order[topk + i] : MAX_INDEX;
    std::shuffle(index + topk, index + total, rng);
    for (unsigned i = 0; i < cand; ++i) {
        const uint32_t id = index[topk + i];
        dist[topk + i] = id == MAX_INDEX ? FLT_MAX : distance(id);
    }
}

/**
 * @brief Check one merged beam against its inputs.
 *
 * Distances must equal the topk smallest input distances. With distinct distances the
 * indices must match exactly; with ties no valid index may repeat and every valid output
 * must carry its own input distance, which rules out dropped or duplicated entries.
 *
 * @param topk beam length
 * @param cand candidate batch length
 * @param tied whether distances repeat
 * @param index trial inputs, topk + cand indices
 * @param dist trial inputs, topk + cand distances
 * @param outindex merged indices, topk
 * @param outdist merged distances, topk
 * @return whether the merged beam is correct
 * SHAME(MANYARG)
 */
static bool checktrial(unsigned topk, unsigned cand, bool tied, const uint32_t *index, const float *dist,
                       const uint32_t *outindex, const float *outdist) {
    std::vector<std::pair<float, uint32_t>> all(topk + cand);
    std::unordered_map<uint32_t, float> source;
    for (unsigned i = 0; i < topk + cand; ++i) {
        all[i] = {dist[i], index[i]};
        if (index[i] != MAX_INDEX) source[index[i]] = dist[i];
    }
    std::sort(all.begin(), all.end());
    std::unordered_set<uint32_t> seen;
    for (unsigned i = 0; i < topk; ++i) {
        if (outdist[i] != all[i].first) return false;
        if (!tied && outindex[i] != all[i].second) return false;
        if (outindex[i] == MAX_INDEX) continue;
        if (!seen.insert(outindex[i]).second) return false;
        const auto hit = source.find(outindex[i]);
        if (hit == source.end() || hit->second != outdist[i]) return false;
    }
    return true;
}

/**
 * @brief Run every trial of one shape on the device and check each merged beam.
 * @param topk beam length
 * @param cand candidate batch length
 * @param tied whether distances repeat
 * @param rng random source
 * @return number of trials whose merged beam is wrong
 * SHAME(WIDEFUNC)
 */
static unsigned checkshape(unsigned topk, unsigned cand, bool tied, std::mt19937 &rng) {
    const unsigned total = topk + cand;
    std::vector<uint32_t> index(static_cast<size_t>(TRIALS) * total);
    std::vector<float> dist(index.size());
    for (unsigned t = 0; t < TRIALS; ++t) filltrial(rng, topk, cand, tied, index.data() + t * total, dist.data() + t * total);

    // [1] device merge
    uint32_t *dindex, *doutindex;
    float *ddist, *doutdist;
    CUDA_SAFE_CALL(cudaMalloc(&dindex, index.size() * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&ddist, dist.size() * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&doutindex, static_cast<size_t>(TRIALS) * topk * sizeof(uint32_t)));
    CUDA_SAFE_CALL(cudaMalloc(&doutdist, static_cast<size_t>(TRIALS) * topk * sizeof(float)));
    CUDA_SAFE_CALL(cudaMemcpy(dindex, index.data(), index.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ddist, dist.data(), dist.size() * sizeof(float), cudaMemcpyHostToDevice));
    size_t bytes = static_cast<size_t>(total) * (sizeof(INDEX_T) + sizeof(DISTANCE_T));
    if (!GPU_RABITQ_USE_BLOCK_CANDIDATE_SORT && cand > 256) {
        bytes = align_up_uintptr(bytes, candidate_radix_sort_scratch_alignment()) + candidate_radix_sort_scratch_bytes();
    }
    CUDA_SAFE_CALL(cudaFuncSetAttribute(reinterpret_cast<const void *>(mergecheck),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
    mergecheck<<<TRIALS, BLOCK_SIZE, bytes>>>(topk, cand, dindex, ddist, doutindex, doutdist);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<uint32_t> outindex(static_cast<size_t>(TRIALS) * topk);
    std::vector<float> outdist(outindex.size());
    CUDA_SAFE_CALL(cudaMemcpy(outindex.data(), doutindex, outindex.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(outdist.data(), doutdist, outdist.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(dindex));
    CUDA_SAFE_CALL(cudaFree(ddist));
    CUDA_SAFE_CALL(cudaFree(doutindex));
    CUDA_SAFE_CALL(cudaFree(doutdist));

    // [2] host check, trial by trial
    unsigned failed = 0;
    for (unsigned t = 0; t < TRIALS; ++t) {
        const bool good = checktrial(topk, cand, tied, index.data() + t * total, dist.data() + t * total,
                                     outindex.data() + t * topk, outdist.data() + t * topk);
        failed += good ? 0 : 1;
    }
    return failed;
}

int main() {
    std::mt19937 rng(SEED);
    unsigned failed = 0;
    unsigned shapes = 0;
    for (const bool tied : {false, true}) {
        for (const unsigned cand : CANDS) {
            for (const unsigned beam : BEAMS) {
                const unsigned topk = effective_sort_beam_size(beam);
                const unsigned bad = checkshape(topk, cand, tied, rng);
                ++shapes;
                failed += bad;
                if (bad) printf("FAIL block=%d beam=%u topk=%u cand=%u tied=%d: %u of %u trials wrong\n",
                                BLOCK_SIZE, beam, topk, cand, tied ? 1 : 0, bad, TRIALS);
            }
        }
    }
    printf("test_merge block=%d: %u shapes x %u trials, %u failing trials\n", BLOCK_SIZE, shapes, TRIALS, failed);
    return failed ? 1 : 0;
}
