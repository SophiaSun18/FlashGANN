#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "src/adaptive_search_utils.cuh"

constexpr unsigned REPEATS = 8;
constexpr unsigned SEED = 11;
constexpr unsigned VALUES = 5;
constexpr unsigned VALIDS = 5;

/**
 * @brief Run the warp top-k keep over one trial per warp.
 *
 * Trial t is taken by warp t of the grid; lane l holds position l, and with two slots also
 * position l + 32, of that trial's value and valid spans.
 *
 * @tparam SLOTS values per lane, 1 or 2
 * @param count number of trials
 * @param value trial values, WARP_SIZE * SLOTS per trial
 * @param valid trial validity flags, same layout
 * @param keep keep count of each trial
 * @param kept receives whether each position is kept, same layout as value
 */
template <int SLOTS>
static __global__ GPU_LAUNCH_BOUNDS(BLOCK_SIZE)
void keepcheck(unsigned count, const float* __restrict__ value, const uint8_t* __restrict__ valid,
               const int* __restrict__ keep, uint8_t* __restrict__ kept) {
    const unsigned trial = blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (trial >= count) return;
    const size_t base = static_cast<size_t>(trial) * WARP_SIZE * SLOTS;

    if constexpr (SLOTS == 1) {
        kept[base + lane] = warp_keep_topk_smallest_f32(value[base + lane], valid[base + lane] != 0, keep[trial]);
    } else {
        const float value0 = value[base + lane];
        const float value1 = value[base + WARP_SIZE + lane];
        const bool valid0 = valid[base + lane] != 0;
        const bool valid1 = valid[base + WARP_SIZE + lane] != 0;
        kept[base + lane] = warp_keep_topk_smallest_pair_f32(
            value0, valid0, lane, value0, valid0, value1, valid1, keep[trial]);
        kept[base + WARP_SIZE + lane] = warp_keep_topk_smallest_pair_f32(
            value1, valid1, lane + WARP_SIZE, value0, valid0, value1, valid1, keep[trial]);
    }
}

/**
 * @brief Draw one value of the given pattern.
 * @param rng random source
 * @param pattern 0 distinct, 1 four tied levels, 2 signed zeros and units, 3 large magnitudes, 4 all equal
 * @return the value
 */
static float drawvalue(std::mt19937 &rng, unsigned pattern) {
    static const float SIGNED[] = {-0.0f, 0.0f, 1.0f, -1.0f};
    static const float LARGE[] = {FLT_MAX, -FLT_MAX, 3.0e38f, 1.0e-38f};
    switch (pattern) {
        case 0: return std::uniform_real_distribution<float>(-1.0e3f, 1.0e3f)(rng);
        case 1: return static_cast<float>(std::uniform_int_distribution<int>(0, 3)(rng));
        case 2: return SIGNED[std::uniform_int_distribution<int>(0, 3)(rng)];
        case 3: return LARGE[std::uniform_int_distribution<int>(0, 3)(rng)];
        default: return 7.0f;
    }
}

/**
 * @brief Draw one validity flag of the given pattern.
 * @param rng random source
 * @param pattern 0 all, 1 none, 2 half, 3 sparse, 4 second half only
 * @param position the flag's position within its trial
 * @return whether the position takes part
 */
static bool drawvalid(std::mt19937 &rng, unsigned pattern, unsigned position) {
    switch (pattern) {
        case 0: return true;
        case 1: return false;
        case 2: return std::bernoulli_distribution(0.5)(rng);
        case 3: return std::bernoulli_distribution(0.1)(rng);
        default: return position >= WARP_SIZE / 2;
    }
}

/**
 * @brief Sweep keep counts, value patterns and validity patterns for one slot count.
 * @tparam SLOTS values per lane, 1 or 2
 * @param rng random source
 * @return number of trials whose kept set is wrong
 */
template <int SLOTS>
static unsigned keepsweep(std::mt19937 &rng) {
    constexpr unsigned width = WARP_SIZE * SLOTS;
    std::vector<float> value;
    std::vector<uint8_t> valid;
    std::vector<int> keep;

    // [1] every keep count from 0 past the width, under every pattern pair
    for (unsigned repeat = 0; repeat < REPEATS; ++repeat) {
        for (unsigned vpattern = 0; vpattern < VALUES; ++vpattern) {
            for (unsigned fpattern = 0; fpattern < VALIDS; ++fpattern) {
                for (int k = 0; k <= static_cast<int>(width) + 1; ++k) {
                    for (unsigned p = 0; p < width; ++p) {
                        value.push_back(drawvalue(rng, vpattern));
                        valid.push_back(drawvalid(rng, fpattern, p) ? 1 : 0);
                    }
                    keep.push_back(k);
                }
            }
        }
    }
    const unsigned count = static_cast<unsigned>(keep.size());

    // [2] device keep
    float *dvalue;
    uint8_t *dvalid, *dkept;
    int *dkeep;
    CUDA_SAFE_CALL(cudaMalloc(&dvalue, value.size() * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc(&dvalid, valid.size()));
    CUDA_SAFE_CALL(cudaMalloc(&dkept, valid.size()));
    CUDA_SAFE_CALL(cudaMalloc(&dkeep, keep.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(dvalue, value.data(), value.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dvalid, valid.data(), valid.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(dkeep, keep.data(), keep.size() * sizeof(int), cudaMemcpyHostToDevice));
    const unsigned blocks = (count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    keepcheck<SLOTS><<<blocks, BLOCK_SIZE>>>(count, dvalue, dvalid, dkeep, dkept);
    CUDA_SAFE_CALL(cudaGetLastError());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<uint8_t> kept(valid.size());
    CUDA_SAFE_CALL(cudaMemcpy(kept.data(), dkept, kept.size(), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(dvalue));
    CUDA_SAFE_CALL(cudaFree(dvalid));
    CUDA_SAFE_CALL(cudaFree(dkept));
    CUDA_SAFE_CALL(cudaFree(dkeep));

    // [3] host reference: stable order by value then position, first min(keep, valid) kept
    unsigned failed = 0;
    for (unsigned t = 0; t < count; ++t) {
        const size_t base = static_cast<size_t>(t) * width;
        std::vector<std::pair<float, unsigned>> order;
        for (unsigned p = 0; p < width; ++p) {
            if (valid[base + p]) order.push_back({value[base + p] + 0.0f, p});
        }
        std::stable_sort(order.begin(), order.end(),
                         [](const auto &a, const auto &b) { return a.first < b.first; });
        std::vector<uint8_t> expect(width, 0);
        const size_t take = std::min(order.size(), static_cast<size_t>(std::max(keep[t], 0)));
        for (size_t i = 0; i < take; ++i) expect[order[i].second] = 1;
        bool good = true;
        for (unsigned p = 0; p < width; ++p) good = good && (kept[base + p] != 0) == (expect[p] != 0);
        if (!good && failed < 4) printf("FAIL block=%d slots=%d trial=%u keep=%d\n", BLOCK_SIZE, SLOTS, t, keep[t]);
        failed += good ? 0 : 1;
    }
    printf("test_keep block=%d slots=%d: %u trials, %u failing\n", BLOCK_SIZE, SLOTS, count, failed);
    return failed;
}

int main() {
    std::mt19937 rng(SEED);
    unsigned failed = 0;
    failed += keepsweep<1>(rng);
    failed += keepsweep<2>(rng);
    return failed ? 1 : 0;
}
