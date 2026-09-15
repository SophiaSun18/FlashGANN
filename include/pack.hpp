#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "quant.hpp"

/**
 * @brief Pack per-dimension codes of one neighbor into the 4-bit scan group layout.
 *
 * Group gi covers dimensions [gi*g, gi*g+g) and occupies one nibble, most significant
 * dimension first, matching fastscan_decode_code_for_neighbor_seq_lut_gpu.
 *
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @param codes paddim per-dimension codes, each below 2^bits
 * @param packed destination of quant_bytes(paddim, bits) bytes
 */
inline void pack_codes(size_t paddim, int bits, const uint8_t* codes, uint8_t* packed) {
    const int group = quant_group(bits);
    const size_t groups = quant_groups(paddim, bits);
    for (size_t i = 0; i < quant_bytes(paddim, bits); ++i) packed[i] = 0;

    for (size_t gi = 0; gi < groups; ++gi) {
        uint8_t nibble = 0;
        for (int j = 0; j < group; ++j) {
            const int shift = QUANT_NIBBLE_DIMS - (j + 1) * bits;
            nibble = static_cast<uint8_t>(nibble | (codes[gi * group + j] << shift));
        }
        uint8_t& slot = packed[gi >> 1];
        slot = static_cast<uint8_t>(slot | ((gi & 1) ? (nibble << 4) : nibble));
    }
}

/**
 * @brief Read a reconstruction level table emitted from the TurboQuant codebooks.
 * @param path binary file of int32 bits, int32 count, then count floats
 * @param bits expected bits per dimension
 * @param levels destination table of 2^bits ascending levels
 */
inline void read_levels(const std::string& path, int bits, std::vector<float>& levels) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open level file: " + path);

    int32_t file_bits = 0;
    int32_t count = 0;
    fin.read(reinterpret_cast<char*>(&file_bits), sizeof(int32_t));
    fin.read(reinterpret_cast<char*>(&count), sizeof(int32_t));
    if (file_bits != bits) throw std::runtime_error("Level file bit width mismatch: " + path);
    if (count != (1 << bits)) throw std::runtime_error("Level file count mismatch: " + path);

    levels.resize(static_cast<size_t>(count));
    fin.read(reinterpret_cast<char*>(levels.data()), sizeof(float) * levels.size());
    if (!fin) throw std::runtime_error("Truncated level file: " + path);
}

/**
 * @brief Quantize one coordinate against an ascending level table.
 *
 * Binary searches the 2^bits - 1 midpoints separating adjacent levels, so the cost is
 * logarithmic in the table size rather than linear.
 *
 * @param levels ascending reconstruction levels
 * @param value coordinate of a unit-norm rotated residual
 * @return index of the nearest level
 */
inline uint8_t quantize_level(const std::vector<float>& levels, float value) {
    size_t lo = 0;
    size_t hi = levels.size() - 1;
    while (lo < hi) {
        const size_t mid = (lo + hi + 1) >> 1;
        if (value >= 0.5f * (levels[mid - 1] + levels[mid])) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return static_cast<uint8_t>(lo);
}

/**
 * @brief Pick the signed uniform grid code whose direction lies closest to a unit vector.
 *
 * Code c reconstructs to 2c - (2^bits - 1), an odd integer, so the sign rides in the code
 * and the magnitude cell m maps to 2m + 1. Sweeps every scale at which a magnitude cell
 * crosses, the Extended RaBitQ search, and keeps the cell set of highest cosine.
 *
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @param unit unit-norm rotated residual
 * @param codes destination of paddim codes, each below 2^bits
 * @return cosine between unit and the reconstructed grid direction
 */
static inline float gridcode(size_t paddim, int bits, const float* unit, uint8_t* codes) {
    using Event = std::pair<float, size_t>;
    const int half = 1 << (bits - 1);
    std::vector<int> mag(paddim, 0);
    std::vector<size_t> order;
    order.reserve(paddim * static_cast<size_t>(half - 1));
    std::priority_queue<Event, std::vector<Event>, std::greater<Event>> heap;

    // [1] start every magnitude at the lowest cell and queue its first crossing
    double num = 0.0, den = 0.0;
    for (size_t k = 0; k < paddim; ++k) {
        const float a = std::fabs(unit[k]);
        num += a;
        den += 1.0;
        if (half > 1 && a > 0.0f) heap.emplace(1.0f / a, k);
    }

    // [2] sweep crossings in increasing scale and remember the best prefix
    double best = num / std::sqrt(den);
    size_t bestn = 0;
    while (!heap.empty()) {
        const size_t k = heap.top().second;
        heap.pop();
        const float a = std::fabs(unit[k]);
        const int m = ++mag[k];
        num += 2.0 * a;
        den += 8.0 * m;
        order.push_back(k);
        if (m + 1 < half) heap.emplace(static_cast<float>(m + 1) / a, k);
        const double cos = num / std::sqrt(den);
        if (cos > best) best = cos, bestn = order.size();
    }

    // [3] replay the best prefix and fold the sign back into the code
    std::fill(mag.begin(), mag.end(), 0);
    for (size_t i = 0; i < bestn; ++i) ++mag[order[i]];
    double ip = 0.0, sq = 0.0;
    for (size_t k = 0; k < paddim; ++k) {
        const bool pos = unit[k] >= 0.0f;
        codes[k] = static_cast<uint8_t>(pos ? half + mag[k] : half - 1 - mag[k]);
        const double y = pos ? 2.0 * mag[k] + 1.0 : -2.0 * mag[k] - 1.0;
        ip += unit[k] * y;
        sq += y * y;
    }
    return static_cast<float>(ip / std::sqrt(sq));
}
