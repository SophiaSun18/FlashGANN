#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
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
 * @param levels ascending reconstruction levels
 * @param value coordinate of a unit-norm rotated residual
 * @return index of the nearest level
 */
inline uint8_t quantize_level(const std::vector<float>& levels, float value) {
    size_t best = 0;
    for (size_t k = 1; k < levels.size(); ++k) {
        if (value >= 0.5f * (levels[k - 1] + levels[k])) best = k;
    }
    return static_cast<uint8_t>(best);
}
