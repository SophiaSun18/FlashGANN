#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

/** @brief Marks the size helpers below callable from host code and CUDA device code. */
#ifdef __CUDACC__
#define QUANT_HD __host__ __device__
#else
#define QUANT_HD
#endif

/**
 * @brief Quantizer family used to build and scan the packed neighbor codes.
 *
 * QUANT_RBQ is RaBitQ over a signed uniform grid, encoded by encode_rbq. QUANT_TBQ is TurboQuant's
 * inner-product algorithm, encoded by encode_tbq: an MSE stage at bits-1 bits over the unit
 * residual, then one QJL sign bit per dimension over what that stage leaves behind.
 */
enum QuantType { QUANT_RBQ, QUANT_TBQ };

/** @brief Quantizer family of the search binary, set by main.cu from the -quant flag. */
inline QuantType g_quant_type = QUANT_RBQ;

/**
 * @brief Bounds on bits per dimension of one scanned code stage, and the bits of one 4-bit scan group.
 *
 * QUANT_NIBBLE_DIMS / bits is the number of dimensions one scan group holds, see quant_group.
 */
inline constexpr int QUANT_MIN_BITS = 1;
inline constexpr int QUANT_MAX_BITS = 4;
inline constexpr int QUANT_NIBBLE_DIMS = 4;

/**
 * @brief Human readable name of a quantizer family, for log lines.
 * @param quant quantizer family
 * @return static string, "rbq" or "tbq"
 */
inline const char* quant_name(QuantType quant) {
    return quant == QUANT_TBQ ? "tbq" : "rbq";
}

/**
 * @brief Parse a quantizer family from a command line token, for main.cu and tools/buildindex.cc.
 * @param name token, "rbq" or "tbq", may be nullptr
 * @return parsed family, QUANT_RBQ when the token is unknown or nullptr
 */
inline QuantType quant_parse(const char* name) {
    return (name && std::strcmp(name, "tbq") == 0) ? QUANT_TBQ : QUANT_RBQ;
}

/**
 * @brief Number of vector dimensions packed into one 4-bit scan group.
 * @param bits bits per dimension
 * @return dimensions per nibble
 */
QUANT_HD inline constexpr int quant_group(int bits) {
    return QUANT_NIBBLE_DIMS / bits;
}

/**
 * @brief Number of 4-bit scan groups spanning one padded vector.
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @return group count, paddim * bits / 4
 */
QUANT_HD inline constexpr size_t quant_groups(size_t paddim, int bits) {
    return (paddim * static_cast<size_t>(bits)) >> 2;
}

/**
 * @brief Packed code footprint of one neighbor in bytes.
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @return bytes per neighbor, paddim * bits / 8
 */
QUANT_HD inline constexpr size_t quant_bytes(size_t paddim, int bits) {
    return (paddim * static_cast<size_t>(bits)) >> 3;
}

/**
 * @brief Packed code footprint of one neighbor in floats, the unit of an index row.
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @return floats per neighbor, paddim * bits / 32
 */
QUANT_HD inline constexpr size_t quant_words(size_t paddim, int bits) {
    return (paddim * static_cast<size_t>(bits)) >> 5;
}

/**
 * @brief Query lookup table footprint for one padded vector, used to size the shared-memory table.
 * @param paddim padded dimension
 * @param bits bits per dimension
 * @return lookup table bytes, 16 one-byte entries per scan group
 */
QUANT_HD inline constexpr size_t quant_lutbytes(size_t paddim, int bits) {
    return quant_groups(paddim, bits) << 4;
}

/**
 * @brief Whether a bit width maps onto the 4-bit scan group decoder.
 * @param bits bits per dimension
 * @return true for 1, 2, or 4, the widths that divide the nibble evenly
 */
QUANT_HD inline constexpr bool quant_valid(int bits) {
    return bits == 1 || bits == 2 || bits == 4;
}

/**
 * @brief Bit width of the TurboQuant MSE stage, the remaining bit going to the QJL sign.
 * @param bits total bits per dimension
 * @return bits spent on the MSE stage, bits - 1
 */
QUANT_HD inline constexpr int quant_stage(int bits) {
    return bits - 1;
}

/**
 * @brief Per-neighbor factor slots a quantizer stores, used to lay out the index row.
 * @param quant quantizer family
 * @return 3 for QUANT_RBQ, 5 for QUANT_TBQ
 */
QUANT_HD inline constexpr int quant_factors(QuantType quant) {
    return quant == QUANT_TBQ ? 5 : 3;
}

/**
 * @brief Whether a TurboQuant total width leaves a decodable MSE stage after the sign bit.
 * @param bits total bits per dimension
 * @return true when bits - 1 is 1, 2, or 4
 */
QUANT_HD inline constexpr bool quant_validprod(int bits) {
    return quant_valid(quant_stage(bits));
}

/**
 * @brief Whether a quantizer and width pair has a scan instantiation, checked by the builder and the index loader.
 * @param quant quantizer family
 * @param bits total bits per dimension
 * @return true when the pair is supported
 */
QUANT_HD inline constexpr bool quant_supported(QuantType quant, int bits) {
    return quant == QUANT_TBQ ? quant_validprod(bits) : quant_valid(bits);
}
