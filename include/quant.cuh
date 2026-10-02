#pragma once

#include "utils.cuh"
#include "quant.hpp"

/*-------------------------------------------- query factors --------------------------------------------*/
/**
 * @brief Packed code layout lanes of the scan: 32 keeps each neighbor's codes contiguous.
 *
 * Smaller values, 2 to 16, interleave WARP_SIZE / lanes neighbors byte by byte in one tile.
 * pack_codes writes the contiguous layout only.
 */
#ifndef GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES
#define GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES 32
#endif

/**
 * @brief Per-query state of the estimate, filled by query preparation and read by every scan.
 */
struct QueryFactors {
    float *rotated_query = nullptr;  // rotated or sketched query, padded_dim floats, live through preparation only
    uint8_t *lut = nullptr;          // lookup table in shared memory, 16 entries per scan group
    float low_val = 0.0f;            // lowest transformed coordinate, the quantizer offset
    float high_val = 0.0f;           // highest transformed coordinate
    float width = 0.0f;              // quantizer step, (high_val - low_val) / (2^QG_BQUERY - 1)
    int32_t sum_q = 0;               // quantized query sum, removed by the scan's sign correction
};

/*-------------------------------------------- query preparation --------------------------------------------*/
/**
 * @brief Rotate a vector into the space the codes were built in, block cooperatively.
 *
 * Flips signs, zero-pads to padded_dim, then runs a fast Walsh-Hadamard transform. It matches
 * FhtRotator in rotator.hpp except for the 1/sqrt(padded_dim) scale, which the per-neighbor
 * factors absorb.
 *
 * @param src raw vector, dim floats
 * @param dst destination of padded_dim floats, also the transform workspace
 * @param signs sign vector of the index rotation
 * @param dim raw dimension
 * @param padded_dim padded dimension, a power of two
 */
static __device__ inline void rotate_vector_gpu(const float *src, float *dst,
                                                const float *signs, int dim, int padded_dim) {

    int tid = tidx();

    // [1] sign flip and zero padding
    for (size_t i = tid; i < dim; i += blockDim.x) {
        dst[i] = src[i] * signs[i];
    }
    for (int i = dim + tid; i < padded_dim; i += blockDim.x) {
        dst[i] = 0.0f;
    }
    __syncthreads();

    // [2] in-place fast Walsh-Hadamard transform, unnormalized
    for (size_t len = 1; len < padded_dim; len <<= 1) {
        int pairs = padded_dim >> 1;
        for (int pair_idx = tid; pair_idx < pairs; pair_idx += blockDim.x) {
            int block = pair_idx / len;
            int j = pair_idx % len;
            int base = block * (len << 1);

            int idx1 = base + j;
            int idx2 = idx1 + len;

            float u = dst[idx1];
            float v = dst[idx2];
            dst[idx1] = u + v;
            dst[idx2] = u - v;
        }
        __syncthreads();
    }
}

/**
 * @brief Apply the Fastfood sketch to one vector for the TurboQuant sign stage, block cooperatively.
 *
 * Mirrors FastfoodSketch::apply including both 1/sqrt(padded_dim) scales, so the sketched
 * query lands on the same scale as the constants the builder folded into the factors.
 * Unlike rotate_vector_gpu this transform is normalized.
 *
 * @param src input vector of dim floats; the search passes the rotated query
 * @param dst destination of padded_dim floats, also the transform workspace
 * @param flipv sign vector applied before the first transform
 * @param spectr diagonal magnitudes applied between the transforms
 * @param flipu sign vector applied with spectr
 * @param dim input dimension; the search passes padded_dim
 * @param padded_dim padded dimension, a power of two
 */
static __device__ inline void sketch_apply(const float *src, float *dst, const float *flipv,
                                           const float *spectr, const float *flipu,
                                           int dim, int padded_dim) {
    const int tid = tidx();
    const float scale = rsqrtf(static_cast<float>(padded_dim));

    // [1] first sign flip and zero padding
    for (int i = tid; i < dim; i += blockDim.x) dst[i] = src[i] * flipv[i];
    for (int i = dim + tid; i < padded_dim; i += blockDim.x) dst[i] = 0.0f;
    __syncthreads();

    // [2] first transform
    for (int len = 1; len < padded_dim; len <<= 1) {
        for (int pair = tid; pair < (padded_dim >> 1); pair += blockDim.x) {
            const int block = pair / len;
            const int j = pair % len;
            const int idx = block * (len << 1) + j;
            const float a = dst[idx];
            const float b = dst[idx + len];
            dst[idx] = a + b;
            dst[idx + len] = a - b;
        }
        __syncthreads();
    }

    // [3] first scale, spectrum, and second sign flip
    for (int i = tid; i < padded_dim; i += blockDim.x) dst[i] *= spectr[i] * scale * flipu[i];
    __syncthreads();

    // [4] second transform and the closing scale
    for (int len = 1; len < padded_dim; len <<= 1) {
        for (int pair = tid; pair < (padded_dim >> 1); pair += blockDim.x) {
            const int block = pair / len;
            const int j = pair % len;
            const int idx = block * (len << 1) + j;
            const float a = dst[idx];
            const float b = dst[idx + len];
            dst[idx] = a + b;
            dst[idx + len] = a - b;
        }
        __syncthreads();
    }

    for (int i = tid; i < padded_dim; i += blockDim.x) dst[i] *= scale;
    __syncthreads();
}

/**
 * @brief Prepare a query for 1-bit RaBitQ codes: rotate, quantize, and build the lookup table.
 *
 * Coordinates are quantized to QG_BQUERY bits over the rotated query's range. Row cb of the
 * table holds the 16 subset sums of coordinates 4cb .. 4cb + 3, one per 4-bit code.
 * scratch.rotated_query and scratch.lut must already point at padded_dim floats and
 * quant_lutbytes(padded_dim, 1) bytes.
 *
 * @param query_raw raw query in shared memory
 * @param scratch query factors, receives the rotated query, its range, the table and the query sum
 * @param signs_ptr sign vector of the index rotation
 * @param dim raw dimension
 * @param padded_dim padded dimension
 */
static __device__ inline void query_prepare_lut_gpu(const float *query_raw, QueryFactors &scratch,
                                                    const float *signs_ptr, int dim, int padded_dim) {
    constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);

    int tid = tidx();
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // [1] rotate the query
    rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
    __syncthreads();

    __shared__ float warp_min[WARPS_PER_BLOCK];
    __shared__ float warp_max[WARPS_PER_BLOCK];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // [2] reduce the rotated query range across the block
    for (size_t i = tid; i < padded_dim; i += blockDim.x) {
        float tmp = scratch.rotated_query[i];
        local_min = fminf(local_min, tmp);
        local_max = fmaxf(local_max, tmp);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
        local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
        }

        if (lane_id == 0) {
            scratch.low_val = local_min;
            scratch.high_val = local_max;
            const float query_span = scratch.high_val - scratch.low_val;
            scratch.width = query_span * kQueryLevelsInv;
        }
    }
    __syncthreads();

    // [3] quantize four coordinates at a time and tabulate their 16 subset sums
    const float inv_width = 1.0f / scratch.width;
    __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
    int32_t local_sum = 0;
    const int num_codebook = padded_dim >> 2;

    for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
        uint8_t q[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = (cb << 2) + j;
            const float scaled = ((scratch.rotated_query[idx] - scratch.low_val) * inv_width) + 0.5f;
            q[j] = static_cast<uint8_t>(lroundf(scaled));
            local_sum += q[j];
        }

        uint8_t *lut_chunk = scratch.lut + (cb << 4);
        lut_chunk[0] = 0;
        for (int mask = 1; mask < 16; ++mask) {
            const int lowbit = mask & -mask;
            const int pos = (lowbit == 8) ? 0 : ((lowbit == 4) ? 1 : ((lowbit == 2) ? 2 : 3));
            lut_chunk[mask] = static_cast<uint8_t>(lut_chunk[mask - lowbit] + q[pos]);
        }
    }

    // [4] reduce the quantized query sum used by the scan correction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
        if (lane_id == 0) {
            scratch.sum_q = local_sum;
        }
    }
}

/**
 * @brief Quantize an already transformed query and build its lookup table against a level codebook.
 *
 * Serves multi-bit RaBitQ, the TurboQuant MSE stage and, with CODEBITS 1, the TurboQuant sign
 * stage. A scan code packs QUANT_NIBBLE_DIMS / CODEBITS dimensions as pack_codes does. Entries
 * and sum_q carry a gain of CODEBITS, which the builder divides out of the table-sum factor.
 *
 * @tparam CODEBITS bits per dimension
 * @param scratch query factors whose rotated_query holds the transformed query, receives its range, the table and the query sum
 * @param levels normalized reconstruction levels, 2^CODEBITS entries spanning [0, 1], or nullptr for an even ladder
 * @param padded_dim padded dimension
 */
template <int CODEBITS>
static __device__ inline void lut_build(QueryFactors &scratch, const float *levels,
                                        int padded_dim) {
    constexpr float kQueryLevelsInv = 1.0f / static_cast<float>((1 << QG_BQUERY) - 1);
    constexpr int GROUP = QUANT_NIBBLE_DIMS / CODEBITS;
    constexpr int LMASK = (1 << CODEBITS) - 1;
    constexpr float kLevelStep = 1.0f / static_cast<float>((1 << CODEBITS) - 1);

    int tid = tidx();
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    __shared__ float warp_min[WARPS_PER_BLOCK];
    __shared__ float warp_max[WARPS_PER_BLOCK];
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // [1] reduce the transformed query range across the block
    for (size_t i = tid; i < padded_dim; i += blockDim.x) {
        float tmp = scratch.rotated_query[i];
        local_min = fminf(local_min, tmp);
        local_max = fmaxf(local_max, tmp);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
        local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
    }

    if (lane_id == 0) {
        warp_min[warp_id] = local_min;
        warp_max[warp_id] = local_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_min = (lane_id < WARPS_PER_BLOCK) ? warp_min[lane_id] : FLT_MAX;
        local_max = (lane_id < WARPS_PER_BLOCK) ? warp_max[lane_id] : -FLT_MAX;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_min = fminf(local_min, __shfl_down_sync(FULL_MASK, local_min, offset));
            local_max = fmaxf(local_max, __shfl_down_sync(FULL_MASK, local_max, offset));
        }

        if (lane_id == 0) {
            scratch.low_val = local_min;
            scratch.high_val = local_max;
            const float query_span = scratch.high_val - scratch.low_val;
            scratch.width = query_span * kQueryLevelsInv;
        }
    }
    __syncthreads();

    // [2] quantize the query and tabulate every 4-bit group code
    const float inv_width = 1.0f / scratch.width;
    __shared__ int32_t warp_sum[WARPS_PER_BLOCK];
    int32_t local_sum = 0;
    const int num_codebook = (padded_dim * CODEBITS) >> 2;

    for (int cb = tid; cb < num_codebook; cb += blockDim.x) {
        float q[GROUP];
#pragma unroll
        for (int j = 0; j < GROUP; ++j) {
            const int idx = cb * GROUP + j;
            const float scaled = ((scratch.rotated_query[idx] - scratch.low_val) * inv_width) + 0.5f;
            const int level = static_cast<int>(scaled);
            q[j] = static_cast<float>(level);
            local_sum += level;
        }

        uint8_t *lut_chunk = scratch.lut + (cb << 4);
        for (int code = 0; code < 16; ++code) {
            float acc = 0.0f;
#pragma unroll
            for (int j = 0; j < GROUP; ++j) {
                constexpr int base_shift = QUANT_NIBBLE_DIMS;
                const int shift = base_shift - (j + 1) * CODEBITS;
                const int slot = (code >> shift) & LMASK;
                acc += q[j] * (levels ? levels[slot] : (slot * kLevelStep));
            }
            lut_chunk[code] = static_cast<uint8_t>(lroundf(acc * static_cast<float>(CODEBITS)));
        }
    }

    // [3] reduce the gain-scaled query sum used by the scan correction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    if (lane_id == 0) {
        warp_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane_id < WARPS_PER_BLOCK) ? warp_sum[lane_id] : 0;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
        if (lane_id == 0) {
            scratch.sum_q = local_sum * CODEBITS;
        }
    }
}

/**
 * @brief Prepare a query for multi-bit RaBitQ or the TurboQuant MSE stage: rotate, then build the level table.
 * @tparam CODEBITS bits per dimension
 * @param query_raw raw query in shared memory
 * @param scratch query factors, receives the rotated query, its range, the table and the query sum
 * @param signs_ptr sign vector of the index rotation
 * @param levels normalized levels spanning [0, 1], or nullptr for an even ladder as multi-bit RaBitQ uses
 * @param dim raw dimension
 * @param padded_dim padded dimension
 */
template <int CODEBITS>
static __device__ inline void turboq_prepare_lut_gpu(
    const float *query_raw, QueryFactors &scratch, const float *signs_ptr,
    const float *levels, int dim, int padded_dim) {
    rotate_vector_gpu(query_raw, scratch.rotated_query, signs_ptr, dim, padded_dim);
    __syncthreads();
    lut_build<CODEBITS>(scratch, levels, padded_dim);
}

/*-------------------------------------------- estimation scan --------------------------------------------*/
/**
 * @brief Read one neighbor's 4-bit code for one scan group from its packed code tile.
 *
 * Two scan groups share a byte, the even one in the low nibble. Below 32 layout lanes a tile
 * interleaves WARP_SIZE / LUT_LAYOUT_LANES neighbors byte by byte.
 *
 * @tparam LUT_LAYOUT_LANES code layout the index was packed with
 * @param code_base tile holding the neighbor
 * @param codebook_idx scan group index
 * @param neighbor_in_tile neighbor slot within an interleaved tile
 * @return the 4-bit code
 */
template <int LUT_LAYOUT_LANES = GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES>
static __device__ __forceinline__ uint8_t fastscan_decode_code_for_neighbor_seq_lut_gpu(
    const uint8_t *code_base, int codebook_idx, int neighbor_in_tile = 0) {
    static_assert(LUT_LAYOUT_LANES == 2 || LUT_LAYOUT_LANES == 4 || LUT_LAYOUT_LANES == 8 || LUT_LAYOUT_LANES == 16 || LUT_LAYOUT_LANES == 32,
                  "GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES must be 2, 4, 8, 16, or 32");
    const int pair_idx = codebook_idx >> 1;
    uint8_t packed;
    if constexpr (LUT_LAYOUT_LANES == 32) {
        packed = code_base[pair_idx];
    } else {
        constexpr int neighbors_per_tile = WARP_SIZE / LUT_LAYOUT_LANES;
        packed = code_base[pair_idx * neighbors_per_tile + neighbor_in_tile];
    }
    return (codebook_idx & 1) ? static_cast<uint8_t>(packed >> 4)
                              : static_cast<uint8_t>(packed & 0x0f);
}

/**
 * @brief Address the packed code tile holding one neighbor under the scan layout.
 * @param code_block base of the parent's packed codes
 * @param neighbor_idx neighbor position within the parent
 * @param bytes_per_neighbor packed code footprint of one neighbor
 * @param in_tile receives the neighbor's slot inside the returned tile
 * @return pointer to the tile
 */
static __device__ __forceinline__ const uint8_t *lut_tile(
    const uint8_t *code_block, int neighbor_idx, int bytes_per_neighbor, int *in_tile) {
    constexpr int lut_layout_neighbors = WARP_SIZE / GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES;
    if constexpr (GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES == 32) {
        *in_tile = 0;
        return code_block + neighbor_idx * bytes_per_neighbor;
    } else {
        *in_tile = neighbor_idx & (lut_layout_neighbors - 1);
        const int tile_idx = neighbor_idx / lut_layout_neighbors;
        return code_block + tile_idx * lut_layout_neighbors * bytes_per_neighbor;
    }
}

/**
 * @brief Load one lookup-table byte through its 32-bit shared-memory address.
 *
 * The table pointer is read from a shared QueryFactors, so the compiler cannot prove it
 * points to shared memory and would emit a generic load with a 64-bit address.
 *
 * @param addr shared-window address of the byte
 * @return the byte, zero-extended
 */
static __device__ __forceinline__ uint32_t lutbyte(uint32_t addr) {
    uint32_t value;
    asm volatile("ld.shared.u8 %0, [%1];" : "=r"(value) : "r"(addr));
    return value;
}

/**
 * @brief Sum one neighbor's lookup entries, reading its packed codes 16 bytes at a time.
 *
 * One 16-byte load holds 32 scan groups and replaces 16 byte loads. Only one chunk is live
 * at a time, which bounds the registers it adds.
 *
 * @param lutaddr shared-window address of the query lookup table, 16 entries per scan group
 * @param tile 16-byte aligned packed codes of the neighbor
 * @param num_codebook scan groups of the neighbor, a multiple of 32
 * @return unsigned sum of the looked-up entries
 */
static __device__ __forceinline__ uint32_t widesum(uint32_t lutaddr, const uint8_t *tile, int num_codebook) {
    const uint4 *chunks = reinterpret_cast<const uint4 *>(tile);
    uint32_t sum = 0;
#pragma unroll 1
    for (int base = 0; base < num_codebook; base += 32) {
        // [1] one load covers scan groups base .. base + 31
        const uint4 chunk = __ldg(chunks + (base >> 5));
        const uint32_t words[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
        const uint32_t row = lutaddr + (static_cast<uint32_t>(base) << 4);

        // [2] byte j holds scan group 2j in its low nibble and 2j + 1 in its high nibble
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint32_t packed = (words[j >> 2] >> ((j & 3) << 3)) & 0xffu;
            sum += lutbyte(row + (j << 5) + (packed & 0x0fu));
            sum += lutbyte(row + (j << 5) + 16 + (packed >> 4));
        }
    }
    return sum;
}

/**
 * @brief Sum one neighbor's lookup entries across a lane group and apply the sign correction, shared by every estimate.
 *
 * Lanes within a group split the scan groups and reduce; only group lane 0 holds the result.
 * A lone lane on contiguous codes whose tile and bytes_per_neighbor are 16-byte aligned reads
 * them through widesum; that alignment makes num_codebook a multiple of 32.
 *
 * @tparam LANES lanes cooperating on one neighbor
 * @tparam CODEBITS bits per dimension of this code block
 * @param qf query factors carrying the shared-memory table and the query sum
 * @param code_block base of the parent's packed codes
 * @param neighbor_idx neighbor position within the parent
 * @param padded_dim padded dimension
 * @param bytes_per_neighbor packed code footprint of one neighbor
 * @return 2 * entry sum - qf.sum_q on group lane 0, zero on the other lanes
 */
template <int LANES, int CODEBITS>
static __device__ __forceinline__ float lut_reduce(
    const QueryFactors &qf, const uint8_t *code_block, int neighbor_idx,
    int padded_dim, int bytes_per_neighbor) {
    const int lane_id = tidx() & (WARP_SIZE - 1);
    const int group_lane = (LANES == 1) ? 0 : (lane_id & (LANES - 1));
    const int num_codebook = (padded_dim * CODEBITS) >> 2;

    // [1] locate the neighbor's tile and pick the wide path
    int in_tile = 0;
    const uint8_t *tile = lut_tile(code_block, neighbor_idx, bytes_per_neighbor, &in_tile);

    bool wide = false;
    if constexpr (LANES == 1 && GPU_RABITQ_FASTSCAN_SEQ_LUT_LAYOUT_LANES == 32) {
        wide = ((reinterpret_cast<uintptr_t>(tile) | static_cast<uintptr_t>(bytes_per_neighbor)) & 15u) == 0;
    }

    // [2] sum the looked-up entries
    const uint32_t lutaddr = static_cast<uint32_t>(__cvta_generic_to_shared(qf.lut));
    uint32_t raw_sum = 0;
    if (wide) {
        raw_sum = widesum(lutaddr, tile, num_codebook);
    } else {
        for (int cb = group_lane; cb < num_codebook; cb += LANES) {
            const uint8_t code = fastscan_decode_code_for_neighbor_seq_lut_gpu(tile, cb, in_tile);
            raw_sum += lutbyte(lutaddr + (static_cast<uint32_t>(cb) << 4) + code);
        }
    }

    // [3] reduce across the lane group
    if constexpr (LANES > 1) {
        unsigned group_mask = FULL_MASK;
        if constexpr (LANES != WARP_SIZE) {
            constexpr unsigned group_bits = (1u << LANES) - 1u;
            group_mask = group_bits << (lane_id - group_lane);
        }
        for (int offset = LANES / 2; offset > 0; offset >>= 1) {
            raw_sum += __shfl_down_sync(group_mask, raw_sum, offset, LANES);
        }
        if (group_lane != 0) return 0.0f;
    }

    // [4] sign correction
    return static_cast<float>((static_cast<int32_t>(raw_sum) << 1) - qf.sum_q);
}

/**
 * @brief Estimate one neighbor's distance from its RaBitQ code, one lane group per neighbor.
 *
 * It adds to the parent's exact distance triple_x, the table sum scaled by factor_dq and the
 * query step, and the query offset scaled by factor_vq. A padding slot, marked by
 * triple_x == FLT_MAX, returns the parent distance on every lane.
 *
 * @tparam LANES_PER_NEIGHBOR lanes cooperating on one neighbor, 2 to 32
 * @tparam CODEBITS bits per dimension
 * @param qf prepared query factors
 * @param packed_codes_block base of the parent's packed codes
 * @param neighbor_idx neighbor position within the parent
 * @param triple_x points at the neighbor's constant factor
 * @param factor_dq points at the neighbor's table-sum coefficient
 * @param factor_vq points at the neighbor's query-offset coefficient
 * @param exact_dist exact distance of the parent
 * @param padded_dim padded dimension
 * @param bytes_per_neighbor packed code footprint of one neighbor
 * @return the estimated distance on group lane 0, zero on the other lanes, exact_dist on every lane for a padding slot
 */
template <int LANES_PER_NEIGHBOR, int CODEBITS = 1>
static __device__ inline DISTANCE_T scan_one_neighbor_lanes_gpu(
    const QueryFactors &qf, const uint8_t *packed_codes_block, int neighbor_idx,
    const float *triple_x, const float *factor_dq, const float *factor_vq,
    float exact_dist, int padded_dim, int bytes_per_neighbor) {
    static_assert(LANES_PER_NEIGHBOR == 2 || LANES_PER_NEIGHBOR == 4 ||
                      LANES_PER_NEIGHBOR == 8 || LANES_PER_NEIGHBOR == 16 ||
                      LANES_PER_NEIGHBOR == 32,
                  "LANES_PER_NEIGHBOR must be one of 2, 4, 8, 16, or 32");
    const float triple_x_value = triple_x[0];
    if (triple_x_value == FLT_MAX)
        return exact_dist;

    const float result_float = lut_reduce<LANES_PER_NEIGHBOR, CODEBITS>(
        qf, packed_codes_block, neighbor_idx, padded_dim, bytes_per_neighbor);

    const int group_lane = tidx() & (LANES_PER_NEIGHBOR - 1);
    if (group_lane != 0)
        return 0.0f;

    DISTANCE_T est_dist = triple_x_value + exact_dist;
    est_dist += factor_dq[0] * qf.width * result_float;
    est_dist += factor_vq[0] * qf.low_val;
    return est_dist;
}

/**
 * @brief Estimate one neighbor's distance from its two-stage TurboQuant code, the counterpart of scan_one_neighbor_lanes_gpu.
 *
 * Combines an MSE-stage lookup against the rotated query with a QJL sign lookup against the
 * sketched query. Factor slots run: shared constant, then a table-sum and an offset coefficient
 * for each stage, as encode_tbq writes them. A padding slot is marked by a shared constant of FLT_MAX.
 *
 * @tparam LANES lanes cooperating on one neighbor
 * @tparam CODEBITS bits per dimension of the MSE stage
 * @param qa query factors for the rotated query
 * @param qb query factors for the sketched query
 * @param code_block base of the parent's MSE codes
 * @param sign_block base of the parent's QJL signs
 * @param neighbor_idx neighbor position within the parent
 * @param factors base of the parent's factor block
 * @param max_degree stride between factor slots
 * @param exact_dist exact distance of the parent
 * @param padded_dim padded dimension
 * @param code_bytes MSE code footprint of one neighbor
 * @param sign_bytes QJL sign footprint of one neighbor
 * @return the estimated distance on group lane 0, zero on the other lanes; for a padding slot FLT_MAX when LANES is 1, else zero
 */
template <int LANES, int CODEBITS>
static __device__ inline DISTANCE_T turbop_scan(
    const QueryFactors &qa, const QueryFactors &qb, const uint8_t *code_block,
    const uint8_t *sign_block, int neighbor_idx, const float *factors, int max_degree,
    float exact_dist, int padded_dim, int code_bytes, int sign_bytes) {
    const float shared = factors[neighbor_idx];
    if (shared == FLT_MAX)
        return (LANES > 1) ? 0.0f : FLT_MAX;

    const float res_a = lut_reduce<LANES, CODEBITS>(qa, code_block, neighbor_idx, padded_dim, code_bytes);
    const float res_b = lut_reduce<LANES, 1>(qb, sign_block, neighbor_idx, padded_dim, sign_bytes);

    if constexpr (LANES > 1) {
        const int group_lane = tidx() & (LANES - 1);
        if (group_lane != 0) return 0.0f;
    }

    DISTANCE_T est_dist = shared + exact_dist;
    est_dist += factors[max_degree + neighbor_idx] * qa.width * res_a;
    est_dist += factors[2 * max_degree + neighbor_idx] * qa.low_val;
    est_dist += factors[3 * max_degree + neighbor_idx] * qb.width * res_b;
    est_dist += factors[4 * max_degree + neighbor_idx] * qb.low_val;
    return est_dist;
}
