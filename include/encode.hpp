#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "pack.hpp"
#include "quant.hpp"
#include "sketch.hpp"

/** @brief Shape of the index being emitted, shared by tools/buildindex.cc and the encoders. */
struct BuildSpec {
    size_t numnode;  // node count
    size_t rawdim;   // raw vector dimension
    size_t paddim;   // rawdim rounded up to a power of two
    int degree;      // neighbors per node
    int codebit;     // total bits per dimension
    QuantType qtype; // quantizer family
};

/**
 * @brief Encode one parent's neighbor list with the RaBitQ signed uniform grid, for tools/buildindex.cc.
 *
 * For neighbor j it writes the packed code at codeoff + j * quant_words(paddim, codebit) and three
 * factors at facoff + j, facoff + degree + j, and facoff + 2 * degree + j, which
 * scan_one_neighbor_lanes_gpu reads as triple_x, factor_dq, and factor_vq.
 *
 * @param spec index shape
 * @param rotu rotated parent vector
 * @param rotv rotated neighbor vectors, degree by paddim
 * @param codes scratch of paddim per-dimension codes
 * @param resid scratch of paddim floats, left holding the last unit residual
 * @param row destination row, positioned at the parent
 * @param codeoff packed code offset within the row
 * @param facoff factor block offset within the row
 */
static inline void encode_rbq(const BuildSpec& spec, const float* rotu, const float* rotv, uint8_t* codes,
                              float* resid, float* row, size_t codeoff, size_t facoff) {
    const int bits = spec.codebit;
    const size_t words = quant_words(spec.paddim, bits);
    const float fhtfix = 1.0f / std::sqrt(static_cast<float>(spec.paddim));
    const float span = static_cast<float>((1 << bits) - 1);
    const float gain = static_cast<float>(bits);
    const int deg = spec.degree;

    for (int j = 0; j < deg; ++j) {
        const float* vec = rotv + static_cast<size_t>(j) * spec.paddim;

        // [1] unit residual against the parent
        float normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            resid[k] = vec[k] - rotu[k];
            normsq += resid[k] * resid[k];
        }
        const float xnorm = std::sqrt(normsq);
        const float inv = (xnorm > 0.0f) ? (1.0f / xnorm) : 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) resid[k] *= inv;

        // [2] grid code and its direction, scanned as y = 2c / span - 1
        const float cos = gridcode(spec.paddim, bits, resid, codes);
        const float facx0 = (xnorm > 0.0f) ? cos : 1.0f;
        float ysum = 0.0f, ysq = 0.0f, ipu = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float y = 2.0f * static_cast<float>(codes[k]) / span - 1.0f;
            ysum += y;
            ysq += y * y;
            ipu += rotu[k] * y;
        }
        const float ynorm = std::sqrt(ysq);

        // [3] factors of |q - u|^2 + |r|^2 - 2 |r| <y, q - u> / <y, o>
        const float xx0 = xnorm / facx0;
        row[facoff + j] = xnorm * xnorm + 2.0f * xx0 * ipu / ynorm;
        row[facoff + deg + j] = -2.0f * xx0 * fhtfix / (ynorm * gain);
        row[facoff + 2 * deg + j] = -2.0f * xx0 * fhtfix * ysum / ynorm;

        // [4] pack the code into the row
        pack_codes(spec.paddim, bits, codes,
                   reinterpret_cast<uint8_t*>(row + codeoff + static_cast<size_t>(j) * words));
    }
}

/**
 * @brief Encode one parent's neighbor list with TurboQuant, an MSE stage then a QJL sign stage, for tools/buildindex.cc.
 *
 * Neighbor j gets its MSE code at codeoff, sign bits at signoff, and factor slot s at
 * facoff + s * degree + j, in turbop_scan's order. Slots 1 to 4 carry 1/sqrt(paddim) since the
 * search scans unnormalized rotated queries.
 *
 * @param spec index shape
 * @param sketch QJL sketch over the padded dimension
 * @param kappa QJL dequantization scale, FastfoodSketch::get_scale
 * @param rotu rotated parent vector
 * @param sku sketched rotated parent vector
 * @param rotv rotated neighbor vectors, degree by paddim
 * @param levels ascending reconstruction levels of the unit-norm residual, 2^(codebit - 1) entries
 * @param codes scratch of paddim per-dimension codes
 * @param signs scratch of paddim sign bits
 * @param resid scratch of paddim floats, left holding the last MSE stage remainder
 * @param proj scratch of paddim floats, receives the sketched remainder
 * @param row destination row, positioned at the parent
 * @param codeoff packed code offset within the row
 * @param signoff packed sign offset within the row
 * @param facoff factor block offset within the row
 */
static inline void encode_tbq(const BuildSpec& spec, const FastfoodSketch& sketch, float kappa,
                              const float* rotu, const float* sku, const float* rotv,
                              const std::vector<float>& levels, uint8_t* codes, uint8_t* signs,
                              float* resid, float* proj, float* row,
                              size_t codeoff, size_t signoff, size_t facoff) {
    const int stage = quant_stage(spec.codebit);
    const size_t words = quant_words(spec.paddim, stage);
    const size_t swords = quant_words(spec.paddim, 1);
    const float crange = levels.back() - levels.front();
    const float fhtfix = 1.0f / std::sqrt(static_cast<float>(spec.paddim));
    const float gain = static_cast<float>(stage);
    const int deg = spec.degree;

    for (int j = 0; j < deg; ++j) {
        const float* vec = rotv + static_cast<size_t>(j) * spec.paddim;

        // [1] residual against the parent and its norm
        float normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            resid[k] = vec[k] - rotu[k];
            normsq += resid[k] * resid[k];
        }
        const float xnorm = std::sqrt(normsq);
        const float inv = (xnorm > 0.0f) ? (1.0f / xnorm) : 0.0f;

        // [2] MSE stage over the unit residual
        float sumhat = 0.0f, ipu = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            codes[k] = quantize_level(levels, resid[k] * inv);
            const float hat = levels[codes[k]];
            sumhat += hat;
            ipu += rotu[k] * hat;
            resid[k] -= xnorm * hat;
        }

        // [3] QJL sign stage over what the MSE stage left behind
        float mnorm = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) mnorm += resid[k] * resid[k];
        mnorm = std::sqrt(mnorm);
        sketch.apply(resid, proj);

        float sumsig = 0.0f, ipsk = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float s = (proj[k] > 0.0f) ? 1.0f : -1.0f;
            signs[k] = (proj[k] > 0.0f) ? 1 : 0;
            sumsig += s;
            ipsk += sku[k] * s;
        }

        // [4] five factors, the query-side coefficients scaled by 1/sqrt(paddim)
        const float qscale = kappa * mnorm;
        row[facoff + j] = xnorm * xnorm + 2.0f * xnorm * ipu + 2.0f * qscale * ipsk;
        row[facoff + deg + j] = -xnorm * crange * fhtfix / gain;
        row[facoff + 2 * deg + j] = -2.0f * xnorm * sumhat * fhtfix;
        row[facoff + 3 * deg + j] = -2.0f * qscale * fhtfix;
        row[facoff + 4 * deg + j] = -2.0f * qscale * sumsig * fhtfix;

        // [5] pack the MSE code and the sign bits into the row
        pack_codes(spec.paddim, stage, codes,
                   reinterpret_cast<uint8_t*>(row + codeoff + static_cast<size_t>(j) * words));
        pack_codes(spec.paddim, 1, signs,
                   reinterpret_cast<uint8_t*>(row + signoff + static_cast<size_t>(j) * swords));
    }
}
