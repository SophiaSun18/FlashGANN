#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "include/encode.hpp"

inline constexpr int QUERY_LEVELS = 63;
inline constexpr long SAMPLE_BEGIN = 1L << 16;
inline constexpr long SAMPLE_LIMIT = 1L << 28;
inline constexpr double STEADY_TOL = 2e-5;
inline constexpr double BIAS_TOL = 2e-4;

/** @brief Fixed vectors of one unbiasedness case, all in the normalized rotated space. */
struct TurboCase {
    size_t paddim;
    int codebit;
    std::vector<float> levels;
    std::vector<float> rotu;
    std::vector<float> rotv;
    std::vector<float> query;
};

/**
 * @brief Read one per-dimension code back from the nibble layout of pack_codes.
 * @param packed packed codes of one neighbor
 * @param bits bits per dimension
 * @param dim dimension index
 * @return the code of that dimension
 */
static int unpack(const uint8_t* packed, int bits, size_t dim) {
    const size_t group = static_cast<size_t>(quant_group(bits));
    const size_t gi = dim / group;
    const int nibble = (gi & 1) ? (packed[gi >> 1] >> 4) : (packed[gi >> 1] & 0x0f);
    const int shift = QUANT_NIBBLE_DIMS - static_cast<int>(dim % group + 1) * bits;
    return (nibble >> shift) & ((1 << bits) - 1);
}

/**
 * @brief Replay the scan correction of lut_reduce on an unquantized query.
 * @param vec query in the scanned space
 * @param packed packed codes of one neighbor
 * @param bits bits per dimension
 * @param norm normalized level table, nullptr for the sign ladder
 * @param dim padded dimension
 * @param width receives the query level width
 * @param low receives the query low value
 * @return the gain-scaled sum 2 * sum(q * level) - sum(q)
 */
static double replay(const std::vector<double>& vec, const uint8_t* packed, int bits,
                     const std::vector<float>* norm, size_t dim, double* width, double* low) { // SHAME(MANYARG)
    double lo = vec[0], hi = vec[0];
    for (double x : vec) lo = std::min(lo, x), hi = std::max(hi, x);
    *low = lo;
    *width = (hi - lo) / QUERY_LEVELS;

    double acc = 0.0, sum = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        const double q = (vec[k] - lo) / *width;
        const int code = unpack(packed, bits, k);
        acc += q * (norm ? (*norm)[code] : code);
        sum += q;
    }
    return static_cast<double>(bits) * (2.0 * acc - sum);
}

/**
 * @brief Encode the case under one sketch and evaluate the turbop_scan distance.
 * @param tc the case
 * @param seed sketch seed
 * @param resid receives the MSE stage remainder
 * @return estimated squared distance from the query to the neighbor
 */
static double estimate(const TurboCase& tc, uint32_t seed, std::vector<float>& resid) { // SHAME(TALLFUNC)
    const size_t dim = tc.paddim;
    const int stage = quant_stage(tc.codebit);
    const BuildSpec spec{1, dim, dim, 1, tc.codebit, QUANT_TBQ};
    const size_t signoff = quant_words(dim, stage);
    const size_t facoff = signoff + quant_words(dim, 1);

    // [1] encode through the builder path into one row
    FastfoodSketch sketch(dim, seed);
    std::vector<float> sku(dim), proj(dim), row(facoff + 5, 0.0f);
    std::vector<uint8_t> codes(dim), signs(dim);
    sketch.apply(tc.rotu.data(), sku.data());
    encode_tbq(spec, sketch, FastfoodSketch::get_scale(dim), tc.rotu.data(), sku.data(),
               tc.rotv.data(), tc.levels, codes.data(), signs.data(), resid.data(), proj.data(),
               row.data(), 0, signoff, facoff);
    const float* fac = row.data() + facoff;

    // [2] the search works on the unnormalized rotated query and its sketch
    const double lift = std::sqrt(static_cast<double>(dim));
    std::vector<float> qraw(dim), qsk(dim);
    for (size_t k = 0; k < dim; ++k) qraw[k] = static_cast<float>(tc.query[k] * lift);
    sketch.apply(qraw.data(), qsk.data());
    std::vector<double> qa(qraw.begin(), qraw.end()), qb(qsk.begin(), qsk.end());

    std::vector<float> norm(tc.levels.size());
    for (size_t k = 0; k < norm.size(); ++k)
        norm[k] = (tc.levels[k] - tc.levels.front()) / (tc.levels.back() - tc.levels.front());

    double wa = 0.0, la = 0.0, wb = 0.0, lb = 0.0;
    const double resa = replay(qa, reinterpret_cast<uint8_t*>(row.data()), stage, &norm, dim, &wa, &la);
    const double resb = replay(qb, reinterpret_cast<uint8_t*>(row.data() + signoff), 1, nullptr, dim, &wb, &lb);

    double exact = 0.0;
    for (size_t k = 0; k < dim; ++k) exact += (tc.query[k] - tc.rotu[k]) * (tc.query[k] - tc.rotu[k]);
    return fac[0] + exact + fac[1] * wa * resa + fac[2] * la + fac[3] * wb * resb + fac[4] * lb;
}

/**
 * @brief Build a case whose query leans on the MSE remainder, so the QJL term dominates.
 * @param dim padded dimension
 * @param codebit total bits per dimension
 * @param seed data seed
 * @param scale receives 2 |<u - q, remainder>|, the magnitude the QJL stage estimates
 * @return the case
 */
static TurboCase makecase(size_t dim, int codebit, uint32_t seed, double* scale) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    TurboCase tc{dim, codebit, {}, std::vector<float>(dim), std::vector<float>(dim), std::vector<float>(dim)};

    const int count = 1 << quant_stage(codebit);
    for (int i = 0; i < count; ++i)
        tc.levels.push_back((2.0f * i + 1.0f - count) / count * 2.0f / std::sqrt(static_cast<float>(dim)));
    for (size_t k = 0; k < dim; ++k) tc.rotu[k] = gauss(gen), tc.rotv[k] = tc.rotu[k] + gauss(gen);

    std::vector<float> resid(dim);
    estimate(tc, 1u, resid);
    double ip = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        tc.query[k] = tc.rotu[k] - 3.0f * resid[k] + 0.3f * gauss(gen);
        ip += (tc.rotu[k] - tc.query[k]) * resid[k];
    }
    *scale = 2.0 * std::fabs(ip);
    return tc;
}

/**
 * @brief Average the estimate over doubling sketch counts until the mean stops moving.
 * @param dim padded dimension
 * @param codebit total bits per dimension
 * @return 0 when the steady mean matches the exact distance
 */
static int runcase(size_t dim, int codebit) {
    double scale = 0.0;
    const TurboCase tc = makecase(dim, codebit, static_cast<uint32_t>(dim * 31 + codebit), &scale);
    double exact = 0.0;
    for (size_t k = 0; k < dim; ++k) exact += (tc.query[k] - tc.rotv[k]) * (tc.query[k] - tc.rotv[k]);

    double total = 0.0, last = 0.0;
    long done = 0;
    int steady = 0;
    for (long target = SAMPLE_BEGIN; target <= SAMPLE_LIMIT && steady < 2; target <<= 1) {
#pragma omp parallel for reduction(+ : total) schedule(static)
        for (long t = done; t < target; ++t) {
            thread_local std::vector<float> resid;
            resid.resize(dim);
            total += estimate(tc, static_cast<uint32_t>(0x9e3779b9u * static_cast<uint32_t>(t) + 101u), resid);
        }
        const double mean = total / static_cast<double>(target);
        steady = (done > 0 && std::fabs(mean - last) < STEADY_TOL * scale) ? steady + 1 : 0;
        last = mean;
        done = target;
    }

    const double bias = (last - exact) / scale;
    const bool pass = steady >= 2 && std::fabs(bias) < BIAS_TOL;
    printf("dim=%zu bits=%d samples=%ld exact=%.6f mean=%.6f rel_bias=%+.2e steady=%d %s\n",
           dim, codebit, done, exact, last, bias, steady, pass ? "ok" : "FAIL");
    return pass ? 0 : 1;
}

int main() {
    int failed = 0;
    for (size_t dim : {32, 64, 128, 256})
        for (int codebit : {2, 3, 5}) failed += runcase(dim, codebit);
    return failed;
}
