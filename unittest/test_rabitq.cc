#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "include/encode.hpp"

inline constexpr int QUERY_LEVELS = 63;
inline constexpr double COS_TOL = 1e-5;
inline constexpr double ALG_TOL = 1e-3;

/** @brief Fixed vectors of one estimator case, all in the normalized rotated space. */
struct RabitqCase {
    size_t paddim;
    int codebit;
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
 * @brief Replay the lut_build and lut_reduce correction on an unquantized query.
 * @param vec query in the scanned space
 * @param packed packed codes of one neighbor
 * @param bits bits per dimension
 * @param dim padded dimension
 * @param width receives the query level width
 * @param low receives the query low value
 * @return the gain-scaled sum bits * (2 * sum(q * c / span) - sum(q))
 */
static double replay(const std::vector<double>& vec, const uint8_t* packed, int bits, size_t dim,
                     double* width, double* low) { // SHAME(MANYARG)
    double lo = vec[0], hi = vec[0];
    for (double x : vec) lo = std::min(lo, x), hi = std::max(hi, x);
    *low = lo;
    *width = (hi - lo) / QUERY_LEVELS;

    const double span = static_cast<double>((1 << bits) - 1);
    double acc = 0.0, sum = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        const double q = (vec[k] - lo) / *width;
        acc += q * unpack(packed, bits, k) / span;
        sum += q;
    }
    return static_cast<double>(bits) * (2.0 * acc - sum);
}

/**
 * @brief Best cosine between a vector and any signed odd grid point, by exhaustive search.
 * @param unit unit vector
 * @param bits bits per dimension
 * @return the highest cosine over all 2^(bits * dim) codes
 */
static double bestcos(const std::vector<float>& unit, int bits) {
    const size_t dim = unit.size();
    const long levels = 1L << bits;
    long total = 1;
    for (size_t k = 0; k < dim; ++k) total *= levels;

    double best = -1.0;
    for (long id = 0; id < total; ++id) {
        double ip = 0.0, sq = 0.0;
        long rest = id;
        for (size_t k = 0; k < dim; ++k) {
            const double y = 2.0 * static_cast<double>(rest % levels) - static_cast<double>(levels - 1);
            rest /= levels;
            ip += unit[k] * y;
            sq += y * y;
        }
        best = std::max(best, ip / std::sqrt(sq));
    }
    return best;
}

/**
 * @brief Encode the case and evaluate the scan distance next to the direct ratio estimator.
 * @param tc the case
 * @param direct receives |q - u|^2 + |r|^2 - 2 |r| <y, q - u> / <y, o>
 * @return estimated squared distance from the query to the neighbor through the scan
 */
static double estimate(const RabitqCase& tc, double* direct) {
    const size_t dim = tc.paddim;
    const BuildSpec spec{1, dim, dim, 1, tc.codebit, QUANT_RBQ};
    const size_t facoff = quant_words(dim, tc.codebit);

    // [1] encode through the builder path into one row
    std::vector<float> row(facoff + 3, 0.0f), resid(dim);
    std::vector<uint8_t> codes(dim);
    encode_rbq(spec, tc.rotu.data(), tc.rotv.data(), codes.data(), resid.data(), row.data(), 0, facoff);
    const float* fac = row.data() + facoff;

    // [2] the scan works on the unnormalized rotated query
    const double lift = std::sqrt(static_cast<double>(dim));
    std::vector<double> qa(dim);
    for (size_t k = 0; k < dim; ++k) qa[k] = tc.query[k] * lift;
    double wa = 0.0, la = 0.0;
    const double res = replay(qa, reinterpret_cast<uint8_t*>(row.data()), tc.codebit, dim, &wa, &la);

    // [3] the direct estimator straight from the unpacked codes
    const double span = static_cast<double>((1 << tc.codebit) - 1);
    double exact = 0.0, rsq = 0.0, ipq = 0.0, ipo = 0.0;
    for (size_t k = 0; k < dim; ++k) {
        const double r = tc.rotv[k] - tc.rotu[k];
        const double y = 2.0 * unpack(reinterpret_cast<uint8_t*>(row.data()), tc.codebit, k) / span - 1.0;
        exact += (tc.query[k] - tc.rotu[k]) * (tc.query[k] - tc.rotu[k]);
        rsq += r * r;
        ipq += y * (tc.query[k] - tc.rotu[k]);
        ipo += y * r;
    }
    *direct = exact + rsq - 2.0 * rsq * ipq / ipo;
    return fac[0] + exact + fac[1] * wa * res + fac[2] * la;
}

/**
 * @brief Build a case with a Gaussian parent, neighbor offset, and query offset.
 * @param dim padded dimension
 * @param codebit bits per dimension
 * @param seed data seed
 * @return the case
 */
static RabitqCase makecase(size_t dim, int codebit, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> gauss(0.0f, 1.0f / std::sqrt(static_cast<float>(dim)));
    RabitqCase tc{dim, codebit, std::vector<float>(dim), std::vector<float>(dim), std::vector<float>(dim)};
    for (size_t k = 0; k < dim; ++k) {
        tc.rotu[k] = gauss(gen);
        tc.rotv[k] = tc.rotu[k] + gauss(gen);
        tc.query[k] = tc.rotu[k] + gauss(gen);
    }
    return tc;
}

/**
 * @brief Check gridcode against exhaustive search and against its own returned cosine.
 * @param dim dimension small enough to enumerate
 * @param bits bits per dimension
 * @return 0 when every trial reaches the exhaustive optimum
 */
static int optimal(size_t dim, int bits) {
    std::mt19937 gen(static_cast<uint32_t>(dim * 131 + bits));
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    int failed = 0;
    double worst = 0.0;
    for (int trial = 0; trial < 20; ++trial) {
        std::vector<float> unit(dim);
        double norm = 0.0;
        for (float& x : unit) x = gauss(gen), norm += x * x;
        for (float& x : unit) x = static_cast<float>(x / std::sqrt(norm));

        std::vector<uint8_t> codes(dim);
        const double got = gridcode(dim, bits, unit.data(), codes.data());
        double ip = 0.0, sq = 0.0;
        for (size_t k = 0; k < dim; ++k) {
            const double y = 2.0 * codes[k] - static_cast<double>((1 << bits) - 1);
            ip += unit[k] * y;
            sq += y * y;
        }
        const double gap = bestcos(unit, bits) - got;
        worst = std::max(worst, gap);
        if (gap > COS_TOL || std::fabs(ip / std::sqrt(sq) - got) > COS_TOL) ++failed;
    }
    printf("optimal dim=%zu bits=%d worst_gap=%+.2e %s\n", dim, bits, worst, failed ? "FAIL" : "ok");
    return failed ? 1 : 0;
}

/**
 * @brief Check that the scan factors reproduce the direct ratio estimator.
 * @param dim padded dimension
 * @param codebit bits per dimension
 * @return 0 when every seed agrees within tolerance
 */
static int runcase(size_t dim, int codebit) {
    int failed = 0;
    double worst = 0.0, err = 0.0;
    for (uint32_t seed = 0; seed < 32; ++seed) {
        const RabitqCase tc = makecase(dim, codebit, static_cast<uint32_t>(dim * 17 + codebit * 7) + seed);
        double direct = 0.0;
        const double est = estimate(tc, &direct);
        double exact = 0.0;
        for (size_t k = 0; k < dim; ++k) exact += (tc.query[k] - tc.rotv[k]) * (tc.query[k] - tc.rotv[k]);
        const double gap = std::fabs(est - direct) / std::max(1.0, std::fabs(direct));
        worst = std::max(worst, gap);
        err += std::fabs(direct - exact) / exact;
        if (gap > ALG_TOL) ++failed;
    }
    printf("scan dim=%zu bits=%d worst_gap=%.2e mean_rel_err=%.4f %s\n", dim, codebit, worst, err / 32.0,
           failed ? "FAIL" : "ok");
    return failed ? 1 : 0;
}

int main() {
    int failed = 0;
    failed += optimal(16, 1);
    failed += optimal(8, 2);
    failed += optimal(4, 4);
    for (size_t dim : {32, 64, 128, 256})
        for (int codebit : {1, 2, 4}) failed += runcase(dim, codebit);
    return failed;
}
