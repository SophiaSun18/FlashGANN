#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * @brief Fastfood sketch S = H flipu spectr H flipv with normalized Walsh-Hadamard H, the QJL projection of encode_tbq.
 *
 * flipu * spectr is an i.i.d. N(0, 1) diagonal, so each coordinate pair of S y and S r is
 * bivariate Gaussian with covariance <y, r> / d, as for a dense N(0, 1/d) matrix; get_scale then
 * makes the QJL estimator unbiased.
 */
class FastfoodSketch {
public:
    /**
     * @brief Draw both sign vectors and the spectrum over dim rounded up to a power of two.
     * @param dim input dimension; tools/buildindex.cc passes the padded dimension
     * @param seed generator seed of the sign vectors; the spectrum uses seed + 7919
     */
    FastfoodSketch(size_t dim, uint32_t seed) : rawdim_(dim) {
        paddim_ = 1;
        while (paddim_ < rawdim_) paddim_ <<= 1;
        sflipv_.resize(paddim_);
        sflipu_.resize(paddim_);
        spectr_.resize(paddim_);

        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        for (size_t i = 0; i < paddim_; ++i) {
            sflipv_[i] = coin(gen) ? 1.0f : -1.0f;
            sflipu_[i] = coin(gen) ? 1.0f : -1.0f;
        }
        fill_spectrum(seed + 7919u);
    }

    /**
     * @brief Rebuild a sketch from the layout store writes.
     * @param dim input dimension
     * @param packed flipv, spectrum, then flipu, paddim floats each
     */
    FastfoodSketch(size_t dim, const float* packed) : rawdim_(dim) {
        paddim_ = 1;
        while (paddim_ < rawdim_) paddim_ <<= 1;
        sflipv_.assign(packed, packed + paddim_);
        spectr_.assign(packed + paddim_, packed + 2 * paddim_);
        sflipu_.assign(packed + 2 * paddim_, packed + 3 * paddim_);
    }

    /**
     * @brief Apply the sketch to one vector, the host counterpart of sketch_apply.
     * @param src input vector of rawdim floats
     * @param dst destination buffer of paddim floats
     */
    void apply(const float* src, float* dst) const {
        // [1] first sign flip, zero padding, and first transform
        for (size_t i = 0; i < rawdim_; ++i) dst[i] = src[i] * sflipv_[i];
        for (size_t i = rawdim_; i < paddim_; ++i) dst[i] = 0.0f;
        transform(dst);

        // [2] spectrum, second sign flip, and second transform, each transform scaled by 1/sqrt(paddim)
        const float scale = 1.0f / std::sqrt(static_cast<float>(paddim_));
        for (size_t i = 0; i < paddim_; ++i) dst[i] *= spectr_[i] * scale * sflipu_[i];
        transform(dst);
        for (size_t i = 0; i < paddim_; ++i) dst[i] *= scale;
    }

    /**
     * @brief Serialize the sketch as flipv, spectrum, then flipu, the layout sketch_apply reads from the index.
     * @param packed destination of get_words(paddim) floats
     */
    void store(float* packed) const {
        for (size_t i = 0; i < paddim_; ++i) packed[i] = sflipv_[i];
        for (size_t i = 0; i < paddim_; ++i) packed[paddim_ + i] = spectr_[i];
        for (size_t i = 0; i < paddim_; ++i) packed[2 * paddim_ + i] = sflipu_[i];
    }

    /** @brief Return the paddim signs, each 1 or -1, applied before the first transform. */
    inline const float* get_flipv() const { return sflipv_.data(); }

    /** @brief Return the paddim half-normal magnitudes applied between the transforms. */
    inline const float* get_spectr() const { return spectr_.data(); }

    /** @brief Return the paddim signs, each 1 or -1, applied with the spectrum before the second transform. */
    inline const float* get_flipu() const { return sflipu_.data(); }

    /** @brief Return the padded dimension, rawdim rounded up to a power of two. */
    inline size_t get_paddim() const { return paddim_; }

    /**
     * @brief Serialized footprint of a sketch, used to size the buffer store fills.
     * @param paddim padded dimension
     * @return floats written by store, 3 * paddim
     */
    static inline size_t get_words(size_t paddim) { return 3 * paddim; }

    /**
     * @brief QJL dequantization scale of this sketch.
     *
     * sqrt(pi/2) / d is the scale for N(0, 1) entries; the normalized Hadamard factors give
     * N(0, 1/d) entries, which contributes a fixed factor of sqrt(d).
     * tools/buildindex.cc passes it to encode_tbq as kappa and stores it in the index.
     *
     * @param paddim padded dimension
     * @return scale c with E[c * ||r|| * <S y, sign(S r)>] = <y, r>
     */
    static inline float get_scale(size_t paddim) {
        const double dim = static_cast<double>(paddim);
        return static_cast<float>(std::sqrt(M_PI / 2.0) / dim * std::sqrt(dim));
    }

private:
    /**
     * @brief Sample the diagonal magnitudes from the half-normal law, called by the seeded constructor.
     *
     * Paired with the independent flipu signs, the diagonal becomes exactly N(0, 1), which
     * the unbiasedness of get_scale requires; a non-Gaussian magnitude law leaves a bias.
     *
     * @param seed generator seed
     */
    void fill_spectrum(uint32_t seed) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> gauss(0.0f, 1.0f);
        for (size_t i = 0; i < paddim_; ++i) spectr_[i] = std::fabs(gauss(gen));
    }

    /**
     * @brief In-place unnormalized Walsh-Hadamard transform, the two transforms of apply.
     * @param buf buffer of paddim floats
     */
    void transform(float* buf) const {
        for (size_t len = 1; len < paddim_; len <<= 1) {
            for (size_t base = 0; base < paddim_; base += (len << 1)) {
                for (size_t j = 0; j < len; ++j) {
                    const float a = buf[base + j];
                    const float b = buf[base + j + len];
                    buf[base + j] = a + b;
                    buf[base + j + len] = a - b;
                }
            }
        }
    }

    size_t rawdim_;
    size_t paddim_;
    std::vector<float> sflipv_;
    std::vector<float> sflipu_;
    std::vector<float> spectr_;
};
