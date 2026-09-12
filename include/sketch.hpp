#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * @brief Fastfood sketch S = H flipu spectr H flipv with normalized Walsh-Hadamard H.
 *
 * flipu * spectr is an i.i.d. N(0, 1) diagonal, so every coordinate pair of S y and S r is
 * exactly bivariate Gaussian with covariance <y, r> / d, the law of a d by d matrix with
 * N(0, 1/d) entries. The QJL sign estimator is then unbiased with get_scale.
 */
class FastfoodSketch {
public:
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

    FastfoodSketch(size_t dim, const float* packed) : rawdim_(dim) {
        paddim_ = 1;
        while (paddim_ < rawdim_) paddim_ <<= 1;
        sflipv_.assign(packed, packed + paddim_);
        spectr_.assign(packed + paddim_, packed + 2 * paddim_);
        sflipu_.assign(packed + 2 * paddim_, packed + 3 * paddim_);
    }

    /**
     * @brief Apply the sketch to one raw vector.
     * @param src raw vector of rawdim floats
     * @param dst destination buffer of paddim floats
     */
    void apply(const float* src, float* dst) const {
        for (size_t i = 0; i < rawdim_; ++i) dst[i] = src[i] * sflipv_[i];
        for (size_t i = rawdim_; i < paddim_; ++i) dst[i] = 0.0f;
        transform(dst);

        const float scale = 1.0f / std::sqrt(static_cast<float>(paddim_));
        for (size_t i = 0; i < paddim_; ++i) dst[i] *= spectr_[i] * scale * sflipu_[i];
        transform(dst);
        for (size_t i = 0; i < paddim_; ++i) dst[i] *= scale;
    }

    /**
     * @brief Serialize the sketch as flipv, spectrum, then flipu.
     * @param packed destination of get_words(paddim) floats
     */
    void store(float* packed) const {
        for (size_t i = 0; i < paddim_; ++i) packed[i] = sflipv_[i];
        for (size_t i = 0; i < paddim_; ++i) packed[paddim_ + i] = spectr_[i];
        for (size_t i = 0; i < paddim_; ++i) packed[2 * paddim_ + i] = sflipu_[i];
    }

    inline const float* get_flipv() const { return sflipv_.data(); }
    inline const float* get_spectr() const { return spectr_.data(); }
    inline const float* get_flipu() const { return sflipu_.data(); }
    inline size_t get_paddim() const { return paddim_; }
    static inline size_t get_words(size_t paddim) { return 3 * paddim; }

    /**
     * @brief QJL dequantization scale of this sketch.
     *
     * TurboQuant's sqrt(pi/2) / d assumes N(0, 1) entries; the normalized Hadamard factors
     * give N(0, 1/d) entries, which contributes a fixed factor of sqrt(d).
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
     * @brief Sample the diagonal magnitudes from the half-normal law.
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
     * @brief In-place unnormalized Walsh-Hadamard transform.
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
