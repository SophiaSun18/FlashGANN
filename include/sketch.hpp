#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * @brief Fastfood approximation of a dense Gaussian sketch.
 *
 * A square Gaussian factors as U * Sigma * V^T with Haar orthogonal U and V and singular
 * values following the Marchenko-Pastur quarter circle. Replacing both orthogonal factors
 * with a random sign flip followed by a Walsh-Hadamard transform keeps the spectrum, and
 * with it the near-independent rows that the sign estimator needs, at O(d log d) instead
 * of O(d^2). The overall scale is left uncalibrated and absorbed downstream.
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

private:
    /**
     * @brief Sample the Marchenko-Pastur quarter circle by rejection.
     *
     * The singular values of a square Gaussian are distributed as sqrt(d) * x with density
     * proportional to sqrt(4 - x^2) on [0, 2], so sampling that law is equivalent to
     * extracting the spectrum without forming or decomposing a matrix.
     *
     * @param seed generator seed
     */
    void fill_spectrum(uint32_t seed) {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> ux(0.0f, 2.0f);
        std::uniform_real_distribution<float> uy(0.0f, 1.0f);
        for (size_t i = 0; i < paddim_; ++i) {
            for (;;) {
                const float x = ux(gen);
                if (uy(gen) * 2.0f <= std::sqrt(4.0f - x * x)) {
                    spectr_[i] = x;
                    break;
                }
            }
        }
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
