#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * @brief Orthonormal random rotation built from a sign flip and a fast Walsh-Hadamard transform, for tools/buildindex.cc.
 *
 * The sign vector written into the index is the one the search kernel reloads, so the host
 * transform here and rotate_vector_gpu must agree on padding and on sign placement. The kernel
 * omits the 1/sqrt(paddim) scale, which the emitted per-neighbor factors absorb.
 */
class FhtRotator {
public:
    /**
     * @brief Draw the sign vector over dim rounded up to a power of two.
     * @param dim raw dimension
     * @param seed generator seed of the sign vector
     */
    FhtRotator(size_t dim, uint32_t seed) : rawdim_(dim) {
        paddim_ = 1;
        while (paddim_ < rawdim_) paddim_ <<= 1;
        sflips_.resize(paddim_);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<int> coin(0, 1);
        for (size_t i = 0; i < paddim_; ++i) {
            sflips_[i] = coin(gen) ? 1.0f : -1.0f;
        }
    }

    /**
     * @brief Rotate one raw vector into padded orthonormal space.
     * @param src raw vector of rawdim floats
     * @param dst destination buffer of paddim floats
     */
    void rotate(const float* src, float* dst) const {
        // [1] sign flip and zero padding
        for (size_t i = 0; i < rawdim_; ++i) dst[i] = src[i] * sflips_[i];
        for (size_t i = rawdim_; i < paddim_; ++i) dst[i] = 0.0f;

        // [2] in-place fast Walsh-Hadamard transform, unnormalized
        for (size_t len = 1; len < paddim_; len <<= 1) {
            for (size_t base = 0; base < paddim_; base += (len << 1)) {
                for (size_t j = 0; j < len; ++j) {
                    const float u = dst[base + j];
                    const float v = dst[base + j + len];
                    dst[base + j] = u + v;
                    dst[base + j + len] = u - v;
                }
            }
        }

        // [3] normalize by 1/sqrt(paddim)
        const float scale = 1.0f / std::sqrt(static_cast<float>(paddim_));
        for (size_t i = 0; i < paddim_; ++i) dst[i] *= scale;
    }

    /** @brief Return the paddim signs, each 1 or -1, written into the index for rotate_vector_gpu. */
    inline const float* get_flips() const { return sflips_.data(); }

    /** @brief Return the padded dimension, rawdim rounded up to a power of two. */
    inline size_t get_paddim() const { return paddim_; }

    /** @brief Return the raw dimension passed to the constructor. */
    inline size_t get_rawdim() const { return rawdim_; }

private:
    size_t rawdim_;
    size_t paddim_;
    std::vector<float> sflips_;
};
