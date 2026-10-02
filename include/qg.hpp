#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "common.hpp"
#include "metric.hpp"
#include "quant.hpp"

/**
 * @brief Host quantized graph codebook for the FlashGANN and RaBitQ modes.
 *
 * Each node row of row_offset_ floats holds the raw vector, packed neighbor codes, TurboQuant
 * sign codes (QUANT_TBQ only), per-neighbor factors and neighbor IDs, at the get_*_offset offsets.
 */
class QuantizationGraph {
public:
    /**
     * @brief Load a codebook file; throws on an unsupported quantizer or a mismatched or truncated file.
     *
     * File order: uint32 entry point, num_node rows, padded_dim rotation signs, then the quantizer tail.
     *
     * @param num_node number of nodes
     * @param dim vector dimension
     * @param degree neighbors per node
     * @param codebook_path codebook file path
     * @param quant quantizer family
     * @param bits code bit width
     */
    QuantizationGraph(size_t num_node, size_t dim, int degree, const std::string& codebook_path,
                      QuantType quant, int bits)
        : num_nodes_(num_node), dim_(dim), degree_(degree), quant_type_(quant), code_bits_(bits) {

        if (!quant_supported(quant_type_, code_bits_)) {
            throw std::runtime_error("Unsupported quantizer bit width");
        }
        init_layout();

        std::ifstream fin(codebook_path, std::ios::binary);
        if (!fin) throw std::runtime_error("Cannot open codebook file: " + codebook_path);

        fin.read(reinterpret_cast<char*>(&entry_point_), sizeof(uint32_t));

        size_t total_floats = num_nodes_ * row_offset_;
        size_t total_bytes = total_floats * sizeof(float);

        size_t alloc_bytes = (total_bytes + 64 - 1) & ~(64 - 1);
        data_ = static_cast<float*>(aligned_alloc(64, alloc_bytes));
        
        fin.read(reinterpret_cast<char*>(data_), total_floats * sizeof(float));

        signs_ptr_ = static_cast<float*>(aligned_alloc(64, padded_dim_ * sizeof(float)));
        fin.read(reinterpret_cast<char*>(signs_ptr_), padded_dim_ * sizeof(float));

        read_quant_tail(fin, codebook_path);
        fin.close();

        printf("Loaded codebook: %zu nodes, dim=%zu, degree=%d, padded_dim=%zu, row_offset=%zu floats "
               "(entry_point=%u, quant=%s, bits=%d)\n",
               num_nodes_, dim_, degree_, padded_dim_, row_offset_, entry_point_,
               quant_name(quant_type_), code_bits_);
    }

    /** @brief Free the host buffers. */
    ~QuantizationGraph() {
        free(data_);
        free(signs_ptr_);
        free(sketch_ptr_);
        free(level_ptr_);
    }

    /** @brief Return the node rows. */
    inline const float* get_data_ptr() const { return data_; }
    /** @brief Return the padded_dim rotation sign vector. */
    inline const float* get_signs_ptr() const { return signs_ptr_; }
    /** @brief Return the TurboQuant Fastfood sketch, or nullptr for QUANT_RBQ. */
    inline const float* get_sketch_ptr() const { return sketch_ptr_; }
    /** @brief Return the reconstruction levels, or nullptr when the file stores none. */
    inline const float* get_level_ptr() const { return level_ptr_; }
    /** @brief Return the byte size of the node rows. */
    inline size_t get_data_bytes() const { return num_nodes_ * row_offset_ * sizeof(float); }
    /** @brief Return the byte size of the sign vector. */
    inline size_t get_signs_bytes() const { return padded_dim_ * sizeof(float); }
    /** @brief Return the byte size of the sketch, 0 when absent. */
    inline size_t get_sketch_bytes() const { return sketch_words_ * sizeof(float); }
    /** @brief Return the byte size of the levels, 0 when absent. */
    inline size_t get_level_bytes() const { return num_levels_ * sizeof(float); }
    /** @brief Return the TurboQuant QJL dequantization scale, 1 for QUANT_RBQ. */
    inline float get_kappa() const { return kappa_; }
    /** @brief Return the search metric. */
    inline MetricType get_metric() const { return metric_; }
    /**
     * @brief Set the search metric.
     * @param metric search metric; main.cu passes g_metric_type
     */
    inline void set_metric(MetricType metric) { metric_ = metric; }
    /** @brief Return the quantizer family. */
    inline QuantType get_quant() const { return quant_type_; }
    /** @brief Return the code bit width. */
    inline int get_bits() const { return code_bits_; }

    /** @brief Return the float offset of the packed codes in a row. */
    inline size_t get_code_offset() const { return code_offset_; }
    /** @brief Return the float offset of the TurboQuant sign codes in a row. */
    inline size_t get_sign_offset() const { return sign_offset_; }
    /** @brief Return the float offset of the per-neighbor factors in a row. */
    inline size_t get_factor_offset() const { return factor_offset_; }
    /** @brief Return the float offset of the neighbor IDs in a row. */
    inline size_t get_neighbor_offset() const { return neighbor_offset_; }
    /** @brief Return the row stride in floats. */
    inline size_t get_row_offset() const { return row_offset_; }
    /** @brief Return the stored entry point, UINT32_MAX when the codebook has none. */
    inline vidType get_entry_point() const { return entry_point_; }

    /**
     * @brief Run the FlashGANN GPU search, implemented in gpu_search_adaptive.cu.
     * @param nq number of queries
     * @param queries nq x dim host queries
     * @param K results per query
     * @param result_idx nq x K host output IDs
     * @param result_dist nq x K host output distances
     * @param iters nq host outputs of per-query iteration counts
     * @param repeat timed kernel launches to average
     * @param beam_sz beam size
     * @param elapsed QG_SHARD_TIMER_COUNT timer slots, seconds
     */
    void gpu_search_adaptive(int nq, const float* queries, int K, vidType* result_idx,
                             float* result_dist, uint32_t* iters, int repeat, int beam_sz,
                             double* elapsed);
    /**
     * @brief Run the RaBitQ GPU search, implemented in gpu_search_rabitq.cu.
     * @param nq number of queries
     * @param queries nq x dim host queries
     * @param K results per query
     * @param result_idx nq x K host output IDs
     * @param result_dist nq x K host output distances
     * @param iters nq host outputs of per-query iteration counts
     * @param repeat timed kernel launches to average
     * @param beam_sz beam size
     * @param elapsed QG_SHARD_TIMER_COUNT timer slots, seconds
     */
    void gpu_search_rabitq(int nq, const float* queries, int K, vidType* result_idx, float* result_dist,
                    uint32_t* iters, int repeat, int beam_sz, double* elapsed);

private:
    /** @brief Compute padded_dim_, the code word counts and the row offsets from dim_, degree_ and the quantizer. */
    void init_layout() {
        const bool prod = quant_type_ == QUANT_TBQ;
        padded_dim_ = 1ULL << static_cast<size_t>(ceil(log2(dim_)));
        bitcode_words_ = quant_words(padded_dim_, prod ? quant_stage(code_bits_) : code_bits_);
        signcode_words_ = prod ? quant_words(padded_dim_, 1) : 0;
        code_offset_ = dim_;
        sign_offset_ = code_offset_ + bitcode_words_ * degree_;
        factor_offset_ = sign_offset_ + signcode_words_ * degree_;
        neighbor_offset_ = factor_offset_ + quant_factors(quant_type_) * degree_;
        row_offset_ = neighbor_offset_ + degree_;
    }

    /**
     * @brief Read and check the quantizer tail after the sign vector: tag, bits, levels and, for QUANT_TBQ, kappa and sketch.
     * @param fin open codebook stream positioned at the tail
     * @param path codebook path, used for error messages
     */
    void read_quant_tail(std::ifstream& fin, const std::string& path) {
        int32_t qtag = 0, qbits = 0, nlevel = 0;
        fin.read(reinterpret_cast<char*>(&qtag), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&qbits), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&nlevel), sizeof(int32_t));
        if (!fin) throw std::runtime_error("Missing quantizer tail in codebook: " + path);
        if (qtag != static_cast<int32_t>(quant_type_) || qbits != code_bits_) {
            throw std::runtime_error("Codebook quantizer does not match the requested one: " + path);
        }

        num_levels_ = static_cast<size_t>(nlevel);
        if (num_levels_ > 0) {
            if (num_levels_ != (1u << quant_stage(code_bits_))) {
                throw std::runtime_error("Level count mismatch: " + path);
            }
            level_ptr_ = static_cast<float*>(aligned_alloc(64, ((num_levels_ * sizeof(float) + 63) / 64) * 64));
            fin.read(reinterpret_cast<char*>(level_ptr_), sizeof(float) * num_levels_);
        }

        if (quant_type_ != QUANT_TBQ) return;
        fin.read(reinterpret_cast<char*>(&kappa_), sizeof(float));
        sketch_words_ = 3 * padded_dim_;
        sketch_ptr_ = static_cast<float*>(aligned_alloc(64, ((sketch_words_ * sizeof(float) + 63) / 64) * 64));
        fin.read(reinterpret_cast<char*>(sketch_ptr_), sizeof(float) * sketch_words_);
        if (!fin) throw std::runtime_error("Truncated sketch in codebook: " + path);
    }

    size_t num_nodes_;
    size_t dim_, padded_dim_;
    int degree_;
    size_t bitcode_words_;
    size_t signcode_words_ = 0;
    vidType entry_point_;

    // QG-style offsets (in units of float)
    size_t code_offset_;
    size_t sign_offset_;
    size_t factor_offset_;
    size_t neighbor_offset_;
    size_t row_offset_;
    float* data_ = nullptr;
    float* signs_ptr_ = nullptr;
    float* sketch_ptr_ = nullptr;
    float* level_ptr_ = nullptr;
    size_t num_levels_ = 0;
    size_t sketch_words_ = 0;
    float kappa_ = 1.0f;
    QuantType quant_type_ = QUANT_RBQ;
    int code_bits_ = 1;
    MetricType metric_ = METRIC_L2;
};
