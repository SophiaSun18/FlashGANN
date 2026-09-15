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

inline constexpr int QG_TIMER_QUERY_TRANSFER = 0;
inline constexpr int QG_TIMER_SEARCH = 1;
inline constexpr int QG_TIMER_RESULT_COPY = 2;
inline constexpr int QG_SHARD_TIMER_COUNT = 3;

class QuantizationGraph {
public:
    // Codebook layout: entry point followed by per-node raw vector, packed codes, factors, and neighbor IDs.
    QuantizationGraph(size_t num_node, size_t dim, int degree, const std::string& codebook_path,
                      QuantType quant, int bits)
        : num_nodes_(num_node), dim_(dim), degree_(degree), quant_type_(quant), code_bits_(bits) {

        if (!quant_valid(code_bits_)) throw std::runtime_error("Unsupported quantizer bit width");
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

    ~QuantizationGraph() {
        free(data_);
        free(signs_ptr_);
        free(level_ptr_);
    }

    inline const float* get_data_ptr() const { return data_; }
    inline const float* get_signs_ptr() const { return signs_ptr_; }
    inline const float* get_level_ptr() const { return level_ptr_; }
    inline size_t get_data_bytes() const { return num_nodes_ * row_offset_ * sizeof(float); }
    inline size_t get_signs_bytes() const { return padded_dim_ * sizeof(float); }
    inline size_t get_level_bytes() const { return num_levels_ * sizeof(float); }
    inline MetricType get_metric() const { return metric_; }
    inline void set_metric(MetricType metric) { metric_ = metric; }
    inline QuantType get_quant() const { return quant_type_; }
    inline int get_bits() const { return code_bits_; }

    inline size_t get_code_offset() const { return code_offset_; }
    inline size_t get_factor_offset() const { return factor_offset_; }
    inline size_t get_neighbor_offset() const { return neighbor_offset_; }
    inline size_t get_row_offset() const { return row_offset_; }
    inline vidType get_entry_point() const { return entry_point_; }

    void gpu_search_adaptive(int nq, const float* queries, int K, vidType* result_idx,
                             float* result_dist, int beam_sz, double *elapsed);

private:
    void init_layout() {
        padded_dim_ = 1ULL << static_cast<size_t>(ceil(log2(dim_)));
        bitcode_words_ = quant_words(padded_dim_, code_bits_);
        code_offset_ = dim_;
        factor_offset_ = code_offset_ + bitcode_words_ * degree_;
        neighbor_offset_ = factor_offset_ + 3 * degree_;
        row_offset_ = neighbor_offset_ + degree_;
    }

    /**
     * @brief Read the quantizer descriptor appended after the sign vector.
     * @param fin open index stream positioned at the tail
     * @param path index path, used for error messages
     */
    void read_quant_tail(std::ifstream& fin, const std::string& path) {
        int32_t qtag = 0, qbits = 0, nlevel = 0;
        fin.read(reinterpret_cast<char*>(&qtag), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&qbits), sizeof(int32_t));
        fin.read(reinterpret_cast<char*>(&nlevel), sizeof(int32_t));
        if (!fin) {
            fin.clear();
            num_levels_ = 0;
            printf("Codebook has no quantizer tail; using requested quantizer %s/%d-bit\n",
                   quant_name(quant_type_), code_bits_);
            return;
        }
        if (qtag != static_cast<int32_t>(quant_type_) || qbits != code_bits_) {
            throw std::runtime_error("Codebook quantizer does not match the requested one: " + path);
        }

        num_levels_ = static_cast<size_t>(nlevel);
        if (num_levels_ == 0) return;
        if (num_levels_ != (1u << code_bits_)) throw std::runtime_error("Level count mismatch: " + path);
        level_ptr_ = static_cast<float*>(aligned_alloc(64, ((num_levels_ * sizeof(float) + 63) / 64) * 64));
        fin.read(reinterpret_cast<char*>(level_ptr_), sizeof(float) * num_levels_);
        if (!fin) throw std::runtime_error("Truncated level table in codebook: " + path);
    }

    size_t num_nodes_;
    size_t dim_, padded_dim_;
    int degree_;
    size_t bitcode_words_;
    vidType entry_point_;
    
    // QG-style offsets (in units of float)
    size_t code_offset_;
    size_t factor_offset_;
    size_t neighbor_offset_;
    size_t row_offset_;
    float* data_ = nullptr;
    float* signs_ptr_ = nullptr;
    float* level_ptr_ = nullptr;
    size_t num_levels_ = 0;
    QuantType quant_type_ = QUANT_RBQ;
    int code_bits_ = 1;
    MetricType metric_ = METRIC_L2;
};
