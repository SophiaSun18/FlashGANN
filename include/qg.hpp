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

class QuantizationGraph {
public:
    // Codebook layout: entry point followed by per-node raw vector, packed codes, factors, and neighbor IDs.
    QuantizationGraph(size_t num_node, size_t dim, int degree, const std::string& codebook_path)
        : num_nodes_(num_node), dim_(dim), degree_(degree) {

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

        fin.close();

        printf("Loaded codebook: %zu nodes, dim=%zu, degree=%d, padded_dim=%zu, row_offset=%zu floats (entry_point=%u)\n",
               num_nodes_, dim_, degree_, padded_dim_, row_offset_, entry_point_);
    }

    ~QuantizationGraph() {
        free(data_);
        free(signs_ptr_);
    }

    inline const float* get_data_ptr() const { return data_; }
    inline const float* get_signs_ptr() const { return signs_ptr_; }
    inline size_t get_data_bytes() const { return num_nodes_ * row_offset_ * sizeof(float); }
    inline size_t get_signs_bytes() const { return padded_dim_ * sizeof(float); }
    inline MetricType get_metric() const { return metric_; }
    inline void set_metric(MetricType metric) { metric_ = metric; }

    inline size_t get_code_offset() const { return code_offset_; }
    inline size_t get_factor_offset() const { return factor_offset_; }
    inline size_t get_neighbor_offset() const { return neighbor_offset_; }
    inline size_t get_row_offset() const { return row_offset_; }
    inline vidType get_entry_point() const { return entry_point_; }

    void gpu_search_adaptive(int nq, const float* queries, int K, vidType* result_idx,
                             int beam_sz, double& elapsed);

private:
    void init_layout() {
        padded_dim_ = 1ULL << static_cast<size_t>(ceil(log2(dim_)));
        bitcode_words_ = padded_dim_ / 64;
        code_offset_ = dim_;
        factor_offset_ = code_offset_ + bitcode_words_ * 2 * degree_;
        neighbor_offset_ = factor_offset_ + 3 * degree_;
        row_offset_ = neighbor_offset_ + degree_;
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
    MetricType metric_ = METRIC_L2;
};
