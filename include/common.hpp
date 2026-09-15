#pragma once

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>

#include "distance.hpp"

typedef uint32_t vidType;

#define QG_BQUERY 6

struct RunStats {
    double runtime = 0.0;
    double latency = 0.0;
    double throughput = 0.0;
    double recall = 0.0;
};

inline vidType compute_rabitq_entry_point(const float *data, size_t npoints, size_t dim, size_t row_offset) {
    std::vector<float> centroid(dim, 0.0f);
    const float inv_npoints = 1.0f / static_cast<float>(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        const float *row = data + i * row_offset;
        for (size_t d = 0; d < dim; ++d) {
            centroid[d] += row[d] * inv_npoints;
        }
    }

    vidType best = 0;
    float best_dist = FLT_MAX;
    for (size_t i = 0; i < npoints; ++i) {
        const float *row = data + i * row_offset;
        float dist = compute_distance(static_cast<int>(dim), row, centroid.data());
        if (dist < best_dist) {
            best_dist = dist;
            best = static_cast<vidType>(i);
        }
    }
    return best;
}

template <typename T>
float compute_recall(const T *predicted, const int *groundtruth, size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    const int eval_k = std::min(topk, gt_k);
    for (size_t i = 0; i < nq; ++i) {
        for (int j = 0; j < topk; ++j) {
            int pred = int(predicted[i * topk + j]);
            for (int k = 0; k < eval_k; ++k) {
                int gt = groundtruth[i * gt_k + k];
                if (pred == gt) {
                    ++correct;
                    break;
                }
            }
        }
    }
    return float(correct) / (nq * topk);
}

template <typename T>
float compute_recall_dedup(const T *predicted, const int *groundtruth, size_t nq, int topk, int gt_k) {
    size_t correct = 0;
    const int eval_k = std::min(topk, gt_k);
    for (size_t i = 0; i < nq; ++i) {
        std::unordered_set<int> final_set;
        final_set.reserve(static_cast<size_t>(topk));
        for (int j = 0; j < topk; ++j) {
            int pred = int(predicted[i * topk + j]);
            if (!final_set.insert(pred).second) {
                continue;
            }
            for (int k = 0; k < eval_k; ++k) {
                int gt = groundtruth[i * gt_k + k];
                if (pred == gt) {
                    ++correct;
                    break;
                }
            }
        }
    }
    return float(correct) / (nq * topk);
}


inline void append_run_stats_to_csv(const std::string &filename, int k, int beam, const RunStats &stats) {
    bool file_exists = std::filesystem::exists(filename);

    std::ofstream out(filename, std::ios::app);
    if (!out.is_open()) {
        fprintf(stderr, "Error: cannot open CSV file %s\n", filename.c_str());
        return;
    }

    if (!file_exists) {
        out << "TopK,"
            << "Beam_size,"
            << "Runtime,"
            << "Latency,"
            << "Throughput,"
            << "Recall\n";
    }

    out << k << ","
        << beam << ","
        << stats.runtime << ","
        << stats.latency << ","
        << stats.throughput << ","
        << stats.recall << "\n";

    out.close();
}
