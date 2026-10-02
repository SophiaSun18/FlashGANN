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

/** @brief Vertex ID type of shard-local and merged search results. */
typedef uint32_t vidType;

/** @brief Bit width of the quantized query that the FlashGANN and RaBitQ lookup tables use. */
#define QG_BQUERY 6

/**
 * @brief Slots of one shard's elapsed-time block, in seconds, filled by each search driver.
 *
 * main.cu gives every shard QG_SHARD_TIMER_COUNT consecutive entries of its elapsed array.
 */
inline constexpr int QG_TIMER_QUERY_TRANSFER = 0;
inline constexpr int QG_TIMER_SEARCH = 1;
inline constexpr int QG_TIMER_RESULT_COPY = 2;
inline constexpr int QG_SHARD_TIMER_COUNT = 3;

/** @brief One benchmark row that append_run_stats_to_csv writes. */
struct RunStats {
    double runtime = 0.0;       // max shard search time, seconds
    double latency = 0.0;       // milliseconds per query
    double throughput = 0.0;    // queries per second
    double recall = 0.0;        // recall@K, percent
};

/**
 * @brief Pick the node closest to the data centroid as the search entry point.
 *
 * The FlashGANN and RaBitQ drivers call it when the codebook stores no entry point (UINT32_MAX).
 *
 * @param data first row of the node data
 * @param npoints number of rows
 * @param dim number of vector coordinates read per row
 * @param row_offset row stride in floats
 * @return index of the row with the smallest compute_distance to the centroid
 */
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

/**
 * @brief Compute recall of row-major top-k results against ground truth.
 *
 * Each predicted ID that appears among the first min(topk, gt_k) ground-truth IDs counts, duplicates included.
 *
 * @tparam T predicted ID type
 * @param predicted nq x topk predicted IDs
 * @param groundtruth nq x gt_k ground-truth IDs
 * @param nq number of queries
 * @param topk predicted IDs per query
 * @param gt_k ground-truth IDs per query
 * @return matched count divided by nq * topk
 */
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

/**
 * @brief Compute recall like compute_recall, counting each distinct predicted ID once per query.
 *
 * main.cu reports this recall for the merged shard results.
 *
 * @tparam T predicted ID type
 * @param predicted nq x topk predicted IDs
 * @param groundtruth nq x gt_k ground-truth IDs
 * @param nq number of queries
 * @param topk predicted IDs per query
 * @param gt_k ground-truth IDs per query
 * @return matched count divided by nq * topk
 */
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

/**
 * @brief Append one result row to a CSV file, writing the header when the file is new.
 * @param filename CSV path
 * @param k top-K of the run
 * @param beam beam size of the run
 * @param stats measured runtime, latency, throughput and recall
 */
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
