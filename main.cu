#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <omp.h>

#include "include/data_io.hpp"
#include "include/metric.hpp"
#include "include/qg.hpp"
#include "include/quant.hpp"
#include "include/common.hpp"

struct CliConfig {
    int total_shards = 1;
    std::vector<std::string> data_files;
    std::vector<std::string> codebook_files;
    std::string query_file;
    std::string gt_file;
    int K = 100;
    int beam_size = 128;
    int degree = 32;
    int code_bits = 1;
    std::string csv_file;
};

static bool is_positive_integer(const char* s) {
    if (s == nullptr || *s == '\0') return false;
    for (const char* p = s; *p != '\0'; ++p) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    }
    return atoi(s) > 0;
}

static std::vector<std::string> split_file_list(const std::string& files) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= files.size()) {
        const size_t comma = files.find(',', begin);
        const size_t end = (comma == std::string::npos) ? files.size() : comma;
        if (end == begin) throw std::runtime_error("Empty path in comma-separated file list");
        out.emplace_back(files.substr(begin, end - begin));
        if (comma == std::string::npos) break;
        begin = comma + 1;
    }
    return out;
}

static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <num_shards> <data1.fvecs[,data2...]> <query.fvecs> <gt.ivecs> "
            "<qg1.index[,qg2...]> [K=100] [beam_size=128] [degree=32] "
            "[-quant rbq|tbq] [-bits 1|2|4] [-csv output.csv]\n"
            "Legacy: %s <data.fvecs> <query.fvecs> <gt.ivecs> <qg_codebook> "
            "[K=100] [beam_size=128] [degree=32] [-quant rbq|tbq] [-bits 1|2|4] [-csv output.csv]\n",
            prog, prog);
}

static bool is_valid_merge_candidate(vidType id, float dist) {
    return id != std::numeric_limits<vidType>::max() && std::isfinite(dist) && dist < FLT_MAX;
}

static bool parse_common_args(int argc, char** argv, int arg_idx, CliConfig& cfg) {
    if (arg_idx < argc && argv[arg_idx][0] != '-') cfg.K = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') cfg.beam_size = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') cfg.degree = atoi(argv[arg_idx++]);
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-csv" && arg_idx + 1 < argc) {
            cfg.csv_file = argv[++arg_idx];
        } else if (arg == "-quant" && arg_idx + 1 < argc) {
            g_quant_type = quant_parse(argv[++arg_idx]);
        } else if (arg == "-bits" && arg_idx + 1 < argc) {
            cfg.code_bits = atoi(argv[++arg_idx]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
        ++arg_idx;
    }
    return true;
}

static bool parse_cli(int argc, char** argv, CliConfig& cfg) {
    if (argc < 5) return false;

    int arg_idx = 1;
    if (is_positive_integer(argv[arg_idx])) {
        if (argc < 6) return false;
        cfg.total_shards = atoi(argv[arg_idx++]);
        cfg.data_files = split_file_list(argv[arg_idx++]);
        cfg.query_file = argv[arg_idx++];
        cfg.gt_file = argv[arg_idx++];
        cfg.codebook_files = split_file_list(argv[arg_idx++]);
    } else {
        cfg.total_shards = 1;
        cfg.data_files = split_file_list(argv[arg_idx++]);
        cfg.query_file = argv[arg_idx++];
        cfg.gt_file = argv[arg_idx++];
        cfg.codebook_files = split_file_list(argv[arg_idx++]);
    }

    if (cfg.data_files.size() != static_cast<size_t>(cfg.total_shards)) {
        fprintf(stderr, "Data partition count %zu does not match num_shards %d\n",
                cfg.data_files.size(), cfg.total_shards);
        return false;
    }
    if (cfg.codebook_files.size() != static_cast<size_t>(cfg.total_shards)) {
        fprintf(stderr, "Codebook count %zu does not match num_shards %d\n",
                cfg.codebook_files.size(), cfg.total_shards);
        return false;
    }
    return parse_common_args(argc, argv, arg_idx, cfg);
}

/** SHAME(TALLFUNC) */
int main(int argc, char** argv) {
    CliConfig cfg;
    try {
        if (!parse_cli(argc, argv, cfg)) {
            print_usage(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        print_usage(argv[0]);
        return 1;
    }

    printf("========================================\n");
    printf("FlashGANN AP Loading from:\n");
    printf("========================================\n");
    printf("Shards: %d\n", cfg.total_shards);
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        printf("Shard %d data file: %s\n", shard, cfg.data_files[shard].c_str());
        printf("Shard %d QG codebook: %s\n", shard, cfg.codebook_files[shard].c_str());
    }
    printf("Query file: %s\n", cfg.query_file.c_str());
    printf("Ground truth: %s\n", cfg.gt_file.c_str());
    printf("K: %d\n", cfg.K);
    printf("Beam size: %d\n", cfg.beam_size);
    printf("Degree: %d\n", cfg.degree);
    printf("Quantizer: %s, bits: %d\n", quant_name(g_quant_type), cfg.code_bits);
    if (!cfg.csv_file.empty()) printf("CSV output: %s\n", cfg.csv_file.c_str());

    g_metric_type = infer_metric_from_dataset_path(cfg.data_files[0]);
    printf("Metric: %s\n", metric_name(g_metric_type));
    printf("========================================\n\n");

    LoadedVectors<float> query_vectors = load_fvecs<float>(cfg.query_file, "queries");
    LoadedVectors<int> groundtruth_vectors = load_ivecs(cfg.gt_file, "ground truth");

    const size_t nq = query_vectors.count;
    const int query_dim = query_vectors.dim;
    const size_t gt_nq = groundtruth_vectors.count;
    const int gt_k = groundtruth_vectors.dim;
    if (gt_nq != nq) {
        fprintf(stderr, "Ground-truth query count %zu does not match query count %zu\n", gt_nq, nq);
        return 1;
    }

    std::vector<LoadedVectors<float>> bases;
    std::vector<size_t> shard_offsets;
    bases.reserve(cfg.total_shards);
    shard_offsets.reserve(cfg.total_shards);
    size_t total_base_count = 0;
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        shard_offsets.push_back(total_base_count);
        bases.push_back(load_fvecs<float>(cfg.data_files[shard], "data"));
        if (query_dim != bases.back().dim) {
            fprintf(stderr, "Query dimension %d does not match shard %d data dimension %d\n",
                    query_dim, shard, bases.back().dim);
            return 1;
        }
        total_base_count += bases.back().count;
    }
    if (total_base_count > static_cast<size_t>(std::numeric_limits<vidType>::max())) {
        fprintf(stderr, "Total sharded data count %zu exceeds vidType capacity\n", total_base_count);
        return 1;
    }

    const std::vector<float>& queries = query_vectors.values;
    const std::vector<int>& groundtruth = groundtruth_vectors.values;

    // record each shard's intermediate results and elapsed time
    std::vector<vidType> results(static_cast<size_t>(cfg.total_shards) * nq * cfg.K);
    std::vector<float> result_dist(static_cast<size_t>(cfg.total_shards) * nq * cfg.K);
    std::vector<vidType> merged_results(nq * cfg.K, std::numeric_limits<vidType>::max());
    std::vector<float> merged_dist(nq * cfg.K, FLT_MAX);
    std::vector<double> elapsed(static_cast<size_t>(cfg.total_shards) * QG_SHARD_TIMER_COUNT + 1);

    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);
    if (dev_err != cudaSuccess || device_count <= 0) {
        fprintf(stderr, "No CUDA devices available: %s\n", cudaGetErrorString(dev_err));
        return 1;
    }
    printf("Found %d CUDA devices\n", device_count);

    int shard_failed = 0;

#pragma omp parallel for schedule(static)
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        const int device_id = shard % device_count;
        const size_t result_base = static_cast<size_t>(shard) * nq * cfg.K;
        const size_t elapsed_base = static_cast<size_t>(shard) * QG_SHARD_TIMER_COUNT;
        try {
            cudaError_t set_err = cudaSetDevice(device_id);
            if (set_err != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(set_err));
            }
            printf("Shard %d running on GPU %d\n", shard, device_id);
            QuantizationGraph qg(bases[shard].count, bases[shard].dim, cfg.degree,
                                 cfg.codebook_files[shard], g_quant_type, cfg.code_bits);
            qg.set_metric(g_metric_type);
            qg.gpu_search_adaptive(static_cast<int>(nq), queries.data(), cfg.K,
                                   results.data() + result_base, result_dist.data() + result_base,
                                   cfg.beam_size, elapsed.data() + elapsed_base);
        } catch (const std::exception& e) {
#pragma omp critical
            {
                fprintf(stderr, "Shard %d failed: %s\n", shard, e.what());
            }
#pragma omp atomic write
            shard_failed = 1;
        }
    }
    if (shard_failed) return 1;

    const size_t merge_timer_idx = static_cast<size_t>(cfg.total_shards) * QG_SHARD_TIMER_COUNT;
    auto merge_start = std::chrono::high_resolution_clock::now();
    std::vector<int> merge_positions(nq * static_cast<size_t>(cfg.total_shards), 0);
#pragma omp parallel for schedule(static)
    for (size_t query_id = 0; query_id < nq; ++query_id) {
        int* positions = merge_positions.data() + query_id * static_cast<size_t>(cfg.total_shards);
        for (int out_rank = 0; out_rank < cfg.K; ++out_rank) {
            int best_shard = -1;
            vidType best_local_id = std::numeric_limits<vidType>::max();
            float best_dist = FLT_MAX;

            for (int shard = 0; shard < cfg.total_shards; ++shard) {
                while (positions[shard] < cfg.K) {
                    const size_t in_idx = (static_cast<size_t>(shard) * nq + query_id) * cfg.K
                                          + positions[shard];
                    const vidType candidate_id = results[in_idx];
                    const float candidate_dist = result_dist[in_idx];
                    if (is_valid_merge_candidate(candidate_id, candidate_dist)) {
                        if (best_shard < 0 || candidate_dist < best_dist ||
                            (candidate_dist == best_dist && shard < best_shard)) {
                            best_shard = shard;
                            best_local_id = candidate_id;
                            best_dist = candidate_dist;
                        }
                        break;
                    }
                    ++positions[shard];
                }
            }

            const size_t out_idx = query_id * cfg.K + out_rank;
            if (best_shard < 0) {
                merged_results[out_idx] = std::numeric_limits<vidType>::max();
                merged_dist[out_idx] = FLT_MAX;
                continue;
            }

            merged_results[out_idx] = static_cast<vidType>(shard_offsets[best_shard] + best_local_id);
            merged_dist[out_idx] = best_dist;
            ++positions[best_shard];
        }
    }
    auto merge_end = std::chrono::high_resolution_clock::now();
    elapsed[merge_timer_idx] = std::chrono::duration<double>(merge_end - merge_start).count();

    double max_search_elapsed = 0.0;
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        const size_t base_idx = static_cast<size_t>(shard) * QG_SHARD_TIMER_COUNT;
        max_search_elapsed = std::max(max_search_elapsed, elapsed[base_idx + QG_TIMER_SEARCH]);
    }

    const float recall = compute_recall_dedup(merged_results.data(), groundtruth.data(), nq, cfg.K, gt_k) * 100.0f;
    const double latency_ms = (nq > 0) ? (max_search_elapsed * 1000.0 / static_cast<double>(nq)) : 0.0;
    const double qps = (max_search_elapsed > 0.0) ? (static_cast<double>(nq) / max_search_elapsed) : 0.0;

    printf("\n========================================\n");
    printf("Timing Breakdown\n");
    printf("========================================\n");
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        const size_t base_idx = static_cast<size_t>(shard) * QG_SHARD_TIMER_COUNT;
        printf("Shard %d query transfer: %.6f ms\n",
               shard, elapsed[base_idx + QG_TIMER_QUERY_TRANSFER] * 1000.0);
        printf("Shard %d search: %.6f ms\n",
               shard, elapsed[base_idx + QG_TIMER_SEARCH] * 1000.0);
        printf("Shard %d result copy: %.6f ms\n",
               shard, elapsed[base_idx + QG_TIMER_RESULT_COPY] * 1000.0);
    }
    printf("Global merge: %.6f ms\n", elapsed[merge_timer_idx] * 1000.0);
    printf("Reported runtime (max shard search): %.6f ms\n", max_search_elapsed * 1000.0);
    printf("========================================\n");

    printf("\n========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("Total time: %.6f ms\n", max_search_elapsed * 1000.0);
    printf("Throughput: %.6f queries/sec\n", qps);
    printf("Avg latency: %.6f ms/query\n", latency_ms);
    printf("Recall@%d: %.6f\n", cfg.K, recall);
    printf("========================================\n");

    RunStats run_stats;
    run_stats.runtime = max_search_elapsed;
    run_stats.latency = latency_ms;
    run_stats.throughput = qps;
    run_stats.recall = recall;

    if (!cfg.csv_file.empty()) {
        append_run_stats_to_csv(cfg.csv_file, cfg.K, cfg.beam_size, run_stats);
        printf("Saved GPU stats CSV row to: %s\n", cfg.csv_file.c_str());
    }

    return 0;
}
