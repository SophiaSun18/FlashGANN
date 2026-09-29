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
#include "include/common.hpp"
#define GPU_SEARCH_MODE_FLASHGANN 1
#define GPU_SEARCH_MODE_PATHW 2

#ifndef GPU_SEARCH_MODE
#error "GPU_SEARCH_MODE must be set by the GPU binary build target"
#endif

#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
#include "include/index.hpp"
#elif GPU_SEARCH_MODE == GPU_SEARCH_MODE_FLASHGANN
#include "include/qg.hpp"
#include "include/quant.hpp"
#else
#error "Unknown GPU_SEARCH_MODE"
#endif

/** @brief Command-line options, one data, index and (PathW) signbit file per shard. */
struct CliConfig {
    int total_shards = 1;
    std::vector<std::string> data_files;
    std::vector<std::string> codebook_files;
    std::vector<std::string> sign_files;
    std::string query_file;
    std::string gt_file;
    int K = 100;
    int beam_size = 128;
    int degree = 32;
    bool degree_explicit = false;
    int code_bits = 1;
    float keep_ratio = 0.0f;
    float prune_ratio = 0.7f;
    std::string csv_file;
    std::string iters_file;
    int repeat = 1;
};

/** @brief Recognize explicit GPU modes, following beam_search_collab's shared-main convention. */
static bool is_gpu_mode_arg(const std::string& value) {
    return value == "gpu_flashgann" || value == "gpu_pathw" || value == "flashgann" || value == "pathw";
}

/** @brief Name the search mode linked into this executable. */
static const char* gpu_search_mode_name() {
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    return "gpu_pathw";
#else
    return "gpu_flashgann";
#endif
}

/**
 * @brief Whether the argument is a positive decimal integer.
 * @param s argument text
 * @return true for a shard count
 */
static bool is_positive_integer(const char* s) {
    if (s == nullptr || *s == '\0') return false;
    for (const char* p = s; *p != '\0'; ++p) {
        if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    }
    return atoi(s) > 0;
}

/**
 * @brief Split a comma-separated path list.
 * @param files the list
 * @return the paths in order
 */
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

/**
 * @brief Print both the sharded and the single-index command line of this mode.
 * @param prog program name
 */
static void print_usage(const char* prog) {
    const char* mode = gpu_search_mode_name();
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    fprintf(stderr,
            "Usage: %s [%s] <num_shards> <data1.fvecs[,data2...]> <query.fvecs> <gt.ivecs> "
            "<graph1.bin[,graph2...]> <signbit1.bin[,signbit2...]> [K=100] [beam_size=128] [degree=graph] "
            "[-p_ratio keep] [-a_ratio prune] [-csv output.csv] [-iters output.txt] [-repeat n]\n"
            "Legacy: %s [%s] <data.fvecs> <query.fvecs> <gt.ivecs> <graph.bin> <signbit.bin> "
            "[K=100] [beam_size=128] [degree=graph] [-p_ratio keep] [-a_ratio prune] "
            "[-csv output.csv] [-iters output.txt] [-repeat n]\n",
            prog, mode, prog, mode);
#else
    fprintf(stderr,
            "Usage: %s [%s] <num_shards> <data1.fvecs[,data2...]> <query.fvecs> <gt.ivecs> "
            "<qg1.index[,qg2...]> [K=100] [beam_size=128] [degree=32] "
            "[-quant rbq|tbq] [-bits 1|2|4] [-csv output.csv] [-iters output.txt] [-repeat n]\n"
            "Legacy: %s [%s] <data.fvecs> <query.fvecs> <gt.ivecs> <qg_codebook> "
            "[K=100] [beam_size=128] [degree=32] [-quant rbq|tbq] [-bits 1|2|4] "
            "[-csv output.csv] [-iters output.txt] [-repeat n]\n",
            prog, mode, prog, mode);
#endif
}

/**
 * @brief Whether a shard result slot holds a real neighbor.
 * @param id local neighbor id
 * @param dist its distance
 * @return true when the slot can take part in the merge
 */
static bool is_valid_merge_candidate(vidType id, float dist) {
    return id != std::numeric_limits<vidType>::max() && std::isfinite(dist) && dist < FLT_MAX;
}

/**
 * @brief Parse the optional positional values and the flags after the file arguments.
 * @param argc argument count
 * @param argv arguments
 * @param arg_idx first argument after the files
 * @param cfg receives the options
 * @return false on an unknown flag
 */
static bool parse_common_args(int argc, char** argv, int arg_idx, CliConfig& cfg) {
    if (arg_idx < argc && argv[arg_idx][0] != '-') cfg.K = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') cfg.beam_size = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        cfg.degree = atoi(argv[arg_idx++]);
        cfg.degree_explicit = true;
    }
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-csv" && arg_idx + 1 < argc) {
            cfg.csv_file = argv[++arg_idx];
        } else if (arg == "-iters" && arg_idx + 1 < argc) {
            cfg.iters_file = argv[++arg_idx];
        } else if (arg == "-repeat" && arg_idx + 1 < argc) {
            cfg.repeat = atoi(argv[++arg_idx]);
            if (cfg.repeat < 1) cfg.repeat = 1;
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
        } else if ((arg == "-p_ratio" || arg == "-a_ratio") && arg_idx + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++arg_idx];
            const float ratio = std::strtof(value, &end);
            if (end == value || *end) {
                fprintf(stderr, "Invalid PathW ratio: %s\n", value);
                return false;
            }
            if (arg == "-p_ratio") cfg.keep_ratio = ratio;
            else cfg.prune_ratio = ratio;
#else
        } else if (arg == "-quant" && arg_idx + 1 < argc) {
            g_quant_type = quant_parse(argv[++arg_idx]);
        } else if (arg == "-bits" && arg_idx + 1 < argc) {
            cfg.code_bits = atoi(argv[++arg_idx]);
#endif
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return false;
        }
        ++arg_idx;
    }
    return true;
}

/**
 * @brief Parse an optional mode word, then the sharded form when a shard count follows, else the single-index form.
 * @param argc argument count
 * @param argv arguments
 * @param cfg receives the options
 * @return false when the arguments do not fit either form
 */
static bool parse_cli(int argc, char** argv, CliConfig& cfg) {
    constexpr bool pathw = GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW;
    constexpr int input_files = pathw ? 5 : 4;

    // [1] optional mode word, which must name this binary's mode
    int arg_idx = 1;
    if (arg_idx < argc && is_gpu_mode_arg(argv[arg_idx])) {
        const std::string word = argv[arg_idx];
        const std::string requested = word.starts_with("gpu_") ? word : "gpu_" + word;
        if (requested != gpu_search_mode_name()) {
            fprintf(stderr, "This binary was built for mode %s, got %s\n", gpu_search_mode_name(), word.c_str());
            return false;
        }
        ++arg_idx;
    }

    // [2] optional shard count, then one comma-separated list per shard file kind
    if (arg_idx < argc && is_positive_integer(argv[arg_idx])) {
        cfg.total_shards = atoi(argv[arg_idx++]);
    } else {
        cfg.total_shards = 1;
    }
    if (argc < arg_idx + input_files) return false;
    cfg.data_files = split_file_list(argv[arg_idx++]);
    cfg.query_file = argv[arg_idx++];
    cfg.gt_file = argv[arg_idx++];
    cfg.codebook_files = split_file_list(argv[arg_idx++]);
    if (pathw) cfg.sign_files = split_file_list(argv[arg_idx++]);

    // [3] every shard needs one file of each kind
    const size_t shards = static_cast<size_t>(cfg.total_shards);
    if (cfg.data_files.size() != shards) {
        fprintf(stderr, "Data partition count %zu does not match num_shards %d\n",
                cfg.data_files.size(), cfg.total_shards);
        return false;
    }
    if (cfg.codebook_files.size() != shards) {
        fprintf(stderr, "%s count %zu does not match num_shards %d\n",
                pathw ? "Graph" : "Codebook", cfg.codebook_files.size(), cfg.total_shards);
        return false;
    }
    if (pathw && cfg.sign_files.size() != shards) {
        fprintf(stderr, "Signbit count %zu does not match num_shards %d\n",
                cfg.sign_files.size(), cfg.total_shards);
        return false;
    }
    return parse_common_args(argc, argv, arg_idx, cfg);
}

int main(int argc, char** argv) {
    constexpr bool pathw = GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW;
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
    if (cfg.K < 1 || cfg.beam_size < 1 || cfg.degree < 1) {
        fprintf(stderr, "K, beam size and degree must be positive\n");
        return 1;
    }

    printf("========================================\n");
    printf("GPU Beam Search Loading from:\nMode: %s\n", gpu_search_mode_name());
    printf("========================================\n");
    printf("Shards: %d\n", cfg.total_shards);
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        printf("Shard %d data file: %s\n", shard, cfg.data_files[shard].c_str());
        printf("Shard %d %s: %s\n", shard, pathw ? "graph" : "QG codebook", cfg.codebook_files[shard].c_str());
        if (pathw) printf("Shard %d signbit: %s\n", shard, cfg.sign_files[shard].c_str());
    }
    printf("Query file: %s\n", cfg.query_file.c_str());
    printf("Ground truth: %s\n", cfg.gt_file.c_str());
    printf("K: %d\n", cfg.K);
    printf("Beam size: %d\n", cfg.beam_size);
    if (!pathw || cfg.degree_explicit) printf("Degree: %d\n", cfg.degree);
    else printf("Degree: from graph header\n");
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    printf("PathW ratios: keep %.4f, prune %.4f\n", cfg.keep_ratio, cfg.prune_ratio);
#else
    printf("Quantizer: %s, bits: %d\n", quant_name(g_quant_type), cfg.code_bits);
#endif
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

    // load every shard's base vectors (and PathW graph); shard ids are offset by the shards before it
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    std::vector<IndexGraph<float>> bases(cfg.total_shards);
#else
    std::vector<LoadedVectors<float>> bases;
    bases.reserve(cfg.total_shards);
#endif
    std::vector<size_t> shard_offsets;
    shard_offsets.reserve(cfg.total_shards);
    size_t total_base_count = 0;
    for (int shard = 0; shard < cfg.total_shards; ++shard) {
        shard_offsets.push_back(total_base_count);
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
        try {
            bases[shard].load_graph_index(cfg.codebook_files[shard].c_str());
            bases[shard].load_data(cfg.data_files[shard]);
        } catch (const std::exception& e) {
            fprintf(stderr, "PathW shard %d input error: %s\n", shard, e.what());
            return 1;
        }
        bases[shard].metric = g_metric_type;
        if (cfg.degree_explicit && bases[shard].maxDeg != cfg.degree) {
            fprintf(stderr, "Requested degree %d does not match shard %d graph degree %d\n",
                    cfg.degree, shard, bases[shard].maxDeg);
            return 1;
        }
        const size_t shard_count = bases[shard].ntotal;
        const int shard_dim = bases[shard].d;
#else
        bases.push_back(load_fvecs<float>(cfg.data_files[shard], "data"));
        const size_t shard_count = bases.back().count;
        const int shard_dim = bases.back().dim;
#endif
        if (query_dim != shard_dim) {
            fprintf(stderr, "Query dimension %d does not match shard %d data dimension %d\n",
                    query_dim, shard, shard_dim);
            return 1;
        }
        total_base_count += shard_count;
    }
    if (total_base_count > static_cast<size_t>(std::numeric_limits<vidType>::max())) {
        fprintf(stderr, "Total sharded data count %zu exceeds vidType capacity\n",
                total_base_count);
        return 1;
    }

    const std::vector<float>& queries = query_vectors.values;
    const std::vector<int>& groundtruth = groundtruth_vectors.values;

    // record each shard's intermediate results, iterations and elapsed time
    const size_t shards = static_cast<size_t>(cfg.total_shards);
    std::vector<vidType> results(shards * nq * cfg.K);
    std::vector<float> result_dist(shards * nq * cfg.K);
    std::vector<uint32_t> shard_iters(shards * nq);
    std::vector<vidType> merged_results(nq * cfg.K, std::numeric_limits<vidType>::max());
    std::vector<float> merged_dist(nq * cfg.K, FLT_MAX);
    std::vector<uint32_t> merged_iters(nq, 0);
    std::vector<double> elapsed(shards * QG_SHARD_TIMER_COUNT + 1, 0.0);

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
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
            bases[shard].search_pathw(static_cast<int>(nq), queries.data(), cfg.K,
                                      results.data() + result_base, result_dist.data() + result_base,
                                      cfg.beam_size, cfg.sign_files[shard].c_str(), cfg.keep_ratio,
                                      cfg.prune_ratio, elapsed.data() + elapsed_base,
                                      shard_iters.data() + static_cast<size_t>(shard) * nq, cfg.repeat);
#else
            QuantizationGraph qg(bases[shard].count, bases[shard].dim, cfg.degree,
                                 cfg.codebook_files[shard], g_quant_type, cfg.code_bits);
            qg.set_metric(g_metric_type);
            qg.gpu_search_adaptive(static_cast<int>(nq), queries.data(), cfg.K,
                                   results.data() + result_base, result_dist.data() + result_base,
                                   shard_iters.data() + static_cast<size_t>(shard) * nq, cfg.repeat,
                                   cfg.beam_size, elapsed.data() + elapsed_base);
#endif
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

    // merge the per-shard sorted lists by distance, lower shard first on ties
    const size_t merge_timer_idx = shards * QG_SHARD_TIMER_COUNT;
    auto merge_start = std::chrono::high_resolution_clock::now();
    std::vector<int> merge_positions(nq * shards, 0);
#pragma omp parallel for schedule(static)
    for (size_t query_id = 0; query_id < nq; ++query_id) {
        int* positions = merge_positions.data() + query_id * shards;
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

            merged_results[out_idx] =
                static_cast<vidType>(shard_offsets[best_shard] + best_local_id);
            merged_dist[out_idx] = best_dist;
            ++positions[best_shard];
        }

        // a query takes as many iterations as its slowest shard
        for (int shard = 0; shard < cfg.total_shards; ++shard) {
            const uint32_t count = shard_iters[static_cast<size_t>(shard) * nq + query_id];
            merged_iters[query_id] = std::max(merged_iters[query_id], count);
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
    const double nqd = static_cast<double>(nq);
    const double latency_ms = (nq > 0) ? (max_search_elapsed * 1000.0 / nqd) : 0.0;
    const double qps = (max_search_elapsed > 0.0) ? (nqd / max_search_elapsed) : 0.0;

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

    if (!cfg.iters_file.empty()) {
        FILE* fout = fopen(cfg.iters_file.c_str(), "w");
        if (fout == nullptr) {
            fprintf(stderr, "Cannot open iteration output: %s\n", cfg.iters_file.c_str());
            return 1;
        }
        for (uint32_t count : merged_iters) fprintf(fout, "%u\n", count);
        fclose(fout);
        printf("Saved per-query iterations to: %s\n", cfg.iters_file.c_str());
    }

    return 0;
}
