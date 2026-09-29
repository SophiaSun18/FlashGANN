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

/** SHAME(TALLFUNC) */
int main(int argc, char** argv) {
    const std::string mode = gpu_search_mode_name();
    int input_arg = 1;
    if (argc > 1 && is_gpu_mode_arg(argv[1])) {
        const std::string requested = std::string(argv[1]).starts_with("gpu_") ? argv[1] : "gpu_" + std::string(argv[1]);
        if (mode != requested) {
            fprintf(stderr, "This binary was built for mode %s, got %s\n", mode.c_str(), argv[1]);
            return 1;
        }
        input_arg = 2;
    }
    constexpr bool pathw = GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW;
    const int input_files = pathw ? 5 : 4;
    if (argc < input_arg + input_files) {
        fprintf(stderr, "Usage: %s [%s] <base.fvecs> <query.fvecs> <gt.ivecs> <graph_or_codebook> [signbit.bin for gpu_pathw] [K=100] [beam=128] [degree=32] [-quant rbq|tbq] [-bits n] [-p_ratio keep] [-a_ratio prune] [-repeat n] [-iters file] [-csv file]\n", argv[0], mode.c_str());
        return 1;
    }

    const char* data_file = argv[input_arg];
    const char* query_file = argv[input_arg + 1];
    const char* gt_file = argv[input_arg + 2];
    const char* index_file = argv[input_arg + 3];
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    const char* sign_file = argv[input_arg + 4];
    float keep_ratio = 0.0f;
    float prune_ratio = 0.7f;
#else
    int code_bits = 1;
#endif

    int K = 100;
    int beam_size = 128;
    int degree = 32;
    bool degree_explicit = false;
    std::string csv_file;
    std::string iters_file;
    int repeat = 1;
};

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
 * @brief Print both the sharded and the single-index command line.
 * @param prog program name
 */
static void print_usage(const char* prog) {
    fprintf(stderr,
            "Usage: %s <num_shards> <data1.fvecs[,data2...]> <query.fvecs> <gt.ivecs> "
            "<qg1.index[,qg2...]> [K=100] [beam_size=128] [degree=32] "
            "[-quant rbq|tbq] [-bits 1|2|4] [-csv output.csv] [-iters output.txt] [-repeat n]\n"
            "Legacy: %s <data.fvecs> <query.fvecs> <gt.ivecs> <qg_codebook> "
            "[K=100] [beam_size=128] [degree=32] [-quant rbq|tbq] [-bits 1|2|4] "
            "[-csv output.csv] [-iters output.txt] [-repeat n]\n",
            prog, prog);
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

    int arg_idx = input_arg + input_files;
    if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') {
        degree = atoi(argv[arg_idx++]);
        degree_explicit = true;
    }
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-csv" && arg_idx + 1 < argc) {
            cfg.csv_file = argv[++arg_idx];
        } else if (arg == "-iters" && arg_idx + 1 < argc) {
            cfg.iters_file = argv[++arg_idx];
        } else if (arg == "-repeat" && arg_idx + 1 < argc) {
            repeat = atoi(argv[++arg_idx]);
            if (repeat < 1) repeat = 1;
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
        } else if ((arg == "-p_ratio" || arg == "-a_ratio") && arg_idx + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++arg_idx];
            const float ratio = std::strtof(value, &end);
            if (end == value || *end) {
                fprintf(stderr, "Invalid PathW ratio: %s\n", value);
                return 1;
            }
            if (arg == "-p_ratio") keep_ratio = ratio;
            else prune_ratio = ratio;
#else
        } else if (arg == "-quant" && arg_idx + 1 < argc) {
            g_quant_type = quant_parse(argv[++arg_idx]);
        } else if (arg == "-bits" && arg_idx + 1 < argc) {
            code_bits = atoi(argv[++arg_idx]);
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
 * @brief Parse the sharded form when argv[1] is a shard count, else the single-index form.
 * @param argc argument count
 * @param argv arguments
 * @param cfg receives the options
 * @return false when the arguments do not fit either form
 */
static bool parse_cli(int argc, char** argv, CliConfig& cfg) {
    if (argc < 5) return false;

    int arg_idx = 1;
    if (is_positive_integer(argv[arg_idx])) {
        if (argc < 6) return false;
        cfg.total_shards = atoi(argv[arg_idx++]);
    } else {
        cfg.total_shards = 1;
    }
    cfg.data_files = split_file_list(argv[arg_idx++]);
    cfg.query_file = argv[arg_idx++];
    cfg.gt_file = argv[arg_idx++];
    cfg.codebook_files = split_file_list(argv[arg_idx++]);

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
    printf("GPU Beam Search Loading from:\nMode: %s\n", mode.c_str());
    printf("========================================\n");
    printf("Data file: %s\n", data_file);
    printf("Query file: %s\n", query_file);
    printf("Ground truth: %s\n", gt_file);
    printf("%s: %s\n", pathw ? "Graph" : "QG codebook", index_file);
    printf("K: %d\n", K);
    printf("Beam size: %d\n", beam_size);
    if (!pathw || degree_explicit) printf("Degree: %d\n", degree);
    else printf("Degree: from graph header\n");
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    printf("PathW signbit: %s\n", sign_file);
#else
    printf("Quantizer: %s, bits: %d\n", quant_name(g_quant_type), code_bits);
#endif
    if (!csv_file.empty()) printf("CSV output: %s\n", csv_file.c_str());

    g_metric_type = infer_metric_from_dataset_path(data_file);
    printf("Metric: %s\n", metric_name(g_metric_type));
    printf("========================================\n\n");

#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    IndexGraph<float> index;
    try {
        index.load_graph_index(index_file);
        index.load_data(data_file);
    } catch (const std::exception& error) {
        fprintf(stderr, "PathW input error: %s\n", error.what());
        return 1;
    }
    index.metric = g_metric_type;
    if (degree_explicit && index.maxDeg != degree) {
        fprintf(stderr, "Requested degree %d does not match graph degree %d\n", degree, index.maxDeg);
        return 1;
    }
    degree = index.maxDeg;
    size_t nq = 0, gt_nq = 0;
    int query_dim = 0, gt_k = 0;
    const std::vector<float> queries = IndexGraph<float>::load_queries(query_file, nq, query_dim);
    const std::vector<int> groundtruth = IndexGraph<float>::load_groundtruth(gt_file, gt_nq, gt_k);
    const int base_dim = index.d;
#else
    LoadedVectors<float> base = load_fvecs<float>(data_file, "data");
    LoadedVectors<float> query_vectors = load_fvecs<float>(query_file, "queries");
    LoadedVectors<int> groundtruth_vectors = load_ivecs(gt_file, "ground truth");
    const size_t nq = query_vectors.count;
    const int query_dim = query_vectors.dim;
    const size_t gt_nq = groundtruth_vectors.count;
    const int gt_k = groundtruth_vectors.dim;
    const int base_dim = base.dim;
    const std::vector<float>& queries = query_vectors.values;
    const std::vector<int>& groundtruth = groundtruth_vectors.values;
#endif
    if (query_dim != base_dim) {
        fprintf(stderr, "Query dimension %d does not match data dimension %d\n", query_dim, base_dim);
        return 1;
    }
    if (gt_nq != nq) {
        fprintf(stderr, "Ground-truth query count %zu does not match query count %zu\n", gt_nq, nq);
        return 1;
    }

    if (K < 1 || beam_size < 1 || degree < 1) {
        fprintf(stderr, "K, beam size and degree must be positive\n");
        return 1;
    }
    std::vector<vidType> results(nq * K);
    std::vector<uint32_t> iters(nq);
    double elapsed = 0.0;
#if GPU_SEARCH_MODE == GPU_SEARCH_MODE_PATHW
    try {
        index.search_pathw(static_cast<int>(nq), queries.data(), K, results.data(), beam_size,
                           sign_file, keep_ratio, prune_ratio, elapsed, iters.data(), repeat);
    } catch (const std::exception& error) {
        fprintf(stderr, "PathW error: %s\n", error.what());
        return 1;
    }

#else
    QuantizationGraph qg(base.count, base.dim, degree, index_file, g_quant_type, code_bits);
    qg.set_metric(g_metric_type);
    qg.gpu_search_adaptive(static_cast<int>(nq), queries.data(), K, results.data(), iters.data(),
                           repeat, beam_size, elapsed);
#endif

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
