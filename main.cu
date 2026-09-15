#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "include/data_io.hpp"
#include "include/metric.hpp"
#include "include/qg.hpp"
#include "include/quant.hpp"
#include "include/common.hpp"

/** SHAME(TALLFUNC) */
int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: %s <data.fvecs> <query.fvecs> <gt.ivecs> <qg_codebook> [K=100] [beam_size=128] [degree=32] [-quant rbq|tbq] [-bits 1|2|4] [-csv output.csv] [-iters output.txt] [-repeat n]\n", argv[0]);
        return 1;
    }

    const char* data_file = argv[1];
    const char* query_file = argv[2];
    const char* gt_file = argv[3];
    const char* qg_codebook = argv[4];

    int K = 100;
    int beam_size = 128;
    int degree = 32;
    int code_bits = 1;
    std::string csv_file;
    std::string iters_file;
    int repeat = 1;

    int arg_idx = 5;
    if (arg_idx < argc && argv[arg_idx][0] != '-') K = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') beam_size = atoi(argv[arg_idx++]);
    if (arg_idx < argc && argv[arg_idx][0] != '-') degree = atoi(argv[arg_idx++]);
    while (arg_idx < argc) {
        std::string arg = argv[arg_idx];
        if (arg == "-csv" && arg_idx + 1 < argc) {
            csv_file = argv[++arg_idx];
        } else if (arg == "-iters" && arg_idx + 1 < argc) {
            iters_file = argv[++arg_idx];
        } else if (arg == "-repeat" && arg_idx + 1 < argc) {
            repeat = atoi(argv[++arg_idx]);
            if (repeat < 1) repeat = 1;
        } else if (arg == "-quant" && arg_idx + 1 < argc) {
            g_quant_type = quant_parse(argv[++arg_idx]);
        } else if (arg == "-bits" && arg_idx + 1 < argc) {
            code_bits = atoi(argv[++arg_idx]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            return 1;
        }
        ++arg_idx;
    }
    printf("========================================\n");
    printf("FlashGANN AP Loading from:\n");
    printf("========================================\n");
    printf("Data file: %s\n", data_file);
    printf("Query file: %s\n", query_file);
    printf("Ground truth: %s\n", gt_file);
    printf("QG codebook: %s\n", qg_codebook);
    printf("K: %d\n", K);
    printf("Beam size: %d\n", beam_size);
    printf("Degree: %d\n", degree);
    printf("Quantizer: %s, bits: %d\n", quant_name(g_quant_type), code_bits);
    if (!csv_file.empty()) printf("CSV output: %s\n", csv_file.c_str());

    g_metric_type = infer_metric_from_dataset_path(data_file);
    printf("Metric: %s\n", metric_name(g_metric_type));
    printf("========================================\n\n");

    LoadedVectors<float> base = load_fvecs<float>(data_file, "data");
    LoadedVectors<float> query_vectors = load_fvecs<float>(query_file, "queries");
    LoadedVectors<int> groundtruth_vectors = load_ivecs(gt_file, "ground truth");

    const size_t nq = query_vectors.count;
    const int query_dim = query_vectors.dim;
    const size_t gt_nq = groundtruth_vectors.count;
    const int gt_k = groundtruth_vectors.dim;
    if (query_dim != base.dim) {
        fprintf(stderr, "Query dimension %d does not match data dimension %d\n", query_dim, base.dim);
        return 1;
    }
    if (gt_nq != nq) {
        fprintf(stderr, "Ground-truth query count %zu does not match query count %zu\n", gt_nq, nq);
        return 1;
    }

    const std::vector<float>& queries = query_vectors.values;
    const std::vector<int>& groundtruth = groundtruth_vectors.values;

    std::vector<vidType> results(nq * K);
    std::vector<uint32_t> iters(nq);
    double elapsed = 0.0;
    QuantizationGraph qg(base.count, base.dim, degree, qg_codebook, g_quant_type, code_bits);
    qg.set_metric(g_metric_type);
    qg.gpu_search_adaptive(static_cast<int>(nq), queries.data(), K, results.data(), iters.data(),
                           repeat, beam_size, elapsed);

    const float recall = compute_recall_dedup(results.data(), groundtruth.data(), nq, K, gt_k) * 100.0f;
    const double latency_ms = (nq > 0) ? (elapsed * 1000.0 / static_cast<double>(nq)) : 0.0;
    const double qps = (elapsed > 0.0) ? (static_cast<double>(nq) / elapsed) : 0.0;

    printf("\n========================================\n");
    printf("Results\n");
    printf("========================================\n");
    printf("Total time: %.6f ms\n", elapsed * 1000.0);
    printf("Throughput: %.6f queries/sec\n", qps);
    printf("Avg latency: %.6f ms/query\n", latency_ms);
    printf("Recall@%d: %.6f\n", K, recall);
    printf("========================================\n");

    RunStats run_stats;
    run_stats.runtime = elapsed;
    run_stats.latency = latency_ms;
    run_stats.throughput = qps;
    run_stats.recall = recall;

    if (!csv_file.empty()) {
        append_run_stats_to_csv(csv_file, K, beam_size, run_stats);
        printf("Saved GPU stats CSV row to: %s\n", csv_file.c_str());
    }

    if (!iters_file.empty()) {
        FILE* fout = fopen(iters_file.c_str(), "w");
        if (fout == nullptr) {
            fprintf(stderr, "Cannot open iteration output: %s\n", iters_file.c_str());
            return 1;
        }
        for (uint32_t count : iters) fprintf(fout, "%u\n", count);
        fclose(fout);
        printf("Saved per-query iterations to: %s\n", iters_file.c_str());
    }

    return 0;
}
