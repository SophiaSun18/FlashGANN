#pragma once
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <string>
#include "metric.hpp"
#include "data_io.hpp"
#include "common.hpp"

/** @brief Vertex ID type of IndexGraph edges and results. */
typedef uint32_t vid_t;

/**
 * @brief Host graph index with fixed-degree adjacency and raw vectors, for the PathW and CAGRA modes.
 *
 * main.cu fills one per shard with load_graph_index and load_data, sets metric, then calls
 * search_pathw or search_cagra.
 *
 * @tparam T vector element type
 */
template <typename T>
class IndexGraph {
public:
    size_t ntotal;           // number of data points
    int d;                   // vector dimension
    int maxDeg;              // maximum degree
    MetricType metric;       // distance metric type
    std::vector<vid_t> edges; // adjacency list (ntotal * maxDeg)
    std::vector<T> data;     // data vectors (ntotal * d)

    /** @brief Construct an empty index with METRIC_L2. */
    IndexGraph() : ntotal(0), d(0), maxDeg(0), metric(METRIC_L2) {}

    /** @brief Return the row-major data vectors. */
    T* get_data_ptr() { return data.data(); }

    /**
     * @brief Load the adjacency from a read_bin graph file, setting ntotal and maxDeg.
     * @param gFile graph file path
     */
    void load_graph_index(const char* gFile) {
        size_t nt = 0, deg = 0;
        vid_t* edge_data = read_bin<vid_t>(gFile, nt, deg);

        ntotal = nt;
        maxDeg = deg;
        edges.assign(edge_data, edge_data + ntotal * maxDeg);

        std::cout << "Loading graph index from " << gFile 
                  << ": ntotal = " << ntotal << ", maxDeg = " << maxDeg << std::endl;

        release_bin_buffer(edge_data, deg);
    }

    /**
     * @brief Load the adjacency from a <prefix>.meta.txt header and a <prefix>.edge.bin edge array.
     *
     * Sets ntotal, d and maxDeg from the meta file; exits when either file cannot be opened.
     *
     * @param prefix path prefix of the two files
     */
    void load_graph(const std::string& prefix) {
        // [1] read the meta file
        std::string meta_file = prefix + ".meta.txt";
        std::ifstream f_meta(meta_file);
        if (!f_meta.is_open()) {
            std::cerr << "Error: cannot open " << meta_file << std::endl;
            exit(1);
        }
        
        int64_t nv, ne;
        int vid_size, eid_size, vlabel_size, elabel_size, feat_len, nvc, nec;
        f_meta >> nv >> ne >> vid_size >> eid_size >> vlabel_size >> elabel_size
               >> maxDeg >> feat_len >> nvc >> nec;
        f_meta.close();
        
        ntotal = nv;
        d = feat_len;
        
        std::cout << "Loading graph: n=" << ntotal << ", maxDeg=" << maxDeg << std::endl;
        
        // [2] read ntotal * maxDeg edges
        std::string edge_file = prefix + ".edge.bin";
        std::ifstream f_edge(edge_file, std::ios::binary);
        if (!f_edge.is_open()) {
            std::cerr << "Error: cannot open " << edge_file << std::endl;
            exit(1);
        }
        
        edges.resize(ntotal * maxDeg);
        f_edge.read(reinterpret_cast<char*>(edges.data()), 
                    sizeof(vid_t) * ntotal * maxDeg);
        f_edge.close();
        
        std::cout << "Graph loaded: " << edges.size() << " edges" << std::endl;
    }

    /**
     * @brief Load the data vectors from a .fvecs file into data.
     *
     * Throws when a graph is already loaded with a different row count; sets ntotal and d when unset.
     *
     * @param data_file .fvecs path
     */
    void load_data(const std::string& data_file) {
        std::ifstream f(data_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << data_file << std::endl;
            exit(1);
        }
        
        // [1] row count from the first dimension and the file size
        int dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), sizeof(int));
        
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t n = file_size / (sizeof(int) + dim_check * sizeof(T));
        
        // [2] check against a loaded graph
        if (ntotal > 0 && n != ntotal) {
            throw std::runtime_error("PathW graph/data row count mismatch");
        }
        
        if (ntotal == 0) ntotal = n;
        if (d == 0) d = dim_check;
        
        std::cout << "Loading data: n=" << n << ", dim=" << d << std::endl;
        
        // [3] read all rows
        data.resize(ntotal * d);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < ntotal; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == d);
            f.read(reinterpret_cast<char*>(&data[i * d]), d * sizeof(T));
        }
        f.close();
        
    }

    /**
     * @brief Read query vectors from a .fvecs file; exits when the file cannot be opened.
     * @param query_file .fvecs path
     * @param nq receives the query count
     * @param dim receives the dimension
     * @return nq x dim row-major queries
     */
    static std::vector<T> load_queries(const std::string& query_file, size_t& nq, int& dim) {
        std::ifstream f(query_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << query_file << std::endl;
            exit(1);
        }
        
        // [1] query count from the first dimension and the file size
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + dim * sizeof(T));
        
        std::cout << "Loading queries: nq=" << nq << ", dim=" << dim << std::endl;
        
        // [2] read all rows
        std::vector<T> queries(nq * dim);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < nq; i++) {
            int dim_i;
            f.read(reinterpret_cast<char*>(&dim_i), sizeof(int));
            assert(dim_i == dim);
            f.read(reinterpret_cast<char*>(&queries[i * dim]), dim * sizeof(T));
        }
        f.close();
        
        return queries;
    }

    /**
     * @brief Read ground-truth IDs from a .ivecs file; exits when the file cannot be opened.
     * @param gt_file .ivecs path
     * @param nq receives the query count
     * @param k receives the IDs per query
     * @return nq x k row-major IDs
     */
    static std::vector<int> load_groundtruth(const std::string& gt_file, size_t& nq, int& k) {
        std::ifstream f(gt_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << gt_file << std::endl;
            exit(1);
        }
        
        // [1] query count from the first row length and the file size
        f.read(reinterpret_cast<char*>(&k), sizeof(int));
        
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + k * sizeof(int));
        
        // [2] read all rows
        std::vector<int> gt(nq * k);
        f.seekg(0, std::ios::beg);
        
        for (size_t i = 0; i < nq; i++) {
            int k_i;
            f.read(reinterpret_cast<char*>(&k_i), sizeof(int));
            f.read(reinterpret_cast<char*>(&gt[i * k]), k * sizeof(int));
        }
        f.close();

        std::cout << "Loading ground truth: nq=" << nq << ", k=" << k << std::endl;
        
        return gt;
    }

    /**
     * @brief Run cuVS CAGRA search over this graph and data, implemented in gpu_search_cagra.cu.
     * @param nq number of queries
     * @param queries nq x d host queries
     * @param K results per query, at most beam_sz
     * @param result_idx nq x K host output IDs
     * @param result_dist nq x K host output distances, smaller is closer
     * @param beam_sz CAGRA itopk_size
     * @param search_width CAGRA search_width
     * @param max_iterations CAGRA max_iterations
     * @param algo auto, single, multi or kernel
     * @param elapsed QG_SHARD_TIMER_COUNT timer slots, seconds
     * @param iters nq host outputs set to 0, or nullptr
     * @param repeat timed search runs to average
     */
    void search_cagra(int nq, const T* queries, int K, vid_t* result_idx, float* result_dist,
                      int beam_sz, int search_width, int max_iterations, const std::string& algo,
                      double* elapsed, uint32_t* iters = nullptr, int repeat = 1);

    /**
     * @brief Run the GPU PathWeaver-style beam search over this graph, implemented in gpu_search_pathw.cu.
     * @param nq number of queries
     * @param queries nq x d host queries
     * @param K results per query, at most beam_sz
     * @param result_idx nq x K host output IDs
     * @param result_dist nq x K host output distances, smaller is closer
     * @param beam_sz beam size, also the iteration limit
     * @param signbit_file read_bin file of ntotal x (maxDeg * ceil(d / 32)) sign words
     * @param neighbor_keep_ratio fraction of neighbors kept by sign-bit pruning; pruning is off unless in (0, 1)
     * @param iteration_prune_ratio fraction of the iteration limit that uses pruning
     * @param elapsed QG_SHARD_TIMER_COUNT timer slots, seconds
     * @param iters nq host outputs of expanded node counts, or nullptr
     * @param repeat timed kernel launches to average
     */
    void search_pathw(int nq, const T* queries, int K, vid_t* result_idx, float* result_dist,
                      int beam_sz, const char* signbit_file, float neighbor_keep_ratio,
                      float iteration_prune_ratio, double* elapsed,
                      uint32_t* iters = nullptr, int repeat = 1);
};
