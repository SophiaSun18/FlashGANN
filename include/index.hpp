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

typedef uint32_t vid_t;

template <typename T>
class IndexGraph {
public:
    size_t ntotal;           // number of data points
    int d;                   // vector dimension
    int maxDeg;              // maximum degree
    MetricType metric;       // distance metric type
    std::vector<vid_t> edges; // adjacency list (ntotal * maxDeg)
    std::vector<T> data;     // data vectors (ntotal * d)

    IndexGraph() : ntotal(0), d(0), maxDeg(0), metric(METRIC_L2) {}

    // Get data pointer
    T* get_data_ptr() { return data.data(); }

    // Load graph file (.bin built from ScaleANN)
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

    // Load graph file (.meta.txt + .edge.bin format)
    void load_graph(const std::string& prefix) {
        // Read meta file
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
        
        // Read edge file
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

    // Load data vectors (.fvecs format)
    void load_data(const std::string& data_file) {
        std::ifstream f(data_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << data_file << std::endl;
            exit(1);
        }
        
        // Read first dimension
        int dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), sizeof(int));
        
        // Get file size to calculate number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        size_t n = file_size / (sizeof(int) + dim_check * sizeof(T));
        
        // Verify count matches if graph is already loaded
        if (ntotal > 0 && n != ntotal) {
            throw std::runtime_error("PathW graph/data row count mismatch");
        }
        
        if (ntotal == 0) ntotal = n;
        if (d == 0) d = dim_check;
        
        std::cout << "Loading data: n=" << n << ", dim=" << d << std::endl;
        
        // Read all vectors
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

    // Load query vectors
    static std::vector<T> load_queries(const std::string& query_file, size_t& nq, int& dim) {
        std::ifstream f(query_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << query_file << std::endl;
            exit(1);
        }
        
        // Read dimension
        f.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        // Get number of vectors
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + dim * sizeof(T));
        
        std::cout << "Loading queries: nq=" << nq << ", dim=" << dim << std::endl;
        
        // Read all queries
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

    // Load ground truth
    static std::vector<int> load_groundtruth(const std::string& gt_file, size_t& nq, int& k) {
        std::ifstream f(gt_file, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Error: cannot open " << gt_file << std::endl;
            exit(1);
        }
        
        // Read k
        f.read(reinterpret_cast<char*>(&k), sizeof(int));
        
        // Get number of queries
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        nq = file_size / (sizeof(int) + k * sizeof(int));
        
        // Read all ground truth
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

    // cuVS CAGRA search over this graph, implemented in gpu_search_cagra.cu.
    void search_cagra(int nq, const T* queries, int K, vid_t* result_idx, float* result_dist,
                      int beam_sz, int search_width, int max_iterations, const std::string& algo,
                      double* elapsed, uint32_t* iters = nullptr, int repeat = 1);

    // GPU PathWeaver-style search, implemented in gpu_search_pathw.cu.
    void search_pathw(int nq, const T* queries, int K, vid_t* result_idx, float* result_dist,
                      int beam_sz, const char* signbit_file, float neighbor_keep_ratio,
                      float iteration_prune_ratio, double* elapsed,
                      uint32_t* iters = nullptr, int repeat = 1);
};
