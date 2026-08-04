#pragma once

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

template <typename T>
struct LoadedVectors {
    size_t count = 0;
    int dim = 0;
    std::vector<T> values;
};

template <typename T>
LoadedVectors<T> load_fvecs(const std::string& filename, const char* label) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", filename.c_str());
        exit(1);
    }

    int dim = 0;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int));
    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    const size_t count = file_size / (sizeof(int) + static_cast<size_t>(dim) * sizeof(T));

    LoadedVectors<T> vectors;
    vectors.count = count;
    vectors.dim = dim;
    vectors.values.resize(count * static_cast<size_t>(dim));

    f.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int row_dim = 0;
        f.read(reinterpret_cast<char*>(&row_dim), sizeof(int));
        assert(row_dim == dim);
        f.read(reinterpret_cast<char*>(&vectors.values[i * static_cast<size_t>(dim)]), static_cast<size_t>(dim) * sizeof(T));
    }

    printf("Loading %s: n=%zu, dim=%d\n", label, count, dim);
    return vectors;
}

inline LoadedVectors<int> load_ivecs(const std::string& filename, const char* label) {
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", filename.c_str());
        exit(1);
    }

    int dim = 0;
    f.read(reinterpret_cast<char*>(&dim), sizeof(int));
    f.seekg(0, std::ios::end);
    const size_t file_size = static_cast<size_t>(f.tellg());
    const size_t count = file_size / (sizeof(int) + static_cast<size_t>(dim) * sizeof(int));

    LoadedVectors<int> vectors;
    vectors.count = count;
    vectors.dim = dim;
    vectors.values.resize(count * static_cast<size_t>(dim));

    f.seekg(0, std::ios::beg);
    for (size_t i = 0; i < count; ++i) {
        int row_dim = 0;
        f.read(reinterpret_cast<char*>(&row_dim), sizeof(int));
        assert(row_dim == dim);
        f.read(reinterpret_cast<char*>(&vectors.values[i * static_cast<size_t>(dim)]), static_cast<size_t>(dim) * sizeof(int));
    }

    printf("Loading %s: n=%zu, dim=%d\n", label, count, dim);
    return vectors;
}
