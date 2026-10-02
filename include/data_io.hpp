#pragma once

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <new>
#include <cstddef>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

/**
 * @brief Row-major vectors read by load_fvecs or load_ivecs.
 * @tparam T element type
 */
template <typename T>
struct LoadedVectors {
    size_t count = 0;        // number of rows
    int dim = 0;             // values per row
    std::vector<T> values;   // count x dim row-major values
};

/**
 * @brief Read a .fvecs-style file, each row an int32 dimension followed by that many T values.
 *
 * main.cu loads queries and FlashGANN/RaBitQ base data with it. The row count comes from the
 * file size and the first row's dimension; the process exits when the file cannot be opened.
 *
 * @tparam T element type
 * @param filename input path
 * @param label name printed in the load message
 * @return the loaded rows
 */
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

/**
 * @brief Read a .ivecs file, each row an int32 dimension followed by that many int32 values.
 *
 * main.cu loads the ground truth with it; behavior otherwise matches load_fvecs.
 *
 * @param filename input path
 * @param label name printed in the load message
 * @return the loaded rows
 */
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

/**
 * @brief Read a .bin matrix: int32 row count, int32 row width, then the row-major T payload.
 *
 * Ported from beam_search_collab. Loads PathW/CAGRA graphs and PathW signbit files. The buffer
 * is 64- or 32-byte aligned when the row size allows, so free it with release_bin_buffer.
 *
 * @tparam T element type
 * @param filename input path
 * @param n_out receives the row count
 * @param d_out receives the row width
 * @return the allocated payload; throws on open failure, bad header or size mismatch
 */
template <typename T = float>
T* read_bin(const char* filename, size_t& n_out, size_t& d_out) {
    std::ifstream input(filename, std::ios::binary);
    printf("Reading bin file: %s\n", filename);
    if (!input) throw std::runtime_error("Cannot open bin file");
    int32_t num = 0, dim = 0;
    input.read(reinterpret_cast<char*>(&num), sizeof(int32_t));
    input.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
    if (num <= 0 || dim <= 0)
        throw std::runtime_error("Invalid dimensions read from file");
    const auto payload_start = input.tellg();
    input.seekg(0, std::ios::end);
    const uint64_t expected_bytes = 2 * sizeof(int32_t) + static_cast<uint64_t>(num) * dim * sizeof(T);
    if (static_cast<uint64_t>(input.tellg()) != expected_bytes)
        throw std::runtime_error("Binary input payload size mismatch: " + std::string(filename));
    input.seekg(payload_start);
    size_t total = static_cast<size_t>(num) * dim;
    size_t row_bytes = dim * sizeof(T);
    T *data;
    if (row_bytes % 64 == 0)
        data = static_cast<T*>(std::aligned_alloc(64, sizeof(T) * total));
    else if (row_bytes % 32 == 0)
        data = static_cast<T*>(std::aligned_alloc(32, sizeof(T) * total));
    else
        data = new T[total];
    if (!data) throw std::bad_alloc();
    input.read(reinterpret_cast<char*>(data), sizeof(T) * total);
    if (!input) throw std::runtime_error("Error reading bin vector data");
    n_out = size_t(num);
    d_out = size_t(dim);
    input.close();
    return data;
}

/**
 * @brief Free a read_bin buffer with the deallocator matching its allocation.
 * @tparam T element type
 * @param ptr buffer from read_bin, or nullptr
 * @param row_dim row width that read_bin returned in d_out
 */
template <typename T>
void release_bin_buffer(T* ptr, size_t row_dim) {
    if (ptr == nullptr) return;
    const size_t row_bytes = row_dim * sizeof(T);
    if (row_bytes % 64 == 0 || row_bytes % 32 == 0) {
        std::free(ptr);
    } else {
        delete[] ptr;
    }
}
