#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../include/common.hpp"
#include "../include/data_io.hpp"
#include "../include/pack.hpp"
#include "../include/quant.hpp"
#include "../include/rotator.hpp"

/** @brief Shape of the index being emitted. */
struct BuildSpec {
    size_t numnode;
    size_t rawdim;
    size_t paddim;
    int degree;
    int codebit;
    QuantType qtype;
};

/**
 * @brief Float offsets of the interleaved row layout consumed by QuantizationGraph.
 * @param spec index shape
 * @param codeoff destination for the packed code offset
 * @param facoff destination for the factor block offset
 * @param nbroff destination for the neighbor id offset
 * @return floats per row
 */
static size_t layout_offsets(const BuildSpec& spec, size_t* codeoff, size_t* facoff, size_t* nbroff) {
    *codeoff = spec.rawdim;
    *facoff = *codeoff + quant_words(spec.paddim, spec.codebit) * static_cast<size_t>(spec.degree);
    *nbroff = *facoff + 3 * static_cast<size_t>(spec.degree);
    return *nbroff + static_cast<size_t>(spec.degree);
}

/**
 * @brief Read the 8-byte header adjacency dump produced by the graph builder.
 * @param path graph file
 * @param spec index shape, validated against the header
 * @param edges destination for numnode * degree neighbor ids
 */
static void read_graph(const std::string& path, const BuildSpec& spec, std::vector<uint32_t>& edges) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) throw std::runtime_error("Cannot open graph file: " + path);

    uint32_t ntotal = 0;
    uint32_t degree = 0;
    fin.read(reinterpret_cast<char*>(&ntotal), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
    if (ntotal != spec.numnode) throw std::runtime_error("Graph node count mismatch");
    if (degree != static_cast<uint32_t>(spec.degree)) throw std::runtime_error("Graph degree mismatch");

    edges.resize(spec.numnode * static_cast<size_t>(spec.degree));
    fin.read(reinterpret_cast<char*>(edges.data()), sizeof(uint32_t) * edges.size());
    if (!fin) throw std::runtime_error("Truncated graph file: " + path);
}

/**
 * @brief Encode one parent's neighbor list with the 1-bit RaBitQ codebook.
 * @param spec index shape
 * @param rotu rotated parent vector
 * @param rotv rotated neighbor vectors, degree by paddim
 * @param codes scratch of paddim per-dimension codes
 * @param row destination row, positioned at the parent
 * @param codeoff packed code offset within the row
 * @param facoff factor block offset within the row
 */
static void encode_rbq(const BuildSpec& spec, const float* rotu, const float* rotv, uint8_t* codes,
                       float* row, size_t codeoff, size_t facoff) { // SHAME(MANYARG)
    const size_t words = quant_words(spec.paddim, spec.codebit);
    const float facnorm = 1.0f / std::sqrt(static_cast<float>(spec.paddim));
    const float fhtfix = 1.0f / std::sqrt(static_cast<float>(spec.paddim));

    for (int j = 0; j < spec.degree; ++j) {
        const float* vec = rotv + static_cast<size_t>(j) * spec.paddim;
        int binsum = 0;
        float sum0 = 0.0f, sum1 = 0.0f, normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float r = vec[k] - rotu[k];
            const float sx = (r > 0.0f) ? 1.0f : -1.0f;
            codes[k] = (r > 0.0f) ? 1 : 0;
            if (r > 0.0f) ++binsum;
            sum0 += r * sx * facnorm;
            sum1 += rotu[k] * sx;
            normsq += r * r;
        }

        const float xnorm = std::sqrt(normsq);
        const float facx0 = (xnorm > 0.0f) ? (sum0 / xnorm) : 1.0f;
        const float facx1 = sum1 * facnorm;
        const float xx0 = xnorm / facx0;

        row[facoff + j] = xnorm * xnorm + 2.0f * xx0 * facx1;
        row[facoff + spec.degree + j] = -2.0f * xx0 * facnorm * fhtfix;
        row[facoff + 2 * spec.degree + j] =
            -2.0f * xx0 * facnorm * fhtfix * static_cast<float>(binsum * 2 - int(spec.paddim));

        pack_codes(spec.paddim, spec.codebit, codes,
                   reinterpret_cast<uint8_t*>(row + codeoff + static_cast<size_t>(j) * words));
    }
}

/**
 * @brief Encode one parent's neighbor list with a TurboQuant level table.
 * @param spec index shape
 * @param rotu rotated parent vector
 * @param rotv rotated neighbor vectors, degree by paddim
 * @param levels ascending reconstruction levels of the unit-norm residual
 * @param codes scratch of paddim per-dimension codes
 * @param row destination row, positioned at the parent
 * @param codeoff packed code offset within the row
 * @param facoff factor block offset within the row
 */
static void encode_tbq(const BuildSpec& spec, const float* rotu, const float* rotv,
                       const std::vector<float>& levels, uint8_t* codes, float* row,
                       size_t codeoff, size_t facoff) { // SHAME(MANYARG)
    const size_t words = quant_words(spec.paddim, spec.codebit);
    const float crange = levels.back() - levels.front();
    const float fhtfix = 1.0f / std::sqrt(static_cast<float>(spec.paddim));
    const float gain = static_cast<float>(spec.codebit);

    for (int j = 0; j < spec.degree; ++j) {
        const float* vec = rotv + static_cast<size_t>(j) * spec.paddim;
        float normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float r = vec[k] - rotu[k];
            normsq += r * r;
        }
        const float xnorm = std::sqrt(normsq);
        const float inv = (xnorm > 0.0f) ? (1.0f / xnorm) : 0.0f;

        float sumhat = 0.0f, ipu = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float y = (vec[k] - rotu[k]) * inv;
            codes[k] = quantize_level(levels, y);
            const float hat = levels[codes[k]];
            sumhat += hat;
            ipu += rotu[k] * hat;
        }

        row[facoff + j] = xnorm * xnorm + 2.0f * xnorm * ipu;
        row[facoff + spec.degree + j] = -xnorm * crange * fhtfix / gain;
        row[facoff + 2 * spec.degree + j] = -2.0f * xnorm * sumhat * fhtfix;

        pack_codes(spec.paddim, spec.codebit, codes,
                   reinterpret_cast<uint8_t*>(row + codeoff + static_cast<size_t>(j) * words));
    }
}

/**
 * @brief Write the interleaved index plus its sign vector and quantizer tail.
 * @param path destination file
 * @param spec index shape
 * @param rows numnode rows of rowoff floats
 * @param rowoff floats per row
 * @param flips paddim sign values
 * @param levels reconstruction levels, normalized for the scan lookup table
 * @param entry entry point node id
 */
static void write_index(const std::string& path, const BuildSpec& spec, const float* rows, size_t rowoff,
                        const float* flips, const std::vector<float>& levels, uint32_t entry) { // SHAME(MANYARG)
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("Cannot open output index: " + path);

    fout.write(reinterpret_cast<const char*>(&entry), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(rows), sizeof(float) * spec.numnode * rowoff);
    fout.write(reinterpret_cast<const char*>(flips), sizeof(float) * spec.paddim);

    const int32_t qtag = static_cast<int32_t>(spec.qtype);
    const int32_t qbits = static_cast<int32_t>(spec.codebit);
    const int32_t nlevel = static_cast<int32_t>(levels.size());
    fout.write(reinterpret_cast<const char*>(&qtag), sizeof(int32_t));
    fout.write(reinterpret_cast<const char*>(&qbits), sizeof(int32_t));
    fout.write(reinterpret_cast<const char*>(&nlevel), sizeof(int32_t));
    if (nlevel > 0) fout.write(reinterpret_cast<const char*>(levels.data()), sizeof(float) * levels.size());
    if (!fout) throw std::runtime_error("Error writing index: " + path);
}

/** SHAME(TALLFUNC) */
int main(int argc, char** argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <base.fvecs> <graph> <out.index> <rbq|tbq> <bits> [levels.bin] [degree=32] [seed=1]\n",
                argv[0]);
        return 1;
    }

    const std::string base_file = argv[1];
    const std::string graph_file = argv[2];
    const std::string out_file = argv[3];
    const QuantType qtype = quant_parse(argv[4]);
    const int codebit = atoi(argv[5]);
    const std::string level_file = (argc >= 7 && argv[6][0] != '-') ? argv[6] : std::string();
    const int degree = (argc >= 8) ? atoi(argv[7]) : 32;
    const uint32_t seed = (argc >= 9) ? static_cast<uint32_t>(atoi(argv[8])) : 1u;

    if (!quant_valid(codebit)) {
        fprintf(stderr, "bits must be one of 1, 2, or 4\n");
        return 1;
    }
    if (qtype == QUANT_RBQ && codebit != 1) {
        fprintf(stderr, "rbq supports bits=1 only\n");
        return 1;
    }

    LoadedVectors<float> base = load_fvecs<float>(base_file.c_str(), "data");

    BuildSpec spec;
    spec.numnode = base.count;
    spec.rawdim = static_cast<size_t>(base.dim);
    spec.degree = degree;
    spec.codebit = codebit;
    spec.qtype = qtype;

    FhtRotator rotator(spec.rawdim, seed);
    spec.paddim = rotator.get_paddim();

    std::vector<float> levels;
    if (qtype == QUANT_TBQ) {
        if (level_file.empty()) {
            fprintf(stderr, "tbq requires a level file\n");
            return 1;
        }
        read_levels(level_file, codebit, levels);
    }

    size_t codeoff = 0, facoff = 0, nbroff = 0;
    const size_t rowoff = layout_offsets(spec, &codeoff, &facoff, &nbroff);
    printf("Building %s index: nodes=%zu dim=%zu paddim=%zu degree=%d bits=%d row_offset=%zu\n",
           quant_name(qtype), spec.numnode, spec.rawdim, spec.paddim, degree, codebit, rowoff);

    std::vector<uint32_t> edges;
    read_graph(graph_file, spec, edges);

    printf("Rotating base vectors...\n");
    std::vector<float> rotated(spec.numnode * spec.paddim);
#pragma omp parallel for schedule(static)
    for (size_t u = 0; u < spec.numnode; ++u) {
        rotator.rotate(base.values.data() + u * spec.rawdim, rotated.data() + u * spec.paddim);
    }

    printf("Encoding neighbor codes...\n");
    std::vector<float> rows(spec.numnode * rowoff, 0.0f);
#pragma omp parallel
    {
        std::vector<float> rotv(static_cast<size_t>(degree) * spec.paddim);
        std::vector<uint8_t> codes(spec.paddim);
#pragma omp for schedule(static)
        for (size_t u = 0; u < spec.numnode; ++u) {
            float* row = rows.data() + u * rowoff;
            memcpy(row, base.values.data() + u * spec.rawdim, sizeof(float) * spec.rawdim);

            uint32_t* nbr = reinterpret_cast<uint32_t*>(row + nbroff);
            for (int j = 0; j < degree; ++j) {
                const uint32_t v = edges[u * static_cast<size_t>(degree) + j];
                nbr[j] = v;
                memcpy(rotv.data() + static_cast<size_t>(j) * spec.paddim,
                       rotated.data() + static_cast<size_t>(v) * spec.paddim,
                       sizeof(float) * spec.paddim);
            }

            const float* rotu = rotated.data() + u * spec.paddim;
            if (qtype == QUANT_TBQ) {
                encode_tbq(spec, rotu, rotv.data(), levels, codes.data(), row, codeoff, facoff);
            } else {
                encode_rbq(spec, rotu, rotv.data(), codes.data(), row, codeoff, facoff);
            }
        }
    }

    printf("Selecting entry point...\n");
    const vidType entry = compute_rabitq_entry_point(rows.data(), spec.numnode, spec.rawdim, rowoff);

    std::vector<float> lut_levels;
    if (qtype == QUANT_TBQ) {
        const float crange = levels.back() - levels.front();
        lut_levels.resize(levels.size());
        for (size_t k = 0; k < levels.size(); ++k) lut_levels[k] = (levels[k] - levels.front()) / crange;
    }

    printf("Writing %s\n", out_file.c_str());
    write_index(out_file, spec, rows.data(), rowoff, rotator.get_flips(), lut_levels, entry);
    printf("Done. entry_point=%u\n", entry);
    return 0;
}
