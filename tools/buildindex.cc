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
#include "../include/sketch.hpp"

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
static size_t layout_offsets(const BuildSpec& spec, size_t* codeoff, size_t* signoff,
                             size_t* facoff, size_t* nbroff) { // SHAME(MANYARG)
    const bool prod = spec.qtype == QUANT_TBQ;
    const int stage = prod ? quant_stage(spec.codebit) : spec.codebit;
    const size_t deg = static_cast<size_t>(spec.degree);
    *codeoff = spec.rawdim;
    *signoff = *codeoff + quant_words(spec.paddim, stage) * deg;
    *facoff = *signoff + (prod ? quant_words(spec.paddim, 1) * deg : 0);
    *nbroff = *facoff + static_cast<size_t>(quant_factors(spec.qtype)) * deg;
    return *nbroff + deg;
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
static void encode_tbq(const BuildSpec& spec, const FastfoodSketch& sketch, float kappa,
                       const float* rotu, const float* sku, const float* rotv,
                       const std::vector<float>& levels, uint8_t* codes, uint8_t* signs,
                       float* resid, float* proj, float* row,
                       size_t codeoff, size_t signoff, size_t facoff) { // SHAME(MANYARG) SHAME(TALLFUNC)
    const int stage = quant_stage(spec.codebit);
    const size_t words = quant_words(spec.paddim, stage);
    const size_t swords = quant_words(spec.paddim, 1);
    const float crange = levels.back() - levels.front();
    const float fhtfix = 1.0f / std::sqrt(static_cast<float>(spec.paddim));
    const float gain = static_cast<float>(stage);
    const int deg = spec.degree;

    for (int j = 0; j < deg; ++j) {
        const float* vec = rotv + static_cast<size_t>(j) * spec.paddim;

        // [1] residual against the parent and its norm
        float normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            resid[k] = vec[k] - rotu[k];
            normsq += resid[k] * resid[k];
        }
        const float xnorm = std::sqrt(normsq);
        const float inv = (xnorm > 0.0f) ? (1.0f / xnorm) : 0.0f;

        // [2] MSE stage over the unit residual
        float sumhat = 0.0f, ipu = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            codes[k] = quantize_level(levels, resid[k] * inv);
            const float hat = levels[codes[k]];
            sumhat += hat;
            ipu += rotu[k] * hat;
            resid[k] -= xnorm * hat;
        }

        // [3] QJL sign stage over what the MSE stage left behind
        float mnorm = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) mnorm += resid[k] * resid[k];
        mnorm = std::sqrt(mnorm);
        sketch.apply(resid, proj);

        float sumsig = 0.0f, ipsk = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            const float s = (proj[k] > 0.0f) ? 1.0f : -1.0f;
            signs[k] = (proj[k] > 0.0f) ? 1 : 0;
            sumsig += s;
            ipsk += sku[k] * s;
        }

        // the search sketches the unnormalized rotated query, so both query-side QJL
        // coefficients carry the same 1/sqrt(paddim) the MSE stage coefficients do
        const float qscale = kappa * mnorm;
        row[facoff + j] = xnorm * xnorm + 2.0f * xnorm * ipu + 2.0f * qscale * ipsk;
        row[facoff + deg + j] = -xnorm * crange * fhtfix / gain;
        row[facoff + 2 * deg + j] = -2.0f * xnorm * sumhat * fhtfix;
        row[facoff + 3 * deg + j] = -2.0f * qscale * fhtfix;
        row[facoff + 4 * deg + j] = -2.0f * qscale * sumsig * fhtfix;

        pack_codes(spec.paddim, stage, codes,
                   reinterpret_cast<uint8_t*>(row + codeoff + static_cast<size_t>(j) * words));
        pack_codes(spec.paddim, 1, signs,
                   reinterpret_cast<uint8_t*>(row + signoff + static_cast<size_t>(j) * swords));
    }
}

/**
 * @brief Fit the QJL scale that makes the sign estimator match exact inner products.
 *
 * The sqrt(pi/2)/d constant assumes exactly Gaussian rows, which the Fastfood sketch only
 * approximates, so the scale is least-squares fitted on sampled residuals against random
 * query directions.
 *
 * @param spec index shape
 * @param sketch the sketch the index will store
 * @param levels MSE stage reconstruction levels
 * @param rotated rotated base vectors
 * @param edges adjacency, numnode by degree
 * @param samples number of parent nodes to draw
 * @return the fitted scale
 */
static float calibrate_kappa(const BuildSpec& spec, const FastfoodSketch& sketch,
                             const std::vector<float>& levels, const float* rotated,
                             const std::vector<uint32_t>& edges, size_t samples) { // SHAME(MANYARG)
    std::mt19937 gen(20260912u);
    std::uniform_int_distribution<size_t> pick(0, spec.numnode - 1);
    std::normal_distribution<float> gauss(0.0f, 1.0f);

    std::vector<float> resid(spec.paddim), proj(spec.paddim), query(spec.paddim), skq(spec.paddim);
    double num = 0.0, den = 0.0;

    for (size_t t = 0; t < samples; ++t) {
        const size_t u = pick(gen);
        const uint32_t v = edges[u * static_cast<size_t>(spec.degree) + (t % spec.degree)];
        const float* rotu = rotated + u * spec.paddim;
        const float* rotv = rotated + static_cast<size_t>(v) * spec.paddim;

        float normsq = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) {
            resid[k] = rotv[k] - rotu[k];
            normsq += resid[k] * resid[k];
        }
        const float xnorm = std::sqrt(normsq);
        if (xnorm <= 0.0f) continue;
        for (size_t k = 0; k < spec.paddim; ++k) {
            resid[k] -= xnorm * levels[quantize_level(levels, resid[k] / xnorm)];
        }

        float mnorm = 0.0f;
        for (size_t k = 0; k < spec.paddim; ++k) mnorm += resid[k] * resid[k];
        mnorm = std::sqrt(mnorm);
        if (mnorm <= 0.0f) continue;

        sketch.apply(resid.data(), proj.data());
        for (size_t k = 0; k < spec.paddim; ++k) query[k] = gauss(gen);
        sketch.apply(query.data(), skq.data());

        double exact = 0.0, sign_ip = 0.0;
        for (size_t k = 0; k < spec.paddim; ++k) {
            exact += static_cast<double>(query[k]) * resid[k];
            sign_ip += static_cast<double>(skq[k]) * ((proj[k] > 0.0f) ? 1.0 : -1.0);
        }
        const double b = mnorm * sign_ip;
        num += exact * b;
        den += b * b;
    }

    return (den > 0.0) ? static_cast<float>(num / den) : 1.0f;
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
                        const float* flips, const std::vector<float>& sketch, float kappa,
                        const std::vector<float>& levels, uint32_t entry) { // SHAME(MANYARG)
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
    if (spec.qtype == QUANT_TBQ) {
        fout.write(reinterpret_cast<const char*>(&kappa), sizeof(float));
        fout.write(reinterpret_cast<const char*>(sketch.data()), sizeof(float) * sketch.size());
    }
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

    if (!quant_supported(qtype, codebit)) {
        fprintf(stderr, "rbq supports bits=1; tbq supports bits of 2, 3, or 5\n");
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
    FastfoodSketch sketch(spec.rawdim, seed + 104729u);

    std::vector<float> levels;
    if (qtype == QUANT_TBQ) {
        if (level_file.empty()) {
            fprintf(stderr, "tbq requires a level file for the bits-1 MSE stage\n");
            return 1;
        }
        read_levels(level_file, quant_stage(codebit), levels);
    }

    size_t codeoff = 0, signoff = 0, facoff = 0, nbroff = 0;
    const size_t rowoff = layout_offsets(spec, &codeoff, &signoff, &facoff, &nbroff);
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

    float kappa = 1.0f;
    if (qtype == QUANT_TBQ) {
        printf("Calibrating QJL scale...\n");
        kappa = calibrate_kappa(spec, sketch, levels, rotated.data(), edges, 20000);
        printf("kappa = %.8f\n", kappa);
    }

    printf("Encoding neighbor codes...\n");
    std::vector<float> rows(spec.numnode * rowoff, 0.0f);
#pragma omp parallel
    {
        std::vector<float> rotv(static_cast<size_t>(degree) * spec.paddim);
        std::vector<float> resid(spec.paddim), proj(spec.paddim), sku(spec.paddim);
        std::vector<uint8_t> codes(spec.paddim), signs(spec.paddim);
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
                sketch.apply(rotu, sku.data());
                encode_tbq(spec, sketch, kappa, rotu, sku.data(), rotv.data(), levels,
                           codes.data(), signs.data(), resid.data(), proj.data(), row,
                           codeoff, signoff, facoff);
            } else {
                encode_rbq(spec, rotu, rotv.data(), codes.data(), row, codeoff, facoff);
            }
        }
    }

    printf("Selecting entry point...\n");
    const vidType entry = compute_rabitq_entry_point(rows.data(), spec.numnode, spec.rawdim, rowoff);

    std::vector<float> lut_levels;
    std::vector<float> packed_sketch;
    if (qtype == QUANT_TBQ) {
        const float crange = levels.back() - levels.front();
        lut_levels.resize(levels.size());
        for (size_t k = 0; k < levels.size(); ++k) lut_levels[k] = (levels[k] - levels.front()) / crange;
        packed_sketch.resize(FastfoodSketch::get_words(spec.paddim));
        sketch.store(packed_sketch.data());
    }

    printf("Writing %s\n", out_file.c_str());
    write_index(out_file, spec, rows.data(), rowoff, rotator.get_flips(), packed_sketch, kappa,
                lut_levels, entry);
    printf("Done. entry_point=%u\n", entry);
    return 0;
}
