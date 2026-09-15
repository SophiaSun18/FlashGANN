# FlashGANN

FlashGANN is a GPU graph-based approximate nearest neighbor search system for high-recall vector retrieval. Our key insight is that different stages of graph search require different levels of distance accuracy. Beam ranking and expander selection are accuracy-sensitive, so FlashGANN keeps exact distances there to avoid accumulated quantization error in the search frontier; neighbor expansion is much more bandwidth- and compute-intensive, so it can use RaBitQ estimated distance as a cheap approximate score to prune candidates before exact evaluation. FlashGANN further proposes an adaptive tuning policy that adjusts the candidate pruning strength and speculative expansion width according to runtime search progress, which achieves near-static-oracle performance without manually tuning the parameters for every test case.

## Required Inputs

The executable expects four file inputs followed by the search arguments:

```text
<base.fvecs>          Base vectors in fvecs format.
<query.fvecs>         Query vectors in fvecs format.
<groundtruth.ivecs>   Ground-truth neighbors in ivecs format.
<qg_codebook.index>   Prebuilt SymphonyQG-style codebook with data layout adjustment.
K                     Number of nearest neighbors to return. Default: 100.
beam_size             Search beam size. Default: 128.
graph_degree          Degree of the prebuilt graph/codebook. Default: 32.
-quant rbq|tbq        Quantizer family of the codebook. Default: rbq.
-bits <n>             Bits per dimension of the codebook, rbq: 1|2|4, tbq: 2|3|5. Default: 1.
-csv <result.csv>     Optional CSV output path for runtime, latency, throughput, and recall.
-iters <output.txt>   Optional per-query search iteration output, one count per line.
```

## Build

Build the FlashGANN executable and the index builder:

```bash
make all
```

Or build them one at a time:

```bash
make gpu_flashgann
make buildindex
```

Optional compile-time flags can be passed through `EXTRA_NVFLAGS`:

```bash
make gpu_flashgann ARCH=sm_89 EXTRA_NVFLAGS="-DINTERNAL_TOPK=1024"
```

Clean generated binaries with:

```bash
make clean
```

## Build Index

`bin/buildindex` encodes a prebuilt graph into a codebook index:

```text
bin/buildindex <base.fvecs> <graph> <out.index> <rbq|tbq> <bits> [levels.bin] [degree=32] [seed=1]
```

```text
<graph>        Adjacency dump: uint32 node count, uint32 degree, then node count * degree uint32 ids.
rbq <bits>     RaBitQ, 1, 2, or 4 bits per dimension. Multi-bit codes use the Extended RaBitQ grid.
tbq <bits>     TurboQuant inner-product variant, 2, 3, or 5 bits: a bits-1 MSE stage plus one QJL sign bit.
[levels.bin]   Reconstruction levels of the tbq MSE stage: int32 bits, int32 count, count floats. Pass - for rbq.
```

Example, a 2-bit RaBitQ index:

```bash
bin/buildindex sift_base.fvecs graph-d32 rbq_b2.index rbq 2 - 32 1
```

## Example Run

```bash
bin/gpu_flashgann \
  <base.fvecs> \
  <query.fvecs> \
  <groundtruth.ivecs> \
  <qg_codebook.index> \
  100 128 32 \
  -quant rbq -bits 2 \
  -csv result.csv
```

## Tests

```bash
cmake -S . -B build
cmake --build build --target test_rabitq test_turbo
ctest --test-dir build -R "test_rabitq|test_turbo" --output-on-failure
```

`test_rabitq` checks the grid code against exhaustive search and the scan factors against the direct estimator. `test_turbo` checks that the TurboQuant estimator is unbiased; it takes several minutes.

## Paper Reproduction Configurations

Figure 13, 8-dataset end-to-end AP curves:

```text
Datasets: SIFT, Deep1M, Deep10M, GIST, OpenAI1M, MSTuring10M, Text2Image1M, SpaceV100M
K: 100
Degree: 32
Beams: 100, 128, 256, 384, 512, 640, 768, 896, 1024
Build flags: -DINTERNAL_TOPK=1024
```

Figure 22, SIFT extra degree/top-k curves:

```text
Case: SIFT, degree 32, top-10
K: 10
Degree: 32
Beams: 16, 32, 64, 96, 128, 256, 384, 512
Build flags: -DINTERNAL_TOPK=512

Case: SIFT, degree 16, top-10
K: 10
Degree: 16
Beams: 16, 32, 64, 96, 128, 256, 384, 512
Build flags: -DINTERNAL_TOPK=512

Case: SIFT, degree 16, top-100
K: 100
Degree: 16
Beams: 100, 128, 256, 384, 512, 640, 768, 896, 1024
Build flags: -DINTERNAL_TOPK=1024
```