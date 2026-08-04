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
-csv <result.csv>     Optional CSV output path for runtime, latency, throughput, and recall.
```

## Build

Build the FlashGANN executable:

```bash
make gpu_flashgann
```

Optional compile-time flags can be passed through `EXTRA_NVFLAGS`:

```bash
make gpu_flashgann ARCH=sm_89 EXTRA_NVFLAGS="-DINTERNAL_TOPK=1024"
```

Clean generated binaries with:

```bash
make clean
```

## Example Run

```bash
bin/gpu_flashgann \
  <base.fvecs> \
  <query.fvecs> \
  <groundtruth.ivecs> \
  <qg_codebook.index> \
  100 128 32 \
  -csv result.csv
```

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