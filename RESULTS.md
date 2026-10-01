# Measured results

Measured on the completed 15,538-image gallery. The user ran `scripts/collect_results.py` locally and supplied the original JSON artifacts. All three retrieval metrics and both latency percentiles were recomputed from those records and match the report. The original files are preserved without modification.

<!-- evaluation:start -->
## Retrieval evaluation

Dataset: 15,538 images · classes: 139 · queries: 695

Method: leave-one-out, frozen CLIP image embeddings; query row is excluded from ranking. Class-stratified sample; seed 42.

- Precision@1: **0.9036** — fraction of queries whose first neighbor has the same class.
- Precision@5: **0.8581** — mean fraction of five neighbors with the same class.
- Top-5 hit rate: **0.9770** — fraction of queries with at least one same-class neighbor in the top five.

Hardware: aarch64; 8 logical cores; 7.75 GiB RAM; Linux-6.12.76-linuxkit-aarch64-with-glibc2.41.

Index: `e74982f744e53ca5`; model: `openai/clip-vit-base-patch32`; revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
These measure class agreement for image queries, not text retrieval or a probability of correctness.

Provenance: [evaluation-20261001T012316563954Z.json](results/evaluation-20261001T012316563954Z.json).
<!-- evaluation:end -->

<!-- benchmark:start -->
## HTTP latency benchmark

Dataset: 15,538 images · classes: 139 · queries attempted: 220 · successful: 220 · distinct uploads: 25

Status: complete.

- Median end-to-end latency: **92.22 ms**.
- P95 end-to-end latency: **517.39 ms** (NumPy linear percentile).

Measured from POST start through receipt of the full JSON response, including upload, encoding, ranking, and HTTP overhead. One warm-up is excluded; sequential keep-alive requests; no thumbnail requests. Idle interval 2.10s excluded; top-k 12; HTTP errors 0; 429 retries excluded: 0.

Server hardware: aarch64; 8 logical cores; 7.75 GiB RAM; Linux-6.12.76-linuxkit-aarch64-with-glibc2.41; PyTorch threads 4.

Client hardware: aarch64; 8 logical cores; 7.75 GiB RAM.

Index: `e74982f744e53ca5`; model `openai/clip-vit-base-patch32`; revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

Provenance: [benchmark-20261001T013145391602Z.json](results/benchmark-20261001T013145391602Z.json).
<!-- benchmark:end -->

## Measurement context

Both the HTTP client and server ran inside the Linux Docker environment. These figures describe sequential loopback HTTP search, not browser-to-host networking or concurrent load. `aarch64` identifies the architecture; the physical host CPU model was not reported by this container. The specific physical host CPU model remains unavailable in these records.
