# Report provenance

The user ran `scripts/collect_results.py` against the completed local Docker backend and supplied its final output in the chat. The measured values and full methodology are preserved in the root `RESULTS.md` and its synchronized README section.

- Index: `e74982f744e53ca5`
- Evaluation artifact: `evaluation-20261001T012316563954Z.json`
- Benchmark artifact: `benchmark-20261001T013145391602Z.json`
- Container collection directory reported: `/tmp/visual-search-results-20261001T012315Z`

The user supplied both original JSON files. They are preserved byte-for-byte in this directory. Verification recomputed precision@1, precision@5 and top-5 hit rate from 695 query relevance records, and median/p95 from 220 recorded latencies. All match the stored full-precision values. Every query excludes its own row and has five distinct neighbors. Both files identify the same index, dataset size and class count; the evaluation index hash matches its manifest. This verifies report consistency, without independently rerunning CLIP inference.

Original artifact SHA-256 hashes:

- `evaluation-20261001T012316563954Z.json`: `ba62ae88e04b550245db363e93e12887816cd37612b8e8f9aa49ca861e16b149`
- `benchmark-20261001T013145391602Z.json`: `5b4108ffd4b790c9ef9687ff7442992e70391d1f3fd34bf65fb46b97da8339d1`

The container exposed `aarch64`, 8 logical cores, 7.75 GiB RAM and 4 PyTorch threads. `aarch64` is an architecture rather than a specific physical CPU model. The benchmark client was also running inside the backend container, using loopback HTTP. These figures are not a concurrent load test or a measurement of browser-to-host round trips.
