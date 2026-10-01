# Report provenance

The user ran `scripts/collect_results.py` against the completed local Docker backend and supplied its final output in the chat. The measured values and full methodology are preserved in the root `RESULTS.md` and its synchronized README section.

- Index: `e74982f744e53ca5`
- Evaluation artifact reported: `evaluation-20261001T012316563954Z.json`
- Benchmark artifact reported: `benchmark-20261001T013145391602Z.json`
- Container collection directory reported: `/tmp/visual-search-results-20261001T012315Z`

The helper copied the actual artifacts to the user's project `results/` directory. They have not yet been supplied to this workspace/repository. Do not replace them with synthesized per-query results or latencies; attach the original files to complete provenance.

The container exposed `aarch64`, 8 logical cores, 7.75 GiB RAM and 4 PyTorch threads. `aarch64` is an architecture rather than a specific physical CPU model. The benchmark client was also running inside the backend container, using loopback HTTP. These figures are not a concurrent load test or a measurement of browser-to-host round trips.
