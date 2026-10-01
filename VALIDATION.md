# Validation

Checked on September 30, 2026. These are software checks, not measurements of CLIP retrieval quality or real model latency.

- Backend: **14 checks passed** using a synthetic index and injected encoder. Covers exact cosine retrieval, query exclusion, metric definitions, model/index compatibility, shared PCA projection, API response contracts, authentication, CORS, MIME and byte/pixel validation, per-IP rate limits, report/README synchronization, safe/idempotent key setup, and result collection/provenance checks (including refusing mismatched indexes and preserving evaluation after benchmark failure).
- Frontend: **TypeScript and Vite production build passed**.
- Packaging: **Docker Compose configuration validated** after generating local configuration. Container images were not built or started because Docker's engine was not running.
- Python source compilation passed. Script help/argument validation was checked.

The checks ran using cached packages in this restricted workspace (Python 3.13 for synthetic checks). The supported application Docker runtime uses Python 3.11 and CPU-only PyTorch. Actual CLIP model loading, torchvision downloads, full indexing, and end-to-end model retrieval could not run because the workspace could not access dependency/dataset download hosts.

Browser visual checks were attempted but could not run: local port binding is restricted and the installed browser could not launch under the workspace sandbox. The responsive UI has been compiled but has not been visually verified here.

At the initial software check, performance numbers were pending. The user subsequently supplied completed measurements, now preserved in `RESULTS.md`; the original JSON artifacts still need to be attached. No raw per-query records or latencies have been invented.

## User-run measurements

The user subsequently completed real indexing, retrieval evaluation (695 queries), and an HTTP benchmark (220 successful searches). Their reported scores and methodology are now in `RESULTS.md`. Original raw JSON artifacts have not yet been attached to this workspace; see `results/run-notes.md`. The implementation checks above and the reported performance measurements are distinct evidence.
