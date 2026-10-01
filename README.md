# Lens — AI Visual Search Tool

A complete local portfolio project: drag in an image or describe a scene to retrieve semantically similar pets and flowers. React + Vite + TypeScript + Tailwind provide the interface; FastAPI and CPU-only OpenAI CLIP ViT-B/32 provide exact retrieval.

**No cloud deployment or paid API calls.** Initial setup downloads public datasets and pretrained model weights. After those are cached, inference stays local.

## Quickstart

Requirements: Docker Desktop / Docker Engine with Compose v2 running, Python 3 for the configuration helper, internet for the initial dependencies/dataset/model downloads, and enough local disk/RAM for the datasets and CLIP. Allocate several GiB of RAM to Docker. CPU indexing can take a substantial amount of time; follow its progress in the backend logs.

From the repository root, one command creates matching random keys, builds the index if missing, and starts both services:

```sh
python3 scripts/setup.py && docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

Open **http://localhost:5174** once the frontend starts. The backend listens on **http://localhost:8001**. The quickstart uses the supplied local override to avoid the common 8000/5173 port conflicts. Compose waits for the index and model to be ready before starting the frontend. Subsequent launches reuse the named `visual-data` volume (dataset, thumbnails, embeddings, model cache). `docker compose down` retains it; deleting that volume removes the downloaded/generated data.

The helper creates `.env`, `backend/.env`, and `frontend/.env` with the same random key, mode 0600. It preserves existing files and refuses mismatched or placeholder keys. All `.env` files are ignored by Git; `.env.example` files document the settings. Run the helper before copying examples to avoid accidentally leaving placeholder keys.

Vite embeds configuration at build time. Keep `API_KEY` in `backend/.env` and `VITE_API_KEY` in the other two `.env` files identical. Rebuild the frontend when changing its key or backend URL. `FRONTEND_ORIGIN` must exactly match the browser's origin, including port (`localhost` and `127.0.0.1` are distinct).

## Port already in use

If Docker reports `bind: address already in use`, another process owns that host port. Keep the existing process running and choose a free port. For example, set these values in the root `.env`:

```dotenv
BACKEND_PORT=8001
VITE_BACKEND_URL=http://localhost:8001
```

Also set `VITE_BACKEND_URL=http://localhost:8001` in `frontend/.env` for native Vite runs and `BACKEND_URL=http://localhost:8001` in `backend/.env` for host-side benchmarks. Restart with `docker compose up --build`; rebuilding updates the backend URL embedded in the frontend. Open `http://localhost:5173` as usual. The backend container still uses port 8000 internally, so its health check and Docker-to-Docker benchmark URL remain unchanged.

If another app such as Nextmove is already using port 5173, set `FRONTEND_PORT=5174` in the root `.env` and `FRONTEND_ORIGIN=http://localhost:5174` in `backend/.env`. Run `docker compose up --build` from this project's root, then open **http://localhost:5174** after the frontend starts. This keeps the existing app available on its own port.

`FRONTEND_PORT` similarly changes the frontend host port (default 5173). If changing it, also update `FRONTEND_ORIGIN` in `backend/.env` to the exact new browser origin. To inspect port 8000 on macOS/Linux, use `lsof -nP -iTCP:8000 -sTCP:LISTEN`.

If the error still names host port 8000, the launched Compose configuration still has the old mapping (for example, an earlier extracted copy). Use the supplied explicit override:

```sh
docker compose -f docker-compose.yml -f docker-compose.local.yml up --build
```

This replaces both port lists with backend host port 8001 and frontend host port 5174, and sets the matching frontend API URL and backend CORS origin. It works with the original hardcoded port configuration too. Open **http://localhost:5174** after the frontend starts. The `!override` tag requires Docker Compose 2.24.4 or newer. Current Docker Desktop versions support it.

## Architecture

```text
Image / text → browser → authenticated FastAPI → shared CPU CLIP encoder
                                               ↓ L2 normalization
                                      index.npy @ query → cosine top-k
                                               ↓
                          labels + protected thumbnails + PCA coordinates
                                               ↓
                                  result grid + similarity bars + 2D plot

Index builder → torchvision datasets → CLIP batches → index.npy
                                      ├─ metadata.json + manifest.json
                                      ├─ thumbnails/*.jpg
                                      └─ pca.json (mean, basis, all points)
```

```text
frontend/              React, TypeScript, Tailwind, SVG PCA visualization
backend/app/           API, security, model wrapper, retrieval and report helpers
backend/scripts/       dataset/index builder and HTTP benchmark
backend/eval.py        deterministic leave-one-out retrieval evaluation
backend/tests/         synthetic-index API, security, PCA and metric checks
scripts/setup.py       matching local key generation
RESULTS.md             measured figures and provenance references
results/               timestamped raw evaluation/benchmark records
```

The default full dataset combines **Oxford-IIIT Pets trainval + test (7,349 images / 37 breeds)** and **Flowers102 train + val + test (8,189 images / 102 categories)** into **15,538 images / 139 classes**. Dataset splits are combined because this is retrieval over a gallery, not supervised training. Evaluation excludes each query image from its gallery. Metadata stores relative source paths, human labels, namespaced class IDs, source dataset/split, and row IDs. Thumbnails have a maximum 320-pixel side. Originals are never served by the API.

## Run without Docker

Use Python 3.11 or 3.12 and Node.js 22. First configure keys:

```sh
python3 scripts/setup.py
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install --upgrade pip
```

On Linux, install CPU-only PyTorch explicitly:

```sh
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

On macOS, use the standard wheels; the encoder always selects `cpu`, including on Apple Silicon:

```sh
pip install torch==2.7.1 torchvision==0.22.1
```

Then install the application dependencies, build the full index, and start the backend:

```sh
pip install -r backend/requirements.txt
python backend/scripts/build_index.py --batch-size 32
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-proxy-headers
```

In a second terminal, from the repository root:

```sh
cd frontend
npm ci
npm run dev
```

Defaults resolve dataset/index paths under `backend/data` irrespective of the launch directory. If overriding them, prefer absolute `DATASET_DIR` and `INDEX_DIR` paths. `TORCH_THREADS` controls CPU inference threads. The model ID/revision must match the built index; changing either requires a rebuild.

## Build or reuse the index

```sh
# Native
python backend/scripts/build_index.py --batch-size 32 --thumbnail-size 320
# Rebuild (stop the API first)
python backend/scripts/build_index.py --force
# Docker: build without starting the API
docker compose run --rm --no-deps backend python scripts/build_index.py
```

Index builds use a filesystem lock, a staging directory, and a completed manifest. An interrupted build leaves no completed new index; rerun to rebuild from cached downloads. The builder refuses unexpected cardinalities, overlapping source paths, or unreadable images rather than silently publishing a partial dataset. A completed compatible index is reused. Generated outputs:

- `index.npy`: float32 `[N, 512]`, L2-normalized rows.
- `metadata.json`: one metadata entry per row.
- `pca.json`: centered PCA mean, two components, variance ratios, and row-aligned 2D coordinates.
- `thumbnails/`: authenticated JPEG previews.
- `manifest.json`: dataset counts, model revision, file hashes, index ID, build timing, and hardware.

To mount a prebuilt index in Docker, place it at `backend/data/index` and replace the backend volume entry with `./backend/data:/app/data`. Keep `datasets` alongside it for benchmarks that select original images. A custom benchmark `--image` can be used when originals are unavailable. The default volume is created with permissions for the non-root backend user; a custom bind mount must allow that user (UID 10001) to write if indexing or downloading is needed.

## API

All endpoints except `GET /health` require `X-API-Key`. Interactive docs/openapi routes are disabled to avoid unauthenticated auxiliary endpoints. CORS preflight is handled for the configured frontend origin. CORS is a browser constraint; it is not authentication.

| Endpoint | Input | Output |
| --- | --- | --- |
| `GET /health` | None | Readiness, dataset size, classes, model/revision, index ID, hardware, rate-limit configuration |
| `POST /search?top_k=12` | Multipart `file` | Neighbors, scores, thumbnail URLs, query/neighbor PCA coordinates, timings |
| `POST /search/text` | JSON `{"query":"pink flowers","top_k":12}` | Same search response |
| `POST /embed` | Multipart `file` | Normalized 512D vector, dimension, model/revision |
| `GET /thumbnails/{id}` | Row ID | Downsized JPEG; requires key |

```sh
# Export the key locally (never put it in a committed script).
export API_KEY='your-local-key'
curl -H "X-API-Key: $API_KEY" -F file=@query.jpg 'http://localhost:8000/search?top_k=12'
curl -H "X-API-Key: $API_KEY" -H 'Content-Type: application/json' \
  -d '{"query":"a golden retriever","top_k":12}' http://localhost:8000/search/text
curl http://localhost:8000/health
```

The browser authenticates thumbnail fetches and uses object URLs; the API key is never added to a URL. Uploads must be genuine JPEG/PNG/WebP matching the declared MIME type, no larger than 10 MiB by default, and within the 25-million-pixel limit. Total request bytes are bounded before multipart parsing, including chunked bodies. Maximum `top_k` is configurable (default 50). A per-IP sliding-window limiter defaults to 30 search/embed requests per minute and 600 thumbnail requests per minute; health is exempt. `429` includes `Retry-After`. Keys use constant-time comparison. Uvicorn proxy header trust is disabled, and the provided deployment uses one worker so rate-limit state is shared within that process. Limits reset on process restart.

Structured per-request logs contain embedding, ranking, and handler total milliseconds; embedding time includes waiting for the CPU encoder slot. Search responses expose encode/rank/handler timing separately. HTTP benchmark measurements also include upload and response receipt.

**Local-key note:** Vite environment variables are visible in the browser bundle. This is acceptable for a local portfolio app; a real deployment would place the API key behind a server-side proxy and use user authentication.

## Evaluation and benchmarks

After the real index is built, collect both evaluations directly from the running Docker backend into this repository with one command:

```sh
python3 scripts/collect_results.py
# Only retrieval accuracy (skip the latency benchmark):
python3 scripts/collect_results.py --evaluation-only
```

The helper uses `ai-visual-search-backend-1` by default, checks the live index ID, runs leave-one-out evaluation and 220 real HTTP image requests, copies raw provenance and `RESULTS.md` back to the host, and refreshes the README. It uses the backend's internal port 8000, so host port overrides do not affect it. No API keys need to be copied into your terminal. Use `--container NAME` if your container has a different name.

For separate native commands:

```sh
# Native, from root with backend venv activated
python backend/eval.py                     # 5 queries/class, 695 queries, seed 42
python backend/eval.py --all               # every row as a leave-one-out query
python backend/scripts/bench.py --base-url http://localhost:8000 --requests 220
# Optional custom query upload
python backend/scripts/bench.py --image query.jpg --requests 220
```

Both scripts update `RESULTS.md`, preserve the other's section, and refresh the README's embedded results block. Timestamped JSON records in `results/` retain exact values, all query/neighbor IDs or per-request latencies, model/index provenance, dataset/class/query counts, configuration, and hardware. Evaluation and latency are measured separately; neither script tunes for a target.

For Docker, bind the report destinations so measured results remain in the repository. Create the directory first with `mkdir -p results`.

Use these report commands from the root (on macOS/Linux):

```sh
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v "$(pwd):/reports" backend python eval.py --results /reports/RESULTS.md
docker compose run --rm --no-deps --user "$(id -u):$(id -g)" \
  -v "$(pwd):/reports" backend python scripts/bench.py \
  --base-url http://backend:8000 --requests 220 --results /reports/RESULTS.md
```

The backend must already be running for the benchmark. In Docker, the report lists the CPU/RAM seen inside the containers and the server's `TORCH_THREADS`; on native runs it records host hardware. This makes figures comparable only with their recorded environment. The benchmark uses sequential keep-alive requests after one unmeasured warm-up, rotates 25 deterministic uploads by default, and requires at least 200 successful requests. Uploads are downsized to a 1024-pixel maximum side before timing. Default pacing respects the rate limit and can take several minutes; idle waits and 429 retries are excluded from latency. Errors remain in the report and make the run incomplete with a nonzero exit status.

## Measured results

<!-- results:start -->
Measured on the completed 15,538-image gallery. The report below was supplied by the user from `scripts/collect_results.py`; the original JSON artifacts are not yet available in this workspace. No raw latencies or neighbor IDs have been reconstructed from rounded values.

<!-- evaluation:start -->
### Retrieval evaluation

Dataset: 15,538 images · classes: 139 · queries: 695

Method: leave-one-out, frozen CLIP image embeddings; query row is excluded from ranking. Class-stratified sample; seed 42.

- Precision@1: **0.9036** — fraction of queries whose first neighbor has the same class.
- Precision@5: **0.8581** — mean fraction of five neighbors with the same class.
- Top-5 hit rate: **0.9770** — fraction of queries with at least one same-class neighbor in the top five.

Hardware: aarch64; 8 logical cores; 7.75 GiB RAM; Linux-6.12.76-linuxkit-aarch64-with-glibc2.41.

Index: `e74982f744e53ca5`; model: `openai/clip-vit-base-patch32`; revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
These measure class agreement for image queries, not text retrieval or a probability of correctness.

Original provenance filename: `results/evaluation-20261001T012316563954Z.json` (awaiting the original artifact).
<!-- evaluation:end -->

<!-- benchmark:start -->
### HTTP latency benchmark

Dataset: 15,538 images · classes: 139 · queries attempted: 220 · successful: 220 · distinct uploads: 25

Status: complete.

- Median end-to-end latency: **92.22 ms**.
- P95 end-to-end latency: **517.39 ms** (NumPy linear percentile).

Measured from POST start through receipt of the full JSON response, including upload, encoding, ranking, and HTTP overhead. One warm-up is excluded; sequential keep-alive requests; no thumbnail requests. Idle interval 2.10s excluded; top-k 12; HTTP errors 0; 429 retries excluded: 0.

Server hardware: aarch64; 8 logical cores; 7.75 GiB RAM; Linux-6.12.76-linuxkit-aarch64-with-glibc2.41; PyTorch threads 4.

Client hardware: aarch64; 8 logical cores; 7.75 GiB RAM.

Index: `e74982f744e53ca5`; model `openai/clip-vit-base-patch32`; revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.

Original provenance filename: `results/benchmark-20261001T013145391602Z.json` (awaiting the original artifact).
<!-- benchmark:end -->

### Measurement context

Both the HTTP client and server ran inside the Linux Docker environment. These figures describe sequential loopback HTTP search, not browser-to-host networking or concurrent load. `aarch64` identifies the architecture; the physical host CPU model was not reported by this container. CPU model and raw artifact attachment remain outstanding provenance details.
<!-- results:end -->

## Design notes

L2 normalization makes cosine similarity equal to the dot product. A float32 15,538 × 512 gallery occupies about 30 MiB, so one dense matrix multiplication plus partial top-k selection is simple and fast enough to evaluate directly on a CPU. A shared, startup-loaded CLIP model embeds both images and text into the same space; no external inference service or approximate vector database is needed. Model encoding is serialized to keep CPU usage predictable while file serving and health remain asynchronous.

Accuracy means **same-class retrieval**, using namespaced breed/flower labels. Precision@1, precision@5, and top-5 hit rate describe different class agreement criteria and are defined alongside their real output values. Leave-one-out excludes the exact query row, while cached frozen embeddings avoid re-encoding during evaluation. This does not measure text retrieval, supervised test-set generalization, or near-duplicate robustness. The default stratified query sample gives each class equal representation.

Text queries follow CLIP's 77-token context; longer queries are truncated. Search ranks in the full 512D space. The scatter plot projects the query with the same centered PCA basis fitted offline on the gallery; 2D distance can distort true similarity. Bars show cosine × 100, not a probability of correctness. Dataset chips filter the returned neighbors; they do not run dataset-specific retrieval. The UI aborts stale searches, revokes thumbnail/preview object URLs, validates uploads locally, and provides backend error/loading/offline states.

## Checks

```sh
pip install -r backend/requirements-dev.txt
cd backend && pytest
# From frontend/
npm ci
npm run build
# From repository root after setup:
docker compose config --quiet
```

Backend checks use a small synthetic index and an injected deterministic encoder: security, exact cosine ranking, self-exclusion, metric definitions, upload constraints, API response contracts, startup model/index consistency, and PCA projection. They do not require dataset/model downloads and do not establish real retrieval performance.

## Demo

**Demo GIF placeholder:** record an image drop, a text query, and scatter hover, then add `docs/demo.gif` and embed it here.

## References

- [CLIP model and immutable revision](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)
- [Torchvision Oxford-IIIT Pets](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html)
- [Torchvision Flowers102](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html)
- [Pinned torchvision dataset source and flower labels](https://github.com/pytorch/vision/blob/v0.22.1/torchvision/datasets/flowers102.py)
