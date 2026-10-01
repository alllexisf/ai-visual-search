#!/usr/bin/env python3
"""Measure 200+ actual image HTTP requests after an unmeasured warm-up."""
import argparse
import io
import hashlib
import json
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
import numpy as np
from PIL import Image, ImageOps
from app.config import Settings
from app.reporting import ROOT, hardware, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default=os.getenv('BACKEND_URL', 'http://localhost:8000'))
    parser.add_argument('--image', type=Path, help='Defaults to 25 deterministic dataset images rotated across requests.')
    parser.add_argument('--requests', type=int, default=220)
    parser.add_argument('--top-k', type=int, default=12)
    parser.add_argument('--interval', type=float, default=None, help='Idle seconds between requests; excluded from measured latency.')
    parser.add_argument('--results', type=Path, default=ROOT / 'RESULTS.md')
    args = parser.parse_args()
    if args.requests < 200 or args.top_k < 1 or (args.interval is not None and args.interval < 0):
        parser.error('Use at least 200 requests, a positive top-k, and a nonnegative interval.')
    cfg = Settings()
    if not cfg.api_key:
        parser.error('API_KEY must be set in the environment or backend/.env.')
    samples = []
    if args.image:
        paths = [args.image]
    else:
        metadata = json.loads((cfg.index_dir / 'metadata.json').read_text())
        ids = np.linspace(0, len(metadata) - 1, min(25, len(metadata)), dtype=int)
        paths = [cfg.dataset_dir / metadata[int(i)]['image_path'] for i in ids]
    for path in paths:
        # Representative upload size; encode once before timing, never during a request.
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        image.thumbnail((1024, 1024))
        stream = io.BytesIO()
        image.save(stream, format='JPEG', quality=90)
        image.close()
        samples.append((path.name, stream.getvalue()))
    latencies, backend_timings, errors, retries = [], [], [], 0
    headers = {'X-API-Key': cfg.api_key}
    with httpx.Client(base_url=args.base_url.rstrip('/'), headers=headers, timeout=120) as client:
        health = client.get('/health')
        health.raise_for_status()
        server = health.json()
        limit = server['rate_limit']
        interval = args.interval if args.interval is not None else limit['window_seconds'] / limit['search_requests'] + 0.1
        def request_sample(sample):
            name, data = sample
            return client.post('/search', params={'top_k': args.top_k}, files={'file': (name, data, 'image/jpeg')})
        warmup = request_sample(samples[0])
        warmup.raise_for_status()
        print(f'Warm-up complete. {args.requests} sequential requests, {interval:.2f}s idle between requests; idle is excluded.', flush=True)
        for i in range(args.requests):
            time.sleep(interval)
            attempts = 0
            while True:
                tick = time.perf_counter()
                try:
                    response = request_sample(samples[i % len(samples)])
                    elapsed = (time.perf_counter() - tick) * 1000
                except httpx.HTTPError as exc:
                    errors.append({'request': i, 'error': type(exc).__name__})
                    break
                if response.status_code == 429 and attempts < 3:
                    # Rate-limited attempts never contaminate inference latency.
                    retries += 1
                    attempts += 1
                    time.sleep(float(response.headers.get('Retry-After', '60')) + 0.1)
                    continue
                if response.status_code != 200:
                    errors.append({'request': i, 'status': response.status_code})
                    break
                body = response.json()
                if body['index_id'] != server['index_id']:
                    raise RuntimeError('Server index changed mid-benchmark; refusing mixed measurements.')
                latencies.append(elapsed)
                backend_timings.append(body['timings'])
                break
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{args.requests} requests | successes {len(latencies)}', flush=True)
    hw = hardware()
    median = float(np.median(latencies)) if latencies else None
    p95 = float(np.percentile(latencies, 95)) if latencies else None
    valid = len(latencies) >= 200 and not errors
    server_hw = server['hardware']
    md = (f'## HTTP latency benchmark\n\n'
          f'Dataset: {server["dataset_size"]:,} images · classes: {server["classes"]} · '
          f'queries attempted: {args.requests} · successful: {len(latencies)} · distinct uploads: {len(samples)}\n\n'
          f'Status: {"complete" if valid else "incomplete (do not cite as a completed 200+ benchmark)"}.\n\n'
          f'- Median end-to-end latency: **{f"{median:.2f} ms" if median is not None else "unavailable"}**.\n'
          f'- P95 end-to-end latency: **{f"{p95:.2f} ms" if p95 is not None else "unavailable"}** (NumPy linear percentile).\n\n'
          'Measured from POST start through receipt of the full JSON response, including upload, encoding, ranking, and HTTP overhead. '
          'One warm-up is excluded; sequential keep-alive requests; no thumbnail requests. '
          f'Idle interval {interval:.2f}s excluded; top-k {args.top_k}; HTTP errors {len(errors)}; 429 retries excluded: {retries}.\n\n'
          f'Server hardware: {server_hw["cpu_model"]}; {server_hw["logical_cores"]} logical cores; {server_hw["ram_gib"]} GiB RAM; '
          f'{server_hw["platform"]}; PyTorch threads {server["torch_threads"]}.\n\n'
          f'Client hardware: {hw["cpu_model"]}; {hw["logical_cores"]} logical cores; {hw["ram_gib"]} GiB RAM.\n\n'
          f'Index: `{server["index_id"]}`; model `{server["model"]}`; revision `{server["model_revision"]}`.')
    write_report('benchmark', dict(valid=valid, median_ms=median, p95_ms=p95, number_of_queries=args.requests,
                 successful_queries=len(latencies), dataset_size=server['dataset_size'], number_of_classes=server['classes'],
                 server=server, client_hardware=hw, interval_seconds=interval, top_k=args.top_k,
                 latencies_ms=latencies, backend_timings=backend_timings, errors=errors, rate_limit_retries=retries,
                 upload_names=[n for n, _ in samples], upload_sizes=[len(d) for _, d in samples],
                 upload_sha256=[hashlib.sha256(d).hexdigest() for _, d in samples],
                 base_url=args.base_url), md, args.results)
    if not valid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
