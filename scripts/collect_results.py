#!/usr/bin/env python3
"""Collect real evaluation/latency results from the already running Docker backend."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def sync_readme():
    readme = ROOT / 'README.md'
    text = readme.read_text()
    report = (ROOT / 'RESULTS.md').read_text()
    if '<!-- results:start -->' not in text:
        return
    left, tail = text.split('<!-- results:start -->', 1)
    _, right = tail.split('<!-- results:end -->', 1)
    embedded = '\n'.join('#' + line if line.startswith('#') else line for line in report.splitlines()[1:]).strip()
    temp = readme.with_suffix('.tmp')
    temp.write_text(left + '<!-- results:start -->\n' + embedded + '\n<!-- results:end -->' + right)
    temp.replace(readme)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--container', default='ai-visual-search-backend-1')
    parser.add_argument('--requests', type=int, default=220)
    parser.add_argument('--evaluation-only', action='store_true')
    args = parser.parse_args()
    if not args.evaluation_only and args.requests < 200:
        parser.error('Use at least 200 requests for a completed benchmark.')
    if not shutil.which('docker'):
        parser.error('Docker is not available. Install/start Docker Desktop first.')
    print(f'Checking the running backend: {args.container}', flush=True)
    health = run(['docker', 'exec', args.container, 'python', '-c',
        "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health', timeout=10).read().decode())"],
        capture_output=True, text=True)
    ready = json.loads(health.stdout)
    print(f"Using real index {ready['index_id']}: {ready['dataset_size']:,} images / {ready['classes']} classes.", flush=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    remote = f'/tmp/visual-search-results-{stamp}'
    run(['docker', 'exec', args.container, 'mkdir', '-p', remote])
    print('Running class-stratified leave-one-out evaluation (5 queries per class)...', flush=True)
    run(['docker', 'exec', args.container, 'python', 'eval.py', '--results', remote + '/RESULTS.md'])
    benchmark_failed = False
    if not args.evaluation_only:
        print(f'Running {args.requests} real HTTP searches inside the backend container. Rate-limit pacing can take several minutes.', flush=True)
        try:
            run(['docker', 'exec', args.container, 'python', 'scripts/bench.py',
                 '--base-url', 'http://localhost:8000', '--requests', str(args.requests), '--results', remote + '/RESULTS.md'])
        except subprocess.CalledProcessError:
            benchmark_failed = True
            print('Benchmark did not complete successfully; preserving evaluation and any incomplete benchmark report.', flush=True)
    with tempfile.TemporaryDirectory(prefix='visual-search-') as temp:
        destination = Path(temp)
        run(['docker', 'cp', args.container + ':' + remote + '/.', str(destination)])
        report = destination / 'RESULTS.md'
        if not report.exists():
            raise RuntimeError('Docker did not produce a report; no host files changed.')
        raw = list((destination / 'results').glob('*.json'))
        if not raw:
            raise RuntimeError('Missing raw provenance; no host files changed.')
        records = [json.loads(path.read_text()) for path in raw]
        for record in records:
            index_id = record.get('index_manifest', record.get('server', {})).get('index_id')
            if index_id != ready['index_id']:
                raise RuntimeError('Results and startup index IDs differ; refusing to publish mismatched results.')
        (ROOT / 'results').mkdir(exist_ok=True)
        for path in raw:
            shutil.copy2(path, ROOT / 'results' / path.name)
        # Preserve the preexisting report as another traceable snapshot.
        previous = ROOT / 'RESULTS.md'
        if previous.exists():
            shutil.copy2(previous, ROOT / 'results' / f'report-before-{stamp}.md')
        shutil.copy2(report, previous)
    sync_readme()
    print('\nSaved real results to RESULTS.md and results/*.json; refreshed the README.', flush=True)
    print((ROOT / 'RESULTS.md').read_text())
    if benchmark_failed:
        return 1
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print('Could not collect complete results. Keep the app running and inspect the container logs.', file=sys.stderr)
        raise SystemExit(exc.returncode)
