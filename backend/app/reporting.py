"""Shared provenance and atomic report writing for evaluation and benchmarks."""
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]


def hardware():
    cpu = platform.processor() or platform.machine()
    try:
        if platform.system() == 'Darwin':
            cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
        elif Path('/proc/cpuinfo').exists():
            cpu = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name'))
    except (OSError, subprocess.SubprocessError, StopIteration):
        pass
    return {'cpu_model': cpu, 'logical_cores': os.cpu_count(),
            'physical_cores': psutil.cpu_count(logical=False),
            'ram_gib': round(psutil.virtual_memory().total / 1024 ** 3, 2),
            'platform': platform.platform()}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(kind: str, payload: dict, markdown: str, results_path: Path):
    import fcntl
    results_path = results_path.resolve()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    payload = dict(payload, recorded_at_utc=stamp)
    artifact_dir = results_path.parent / 'results'
    artifact_dir.mkdir(exist_ok=True)
    artifact = artifact_dir / f'{kind}-{stamp}.json'
    artifact.write_text(json.dumps(payload, indent=2) + '\n')
    start, end = f'<!-- {kind}:start -->', f'<!-- {kind}:end -->'
    section = f'{start}\n{markdown}\n\nProvenance: `{artifact.relative_to(results_path.parent)}`\n{end}'
    # Separate lock survives os.replace; protects simultaneous eval/bench writers.
    with (artifact_dir / '.reports.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        text = results_path.read_text() if results_path.exists() else '# Measured results\n'
        if start in text and end in text:
            left, tail = text.split(start, 1)
            _, right = tail.split(end, 1)
            text = left + section + right
        else:
            text += '\n\n' + section + '\n'
        temp = results_path.with_suffix('.tmp')
        temp.write_text(text)
        temp.replace(results_path)
        readme = results_path.parent / 'README.md'
        if readme.exists() and '<!-- results:start -->' in readme.read_text():
            content = readme.read_text()
            left, tail = content.split('<!-- results:start -->', 1)
            _, right = tail.split('<!-- results:end -->', 1)
            embedded = '\n'.join('#' + line if line.startswith('#') else line for line in text.splitlines()[1:]).strip()
            updated = left + '<!-- results:start -->\n' + embedded + '\n<!-- results:end -->' + right
            temp_readme = readme.with_suffix('.tmp')
            temp_readme.write_text(updated)
            temp_readme.replace(readme)
    print(markdown)
    print(f'Report written: {results_path}; raw provenance: {artifact}')
