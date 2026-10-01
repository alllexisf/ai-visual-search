import json
import subprocess
import sys
from pathlib import Path
from app.reporting import write_report


def test_report_sections_and_readme_sync(tmp_path):
    results = tmp_path / 'RESULTS.md'
    results.write_text('# Results\n<!-- evaluation:start -->pending<!-- evaluation:end -->\n<!-- benchmark:start -->pending<!-- benchmark:end -->')
    readme = tmp_path / 'README.md'
    readme.write_text('Intro\n<!-- results:start -->old<!-- results:end -->\nOutro')
    write_report('evaluation', {'number_of_queries': 3}, 'evaluation measured', results)
    write_report('benchmark', {'number_of_queries': 220}, 'benchmark measured', results)
    assert 'evaluation measured' in results.read_text()
    assert 'benchmark measured' in results.read_text()
    assert 'evaluation measured' in readme.read_text()
    assert 'benchmark measured' in readme.read_text()
    assert 'old' not in readme.read_text()
    assert readme.read_text().endswith('Outro')
    artifacts = list((tmp_path / 'results').glob('*.json'))
    assert len(artifacts) == 2
    assert all('recorded_at_utc' in json.loads(path.read_text()) for path in artifacts)


def test_setup_matching_keys_idempotent_and_mismatch_safe(tmp_path):
    source = Path(__file__).resolve().parents[2]
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'backend').mkdir()
    (tmp_path / 'frontend').mkdir()
    (tmp_path / 'scripts/setup.py').write_text((source / 'scripts/setup.py').read_text())
    for name in ('backend', 'frontend'):
        (tmp_path / name / '.env.example').write_text((source / name / '.env.example').read_text())
    command = [sys.executable, str(tmp_path / 'scripts/setup.py')]
    subprocess.run(command, check=True, capture_output=True)
    files = [tmp_path / name for name in ('backend/.env', 'frontend/.env', '.env')]
    contents = [path.read_text() for path in files]
    keys = [next(line.split('=', 1)[1] for line in text.splitlines() if line.startswith(('API_KEY=', 'VITE_API_KEY='))) for text in contents]
    assert len(set(keys)) == 1 and len(keys[0]) >= 32
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)
    subprocess.run(command, check=True, capture_output=True)
    assert [path.read_text() for path in files] == contents
    files[1].write_text('VITE_API_KEY=mismatched\n')
    before = [path.read_text() for path in files]
    failed = subprocess.run(command, capture_output=True)
    assert failed.returncode != 0
    assert [path.read_text() for path in files] == before
