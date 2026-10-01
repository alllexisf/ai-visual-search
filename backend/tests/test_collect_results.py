import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import pytest


def collector():
    source = Path(__file__).resolve().parents[2] / 'scripts/collect_results.py'
    spec = importlib.util.spec_from_file_location('collect_results_for_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(tmp_path):
    (tmp_path / 'README.md').write_text('Intro\n<!-- results:start -->pending<!-- results:end -->\nOutro')
    (tmp_path / 'RESULTS.md').write_text('# Measured results\nPrevious report')


def fake_docker(monkeypatch, module, remote_index='real-index', failed_benchmark=False):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ['docker', 'cp']:
            destination = Path(command[-1])
            (destination / 'results').mkdir()
            (destination / 'RESULTS.md').write_text('# Measured results\n## Retrieval evaluation\nFixture report for helper tests')
            (destination / 'results/evaluation-test.json').write_text(json.dumps({
                'index_manifest': {'index_id': remote_index}, 'number_of_queries': 695}))
            return subprocess.CompletedProcess(command, 0)
        if command[-2:-1] == ['-c']:
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps({
                'index_id': 'real-index', 'dataset_size': 15538, 'classes': 139}))
        if failed_benchmark and 'scripts/bench.py' in command:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(module, 'run', run)
    monkeypatch.setattr(module.shutil, 'which', lambda _: '/usr/bin/docker')
    return calls


def test_collect_copies_provenance_preserves_old_report_and_syncs_readme(tmp_path, monkeypatch):
    prepare(tmp_path)
    module = collector()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['collect_results.py', '--evaluation-only'])
    calls = fake_docker(monkeypatch, module)
    assert module.main() == 0
    assert any('eval.py' in call for call in calls)
    assert not any('scripts/bench.py' in call for call in calls)
    assert len(list((tmp_path / 'results').glob('evaluation-*.json'))) == 1
    backup = next((tmp_path / 'results').glob('report-before-*.md'))
    assert backup.read_text() == '# Measured results\nPrevious report'
    assert 'Fixture report for helper tests' in (tmp_path / 'README.md').read_text()
    assert (tmp_path / 'README.md').read_text().endswith('Outro')


def test_collect_refuses_mismatched_index_before_modifying_host_report(tmp_path, monkeypatch):
    prepare(tmp_path)
    module = collector()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['collect_results.py', '--evaluation-only'])
    fake_docker(monkeypatch, module, remote_index='different-index')
    with pytest.raises(RuntimeError, match='index IDs differ'):
        module.main()
    assert (tmp_path / 'RESULTS.md').read_text() == '# Measured results\nPrevious report'
    assert not (tmp_path / 'results').exists()


def test_failed_benchmark_retains_evaluation_but_returns_failure(tmp_path, monkeypatch):
    prepare(tmp_path)
    module = collector()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['collect_results.py'])
    calls = fake_docker(monkeypatch, module, failed_benchmark=True)
    assert module.main() == 1
    assert any('scripts/bench.py' in call for call in calls)
    assert 'Fixture report for helper tests' in (tmp_path / 'RESULTS.md').read_text()
