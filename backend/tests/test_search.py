import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app
from app.retrieval import SearchIndex, top_k
from eval import evaluate
from scripts.build_index import pca

KEY = 'test-only-not-a-real-secret'


class Encoder:
    loads = 0
    def __init__(self, cfg):
        Encoder.loads += 1
    def images(self, images):
        vector = np.zeros((len(images), 512), dtype=np.float32)
        vector[:, 0] = 1
        return vector
    def text(self, query):
        return self.images([None])[0]


@pytest.fixture
def cfg(tmp_path):
    matrix = np.zeros((10, 512), dtype=np.float32)
    for i in range(10):
        matrix[i, i // 2] = 1
        matrix[i, 10 + i] = .1
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.save(tmp_path / 'index.npy', matrix)
    rows = [dict(id=i, label=f'class {i // 2}', class_id=f'pets:{i // 2}', dataset='Pets', image_path=f'{i}.jpg') for i in range(10)]
    (tmp_path / 'metadata.json').write_text(json.dumps(rows))
    (tmp_path / 'pca.json').write_text(json.dumps(pca(matrix)))
    config = Settings(api_key=KEY, index_dir=tmp_path, max_upload_bytes=2048, rate_limit=100)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(index_id='fixture', model_id=config.model_id,
        model_revision=config.model_revision, dataset_size=10, number_of_classes=5)))
    (tmp_path / 'thumbnails').mkdir()
    Image.new('RGB', (8, 8)).save(tmp_path / 'thumbnails/0.jpg')
    return config


@pytest.fixture
def client(cfg):
    with TestClient(create_app(cfg, encoder_factory=Encoder)) as client:
        yield client


def upload():
    stream = io.BytesIO()
    Image.new('RGB', (16, 16), 'red').save(stream, 'PNG')
    return ('query.png', stream.getvalue(), 'image/png')


def test_exact_cosine_and_self_exclusion(cfg):
    index = SearchIndex(cfg.index_dir, cfg.model_id, cfg.model_revision)
    ids, scores = top_k(index.matrix, index.matrix[0], 5, exclude=0)
    assert 0 not in ids
    assert ids[0] == 1
    assert scores[0] < 1
    assert np.all(np.diff(scores) <= 0)
    assert np.allclose(scores, (index.matrix @ index.matrix[0])[ids])
    metrics, details = evaluate(index, [0, 2, 4])
    assert metrics == pytest.approx({'precision_at_1': 1, 'precision_at_5': .2, 'top_5_hit_rate': 1})
    assert all(q['query_id'] not in q['neighbor_ids'] for q in details)


def test_pca_uses_shared_basis(cfg):
    index = SearchIndex(cfg.index_dir, cfg.model_id, cfg.model_revision)
    assert np.allclose(index.project(index.matrix[3]), index.pca['points'][3], atol=1e-6)
    assert index.components.shape == (2, 512)


def test_health_public_everything_else_protected(client):
    assert client.get('/health').status_code == 200
    assert client.get('/thumbnails/0').status_code == 401
    assert client.post('/search', files={'file': upload()}).status_code == 401
    assert client.post('/search/text', json={'query': 'flower'}).status_code == 401
    assert client.post('/embed', files={'file': upload()}).status_code == 401
    assert client.get('/docs').status_code == 401
    assert client.get('/health', headers={'X-API-Key': 'wrong'}).status_code == 200


def test_real_api_response_contract_and_single_load(cfg):
    before = Encoder.loads
    with TestClient(create_app(cfg, encoder_factory=Encoder)) as client:
        headers = {'X-API-Key': KEY}
        image = client.post('/search?top_k=5', files={'file': upload()}, headers=headers)
        assert image.status_code == 200
        payload = image.json()
        assert len(payload['results']) == 5
        assert set(payload['timings']) == {'embedding_ms', 'ranking_ms', 'total_ms'}
        assert len(payload['query_point']) == 2
        assert payload['results'][0]['thumbnail_url'] == '/thumbnails/0'
        text = client.post('/search/text', json={'query': 'a flower', 'top_k': 5}, headers=headers)
        assert text.status_code == 200
        assert text.json()['results'] == payload['results']
        embed = client.post('/embed', files={'file': upload()}, headers=headers)
        assert len(embed.json()['embedding']) == 512
        assert np.linalg.norm(embed.json()['embedding']) == pytest.approx(1)
        assert client.get('/thumbnails/0', headers=headers).status_code == 200
        assert client.get('/thumbnails/-1', headers=headers).status_code == 404
    assert Encoder.loads == before + 1


def test_validation_and_actual_body_limits(client, cfg):
    headers = {'X-API-Key': KEY}
    assert client.post('/search', files={'file': ('bad.svg', b'<svg/>', 'image/svg+xml')}, headers=headers).status_code == 415
    assert client.post('/search', files={'file': ('bad.png', b'not an image', 'image/png')}, headers=headers).status_code == 400
    assert client.post('/search', files={'file': ('wrong.jpg', upload()[1], 'image/jpeg')}, headers=headers).status_code == 415
    assert client.post('/search', files={'file': ('empty.png', b'', 'image/png')}, headers=headers).status_code == 400
    assert client.post('/search', files={'file': ('large.png', b'x' * 3000, 'image/png')}, headers=headers).status_code == 413
    assert client.post('/search', content=b'x' * 100_000, headers=headers).status_code == 413
    assert client.post('/search?top_k=0', files={'file': upload()}, headers=headers).status_code == 422
    assert client.post('/search/text', json={'query': '  '}, headers=headers).status_code == 422
    assert client.post('/search/text', json={'query': 'flower', 'top_k': cfg.max_top_k + 1}, headers=headers).status_code == 422


def test_pixel_limit(cfg):
    with TestClient(create_app(replace(cfg, max_image_pixels=10), encoder_factory=Encoder)) as client:
        assert client.post('/search', files={'file': upload()}, headers={'X-API-Key': KEY}).status_code == 413


def test_rate_limiting_health_exempt_and_thumbnail_bucket(cfg):
    with TestClient(create_app(replace(cfg, rate_limit=1, asset_rate_limit=1), encoder_factory=Encoder)) as client:
        headers = {'X-API-Key': KEY}
        assert client.post('/search/text', json={'query': 'cat'}, headers=headers).status_code == 200
        limited = client.post('/search/text', json={'query': 'cat'}, headers=headers)
        assert limited.status_code == 429 and int(limited.headers['Retry-After']) > 0
        assert client.get('/thumbnails/0', headers=headers).status_code == 200
        assert client.get('/thumbnails/0', headers=headers).status_code == 429
        assert client.get('/health').status_code == 200


def test_cors_explicit_origin_even_on_auth_error(client, cfg):
    allowed = client.post('/search/text', headers={'Origin': cfg.frontend_origin}, json={'query': 'cat'})
    assert allowed.status_code == 401
    assert allowed.headers['access-control-allow-origin'] == cfg.frontend_origin
    denied = client.get('/health', headers={'Origin': 'https://untrusted.example'})
    assert 'access-control-allow-origin' not in denied.headers
    preflight = client.options('/search', headers={'Origin': cfg.frontend_origin, 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'X-API-Key'})
    assert preflight.status_code == 200


def test_wrong_model_and_invalid_index_refused(cfg):
    with pytest.raises(ValueError, match='model/revision'):
        SearchIndex(cfg.index_dir, 'wrong-model', cfg.model_revision)
    np.save(cfg.index_dir / 'index.npy', np.zeros((10, 512), dtype=np.float32))
    with pytest.raises(ValueError, match='normalized'):
        SearchIndex(cfg.index_dir, cfg.model_id, cfg.model_revision)
