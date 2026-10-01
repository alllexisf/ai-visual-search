import json
from pathlib import Path
import numpy as np


def top_k(matrix: np.ndarray, vector: np.ndarray, k: int, exclude: int | None = None):
    """One exact cosine matrix multiply; exclude the query row for evaluation."""
    scores = matrix @ vector
    if exclude is not None:
        scores[exclude] = -np.inf
    k = min(k, len(scores) - (exclude is not None))
    candidates = np.argpartition(-scores, k - 1)[:k]
    order = np.lexsort((candidates, -scores[candidates]))
    ids = candidates[order]
    return ids, np.clip(scores[ids], -1.0, 1.0)


class SearchIndex:
    def __init__(self, directory: Path, model_id: str, revision: str):
        self.directory = directory
        self.manifest = json.loads((directory / 'manifest.json').read_text())
        if self.manifest['model_id'] != model_id or self.manifest['model_revision'] != revision:
            raise ValueError('Index model/revision differs from configuration. Rebuild the index.')
        self.matrix = np.load(directory / 'index.npy', allow_pickle=False)
        self.metadata = json.loads((directory / 'metadata.json').read_text())
        self.pca = json.loads((directory / 'pca.json').read_text())
        n = len(self.metadata)
        if self.matrix.shape != (n, 512) or n < 6 or self.manifest['dataset_size'] != n:
            raise ValueError('Incomplete or misaligned index. Rebuild the index.')
        if not np.isfinite(self.matrix).all() or not np.allclose(np.linalg.norm(self.matrix, axis=1), 1, atol=1e-4):
            raise ValueError('Embeddings must be finite and L2 normalized.')
        if len(self.pca['points']) != n or any(row['id'] != i for i, row in enumerate(self.metadata)):
            raise ValueError('PCA/metadata row alignment is invalid.')
        self.mean = np.asarray(self.pca['mean'], dtype=np.float32)
        self.components = np.asarray(self.pca['components'], dtype=np.float32)
        if self.mean.shape != (512,) or self.components.shape != (2, 512):
            raise ValueError('Invalid PCA basis.')

    def project(self, vector):
        return ((vector - self.mean) @ self.components.T).tolist()

    def rank(self, vector, k):
        ids, scores = top_k(self.matrix, vector, k)
        return [dict(id=int(i), label=self.metadata[i]['label'], dataset=self.metadata[i]['dataset'],
                     similarity=float(score), thumbnail_url=f'/thumbnails/{i}',
                     point=self.pca['points'][int(i)]) for i, score in zip(ids, scores)]
