#!/usr/bin/env python3
"""Download all five torchvision splits and build an immutable CLIP/PCA index."""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageOps
from app.config import Settings
from app.encoder import ClipEncoder
from app.reporting import hardware, sha256


def collect(dataset_dir):
    from torchvision.datasets import Flowers102, OxfordIIITPet
    rows = []
    # Private path/label arrays are intentional and covered by the pinned torchvision version.
    for split in ('trainval', 'test'):
        ds = OxfordIIITPet(dataset_dir, split=split, target_types='category', download=True)
        rows.extend({'image_path': str(path.relative_to(dataset_dir)), 'label': ds.classes[label],
                     'class_id': f'pets:{label}', 'dataset': 'Pets', 'source_split': split}
                    for path, label in zip(ds._images, ds._labels))
    for split in ('train', 'val', 'test'):
        ds = Flowers102(dataset_dir, split=split, download=True)
        rows.extend({'image_path': str(path.relative_to(dataset_dir)), 'label': ds.classes[label],
                     'class_id': f'flowers:{label}', 'dataset': 'Flowers', 'source_split': split}
                    for path, label in zip(ds._image_files, ds._labels))
    rows.sort(key=lambda row: (row['dataset'], row['image_path']))
    if len({row['image_path'] for row in rows}) != len(rows):
        raise RuntimeError('Dataset splits overlap by path; refusing a duplicate index.')
    if len(rows) != 15538 or len({row['class_id'] for row in rows}) != 139:
        raise RuntimeError(f'Unexpected full dataset cardinality: {len(rows)} images. Expected 15,538 / 139 classes.')
    return [dict(row, id=i) for i, row in enumerate(rows)]


def pca(matrix):
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    covariance = centered.T.astype(np.float64) @ centered.astype(np.float64) / (len(matrix) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    components = eigenvectors[:, -2:][:, ::-1].T.astype(np.float32)
    # Fix arbitrary eigenvector signs for stable presentation.
    for row in components:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    return {'mean': mean.tolist(), 'components': components.tolist(),
            'points': (centered @ components.T).tolist(),
            'explained_variance_ratio': (eigenvalues[-2:][::-1] / eigenvalues.sum()).tolist()}


def main():
    cfg = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=int(os.getenv('INDEX_BATCH_SIZE', '32')))
    parser.add_argument('--thumbnail-size', type=int, default=int(os.getenv('THUMBNAIL_SIZE', '320')))
    parser.add_argument('--force', action='store_true', help='Rebuild; stop the API before replacing a live index.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.thumbnail_size < 32:
        parser.error('batch-size must be positive and thumbnail-size must be >= 32')
    cfg.index_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.dataset_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.index_dir.parent / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (cfg.index_dir / 'manifest.json').exists() and not args.force:
            from app.retrieval import SearchIndex
            index = SearchIndex(cfg.index_dir, cfg.model_id, cfg.model_revision)
            missing = [row['id'] for row in index.metadata if not (cfg.index_dir / 'thumbnails' / f"{row['id']}.jpg").exists()]
            if missing:
                raise RuntimeError('Thumbnails missing; rebuild with --force.')
            print(f'Using completed index: {index.manifest["index_id"]} ({len(index.metadata)} images)', flush=True)
            return
        stage = cfg.index_dir.with_name(cfg.index_dir.name + '.building')
        shutil.rmtree(stage, ignore_errors=True)
        (stage / 'thumbnails').mkdir(parents=True)
        start = time.perf_counter()
        rows = collect(cfg.dataset_dir)
        encoder = ClipEncoder(cfg)
        matrix = np.empty((len(rows), 512), dtype=np.float32)
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            images = []
            try:
                for row in batch:
                    path = cfg.dataset_dir / row['image_path']
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert('RGB')
                    images.append(image)
                    thumb = image.copy()
                    thumb.thumbnail((args.thumbnail_size, args.thumbnail_size), Image.Resampling.LANCZOS)
                    thumb.save(stage / 'thumbnails' / f"{row['id']}.jpg", quality=82, optimize=True)
                    thumb.close()
                matrix[offset:offset + len(batch)] = encoder.images(images)
            except Exception as exc:
                raise RuntimeError(f'Index batch at row {offset} failed; no completed index was published. {exc}') from exc
            finally:
                for image in images:
                    image.close()
            print(f'{offset + len(batch):,}/{len(rows):,} embedded | elapsed {time.perf_counter() - start:.0f}s', flush=True)
        np.save(stage / 'index.npy', matrix, allow_pickle=False)
        (stage / 'metadata.json').write_text(json.dumps(rows, indent=2) + '\n')
        (stage / 'pca.json').write_text(json.dumps(pca(matrix)) + '\n')
        hashes = {name: sha256(stage / name) for name in ('index.npy', 'metadata.json', 'pca.json')}
        index_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        manifest = {'schema_version': 1, 'index_id': index_id, 'model_id': cfg.model_id,
                    'model_revision': cfg.model_revision, 'dataset_size': len(rows), 'number_of_classes': 139,
                    'datasets': dict(Counter(row['dataset'] for row in rows)), 'embedding_dimension': 512,
                    'normalized': True, 'thumbnail_size': args.thumbnail_size,
                    'built_at_utc': datetime.now(timezone.utc).isoformat(), 'hardware': hardware(),
                    'torch_threads': cfg.torch_threads, 'batch_size': args.batch_size,
                    'build_seconds': time.perf_counter() - start, 'sha256': hashes}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Stage on same filesystem; completed manifest appears only after all files exist.
        backup = cfg.index_dir.with_name(cfg.index_dir.name + '.previous')
        shutil.rmtree(backup, ignore_errors=True)
        if cfg.index_dir.exists():
            cfg.index_dir.rename(backup)
        stage.rename(cfg.index_dir)
        shutil.rmtree(backup, ignore_errors=True)
        print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
