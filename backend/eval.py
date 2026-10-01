#!/usr/bin/env python3
"""Evaluate frozen embeddings with self-excluded, class-stratified queries."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from app.config import Settings
from app.reporting import ROOT, hardware, sha256, write_report
from app.retrieval import SearchIndex, top_k


def evaluate(index, ids):
    hits = []
    query_details = []
    for row_id in ids:
        neighbors, _ = top_k(index.matrix, index.matrix[row_id], 5, exclude=int(row_id))
        relevant = [index.metadata[int(i)]['class_id'] == index.metadata[int(row_id)]['class_id'] for i in neighbors]
        hits.append((float(relevant[0]), sum(relevant) / 5, float(any(relevant))))
        query_details.append({'query_id': int(row_id), 'neighbor_ids': neighbors.tolist(), 'relevant': relevant})
    mean = np.mean(hits, axis=0)
    return dict(precision_at_1=float(mean[0]), precision_at_5=float(mean[1]), top_5_hit_rate=float(mean[2])), query_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries-per-class', type=int, default=5, help='Default 695 queries; equal class weighting.')
    parser.add_argument('--all', action='store_true', help='Use every image as a leave-one-out query.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results', type=Path, default=ROOT / 'RESULTS.md')
    args = parser.parse_args()
    if args.queries_per_class < 1:
        parser.error('queries-per-class must be >= 1')
    cfg = Settings()
    index = SearchIndex(cfg.index_dir, cfg.model_id, cfg.model_revision)
    groups = defaultdict(list)
    for row in index.metadata:
        groups[row['class_id']].append(row['id'])
    rng = np.random.default_rng(args.seed)
    ids = list(range(len(index.metadata))) if args.all else [int(i) for key in sorted(groups)
           for i in rng.choice(groups[key], min(args.queries_per_class, len(groups[key])), replace=False)]
    metrics, details = evaluate(index, ids)
    hw = hardware()
    counts = f'Dataset: {len(index.metadata):,} images · classes: {len(groups)} · queries: {len(ids):,}'
    md = (f'## Retrieval evaluation\n\n{counts}\n\n'
          f'Method: leave-one-out, frozen CLIP image embeddings; query row is excluded from ranking. '
          f'{"All images" if args.all else "Class-stratified sample"}; seed {args.seed}.\n\n'
          f'- Precision@1: **{metrics["precision_at_1"]:.4f}** — fraction of queries whose first neighbor has the same class.\n'
          f'- Precision@5: **{metrics["precision_at_5"]:.4f}** — mean fraction of five neighbors with the same class.\n'
          f'- Top-5 hit rate: **{metrics["top_5_hit_rate"]:.4f}** — fraction of queries with at least one same-class neighbor in the top five.\n\n'
          f'Hardware: {hw["cpu_model"]}; {hw["logical_cores"]} logical cores; {hw["ram_gib"]} GiB RAM; {hw["platform"]}.\n\n'
          f'Index: `{index.manifest["index_id"]}`; model: `{cfg.model_id}`; revision: `{cfg.model_revision}`.\n'
          'These measure class agreement for image queries, not text retrieval or a probability of correctness.')
    write_report('evaluation', dict(metrics=metrics, dataset_size=len(index.metadata), number_of_classes=len(groups),
                 number_of_queries=len(ids), seed=args.seed, all_images=args.all, hardware=hw,
                 index_manifest=index.manifest, index_sha256=sha256(cfg.index_dir / 'index.npy'),
                 queries=details, methodology='leave-one-out class agreement using cached frozen image embeddings'), md, args.results)


if __name__ == '__main__':
    main()
