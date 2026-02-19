#!/usr/bin/env python3
"""
scripts/tune_retrieval.py

Grid-search the retrieval hyperparameters on a validation split.

Example:
    python scripts/tune_retrieval.py \
        --val-csv data/IUXRAY/indiana_merged_cleaned_val.csv \
        --projection-col projection \
        --report-col report_text \
        --image-col image_path \
        --image-root data/IUXRAY/images_normalized \
        --sim-thresholds 0.55 0.65 0.75 \
        --hybrid-weights 0.6 0.4 0.7 0.3 \
        --min-k 2 3 4 \
        --max-candidates 30 40 50 \
        --top-k 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import weaviate
from backend.app.core.rag_pipeline import (
    embed_image,
    embed_text,
)

DEFAULT_WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
VECTOR_CLASS = os.getenv("WEAVIATE_CLASS", "MedicalReport")


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0 or math.isnan(norm):
        return vec
    return vec / norm


def _to_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        if ";" in value:
            return [item.strip() for item in value.split(";") if item.strip()]
        return [value]
    if isinstance(value, Sequence):
        return [str(v) for v in value]
    return [str(value)]


def load_validation_examples(
    csv_path: Path,
    report_col: str,
    projection_col: str,
    image_col: str | None,
    image_root: Path | None,
    limit: int | None,
) -> List[dict]:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit]
    examples: List[dict] = []
    for _, row in df.iterrows():
        question = str(row[report_col]) if report_col in df.columns else ""
        projections = _to_list(row.get(projection_col))
        images = []
        if image_col and image_col in df.columns:
            for img_name in _to_list(row[image_col]):
                if not img_name:
                    continue
                if image_root:
                    images.append(str((image_root / img_name).resolve()))
                else:
                    images.append(img_name)
        if not question and not images:
            continue
        examples.append(
            {
                "question": question,
                "images": images,
                "targets": set(projections),
            }
        )
    return examples


def fetch_candidates(
    client: weaviate.Client,
    query_vec: np.ndarray,
    max_candidates: int,
    domain_filter: str | None,
) -> List[dict]:
    query = (
        client.query.get(
            VECTOR_CLASS,
            [
                "projection",
                "image_path",
                "left_image_path",
                "right_image_path",
                "text_embedding",
                "image_embedding",
                "domain",
            ],
        )
        .with_near_vector({"vector": query_vec.tolist()})
        .with_limit(max_candidates)
        .with_additional(["certainty", "distance", "id"])
    )
    if domain_filter:
        query = query.with_where(
            {
                "path": ["domain"],
                "operator": "Equal",
                "valueText": domain_filter,
            }
        )
    resp = query.do()
    errors = resp.get("errors")
    if errors:
        raise RuntimeError(f"Weaviate query failed: {errors}")
    return resp.get("data", {}).get("Get", {}).get(VECTOR_CLASS, [])


def compute_hybrid_scores(
    candidates: Iterable[dict],
    query_vec: np.ndarray,
    modality: str,
    weight_emb: float,
    weight_cert: float,
) -> List[Tuple[float, dict]]:
    scored: List[Tuple[float, dict]] = []
    q = _normalise(query_vec)

    for cand in candidates:
        props = cand.get("properties", cand)
        certainty = float(cand.get("_additional", {}).get("certainty", 0.0))

        emb_score = 0.0
        try:
            if modality == "image" and props.get("image_embedding"):
                emb = _normalise(np.array(props["image_embedding"], dtype=float))
                emb_score = float(np.dot(q, emb))
            elif modality == "text" and props.get("text_embedding"):
                emb = _normalise(np.array(props["text_embedding"], dtype=float))
                emb_score = float(np.dot(q, emb))
        except Exception:
            emb_score = 0.0

        hybrid = weight_emb * emb_score + weight_cert * certainty
        scored.append((hybrid, cand))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored


def precision_at_k(retrieved: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for item in retrieved_k if item in ground_truth)
    return hits / k


def recall_at_k(retrieved: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    if not ground_truth:
        return 1.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for item in retrieved_k if item in ground_truth)
    return hits / len(ground_truth)


def mrr_at_k(retrieved: Sequence[str], ground_truth: Sequence[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    for idx, item in enumerate(retrieved_k, start=1):
        if item in ground_truth:
            return 1.0 / idx
    return 0.0


def evaluate_configuration(
    client: weaviate.Client,
    examples: Sequence[dict],
    weight_emb: float,
    weight_cert: float,
    sim_threshold: float,
    min_k: int,
    max_candidates: int,
    top_k: int,
    domain_filter: str | None,
) -> dict:
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    mrr_scores: List[float] = []

    for example in tqdm(examples, desc="Evaluating", leave=False):
        question = example["question"]
        images = example["images"]
        targets = example["targets"]

        modality = "text"
        query_vec = None

        if images:
            chosen = None
            for img_path in images:
                if Path(img_path).exists():
                    chosen = img_path
                    break
            if chosen:
                query_vec = embed_image(chosen)
                modality = "image"

        if query_vec is None:
            query_vec = embed_text(question)
            modality = "text"

        candidates = fetch_candidates(
            client=client,
            query_vec=query_vec,
            max_candidates=max_candidates,
            domain_filter=domain_filter,
        )

        scored = compute_hybrid_scores(
            candidates,
            query_vec=query_vec,
            modality=modality,
            weight_emb=weight_emb,
            weight_cert=weight_cert,
        )

        certs = [float(cand.get("_additional", {}).get("certainty", 0.0)) for _, cand in scored]
        if not certs:
            retrieved_ids: List[str] = []
        else:
            adaptive_k = max(min_k, min(len(scored), top_k))
            for idx, cert in enumerate(certs[:-1]):
                if cert - certs[idx + 1] > sim_threshold:
                    adaptive_k = max(min_k, idx + 1)
                    break
            top_scored = scored[:adaptive_k]
            retrieved_ids = []
            for _, cand in top_scored:
                props = cand.get("properties", cand)
                projection = props.get("projection")
                if projection:
                    retrieved_ids.append(str(projection))
                else:
                    image_path = props.get("image_path") or props.get("left_image_path") or props.get("right_image_path")
                    if image_path:
                        retrieved_ids.append(str(image_path))

        gt_list = list(targets)
        precision_scores.append(precision_at_k(retrieved_ids, gt_list, top_k))
        recall_scores.append(recall_at_k(retrieved_ids, gt_list, top_k))
        mrr_scores.append(mrr_at_k(retrieved_ids, gt_list, top_k))

    return {
        "precision_at_k": float(np.mean(precision_scores)),
        "recall_at_k": float(np.mean(recall_scores)),
        "mrr_at_k": float(np.mean(mrr_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuner for MED-RAG retrieval")
    parser.add_argument("--val-csv", required=True, type=Path, help="Path to validation CSV")
    parser.add_argument("--projection-col", required=True, help="Column containing unique projection IDs")
    parser.add_argument("--report-col", default="report_text", help="Column containing the question/report text")
    parser.add_argument("--image-col", default=None, help="Column with image filenames (optional)")
    parser.add_argument("--image-root", type=Path, default=None, help="Root folder prepended to image filenames")
    parser.add_argument("--domain-filter", default=None, help="Optional domain filter (radiology|ophthalmology|pathology)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of validation rows (for quick runs)")
    parser.add_argument("--sim-thresholds", nargs="+", type=float, required=True, help="List of SIM_THRESHOLD values to try")
    parser.add_argument("--min-k", nargs="+", type=int, required=True, help="List of MIN_K values to try")
    parser.add_argument("--max-candidates", nargs="+", type=int, required=True, help="List of MAX_CANDIDATES values to try")
    parser.add_argument("--hybrid-weights", nargs=2, action="append", metavar=("EMB", "CERT"), required=True,
                        help="Pairs of weights for embedding score and certainty (will be normalised)")
    parser.add_argument("--top-k", type=int, default=5, help="K for evaluation metrics")
    parser.add_argument("--output", type=Path, default=Path("tuning_results.csv"), help="CSV file to append results to")
    parser.add_argument("--weaviate-url", default=DEFAULT_WEAVIATE_URL, help="Weaviate endpoint (default: %(default)s)")
    args = parser.parse_args()

    examples = load_validation_examples(
        csv_path=args.val_csv,
        report_col=args.report_col,
        projection_col=args.projection_col,
        image_col=args.image_col,
        image_root=args.image_root,
        limit=args.limit,
    )

    client = weaviate.Client(url=args.weaviate_url)

    grid = list(
        itertools.product(
            args.sim_thresholds,
            args.min_k,
            args.max_candidates,
            args.hybrid_weights,
        )
    )

    records = []
    for sim_threshold, min_k, max_candidates, weight_pair in tqdm(grid, desc="Grid search"):
        w_emb = float(weight_pair[0])
        w_cert = float(weight_pair[1])
        total = w_emb + w_cert
        if total == 0:
            continue
        w_emb /= total
        w_cert /= total

        metrics = evaluate_configuration(
            client=client,
            examples=examples,
            weight_emb=w_emb,
            weight_cert=w_cert,
            sim_threshold=sim_threshold,
            min_k=min_k,
            max_candidates=max_candidates,
            top_k=args.top_k,
            domain_filter=args.domain_filter,
        )
        record = {
            "sim_threshold": sim_threshold,
            "min_k": min_k,
            "max_candidates": max_candidates,
            "weight_emb": w_emb,
            "weight_cert": w_cert,
            **metrics,
        }
        records.append(record)
        print(
            f"sim={sim_threshold:.2f}, min_k={min_k}, maxCand={max_candidates}, "
            f"w_emb={w_emb:.2f}, w_cert={w_cert:.2f} -> "
            f"P@{args.top_k}={metrics['precision_at_k']:.3f}, "
            f"R@{args.top_k}={metrics['recall_at_k']:.3f}, "
            f"MRR@{args.top_k}={metrics['mrr_at_k']:.3f}"
        )

    results_df = pd.DataFrame(records)
    if args.output.exists():
        existing = pd.read_csv(args.output)
        results_df = pd.concat([existing, results_df], ignore_index=True)
    results_df.sort_values(by=["precision_at_k", "recall_at_k", "mrr_at_k"], ascending=False, inplace=True)
    results_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(records)} configurations to {args.output}")
    print(results_df.head())
    

if __name__ == "__main__":
    main()