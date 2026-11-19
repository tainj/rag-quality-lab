"""
Shared evaluation harness.

Public benchmarks (BEIR, HotpotQA) and synthetic datasets flow through the
same retrieval grid and land in the same metrics table, distinguished only
by a `dataset_source` tag.

Two evaluation strategies are supported:
  - span: gold_spans have char offsets -> retrieve chunks and match by overlap
  - document: gold_spans have document-level relevance -> check if gold doc
              appears in top-k results (BEIR style)

The runner auto-detects the strategy per question based on `span_granularity`.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from evalkit.retrieval_metrics import eval_single_query
from evalkit.span_matching import evidence_metrics


def run_retrieval_grid(
    qa_dataset: list[dict],
    client: Any,
    top_k: int = 5,
    k_set: tuple[int, ...] = (1, 3, 5),
    dataset_collections: dict[str, str] | None = None,
    relevance_threshold: float = 0.3,
    relevance_metric: str = "coverage",
    continue_on_error: bool = False,
) -> pd.DataFrame:
    """Run retrieval evaluation across collections and datasets.

    Supports both span-level and document-level ground truth.
    Span-level: retrieve all chunks, match by char-span overlap.
    Document-level: search per gold document, check if gold doc in top-k.

    ``dataset_collections`` maps dataset source (key) -> collection name.
    Each source is evaluated only against its own collection.
    """
    if dataset_collections is None:
        dataset_collections = {
            source: f"auto_{source}"
            for source in sorted(
                {q.get("dataset_source", "unknown") for q in qa_dataset}
            )
        }

    rows: list[dict] = []

    total = 0
    for source, coll in dataset_collections.items():
        qa = [
            q
            for q in qa_dataset
            if q.get("dataset_source") == source and q.get("gold_spans")
        ]
        total += len(qa)
        print(f"  {coll}: {len(qa)} answerable questions (source={source})")

    print(f"Total: {total} queries across {len(dataset_collections)} collection(s)")

    t0 = time.perf_counter()
    for source, collection_name in dataset_collections.items():
        qa = [
            q
            for q in qa_dataset
            if q.get("dataset_source") == source and q.get("gold_spans")
        ]
        sub_pbar = tqdm(qa, desc=f"{collection_name}", unit="q", leave=False)
        for q in sub_pbar:
            error = None
            try:
                results = client.search(
                    q["question"], collection_name=collection_name, top_k=top_k
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                error = f"{type(exc).__name__}: {exc}"
                rows.append(
                    {
                        "dataset_source": source,
                        "collection": collection_name,
                        "qid": q["qid"],
                        "question_type": q.get("question_type", "unknown"),
                        "span_granularity": q.get("span_granularity", "unknown"),
                        **{
                            f"{metric}@{k}": None
                            for k in k_set
                            for metric in (
                                "recall",
                                "precision",
                                "ndcg",
                                "evidence_precision",
                                "f1",
                            )
                        },
                        "mrr": None,
                        "error": error,
                    }
                )
                continue
            is_document_level = q.get("span_granularity") == "document"

            if is_document_level:
                gold_doc_ids = {gs["doc_id"] for gs in q["gold_spans"]}
                # Build relevance dict: gold doc_id -> 1.0
                relevant_docs = {doc_id: 1.0 for doc_id in gold_doc_ids}
                # Extract unique doc_ids in result order
                seen_doc_ids = []
                seen = set()
                for r in results:
                    did = r.get("doc_id")
                    if did and did not in seen:
                        seen.add(did)
                        seen_doc_ids.append(did)
                metrics = eval_single_query(seen_doc_ids, relevant_docs, ks=k_set)
            else:
                metrics = evidence_metrics(q["gold_spans"], results, ks=k_set)

            rows.append(
                {
                    "dataset_source": source,
                    "collection": collection_name,
                    "qid": q["qid"],
                    "question_type": q.get("question_type", "unknown"),
                    "span_granularity": q.get("span_granularity", "unknown"),
                    **metrics,
                    "error": error,
                }
            )
        sub_pbar.close()

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({total / max(elapsed, 1e-9):.1f} q/s)")
    return pd.DataFrame(rows)
