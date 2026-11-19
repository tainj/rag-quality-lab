"""
Standalone RAGAS evaluation script.

Usage:
    python evalkit/run_ragas_eval.py

All settings come from config.py.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    DATA_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RAGAS_EMBED_MODEL,
    RAGAS_METRICS,
    RAGAS_NUM_WORKERS,
    RAGAS_TIMEOUT,
)
from evalkit.ragas_eval import run_ragas_eval


def load_traces(traces_path: Path):
    """Load traces from JSONL file."""
    if not traces_path.exists():
        print(f"ERROR: Traces file not found: {traces_path}", file=sys.stderr)
        sys.exit(1)

    traces = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))

    print(f"Loaded {len(traces)} traces from {traces_path}")
    return traces


def build_ragas_rows(traces, include_unanswerable=False):
    """Build RAGAS-compatible rows from traces."""
    rows = []
    for t in traces:
        if not include_unanswerable and t.get("question_type") == "unanswerable":
            continue

        contexts = [c.get("content", "") for c in t.get("retrieved_chunks", [])]

        rows.append(
            {
                "question": t.get("question", ""),
                "answer": t.get("answer", ""),
                "contexts": contexts or ["(no context retrieved)"],
                "ground_truth": t.get("reference_answer", ""),
            }
        )

    return rows


def save_results(collection_name: str, result_df, output_dir: Path):
    """Save evaluation results to multiple formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"generation_report_{collection_name}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": result_df.to_dict()}, f, ensure_ascii=False, indent=2)
    print(f"Saved report -> {report_path}")

    csv_path = output_dir / f"generation_results_{collection_name}.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Saved results -> {csv_path}")

    jsonl_path = output_dir / f"generation_ragas_{collection_name}.jsonl"
    if hasattr(result_df, "dataset") and result_df.dataset is not None:
        result_df.dataset.to_json(
            str(jsonl_path),
            force_ascii=False,
            orient="records",
            lines=True,
        )
        print(f"Saved ragas dataset -> {jsonl_path}")


def run_collection_eval(
    collection_name: str,
    rows,
    metrics,
    embed_model: str,
    num_workers: int,
    timeout: int,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
):
    """Run RAGAS evaluation for a single collection."""
    print(f"\n{'=' * 60}")
    print(f"Running RAGAS for {collection_name} ({len(rows)} questions)")
    print(f"{'=' * 60}")

    start_time = time.time()

    result = run_ragas_eval(
        rows=rows,
        metrics=metrics,
        embed_model=embed_model,
        num_workers=num_workers,
        timeout=timeout,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
    )

    elapsed = time.time() - start_time
    result_df = result.to_pandas()

    print(f"\nCompleted in {elapsed:.1f}s")
    print("\nResults:")
    print(result_df)

    return result_df


def main():
    traces_path = DATA_DIR / "generation_traces.jsonl"
    all_traces = load_traces(traces_path)

    # Determine which collections to evaluate
    collections = sorted(
        {t.get("collection") for t in all_traces if t.get("collection")}
    )

    print(f"\nEvaluating {len(collections)} collections:")
    for c in collections:
        print(f"  - {c}")

    for ci, collection_name in enumerate(collections, 1):
        col_traces = [t for t in all_traces if t.get("collection") == collection_name]
        rows = build_ragas_rows(col_traces, include_unanswerable=False)

        if not rows:
            print(f"\nSkipping {collection_name}: no answerable questions")
            continue

        print(f"\n[{ci}/{len(collections)}] Processing {collection_name}...")

        try:
            result_df = run_collection_eval(
                collection_name=collection_name,
                rows=rows,
                metrics=RAGAS_METRICS,
                embed_model=RAGAS_EMBED_MODEL,
                num_workers=RAGAS_NUM_WORKERS,
                timeout=RAGAS_TIMEOUT,
                llm_model=LLM_MODEL,
                llm_base_url=LLM_BASE_URL,
                llm_api_key=LLM_API_KEY,
            )

            save_results(
                collection_name=collection_name,
                result_df=result_df,
                output_dir=DATA_DIR,
            )

        except Exception as e:
            print(f"\nERROR evaluating {collection_name}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
