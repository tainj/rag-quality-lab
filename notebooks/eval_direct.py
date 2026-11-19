"""
Direct generation evaluation: manual ThreadPoolExecutor + tqdm per metric.
Replaces deepeval's evaluate() with per-metric parallel execution and sampling.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import json
from pathlib import Path
import random
import sys

from deepeval.models.llms import GPTModel
from deepeval.test_case import LLMTestCase
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RAGAS_EVAL_SAMPLE_FRACTION,
    RAGAS_METRICS,
    RAGAS_NUM_WORKERS,
    RAGAS_TIMEOUT,
)
from evalkit.llm_http import http_clients
from evalkit.llm_judge import is_declined_verdict, judge_negative_rejection
from evalkit.metrics import (
    get_answer_correctness_embeddings,
    make_metric,
    patch_model,
)

DATA_DIR = (Path(__file__).resolve().parent.parent / "dataset").resolve()


def _measure_one(metric_name, model, ac_kwargs, test_case):
    metric = make_metric(metric_name, model=model, answer_correctness_kwargs=ac_kwargs)
    try:
        metric.measure(test_case, _show_indicator=False)
        return metric.score, metric.reason, metric.error
    except Exception as e:  # noqa: BLE001 — one metric must not stop the evaluation
        return None, None, str(e)


def judge_negative_rejection_section(traces: list[dict], agents: list[str]):
    print("\n=== Section 4: Negative rejection check ===")
    print(f"{'collection':<35} {'unanswerable':>12} {'declined':>10} {'rate':>8}")
    print("-" * 68)

    results = {}

    for collection_name in agents:
        col_traces = [t for t in traces if t.get("collection") == collection_name]
        unanswerable = [
            t for t in col_traces if t.get("question_type") == "unanswerable"
        ]
        declined = 0

        if unanswerable:
            with ThreadPoolExecutor(max_workers=max(1, RAGAS_NUM_WORKERS)) as executor:
                futures = [
                    executor.submit(
                        judge_negative_rejection,
                        t.get("question", ""),
                        t.get("answer", ""),
                    )
                    for t in unanswerable
                ]
                for fut in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"judge {collection_name}",
                    unit="q",
                ):
                    try:
                        verdict = fut.result()
                        if is_declined_verdict(verdict):
                            declined += 1
                    except Exception as e:  # noqa: BLE001 — keep failures of independent judge calls
                        tqdm.write(f"WARNING: judge failed: {e}")

        rate = declined / len(unanswerable) if unanswerable else 0.0
        results[collection_name] = {
            "unanswerable": len(unanswerable),
            "declined": declined,
            "rate": rate,
        }
        print(
            f"{collection_name:<35} {len(unanswerable):>12} {declined:>10} {rate:>7.0%}"
        )

    out_path = DATA_DIR / "generation_negative_rejection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved negative rejection results -> {out_path}")


def run_direct_eval(traces: list[dict], agents: list[str]):
    print("\n=== Section 5: Direct evaluation (manual parallel) ===")

    model_name = LLM_MODEL
    if model_name.startswith("openai/"):
        model_name = model_name.split("/", 1)[1]

    model = GPTModel(
        model=model_name,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        generation_kwargs={"max_tokens": 2000},
        timeout=RAGAS_TIMEOUT,
        **http_clients(RAGAS_TIMEOUT),
    )
    model = patch_model(model)

    valid_metrics = {
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    }
    metric_names = [
        m for m in RAGAS_METRICS if isinstance(m, str) and m in valid_metrics
    ]

    if not metric_names:
        print("No valid metrics selected. Skipping Section 5.")
        return

    print(f"Metrics: {metric_names}")

    ac_embeddings, ac_semantic_weight = get_answer_correctness_embeddings()
    answer_correctness_kwargs = {
        "threshold": 0.0,
        "include_reason": True,
        "factual_weight": 1.0 if ac_semantic_weight <= 0 else 0.75,
        "semantic_weight": ac_semantic_weight,
        "embeddings": ac_embeddings,
    }

    random.seed(42)

    for ci, collection_name in enumerate(agents, 1):
        col_traces = [t for t in traces if t.get("collection") == collection_name]
        answerable = [t for t in col_traces if t.get("question_type") != "unanswerable"]

        print(
            f"\n=== [{ci}/{len(agents)}] Direct eval for {collection_name} "
            f"({len(answerable)} answerable) ==="
        )

        if not answerable:
            print(f"[skip] {collection_name}: no answerable traces")
            continue

        # Build test cases
        all_test_cases = []
        all_metadata = []

        for trace in answerable:
            # Skip retrieval-only questions — they have no reference answer,
            # so metrics like context_recall/answer_correctness are meaningless
            if trace.get("question_type") == "retrieval_only":
                continue

            question = (trace.get("question") or "").strip()
            answer = str(trace.get("answer") or "")
            reference = str(trace.get("reference_answer") or "")
            contexts = [
                c.get("content", "")
                for c in (trace.get("retrieved_chunks") or [])
                if c.get("content")
            ]

            if not question:
                continue

            tc = LLMTestCase(
                input=question,
                actual_output=answer or "(no answer)",
                expected_output=reference or "(no reference)",
                retrieval_context=contexts or ["No retrieval context provided."],
            )
            all_test_cases.append(tc)
            all_metadata.append(
                {
                    "collection": collection_name,
                    "question": question,
                    "answer": answer,
                    "reference": reference,
                    "has_reference": bool(reference),
                    "num_contexts": len(contexts),
                }
            )

        if not all_test_cases:
            continue

        # Sampling
        n_total = len(all_test_cases)
        sample_n = max(1, int(n_total * RAGAS_EVAL_SAMPLE_FRACTION))
        if RAGAS_EVAL_SAMPLE_FRACTION <= 0 or sample_n >= n_total:
            test_cases = all_test_cases
            orig_by_sampled = dict(enumerate(range(n_total)))
            print(f"  All {n_total} test cases (no sampling)")
        else:
            sampled_indices = set(random.sample(range(n_total), sample_n))
            test_cases = [
                tc for i, tc in enumerate(all_test_cases) if i in sampled_indices
            ]
            orig_by_sampled = dict(enumerate(sorted(sampled_indices)))
            print(
                f"  Sampled {len(test_cases)}/{n_total} ({RAGAS_EVAL_SAMPLE_FRACTION:.0%})"
            )

        # Initialize rows for ALL original test cases (NaN for non-sampled)
        rows = []
        for meta in all_metadata:
            rows.append(
                {
                    "collection": meta["collection"],
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "reference": meta["reference"],
                    "num_contexts": meta["num_contexts"],
                }
            )

        # Single progress bar across all metrics × test cases
        ref_dependent = {"context_recall", "answer_correctness"}
        scores: dict[str, list] = {}
        reasons: dict[str, list] = {}
        n_has_ref = sum(
            1
            for i in range(len(test_cases))
            if all_metadata[orig_by_sampled[i]]["has_reference"]
        )
        ref_dep_count = sum(1 for m in metric_names if m in ref_dependent) * n_has_ref
        ref_indep_count = sum(1 for m in metric_names if m not in ref_dependent) * len(
            test_cases
        )
        total_tasks = ref_indep_count + ref_dep_count

        for m in metric_names:
            scores[m] = [None] * len(test_cases)
            reasons[m] = [None] * len(test_cases)

        # Metrics that require a reference answer
        ref_dependent = {"context_recall", "answer_correctness"}

        with ThreadPoolExecutor(max_workers=max(1, RAGAS_NUM_WORKERS)) as ex:
            futures = {}
            for m in metric_names:
                for i, tc in enumerate(test_cases):
                    if (
                        m in ref_dependent
                        and not all_metadata[orig_by_sampled[i]]["has_reference"]
                    ):
                        continue
                    futures[
                        ex.submit(_measure_one, m, model, answer_correctness_kwargs, tc)
                    ] = (m, i)

            for f in tqdm(as_completed(futures), total=total_tasks, desc="Evaluating"):
                m, i = futures[f]
                try:
                    score, reason, error = f.result(timeout=RAGAS_TIMEOUT + 60)
                    scores[m][i] = score
                    reasons[m][i] = reason if error is None else f"Error: {error}"
                except TimeoutError:
                    scores[m][i] = None
                    reasons[m][i] = f"Timeout after {RAGAS_TIMEOUT}s"
                except Exception as e:  # noqa: BLE001 — persist failure of this independent metric task
                    scores[m][i] = None
                    reasons[m][i] = str(e)

        # Write sampled results back to original rows
        for sampled_idx, orig_idx in orig_by_sampled.items():
            for m in metric_names:
                rows[orig_idx][m] = scores[m][sampled_idx]
                rows[orig_idx][f"{m}_reason"] = reasons[m][sampled_idx]

        # Post-processing
        for row in rows:
            if row["num_contexts"] == 0 and "faithfulness" in row:
                row["faithfulness"] = 0.0
                row["faithfulness_reason"] = "No retrieval context provided."

        df = pd.DataFrame(rows)

        print(df.head(20).to_string())

        out_report = DATA_DIR / f"generation_report_deepeval_{collection_name}.json"
        out_csv = DATA_DIR / f"generation_results_deepeval_{collection_name}.csv"
        out_jsonl = DATA_DIR / f"generation_deepeval_{collection_name}.jsonl"

        summary = {}
        score_cols = [c for c in metric_names if c in df.columns]
        if score_cols:
            numeric_df = df[score_cols].select_dtypes(include="number")
            if not numeric_df.empty:
                summary = json.loads(numeric_df.mean().to_json())

        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "collection": collection_name,
                    "framework": "deepeval",
                    "metrics": metric_names,
                    "summary": summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        df.to_csv(out_csv, index=False)
        df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)

        print(f"Saved report -> {out_report}")
        print(f"Saved results -> {out_csv}")
        print(f"Saved DeepEval dataset -> {out_jsonl}")


def main():
    traces_path = DATA_DIR / "generation_traces.jsonl"

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found.")
        sys.exit(1)

    print(f"Loading traces from {traces_path}")

    traces = []
    with open(traces_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                traces.append(json.loads(line))
            except json.JSONDecodeError as e:
                tqdm.write(f"WARNING: cannot parse trace line: {e}")

    print(f"Loaded {len(traces)} traces")

    manifest_path = DATA_DIR / "eval_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        agents = manifest.get("collections", [])
        if not isinstance(agents, list) or not agents:
            agents = sorted({t["collection"] for t in traces if "collection" in t})
    else:
        agents = sorted({t["collection"] for t in traces if "collection" in t})

    print(f"Evaluating {len(agents)} collections: {agents}")

    judge_negative_rejection_section(traces, agents)
    run_direct_eval(traces, agents)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
