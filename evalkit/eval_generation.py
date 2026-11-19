"""
Generation evaluation script using DeepEval (Official evaluate() API).
Исправлено для новых версий DeepEval (v1.x+) с использованием AsyncConfig.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RAGAS_METRICS,
    RAGAS_NUM_WORKERS,
    RAGAS_TIMEOUT,
)

os.environ["DEEPEVAL_LOG_STACK_TRACES"] = "1"
os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS"] = str(RAGAS_TIMEOUT)
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = str(RAGAS_TIMEOUT)

try:
    from config import RAGAS_EMBED_MODEL
except ImportError:
    RAGAS_EMBED_MODEL = None

# Официальный API для запуска оценки
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.models.llms import GPTModel
from deepeval.test_case import LLMTestCase

from evalkit.llm_http import http_clients
from evalkit.llm_judge import is_declined_verdict, judge_negative_rejection
from evalkit.metrics import (
    get_answer_correctness_embeddings,
    make_metric,
    patch_model,
)

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

DATA_DIR = (Path(__file__).resolve().parent.parent / "dataset").resolve()


def judge_negative_rejection_section(traces: list[dict], agents: list[str]):
    print("\n=== Section 4: Negative rejection check ===")
    print(f"{'collection':<35} {'unanswerable':>12} {'declined':>10} {'rate':>8}")
    print("-" * 68)

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
        print(
            f"{collection_name:<35} {len(unanswerable):>12} {declined:>10} {rate:>7.0%}"
        )


def run_deepeval(traces: list[dict], agents: list[str]):
    print("\n=== Section 5: DeepEval evaluation ===")

    model_name = LLM_MODEL
    if model_name.startswith("openai/"):
        model_name = model_name.split("/", 1)[1]

    model = GPTModel(
        model=model_name,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
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
        print("No valid DeepEval metrics selected. Skipping Section 5.")
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

    for ci, collection_name in enumerate(agents, 1):
        col_traces = [t for t in traces if t.get("collection") == collection_name]
        answerable = [t for t in col_traces if t.get("question_type") != "unanswerable"]

        print(
            f"\n=== [{ci}/{len(agents)}] DeepEval for {collection_name} ({len(answerable)} answerable) ==="
        )

        if not answerable:
            print(f"[skip] {collection_name}: no answerable traces")
            continue

        test_cases = []
        metadata = []

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
            test_cases.append(tc)
            metadata.append(
                {
                    "collection": collection_name,
                    "question": question,
                    "answer": answer,
                    "reference": reference,
                    "num_contexts": len(contexts),
                }
            )

        if not test_cases:
            continue

        metrics = [
            make_metric(
                m, model=model, answer_correctness_kwargs=answer_correctness_kwargs
            )
            for m in metric_names
        ]

        print(
            f"Running deepeval.evaluate() with throttle_value={max(1, RAGAS_NUM_WORKERS)}..."
        )

        # ИСПРАВЛЕННЫЙ ВЫЗОВ: используем AsyncConfig
        result = evaluate(
            test_cases=test_cases,
            metrics=metrics,
            # 1. Скорость и лимиты (конкурентность)
            async_config=AsyncConfig(
                run_async=True,
                max_concurrent=max(1, RAGAS_NUM_WORKERS),
                throttle_value=0.0,
            ),
            # 2. Кэширование (спасение при сбоях и ускорение повторных запусков)
            cache_config=CacheConfig(
                use_cache=True  # Результаты будут сохраняться в .deepeval/
            ),
            # 3. Устойчивость к ошибкам (чтобы один таймаут не ронял всё)
            error_config=ErrorConfig(
                ignore_errors=False  # При ошибке метрика вернет None/0, но цикл продолжится
            ),
            # 4. Визуализация (чистый вывод)
            display_config=DisplayConfig(
                show_indicator=True,  # Гарантируем, что бар будет виден
                print_results=False,
            ),
        )

        rows = []
        test_result_by_index = {
            tr.index: tr for tr in result.test_results if tr.index is not None
        }
        for i, meta in enumerate(metadata):
            row = {
                "collection": meta["collection"],
                "question": meta["question"],
                "answer": meta["answer"],
                "reference": meta["reference"],
                "num_contexts": meta["num_contexts"],
            }

            test_result = test_result_by_index.get(i)
            if test_result and test_result.metrics_data:
                for metric_data in test_result.metrics_data:
                    m_name = metric_data.name
                    row[m_name] = metric_data.score
                    row[f"{m_name}_reason"] = metric_data.reason

            # Пост-обработка для возврата None там, где это требовала исходная логика
            if not meta["reference"]:
                for m in ["context_precision", "context_recall", "answer_correctness"]:
                    if m in row:
                        row[m] = None

            if meta["num_contexts"] == 0 and "faithfulness" in row:
                row["faithfulness"] = 0.0
                row["faithfulness_reason"] = "No retrieval context provided."

            rows.append(row)

        df = pd.DataFrame(rows)

        if df.empty:
            print(f"[skip] {collection_name}: empty results")
            continue

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
        print(
            f"ERROR: {traces_path} not found. Run Section 3 first to generate traces."
        )
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
    run_deepeval(traces, agents)

    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
