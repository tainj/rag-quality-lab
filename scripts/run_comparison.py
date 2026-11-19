"""Bounded agent-core comparison of retrieval configurations and prompts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="synthetic_bpmsoft")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--sample", type=int, default=3)
    parser.add_argument(
        "--run-id", default=time.strftime("agent-comparison-%Y%m%d-%H%M%S")
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--request-timeout",
        type=int,
        help="Timeout for platform and judge requests in seconds",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        help="Output budget for LLM judging; does not change saved agent answers",
    )
    parser.add_argument(
        "--preset",
        action="append",
        help="Restrict retrieval presets; repeat to compare",
    )
    parser.add_argument(
        "--one-prompt", action="store_true", help="Use only the saved base prompt"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry failed queries/judgements, preserving successful answers",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All QA, all retrieval presets, both prompts, dev/test generation",
    )
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("Invalid run id")
    inputs = ROOT / "runs" / f"{args.run_id}-inputs"
    inputs.mkdir(parents=True, exist_ok=args.resume)
    settings_file = inputs / "settings.json"
    settings = json.loads(
        (settings_file if args.resume else ROOT / "configs/settings.json").read_text()
    )
    initial_settings = inputs / "settings.initial.json"
    if not initial_settings.exists():
        initial_settings.write_text(
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n"
        )
    budget = (
        args.judge_max_tokens
        if args.judge_max_tokens is not None
        else settings.get("JUDGE_MAX_TOKENS", settings.get("LLM_MAX_TOKENS", 4096))
    )
    if budget <= 0:
        raise ValueError("Judge token budget must be positive")
    timeout = (
        args.request_timeout
        if args.request_timeout is not None
        else settings.get("JUDGE_TIMEOUT", settings.get("RAGAS_TIMEOUT", 60))
    )
    if timeout <= 0:
        raise ValueError("Request timeout must be positive")
    settings.update(
        LLM_VIA_PLATFORM=True,
        LLM_REASONING_EFFORT="none",
        LLM_MAX_RETRIES=0,
        RAGAS_TIMEOUT=timeout,
        JUDGE_MAX_TOKENS=budget,
        JUDGE_TIMEOUT=timeout,
        RAGAS_METRICS=["answer_correctness", "faithfulness"],
    )
    settings_file.write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    os.environ["RAG_EVAL_CONFIG"] = str(settings_file)
    os.environ["LLM_VIA_PLATFORM"] = "1"
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "1"
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
    import config
    from evalkit import pipeline as pl
    from evalkit.client import make_client
    from evalkit.data_catalog import available_datasets
    from evalkit.rag_config import derive_config, preset_configs

    catalog = available_datasets()
    if args.dataset not in catalog:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    client = make_client()
    variants = []
    prompt_b = (
        "Ты — помощник по BPMSoft. Всегда сначала используй поиск. Отвечай только по найденным источникам. "
        "Начни с прямого ответа. Для вопросов о настройке дай короткие нумерованные шаги. "
        "Укажи названия или идентификаторы источников. Если подтверждения нет, явно скажи, что данных недостаточно. "
        "Не придумывай названия полей, версии и действия."
    )
    bases = preset_configs(
        args.preset or (None if args.full else ["stand_dense_512", "stand_hybrid_512"])
    )
    for base in bases:
        variants.append(
            derive_config(
                base,
                {"generation.system_prompt": config.RAG_SYSTEM_PROMPT},
                name=base["name"] + "__concise",
            )
        )
        if args.one_prompt:
            continue
        variants.append(
            derive_config(
                base,
                {"generation.system_prompt": prompt_b},
                name=base["name"] + "__steps_sources",
            )
        )
    opts = pl.RunOptions(
        mode="real",
        generation=True,
        max_questions=None if args.full else args.questions,
        generation_sample=None if args.full else args.sample,
        generation_full=args.full,
        top_n_generation=len(bases),
        select_metric="recall@5",
        retry_errors=args.retry_errors,
    )
    run = (
        pl.load_run(args.run_id)
        if args.resume
        else pl.new_run(
            {args.dataset: catalog[args.dataset]}, variants, opts, args.run_id
        )
    )
    run.state["options"]["continue_on_error"] = True
    run.state.setdefault("evaluation_settings_history", []).append(
        {
            "time": time.time(),
            "judge_max_tokens": budget,
            "request_timeout": timeout,
            "reasoning_effort": settings["LLM_REASONING_EFFORT"],
            "retry_errors": args.retry_errors,
        }
    )
    run.state["options"]["retry_errors"] = args.retry_errors
    variants = run.configs
    retrieval_count = len({pl.retrieval_id(c["config"]) for c in variants})
    run.state["experiment_scope"] = {
        "backend": "agent-core",
        "retrieval_questions_limit": None if args.full else args.questions,
        "generation_questions_per_variant": "all QA" if args.full else args.sample,
        "retrieval_configs": retrieval_count,
        "full": args.full,
        "prompt_variants": len(variants) // retrieval_count,
        "dataset_id": args.dataset,
        "settings_file": str(settings_file.resolve()),
    }
    run.save()
    print("RUN", run.path, flush=True)
    pl.run_pipeline(run, client=client)
    report = [
        f"# Agent-core comparison {run.id}",
        f"Dataset: {args.dataset}. Retrieval: {'all questions' if args.full else args.questions}. Generation: {'all dev/test questions' if args.full else args.sample} per variant.",
        f"All indexing, retrieval and agent answers use agent-core. {retrieval_count} retrieval setups, {len(variants)} variants total. Reranker disabled.",
        "Primary retrieval metric: gold-character coverage recall@5; ties use evidence precision.",
        "Status: " + run.status("pipeline"),
        "Selected: " + run.state.get("best", {}).get("name", "none"),
        run.state.get("best", {}).get(
            "reason", run.state.get("selection_skipped_reason", "")
        ),
        "Failed jobs: "
        + json.dumps(run.state.get("job_errors", {}), ensure_ascii=False),
        (
            "Full generated dataset; dev-only selection, test scores reported separately."
            if args.full
            else "Pilot sample; not a statistically conclusive full-dataset prompt benchmark."
        )
        + " Provider usage excludes judge calls.",
    ]
    run.file("comparison-report.md").write_text("\n\n".join(report) + "\n")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
