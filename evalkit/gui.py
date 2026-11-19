"""
Gradio UI for the config search pipeline.

    uv run python -m evalkit.cli gui        # http://127.0.0.1:7860

Sidebar: navigation, run selector, live status and recent log; the main
area shows the chosen section (setup, results, chat, full log, deploy).
The pipeline runs in a background thread; a 1 s timer renders its events
and the run state (runs/<id>/run.json), same as the CLI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import fcntl
import html
import json
from pathlib import Path
import shutil
import threading
import time

import gradio as gr
import pandas as pd

import config
from evalkit import pipeline as pl
from evalkit.data_catalog import available_datasets, generated_datasets
from evalkit.rag_config import (
    agent_settings,
    available_presets,
    derive_config,
    describe,
    parse_value,
    preset_configs,
    system_prompt,
)

STEP_TITLES = {
    "prepare": "Данные",
    "index": "Индексация",
    "retrieval": "Retrieval",
    "generation": "Генерация",
    "select": "Выбор",
}
STATUS_WORDS = {
    "done": "готово",
    "done_with_errors": "завершён с ошибками",
    "running": "идёт",
    "failed": "ошибка (см. журнал)",
    "stopped": "остановлен",
    "pending": "не запускался",
}
METRICS_LEGEND = (
    "**Recall@k** — доля символов всех эталонных свидетельств, покрытая top-k. "
    "**Evidence precision** — доля полезного текста в контексте; **F1** балансирует покрытие и лишний текст. "
    "Пересечения одного документа учитываются один раз. nDCG доступен для документальной разметки. "
    "Токены берутся из usage провайдера: все LLM-вызовы ответа агента, без судьи. "
    "Если usage отсутствует или неполон, он не заменяется нулём. Исторические запуски сохраняют прежние метрики."
)


@dataclass
class Job:
    run_id: str
    kind: str  # pipeline | deploy
    stop: threading.Event = field(default_factory=threading.Event)
    events: deque = field(default_factory=lambda: deque(maxlen=400))
    thread: threading.Thread | None = None
    started: float = field(default_factory=time.time)
    error: str | None = None
    result: dict | None = None

    @property
    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _busy() -> Job | None:
    with _LOCK:
        return next((j for j in JOBS.values() if j.alive), None)


def _start(job: Job, target) -> None:
    def body():
        try:
            target(job)
        except Exception as e:  # noqa: BLE001 — shown in the UI log
            job.error = f"{type(e).__name__}: {e}"
            job.events.append(
                pl.Event("pipeline", f"Ошибка: {job.error}", level="error")
            )

    job.thread = threading.Thread(target=body, daemon=True)
    with _LOCK:
        JOBS[job.run_id] = job
    job.thread.start()


def _run_job(job: Job) -> None:
    from evalkit.llm import DEEPEVAL_METRICS

    run = pl.load_run(job.run_id)
    evaluation = {
        "time": time.time(),
        "judge_model": config.LLM_MODEL,
        "judge_max_tokens": config.JUDGE_MAX_TOKENS,
        "judge_include_reason": config.JUDGE_INCLUDE_REASON,
        "judge_max_retries": config.JUDGE_MAX_RETRIES,
        "request_timeout": config.JUDGE_TIMEOUT,
        "metrics": [name for name in config.RAGAS_METRICS if name in DEEPEVAL_METRICS],
        "metrics_requested": config.RAGAS_METRICS,
        "reasoning_effort": config.LLM_REASONING_EFFORT,
        "agent_tool_prompt": config.AGENT_TOOL_PROMPT,
        "retry_errors": run.options.retry_errors,
    }
    run.state.setdefault("evaluation_settings_history", []).append(evaluation)
    run.file("evaluation_settings.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n"
    )
    run.save()
    pl.run_pipeline(run, on_event=job.events.append, stop=job.stop)


def _parse_changes(text: str) -> dict:
    changes = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise gr.Error(f"Ожидается KEY=VALUE, получено: {line}")
        key, raw = line.split("=", 1)
        changes[key.strip()] = parse_value(raw.strip())
    return changes


# ── progress panel ────────────────────────────────────────────


def _recent_events(run_id: str, limit: int = 400) -> list[dict]:
    """Events of the run: live ones if it runs here, else from events.log."""
    job = JOBS.get(run_id)
    if job and job.kind != "pipeline" and job.events:
        return [asdict(ev) for ev in list(job.events)[-limit:]]
    path = config.RUNS_DIR / run_id / "events.log"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    events = [json.loads(line) for line in lines if line.strip()]
    if job and job.error and job.events:
        error = asdict(job.events[-1])
        if not events or error["ts"] > events[-1]["ts"]:
            events.append(error)
    return events[-limit:]


# BPMSoft palette (bpmsoft.ru): orange #f9763d / #eb6012, graphite #2e363e, grey #abafb2
BPM_ORANGE, BPM_ORANGE_DARK = "#f9763d", "#eb6012"
BPM_GRAPHITE, BPM_GREY = "#2e363e", "#abafb2"
THEME = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="slate",
    font=["system-ui", "Segoe UI", "sans-serif"],
    font_mono=["JetBrains Mono", "Fira Code", "ui-monospace", "monospace"],
).set(
    body_background_fill_dark="#0c111b",
    background_fill_primary_dark="#121a28",
    background_fill_secondary_dark="#101724",
    block_background_fill_dark="#121a28",
    block_label_background_fill_dark=BPM_GRAPHITE,
    input_background_fill_dark="#0d1420",
    border_color_primary_dark="#263246",
    table_even_background_fill_dark=BPM_GRAPHITE,
    table_odd_background_fill_dark="#333c45",
    button_primary_background_fill=BPM_ORANGE,
    button_primary_background_fill_dark=BPM_ORANGE,
    button_primary_background_fill_hover=BPM_ORANGE_DARK,
    button_primary_background_fill_hover_dark=BPM_ORANGE_DARK,
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    checkbox_background_color_selected=BPM_ORANGE,
    checkbox_background_color_selected_dark=BPM_ORANGE,
    checkbox_border_color_selected_dark=BPM_ORANGE,
    slider_color=BPM_ORANGE,
    slider_color_dark=BPM_ORANGE,
    color_accent=BPM_ORANGE,
    color_accent_soft_dark="rgba(249,118,61,.18)",
    link_text_color_dark=BPM_ORANGE,
)
FORCE_DARK = """() => {
 document.body.classList.add('dark');
 if (window.innerWidth < 800) setTimeout(() => {
   document.querySelector('.sidebar.open button[aria-label="Toggle Sidebar"]')?.click();
 }, 500);
}"""
CSS = """
.gradio-container { max-width: 100% !important; overflow-x: hidden; }
.prose table { width: 100%; }
.prose td, .prose th { white-space: normal; word-break: break-word; }
.brand { font-size: 20px; font-weight: 700; letter-spacing: .2px; }
.brand-sub { font-size: 12px; opacity: .65; margin-top: 2px; word-break: break-all; }
.nav label { width: 100%; }
.status { font-size: 13px; }
.status .head { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.status .dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.status .run { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.status .pct { margin-left: auto; font-variant-numeric: tabular-nums; opacity: .85; }
.status .total { height: 6px; border-radius: 3px; background: rgba(148,163,184,.18);
                 overflow: hidden; margin-bottom: 12px; }
.status .total > div { height: 100%; background: #f9763d; transition: width .6s; }
.status .step { display: grid; grid-template-columns: 14px 1fr 44px; align-items: center;
                gap: 2px 8px; margin: 6px 0; }
.status .step .bar { grid-column: 2 / 4; height: 3px; border-radius: 2px;
                     background: rgba(148,163,184,.15); overflow: hidden; }
.status .step .bar > div { height: 100%; background: #f9763d; transition: width .6s; }
.status .muted { opacity: .45; }
.status .now { margin-top: 10px; padding: 8px 10px; border-radius: 8px;
               background: rgba(249,118,61,.14); line-height: 1.35; }
.status .err { margin-top: 10px; padding: 8px 10px; border-radius: 8px;
               background: rgba(239,68,68,.14); color: #fca5a5; line-height: 1.35; }
.log { font-family: var(--font-mono); font-size: 12px; line-height: 1.5;
       max-height: 560px; overflow-y: auto; }
.log.short { max-height: 280px; }
.log .row { display: grid; grid-template-columns: 62px 78px 1fr; gap: 8px;
            padding: 2px 0; border-bottom: 1px solid rgba(148,163,184,.08); }
.log.short .row { grid-template-columns: 58px 1fr; }
.log.short .s { display: none; }
.log .t { opacity: .5; }
.log .s { color: #abafb2; }
.log .warn .m { color: #fcd34d; }
.log .error .m { color: #fca5a5; }
.log .done .m { color: #86efac; }
"""

CSS += "\n.gradio-container { max-width: 1540px !important; margin: auto; font-size: 14px; }\n.hero { padding: 30px 8px 24px; }\n.eyebrow { color: #92a0b9; font-size: 11px; letter-spacing: .2em; font-weight: 700; }\n.hero h1 { font-size: clamp(26px, 3vw, 42px); font-weight: 650; letter-spacing: -.045em; margin: 12px 0; line-height: 1.18; color: #f3f6fc; }\n.hero h1 span { color: #f9763d; }\n.hero p { color: #96a4ba; font-size: 14px; margin: 0; line-height: 1.65; }\n.brand { padding: 14px 0 8px; font-size: 22px; letter-spacing: -.04em; display: flex; align-items: center; gap: 9px; }\n.brand span { color: #8795ad; font-weight: 400; margin-left: -6px; }\n.brand i { font-style: normal; display: inline-grid; place-items: center; background: #f9763d; color: #111827; height: 32px; width: 32px; border-radius: 9px; font-size: 22px; }\n.brand-sub { font-size: 11px; color: #8290a7; padding-bottom: 25px; }\n.connection { margin: 10px 0; padding: 10px 12px; border: 1px solid #263246; border-radius: 10px; color: #aebbd0; font-size: 12px; }\n.connection span { display: inline-block; width: 6px; height: 6px; background: #7b91b0; border-radius: 50%; margin-right: 9px; }\n.panel { border: 1px solid #253147 !important; border-radius: 18px !important; padding: 22px !important; background: #111a29 !important; gap: 16px !important; }\n.section-label { display: flex; gap: 12px; align-items: flex-start; margin: 4px 0 12px; }\n.section-label b { color: #f9763d; background: #f9763d12; border: 1px solid #f9763d30; border-radius: 9px; min-width: 32px; height: 32px; display: grid; place-items: center; font-size: 12px; font-weight: 600; }\n.section-label h2 { font-size: 19px; font-weight: 600; letter-spacing: -.025em; margin: 0; color: #e8edf7; }\n.section-label p { font-size: 12px; color: #8b9ab2; line-height: 1.5; margin: 5px 0 0; }\n.run-panel { margin-top: 12px; }\n.kpi-grid { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 14px; margin-bottom: 22px; }\n.kpi { padding: 18px 20px; border: 1px solid #253147; border-radius: 14px; background: linear-gradient(135deg,#151f30,#101724); }\n.kpi span { display: block; font-size: 10px; font-weight: 650; color: #9caac1; letter-spacing: .12em; }\n.kpi strong { display: block; color: #f1f5fb; font-size: 30px; letter-spacing: -.04em; font-weight: 600; line-height: 1.6; font-variant-numeric: tabular-nums; }\n.kpi small { display: block; font-size: 11px; color: #788aa5; }\n.side-note { font-size: 11px; color: #64758f; line-height: 1.7; padding: 20px 0; }\n.nav label { border-radius: 10px !important; padding: 10px 14px !important; }\n.nav label:has(input:checked) { border-color: #f9763d55 !important; background: #f9763d12 !important; }\n@media (max-width: 800px) { .kpi-grid { grid-template-columns: repeat(2,minmax(0,1fr)); } .panel { padding: 16px !important; } }\n"

CSS += '\nmain.contain:has(.sidebar.open) { margin-left: 290px !important; width: calc(100% - 290px) !important; box-sizing: border-box; }\nmain.contain { padding: 20px 28px 40px !important; }\n.column:has(> .block > .html-container > .prose > .section-label) { padding: 22px !important; border: 1px solid #253147 !important; border-radius: 18px !important; background: #111a29 !important; gap: 16px !important; }\n.sidebar label:has(input[type="radio"]) { background: transparent !important; border: 1px solid transparent !important; color: #aab8ce !important; border-radius: 9px !important; }\n.sidebar label:has(input[type="radio"]:checked) { background: #f9763d12 !important; border-color: #f9763d40 !important; color: #f4f6fa !important; }\n@media(max-width:800px) { main.contain { padding: 12px !important; } main.contain:has(.sidebar.open) { margin-left: 0 !important; width:100% !important; } }\n'

CSS += '\n.sidebar.open { position: fixed !important; left: 0 !important; top: 0 !important; bottom: 0 !important; transform: none !important; width: 290px !important; z-index: 50 !important; }\n.gradio-container label:has(input[type="checkbox"]) { background: #192438 !important; color: #c6d3e6 !important; border: 1px solid #2b3a52 !important; border-radius: 8px !important; }\n.gradio-container label:has(input[type="checkbox"]) span { color: #c6d3e6 !important; }\n.gradio-container label:has(input[type="checkbox"]:checked) { background: #f9763d10 !important; border-color: #f9763d60 !important; }\n'


def _progress_html(run_id: str | None) -> str:
    if not run_id:
        return "<div class='status muted'>Запуск не выбран</div>"
    try:
        run = pl.load_run(run_id)
    except FileNotFoundError:
        return f"<div class='status muted'>Запуск {html.escape(run_id)} не найден</div>"
    status = run.state.get("status", {})
    job = JOBS.get(run_id)
    pipeline = status.get("pipeline", "pending")
    running = bool(job and job.alive) or pipeline == "running"
    events = _recent_events(run_id)
    last = {}
    for ev in events:
        last[ev["step"]] = ev
    current = events[-1]["step"] if events and running else None

    steps, total = [], 0.0
    for step, title in STEP_TITLES.items():
        st = status.get(step, "pending")
        if step == current:
            frac, icon, cls = last[step].get("progress") or 0.0, "●", ""
        elif st in ("done", "skipped", "partial"):
            frac = 1.0
            icon = "✓" if st == "done" else "!" if st == "partial" else "–"
            cls = "" if st == "done" else "muted"
        else:
            frac, icon, cls = 0.0, "○", "muted"
        total += frac
        pct = f"{frac:.0%}" if step == current else ""
        steps.append(
            f"<div class='step {cls}'><span>{icon}</span><span>{title}</span>"
            f"<span class='pct'>{pct}</span>"
            f"<div class='bar'><div style='width:{frac * 100:.0f}%'></div></div></div>"
        )
    overall = total / len(STEP_TITLES)
    color = {
        "done": "#22c55e",
        "done_with_errors": "#f59e0b",
        "failed": "#ef4444",
        "stopped": "#f59e0b",
    }.get(pipeline, "#f9763d" if running else "#64748b")
    state = "идёт" if running else STATUS_WORDS.get(pipeline, pipeline)
    out = [
        "<div class='status'>",
        (
            f"<div class='head'><span class='dot' style='background:{color}'></span>"
            f"<span class='run'>{html.escape(run_id)}</span>"
            f"<span class='pct'>{state} · {overall:.0%}</span></div>"
        ),
        f"<div class='total'><div style='width:{overall * 100:.0f}%'></div></div>",
        *steps,
    ]
    if running and events:
        out.append(f"<div class='now'>{html.escape(events[-1]['message'])}</div>")
    errors = run.state.get("errors") or []
    if pipeline == "failed" and errors:
        out.append(f"<div class='err'>{html.escape(errors[-1][:300])}</div>")
    out.append("</div>")
    return "".join(out)


def _log_html(run_id: str | None, limit: int = 300, short: bool = False) -> str:
    """Newest first; per-batch progress lines collapsed to the latest one."""
    if not run_id:
        return "<div class='log'>—</div>"
    rows: list[dict] = []
    for ev in _recent_events(run_id, limit=2000):
        batch = "чанков" in ev["message"] or "Индексация на сервере" in ev["message"]
        if batch and rows and rows[-1].get("batch"):
            rows[-1] = {**ev, "batch": True}
        else:
            rows.append({**ev, "batch": batch})
    rows = rows[-limit:][::-1]
    body = "".join(
        f"<div class='row {r.get('level', 'info')}'>"
        f"<span class='t'>{time.strftime('%H:%M:%S', time.localtime(r['ts']))}</span>"
        f"<span class='s'>{html.escape(r['step'])}</span>"
        f"<span class='m'>{html.escape(r['message'])}</span></div>"
        for r in rows
    )
    return f"<div class='log{' short' if short else ''}'>{body or '—'}</div>"


def _summary_df(run_id: str | None) -> pd.DataFrame:
    if not run_id:
        return pd.DataFrame()
    try:
        run = pl.load_run(run_id)
    except FileNotFoundError:
        return pd.DataFrame()
    table = pl.summarize(run)
    if table.empty:
        return pd.DataFrame()
    metric = run.options.select_metric
    best = (run.state.get("best") or {}).get("config_id")
    cols = [
        "rank",
        "name",
        "setup",
        f"dev_{metric}",
        f"test_{metric}",
        "dev_recall@1",
        "dev_recall@5",
        "dev_evidence_precision@5",
        "dev_f1@5",
        "dev_mrr",
    ]
    cols = list(dict.fromkeys(cols))
    df = table.reset_index()[["config_id", *[c for c in cols if c in table.columns]]]
    df = df.drop(columns=["config_id"])
    df["name"] = [
        ("★ " if cid == best else "") + n for cid, n in zip(table.index, df["name"])
    ]
    return df.round(3)


def _best_md(run_id: str | None) -> str:
    """Best config (or the current retrieval leader while the run goes)."""
    if not run_id:
        return "Запусков ещё нет."
    run = pl.load_run(run_id)
    best = run.state.get("best")
    metric = run.options.select_metric
    if best:
        title = f"### ★ Лучший: `{best['name']}`"
        setup, reason, r = (
            best["setup"],
            f"Выбран: {best['reason']}.",
            best["retrieval"],
        )
        gen = best.get("generation")
    else:
        table = pl.summarize(run)
        if table.empty:
            return "Результатов retrieval пока нет."
        cid = table.index[0]
        row = table.loc[cid]
        title = f"### Лидер сейчас: `{row['name']}`"
        setup, reason = (
            row["setup"],
            "Запуск не завершён — лидер по dev " + metric + ".",
        )
        r = {k: row[k] for k in table.columns if k.startswith(("dev_", "test_"))}
        gen = None
    rows = [
        f"| {m} | {_fmt(r.get('dev_' + m))} | {_fmt(r.get('test_' + m))} |"
        for m in (metric, "recall@1", "recall@5", "mrr")
    ]
    md = [
        title,
        f"`{setup}`",
        reason,
        "| метрика | dev | test |\n|---|---|---|\n" + "\n".join(rows),
    ]
    if best:
        settings = agent_settings(best["config"])
        md.append(f"Модель: `{settings['model']}` · {settings['description']}")
        if (
            pl.load_run(run_id).state.get("evaluation_version")
            != "evidence-coverage-v2"
        ):
            md.append("Исторический запуск: recall рассчитан по прежней методике.")
    if gen:
        md.append(
            "Генерация: "
            + ", ".join(
                f"{k} {_fmt(gen.get(k))}"
                for k in ("answer_correctness", "faithfulness", "context_recall")
                if gen.get(k) is not None
            )
        )
    skipped = run.state.get("skipped", {})
    if skipped:
        md.append(
            "Пропущены: "
            + "; ".join(f"`{run.config(c)['name']}` — {r}" for c, r in skipped.items())
        )
    return "\n\n".join(md)


def _best_json(run_id: str | None) -> str:
    if not run_id:
        return ""
    best = pl.load_run(run_id).state.get("best")
    return json.dumps(best["config"], ensure_ascii=False, indent=2) if best else ""


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _plot_df(run_id: str | None) -> pd.DataFrame:
    df = _summary_df(run_id)
    if df.empty:
        return pd.DataFrame({"config": [], "Значение": []})
    metric = pl.load_run(run_id).options.select_metric
    return pd.DataFrame({"config": df["name"], "Значение": df[f"dev_{metric}"]})


def refresh(run_id):
    """Timer tick: status, logs and everything that changes while a run goes."""
    job = JOBS.get(run_id or "")
    return (
        _progress_html(run_id),
        _log_html(run_id, limit=12, short=True),
        _log_html(run_id),
        gr.update(interactive=bool(job and job.alive)),
        *results(run_id)[:5],
        _kpi_html(run_id),
        export_config(run_id),
    )


# ── handlers ──────────────────────────────────────────────────


def start_run(
    datasets,
    upload_corpus,
    upload_qa,
    upload_name,
    presets,
    changes_text,
    mode,
    max_q,
    generation,
    top_n,
    sample=None,
    prompt_text=None,
    model=None,
    agent_description=None,
    tool_description=None,
    prompt_alt=None,
    generation_full=False,
):
    if _busy():
        raise gr.Error(
            f"Уже идёт запуск {_busy().run_id}. Дождитесь или остановите его."
        )
    if not presets:
        raise gr.Error("Выберите хотя бы один конфиг.")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    catalog = available_datasets()
    ds = {k: catalog[k] for k in datasets or []}
    if upload_corpus is not None:
        up_dir = config.RUNS_DIR / "uploads" / run_id
        up_dir.mkdir(parents=True, exist_ok=True)
        corpus_path = up_dir / Path(upload_corpus).name
        shutil.copy(upload_corpus, corpus_path)
        qa_path = None
        if upload_qa is not None:
            qa_path = up_dir / Path(upload_qa).name
            shutil.copy(upload_qa, qa_path)
        key = (
            "upload_"
            + "".join(ch if ch.isalnum() else "_" for ch in (upload_name or "data"))[
                :30
            ]
        )
        ds[key] = {
            "type": "upload",
            "corpus_path": str(corpus_path),
            "qa_path": str(qa_path) if qa_path else None,
        }
    if not ds:
        raise gr.Error("Выберите базу знаний или загрузите свою.")

    names = [p.split(" — ")[0] for p in presets]
    cfgs = preset_configs(names)
    changes = _parse_changes(changes_text)
    if prompt_text is not None and prompt_text.strip():
        changes["generation.system_prompt"] = prompt_text
    for key, value in (
        ("model", model),
        ("description", agent_description),
        ("tool_description", tool_description),
    ):
        if value is not None and value.strip():
            changes[f"agent.{key}"] = value
    if changes:
        cfgs = [derive_config(c, changes) for c in cfgs]
    if prompt_alt and prompt_alt.strip():
        cfgs += [
            derive_config(c, {"generation.system_prompt": prompt_alt})
            for c in list(cfgs)
        ]
    opts = pl.RunOptions(
        mode=mode,
        generation=generation,
        max_questions=int(max_q) if max_q else None,
        top_n_generation=int(top_n),
        generation_sample=int(sample) if sample else None,
        generation_full=bool(generation_full),
        retry_errors=True,
    )
    if (config.RUNS_DIR / run_id).exists():
        raise gr.Error(
            "Запуск с таким именем уже есть, попробуйте ещё раз через секунду."
        )
    run = pl.new_run(ds, cfgs, opts, run_id)
    _start(Job(run.id, "pipeline"), _run_job)
    return run.id, gr.update(choices=_available_runs(), value=run.id)


def stop_run(run_id):
    job = JOBS.get(run_id or "")
    if job and job.alive:
        job.stop.set()
        gr.Info("Остановка после текущего шага…")


def resume_run(run_id):
    if not run_id:
        raise gr.Error("Выберите запуск.")
    lock_path = config.RUNS_DIR / run_id / "pipeline.lock"
    if lock_path.is_file():
        with lock_path.open("r") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gr.Info("Этот прогон уже выполняется. Прогресс показан в журнале.")
                return run_id
    busy = _busy()
    if busy:
        if busy.kind == "pipeline" and busy.run_id == run_id:
            gr.Info("Этот прогон уже выполняется. Прогресс показан в журнале.")
            return run_id
        raise gr.Error(f"Уже идёт запуск {busy.run_id}.")
    _require_current_backend(run_id)
    run = pl.load_run(run_id)
    run.state["options"]["retry_errors"] = True
    run.save()
    _start(Job(run_id, "pipeline"), _run_job)
    return run_id


GEN_COLUMNS = {
    "name": "name",
    "n": "n",
    "errors": "errors",
    "answer_correctness": "correctness",
    "faithfulness": "faithfulness",
    "answer_relevancy": "relevancy",
    "context_precision": "ctx precision",
    "context_recall": "ctx recall",
    "false_decline_rate": "ложные отказы",
    "unanswerable_decline_rate": "верные отказы",
    "gen_score": "gen_score",
    "mean_input_tokens": "Вход / ответ",
    "mean_output_tokens": "Выход / ответ",
    "mean_total_tokens": "Токены / ответ",
    "usage_answers": "Ответов с usage",
    "usage_missing": "Без полного usage",
    "mean_answer_seconds": "Секунд / ответ",
}


def _gen_df(run: pl.Run) -> pd.DataFrame:
    df = pl.generation_summary(run)
    if df.empty:
        return df
    cols = [c for c in GEN_COLUMNS if c in df.columns]
    return df[cols].rename(columns=GEN_COLUMNS).round(3)


def results(run_id):
    if not run_id:
        empty = pd.DataFrame()
        return (
            _best_md(None),
            _plot_df(None),
            empty,
            empty,
            empty,
            gr.update(choices=[]),
            "",
            None,
        )
    run = pl.load_run(run_id)
    by_type_path = run.path / "summary_by_type.csv"
    by_type = (
        pd.read_csv(by_type_path).drop(columns=["config_id"], errors="ignore").round(3)
        if by_type_path.exists()
        else pd.DataFrame()
    )
    choices = [(f"{c['name']} ({c['id']})", c["id"]) for c in run.configs]
    return (
        _best_md(run_id),
        _plot_df(run_id),
        _summary_df(run_id),
        by_type,
        _gen_df(run),
        gr.update(choices=choices, value=choices[0][1] if choices else None),
        _best_json(run_id),
        export_config(run_id),
    )


def tweak(
    run_id,
    parent_id,
    changes_text,
    prompt_text=None,
    model=None,
    agent_description=None,
    tool_description=None,
):
    if not run_id or not parent_id:
        raise gr.Error("Выберите запуск и конфиг.")
    if _busy():
        raise gr.Error(f"Уже идёт запуск {_busy().run_id}.")
    run = pl.load_run(run_id)
    _require_current_backend(run_id)
    changes = _parse_changes(changes_text)
    saved = agent_settings(run.config(parent_id)["config"])
    for key, value in (
        ("model", model),
        ("description", agent_description),
        ("tool_description", tool_description),
    ):
        if value is not None and value != saved[key]:
            changes[f"agent.{key}"] = value
    if prompt_text is not None:
        if not prompt_text.strip():
            raise gr.Error("Промпт не должен быть пустым.")
        if prompt_text != system_prompt(run.config(parent_id)["config"]):
            changes["generation.system_prompt"] = prompt_text
    if not changes:
        raise gr.Error(
            "Измените промпт или настройки, например chunking.chunk_size=384"
        )
    new = derive_config(run.config(parent_id), changes)
    if not pl.add_configs(run, [new]):
        raise gr.Error(f"Такой конфиг уже есть: {new['id']}")
    _start(Job(run_id, "pipeline"), _run_job)
    gr.Info(
        f"Добавлен {new['name']}: {describe(new['config'])}. Прогресс — вверху страницы."
    )
    return run_id


def config_prompt(run_id, config_id):
    if not run_id or not config_id:
        return ""
    return system_prompt(pl.load_run(run_id).config(config_id)["config"])


def editor_agent(run_id, config_id):
    if not run_id or not config_id:
        return "", "", ""
    saved = agent_settings(pl.load_run(run_id).config(config_id)["config"])
    return saved["model"], saved["description"], saved["tool_description"]


def preset_editor_values(names):
    if not names:
        return "", "", "", ""
    configs = preset_configs(names)
    values = [
        (*agent_settings(c["config"]).values(), system_prompt(c["config"]))
        for c in configs
    ]
    return tuple(column[0] if len(set(column)) == 1 else "" for column in zip(*values))


def start_synthetic(name, size, corpus_file, kinds):
    if _busy():
        raise gr.Error("Дождитесь завершения текущей задачи.")
    if not kinds:
        raise gr.Error("Выберите типы вопросов.")
    dataset_id = (name or "").strip() or f"synthetic-{time.strftime('%Y%m%d-%H%M%S')}"
    job = Job(dataset_id, "synthetic")

    def body(current):
        from evalkit.synthetic import generate_synthetic

        current.result = generate_synthetic(
            size=int(size),
            name=dataset_id,
            corpus_path=corpus_file,
            question_types=kinds,
            stop=current.stop,
            on_event=lambda message, progress=None: current.events.append(
                pl.Event("synthetic", message, progress)
            ),
        )

    _start(job, body)
    return dataset_id


def synthetic_status(dataset_id):
    job = JOBS.get(dataset_id or "")
    record = generated_datasets().get(dataset_id)
    if job is None and record is None:
        return (
            "Готов к генерации. Корпус по умолчанию — документация BPMSoft.",
            gr.update(interactive=True),
            [],
            pd.DataFrame(),
        )
    lines = (
        [event.message for event in list(job.events)[-6:]]
        if job
        else ["Сохранённый набор загружен с диска."]
    )
    files, preview = [], pd.DataFrame()
    record = record or (job.result if job else None)
    if record:
        ds = record["dataset_config"]
        files = [ds["qa_path"], str(Path(ds["qa_path"]).parent / "manifest.json")]
        rows = [
            json.loads(line)
            for line in Path(ds["qa_path"]).read_text(encoding="utf-8").splitlines()[:8]
        ]
        preview = pd.DataFrame(
            [
                {
                    "Тип": r["question_type"],
                    "Вопрос": r["question"],
                    "Ответ": r["reference_answer"],
                    "Свидетельств": len(r["gold_spans"]),
                }
                for r in rows
            ]
        )
        tokens = record.get("generation_total_tokens")
        lines.append(
            f"**Готово: {record['questions']} вопросов.** Токенов генерации: {tokens if tokens is not None else 'usage недоступен'}."
        )
        lines.append(f"Сохранено: `{Path(ds['qa_path']).parent}`")
    if job and job.error:
        lines.append(f"**Ошибка:** {job.error}")
    return (
        "\n\n".join(lines),
        gr.update(interactive=not (job and job.alive)),
        files,
        preview,
    )


def _synthetic_choices():
    choices = [
        (f"{key} · {record['questions']} вопросов", key)
        for key, record in generated_datasets().items()
    ]
    for key, job in list(JOBS.items()):
        if job.kind != "synthetic" or key in {value for _, value in choices}:
            continue
        if job.alive:
            choices.insert(0, (f"{key} · генерируется", key))
        elif job.error:
            choices.append((f"{key} · ошибка генерации", key))
    return choices


def refresh_synthetic_panel(dataset_id):
    choices = _synthetic_choices()
    ids = [value for _, value in choices]
    if dataset_id not in ids:
        dataset_id = ids[0] if ids else None
    catalog = available_datasets()
    return (
        dataset_id,
        gr.update(choices=choices, value=dataset_id),
        *synthetic_status(dataset_id),
        gr.update(choices=list(catalog)),
    )


def restore_synthetic_panel():
    choices = _synthetic_choices()
    dataset_id = choices[0][1] if choices else None
    catalog = available_datasets()
    selection = [dataset_id] if dataset_id in catalog else list(catalog)[:1]
    panel = list(refresh_synthetic_panel(dataset_id))
    panel[-1] = gr.update(choices=list(catalog), value=selection)
    return tuple(panel)


def poll_synthetic_panel(dataset_id):
    # Background polling must not overwrite a user's selection.
    return (
        gr.update(choices=_synthetic_choices()),
        *synthetic_status(dataset_id),
        gr.update(choices=list(available_datasets())),
    )


def use_synthetic(dataset_id):
    if not dataset_id or dataset_id not in available_datasets():
        raise gr.Error("Сначала завершите генерацию датасета.")
    return gr.update(choices=list(available_datasets()), value=[dataset_id]), "Подбор"


def export_config(run_id):
    if not run_id:
        return None
    path = pl.load_run(run_id).path / "best_config.json"
    return str(path) if path.exists() else None


def _kpi_html(run_id):
    best = pl.load_run(run_id).state.get("best", {}) if run_id else {}
    retrieval, generation = best.get("retrieval", {}), best.get("generation") or {}
    if best.get("mode") == "stub":
        generation = {}
    legacy = bool(
        run_id
        and pl.load_run(run_id).state.get("evaluation_version")
        != "evidence-coverage-v2"
    )
    cards = (
        (
            "RECALL · DEV",
            retrieval.get("dev_recall@5"),
            "Прежняя методика" if legacy else "Покрытие gold · top-5",
        ),
        ("КАЧЕСТВО ОТВЕТА", generation.get("gen_score"), "Correctness + faithfulness"),
        (
            "ТОКЕНОВ / ОТВЕТ",
            generation.get("mean_total_tokens"),
            f"С полным usage: {generation.get('usage_answers', 0)}",
        ),
        (
            "СЕКУНД / ОТВЕТ",
            generation.get("mean_answer_seconds"),
            "Генерация без LLM-судьи",
        ),
    )
    return (
        '<div class="kpi-grid">'
        + "".join(
            '<div class="kpi"><span>'
            + label
            + "</span><strong>"
            + (_fmt(value) if value is not None else "—")
            + "</strong><small>"
            + html.escape(note)
            + "</small></div>"
            for label, value, note in cards
        )
        + "</div>"
    )


_CHAT_CLIENTS: dict[str, object] = {}


def _require_current_backend(run_id: str) -> pl.Run:
    run = pl.load_run(run_id)
    current = pl.current_backend()
    if run.backend != current:
        raise gr.Error(
            f"Запуск «{run_id}» использует backend «{run.backend}», а GUI "
            f"запущен с «{current}». Для работы с этим запуском перезапустите "
            "GUI поддерживает только agent-core."
        )
    return run


def chat_configs(run_id):
    if not run_id:
        return gr.update(choices=[])
    run = pl.load_run(run_id)
    best = (run.state.get("best") or {}).get("config_id")
    choices = [
        (("★ " if c["id"] == best else "") + f"{c['name']} ({c['id']})", c["id"])
        for c in pl.active_configs(run)
    ]
    return gr.update(
        choices=choices, value=best or (choices[0][1] if choices else None)
    )


def chat_send(message, history, run_id, config_id, mode, n_sources):
    from evalkit.client import make_client

    if not message.strip():
        return "", history
    if not run_id or not config_id:
        raise gr.Error("Выберите запуск и конфиг.")
    run = _require_current_backend(run_id)
    client = _CHAT_CLIENTS.get(mode)
    if client is None:
        client = _CHAT_CLIENTS[mode] = make_client()
    res = pl.chat(run, config_id, message, client=client)
    sources = "\n".join(
        f"{i}. doc `{c.get('metadata', {}).get('eval_doc_id')}` "
        f"[{c.get('metadata', {}).get('char_start')}:{c.get('metadata', {}).get('char_end')}] "
        f"score {c.get('score', 0):.3f} — {c.get('content', '')[:200].strip()}…"
        for i, c in enumerate(res["retrieved_chunks"][: int(n_sources)], 1)
    )
    answer = res["answer"] + (f"\n\n**Источники:**\n{sources}" if sources else "")
    history = [
        *history,
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history


def deploy_start(run_id, config_id, name, confirmed):
    if not confirmed:
        raise gr.Error("Подтвердите выкладку галочкой.")
    if not run_id or not name.strip():
        raise gr.Error("Выберите запуск и укажите имя коллекции.")
    if _busy():
        raise gr.Error(f"Уже идёт запуск {_busy().run_id}.")
    _require_current_backend(run_id)

    def body(job: Job):
        pl.deploy(
            pl.load_run(run_id),
            name.strip(),
            config_id=config_id or None,
            on_event=job.events.append,
        )

    _start(Job(run_id, "deploy"), body)
    return run_id


# ── layout ────────────────────────────────────────────────────


SECTIONS = ["Подбор", "Синтетика", "Результаты", "Чат", "Журнал", "Выкладка"]
EXTRA_RUNS = ("stand-rerank", "stand-full")


def _available_runs() -> list[str]:
    runs = pl.list_runs(pl.current_backend())
    return runs + [
        run_id
        for run_id in EXTRA_RUNS
        if run_id not in runs and (config.RUNS_DIR / run_id / "run.json").exists()
    ]


def build_app() -> gr.Blocks:
    preset_list = available_presets()
    preset_choices = [(p.get("title", p["name"]), p["name"]) for p in preset_list]
    runs = _available_runs()
    last = runs[0] if runs else None
    backend = "Agent Core"
    initial = "Подбор"
    with gr.Blocks(title="RAG Lab · BPMSoft", fill_width=True) as app:
        active = gr.State(last)
        with gr.Sidebar(open=True, width=290):
            gr.HTML(
                '<div class="brand"><i>R</i> RAG<span>LAB</span></div><div class="brand-sub">BPMSoft · Лаборатория качества</div>'
            )
            nav = gr.Radio(
                SECTIONS, value=initial, show_label=False, elem_classes=["nav"]
            )
            gr.HTML(
                f'<div class="connection"><span></span>{html.escape(backend)}</div>'
            )
            run_dd = gr.Dropdown(runs, value=last, label="История экспериментов")
            progress = gr.HTML(_progress_html(last))
            with gr.Row():
                resume_btn = gr.Button("Продолжить", size="sm")
                stop_btn = gr.Button(
                    "Остановить", variant="stop", interactive=False, size="sm"
                )
            with gr.Accordion("Последние события", open=False):
                log_side = gr.HTML(_log_html(last, limit=12, short=True))
            gr.HTML(
                '<div class="side-note">Один датасет. Разные стратегии.<br>Измеримый результат.</div>'
            )
        gr.HTML(
            '<div class="hero"><div class="eyebrow">BPMSoft / RAG EXPERIMENTS</div><h1>От поиска — к точному ответу<span>.</span></h1><p>Сравните поиск, инструкции агента и расход токенов на одном наборе вопросов.</p></div>'
        )
        kpis = gr.HTML(_kpi_html(last))
        with gr.Column(visible=True) as sec_setup:
            with gr.Row(equal_height=False):
                with gr.Column(scale=6, elem_classes=["panel"]):
                    gr.HTML(
                        '<div class="section-label"><b>01</b><div><h2>Данные и поиск</h2><p>Выберите базу знаний и варианты для сравнения</p></div></div>'
                    )
                    datasets = gr.CheckboxGroup(
                        [
                            (
                                "Синтетика · BPMSoft"
                                if k == "synthetic_bpmsoft"
                                else k,
                                k,
                            )
                            for k in available_datasets()
                        ],
                        value=list(available_datasets())[:1],
                        label="Набор вопросов",
                    )
                    presets = gr.CheckboxGroup(
                        preset_choices,
                        value=[p["name"] for p in preset_list],
                        label="Стратегии поиска",
                    )
                    with gr.Accordion("Свои данные", open=False):
                        upload_corpus = gr.File(
                            label="Корпус · JSONL или CSV",
                            file_types=[".jsonl", ".csv"],
                            type="filepath",
                        )
                        upload_qa = gr.File(
                            label="Вопросы и gold evidence · JSONL",
                            file_types=[".jsonl"],
                            type="filepath",
                        )
                        upload_name = gr.Textbox(
                            label="Название набора", value="my_data"
                        )
                    with gr.Accordion("Параметры поиска", open=False):
                        changes = gr.Textbox(
                            label="Изменения KEY=VALUE",
                            placeholder="chunking.chunk_overlap=80",
                            lines=3,
                        )
                        gr.Markdown(
                            "Пресеты хранятся в `configs/presets.json`. Конфигурация выбранного результата доступна в JSON."
                        )
                with gr.Column(scale=7, elem_classes=["panel"]):
                    gr.HTML(
                        '<div class="section-label"><b>02</b><div><h2>Настройки агента</h2><p>Инструкции и инструмент тоже часть эксперимента</p></div></div>'
                    )
                    agent_model = gr.Textbox(
                        label="Модель агента",
                        value=config.AGENT_MODEL,
                        placeholder=config.AGENT_MODEL,
                    )
                    agent_desc = gr.Textbox(
                        label="Описание агента",
                        value=config.AGENT_DESCRIPTION,
                        placeholder=config.AGENT_DESCRIPTION,
                        lines=2,
                    )
                    tool_desc = gr.Textbox(
                        label="Описание поискового инструмента",
                        value=config.RAG_TOOL_DESCRIPTION,
                        placeholder=config.RAG_TOOL_DESCRIPTION,
                        lines=3,
                    )
                    prompt_text = gr.Textbox(
                        label="Системный промпт RAG",
                        value=config.RAG_SYSTEM_PROMPT,
                        placeholder=config.RAG_SYSTEM_PROMPT,
                        lines=7,
                    )
                    with gr.Accordion("Сравнить с другим промптом", open=False):
                        prompt_alt = gr.Textbox(
                            label="Альтернативные инструкции · вариант B",
                            lines=5,
                            placeholder="Укажите вторую версию инструкций. Оба варианта получат одинаковые вопросы.",
                        )
                    gr.Markdown(
                        "Пустое поле использует значение из JSON. Изменения сохраняются в новом варианте конфигурации."
                    )
            with gr.Column(elem_classes=["panel", "run-panel"]):
                gr.HTML(
                    '<div class="section-label"><b>03</b><div><h2>Параметры эксперимента</h2><p>Покрытие свидетельств · качество ответа · токены на ответ</p></div></div>'
                )
                with gr.Row():
                    mode = gr.State("real")
                    max_q = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        label="Вопросов для поиска",
                        info="0 или пусто — весь набор",
                        scale=1,
                    )
                    sample = gr.Number(
                        value=0,
                        precision=0,
                        minimum=0,
                        label="Вопросов для ответов",
                        info="0 — выборка из настроек. Для всех вопросов включите dev + test ниже.",
                        scale=1,
                    )
                    top_n = gr.Number(
                        value=config.PIPELINE_TOP_N_GENERATION,
                        precision=0,
                        minimum=1,
                        label="Лучших стратегий в генерацию",
                        scale=1,
                    )
                with gr.Row():
                    generation = gr.Checkbox(
                        value=True, label="Оценивать ответы и собирать usage", scale=3
                    )
                    start_btn = gr.Button(
                        "Начать эксперимент →", variant="primary", size="lg", scale=2
                    )
                generation_full = gr.Checkbox(
                    value=False,
                    label="Все вопросы для ответов · dev + test",
                    info="Полный прогон ответов по выбранному набору. Поле «Вопросов для ответов» при этом не ограничивает выборку.",
                )
                gr.Markdown(
                    "Поиск и ответы выполняются на agent-core. Recall показывает покрытие gold, evidence precision — полезную долю контекста."
                )
        with gr.Column(visible=False) as sec_synthetic:
            gr.HTML(
                '<div class="section-label"><b>+</b><div><h2>Синтетический датасет</h2><p>Вопросы, эталонные ответы и короткие проверяемые цитаты из вашей базы знаний</p></div></div>'
            )
            synth_id = gr.State(None)
            synth_saved = gr.Dropdown(
                label="Сохранённые наборы · восстанавливаются после обновления страницы",
                choices=_synthetic_choices(),
                interactive=True,
            )
            with gr.Row():
                with gr.Column(scale=4):
                    synth_name = gr.Textbox(
                        label="Название нового набора",
                        placeholder="bpmsoft-evidence-v2",
                    )
                    synth_size = gr.Number(
                        label="Вопросов",
                        value=config.SYNTHESIS_DEFAULT_SIZE,
                        precision=0,
                        minimum=1,
                        maximum=10000,
                    )
                    synth_corpus = gr.File(
                        label="Свой корпус · необязательно",
                        type="filepath",
                        file_types=[".csv", ".jsonl"],
                    )
                    synth_kinds = gr.CheckboxGroup(
                        [
                            ("Один факт", "single_hop"),
                            ("Два свидетельства", "multi_hop"),
                            ("Точная деталь", "exact_match"),
                        ],
                        value=["single_hop", "multi_hop", "exact_match"],
                        label="Типы вопросов",
                    )
                    synth_btn = gr.Button("Сгенерировать датасет →", variant="primary")
                    synth_use = gr.Button("Использовать в эксперименте")
                with gr.Column(scale=6):
                    synth_status_md = gr.Markdown(
                        "Источники по умолчанию — BPMSoft CSV. Генератор использует LLM стенда и принимает только дословные цитаты, найденные в исходном тексте."
                    )
                    synth_files = gr.File(
                        label="Скачать QA и метаданные",
                        file_count="multiple",
                        interactive=False,
                    )
            synth_preview = gr.Dataframe(
                label="Предпросмотр вопросов и эталонных ответов",
                interactive=False,
                wrap=True,
            )
            gr.Markdown(
                "Gold хранит короткие исходные цитаты со смещениями символов. Для multi-hop проверяются оба отрывка. После генерации проверьте предпросмотр перед сравнением конфигураций."
            )
        with gr.Column(visible=False) as sec_results:
            gr.HTML(
                '<div class="section-label"><b>↗</b><div><h2>Результаты эксперимента</h2><p>Сравнение на dev и отложенной test-части</p></div></div>'
            )
            with gr.Row():
                with gr.Column(scale=5, elem_classes=["panel"]):
                    best_md = gr.Markdown(_best_md(last))
                with gr.Column(scale=6, elem_classes=["panel"]):
                    plot = gr.BarPlot(
                        _plot_df(last),
                        x="config",
                        y="Значение",
                        sort="-y",
                        y_lim=[0, 1],
                        x_label_angle=-15,
                        height=290,
                        label="Выбранная retrieval-метрика · dev",
                    )
            with gr.Tabs():
                with gr.Tab("Поиск и recall"):
                    summary_df = gr.Dataframe(
                        _summary_df(last),
                        label="Сравнение стратегий",
                        interactive=False,
                        wrap=True,
                    )
                    by_type_df = gr.Dataframe(
                        label="Покрытие по типам вопросов", interactive=False, wrap=True
                    )
                with gr.Tab("Ответы и токены"):
                    gen_df = gr.Dataframe(
                        label="Качество, usage и время ответа",
                        interactive=False,
                        wrap=True,
                    )
                    gr.Markdown(
                        "Средние токены рассчитаны только по ответам с полным usage. Вход и выход включают все LLM-шаги агента, расходы судьи считаются отдельно от ответа."
                    )
                with gr.Tab("Конфигурация JSON"):
                    best_json = gr.Code(
                        _best_json(last),
                        language="json",
                        label="Лучший RAG и настройки агента",
                        max_lines=40,
                    )
            best_file = gr.File(
                value=export_config(last),
                label="Скачать выбранную конфигурацию · JSON",
                interactive=False,
            )
            with gr.Accordion("Создать улучшенную версию", open=False):
                parent_dd = gr.Dropdown(label="Исходный вариант")
                with gr.Row():
                    tweak_model = gr.Textbox(label="Модель агента")
                    tweak_desc = gr.Textbox(label="Описание агента", lines=2)
                tweak_tool = gr.Textbox(
                    label="Описание поискового инструмента", lines=3
                )
                tweak_prompt = gr.Textbox(
                    label="Системный промпт выбранного конфига", lines=7
                )
                tweak_changes = gr.Textbox(
                    label="Дополнительные изменения KEY=VALUE",
                    placeholder="chunking.chunk_size=384",
                )
                tweak_btn = gr.Button("Сравнить новую версию", variant="primary")
            with gr.Accordion("Как читать метрики", open=False):
                gr.Markdown(METRICS_LEGEND)
        with gr.Column(visible=False) as sec_chat:
            gr.HTML(
                '<div class="section-label"><b>↗</b><div><h2>Проверить в диалоге</h2><p>Ответы выбранного агента с источниками</p></div></div>'
            )
            with gr.Row():
                chat_cfg = gr.Dropdown(label="Конфигурация", scale=3)
                chat_mode = gr.State("real")
                n_sources = gr.Number(
                    value=3, precision=0, minimum=0, label="Источников", scale=1
                )
            chatbot = gr.Chatbot(height=500, label="Вопрос → ответ → источники")
            with gr.Row():
                chat_in = gr.Textbox(
                    placeholder="Задайте вопрос по документации BPMSoft",
                    scale=6,
                    show_label=False,
                )
                chat_btn = gr.Button("Отправить →", variant="primary", scale=1)
        with gr.Column(visible=False) as sec_log:
            gr.HTML(
                '<div class="section-label"><b>≡</b><div><h2>Журнал эксперимента</h2><p>События индексации, поиска и генерации</p></div></div>'
            )
            log_full = gr.HTML(_log_html(last))
        with gr.Column(visible=False, elem_classes=["panel"]) as sec_deploy:
            gr.HTML(
                '<div class="section-label"><b>↗</b><div><h2>Внедрить результат</h2><p>Коллекция, сохранённый промпт и агент</p></div></div>'
            )
            gr.Markdown(f"Целевой backend: **{backend}**")
            with gr.Row():
                dep_cfg = gr.Dropdown(label="Конфигурация", scale=3)
                dep_name = gr.Textbox(
                    label="Название коллекции", value="rag_best", scale=2
                )
            dep_ok = gr.Checkbox(label="Подтверждаю создание коллекции и агента")
            dep_btn = gr.Button("Создать RAG", variant="primary")
        sections = [
            sec_setup,
            sec_synthetic,
            sec_results,
            sec_chat,
            sec_log,
            sec_deploy,
        ]
        nav.change(
            lambda value: [gr.update(visible=value == name) for name in SECTIONS],
            nav,
            sections,
        )
        result_outputs = [
            best_md,
            plot,
            summary_df,
            by_type_df,
            gen_df,
            parent_dd,
            best_json,
            best_file,
        ]
        gr.Timer(2.0).tick(
            refresh,
            active,
            [
                progress,
                log_side,
                log_full,
                stop_btn,
                best_md,
                plot,
                summary_df,
                by_type_df,
                gen_df,
                kpis,
                best_file,
            ],
        )
        start_btn.click(
            start_run,
            [
                datasets,
                upload_corpus,
                upload_qa,
                upload_name,
                presets,
                changes,
                mode,
                max_q,
                generation,
                top_n,
                sample,
                prompt_text,
                agent_model,
                agent_desc,
                tool_desc,
                prompt_alt,
                generation_full,
            ],
            [active, run_dd],
        ).then(lambda: "Результаты", None, nav)
        generation_full.change(
            lambda full: gr.update(interactive=not full), generation_full, sample
        )
        synth_start = synth_btn.click(
            start_synthetic,
            [synth_name, synth_size, synth_corpus, synth_kinds],
            synth_id,
        )
        synthetic_outputs = [
            synth_id,
            synth_saved,
            synth_status_md,
            synth_btn,
            synth_files,
            synth_preview,
            datasets,
        ]
        synth_start.then(refresh_synthetic_panel, synth_id, synthetic_outputs)
        gr.Timer(2.0).tick(
            poll_synthetic_panel,
            synth_id,
            [
                synth_saved,
                synth_status_md,
                synth_btn,
                synth_files,
                synth_preview,
                datasets,
            ],
        )
        app.load(restore_synthetic_panel, None, synthetic_outputs)
        synth_saved.input(lambda value: value, synth_saved, synth_id).then(
            refresh_synthetic_panel,
            synth_id,
            synthetic_outputs,
        )
        synth_use.click(use_synthetic, synth_id, [datasets, nav])
        stop_btn.click(stop_run, active)
        resume_btn.click(resume_run, run_dd, active)

        def runs_update(current):
            available = _available_runs()
            return gr.update(
                choices=available,
                value=current
                if current in available
                else (available[0] if available else None),
            )

        run_dd.focus(runs_update, run_dd, run_dd)
        presets.change(
            preset_editor_values,
            presets,
            [agent_model, agent_desc, tool_desc, prompt_text],
        )
        run_dd.change(lambda value: value, run_dd, active)
        run_dd.change(results, run_dd, result_outputs)
        run_dd.change(chat_configs, run_dd, chat_cfg)
        run_dd.change(chat_configs, run_dd, dep_cfg)
        app.load(results, run_dd, result_outputs)
        app.load(chat_configs, run_dd, chat_cfg)
        app.load(chat_configs, run_dd, dep_cfg)
        parent_dd.change(config_prompt, [run_dd, parent_dd], tweak_prompt)
        parent_dd.change(
            editor_agent, [run_dd, parent_dd], [tweak_model, tweak_desc, tweak_tool]
        )
        tweak_btn.click(
            tweak,
            [
                run_dd,
                parent_dd,
                tweak_changes,
                tweak_prompt,
                tweak_model,
                tweak_desc,
                tweak_tool,
            ],
            active,
        )
        chat_inputs = [chat_in, chatbot, run_dd, chat_cfg, chat_mode, n_sources]
        chat_btn.click(chat_send, chat_inputs, [chat_in, chatbot])
        chat_in.submit(chat_send, chat_inputs, [chat_in, chatbot])
        dep_btn.click(deploy_start, [run_dd, dep_cfg, dep_name, dep_ok], active)
    return app


def launch(**kwargs) -> None:
    build_app().queue().launch(theme=THEME, css=CSS, js=FORCE_DARK, **kwargs)


if __name__ == "__main__":
    launch()
