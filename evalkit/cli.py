"""
Terminal interface for the config search pipeline.

    uv run python -m evalkit.cli configs                 # 7 presets + forecasts
    uv run python -m evalkit.cli run                     # all agent-core presets
    uv run python -m evalkit.cli run -p e5_512 -p hybrid_e5_512 --max-questions 100
    uv run python -m evalkit.cli run --real              # real LLM on a few questions
    uv run python -m evalkit.cli show [RUN]              # results of a run
    uv run python -m evalkit.cli tweak RUN CONFIG chunking.chunk_size=384
    uv run python -m evalkit.cli chat RUN [--config ID]  # talk to a config
    uv run python -m evalkit.cli deploy RUN --name rag_best
    uv run python -m evalkit.cli gui                     # Gradio UI

All retrieval, indexing and answers run on agent-core. Values in KEY=VALUE
are JSON when possible.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table
import typer

import config
from evalkit import pipeline as pl
from evalkit.data_catalog import available_datasets
from evalkit.rag_config import (
    available_presets,
    derive_config,
    describe,
    parse_value,
    preset_configs,
)

app = typer.Typer(
    add_completion=False, help="Подбор конфигурации RAG", no_args_is_help=True
)
console = Console()


@app.command("synthesize")
def synthesize(
    size: Annotated[
        int, typer.Option(min=1, max=10000, help="Число вопросов.")
    ] = config.SYNTHESIS_DEFAULT_SIZE,
    name: Annotated[str | None, typer.Option(help="Имя нового датасета.")] = None,
    corpus: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="BPMSoft CSV или корпус JSONL/CSV doc_id,text.",
        ),
    ] = None,
    seed: int = 42,
) -> None:
    """Сгенерировать синтетику через LLM стенда с дословной проверкой gold-цитат."""
    from evalkit.synthetic import generate_synthetic

    record = generate_synthetic(
        size=size,
        name=name,
        corpus_path=corpus,
        seed=seed,
        on_event=lambda message, progress=None: console.print(message),
    )
    console.print_json(json.dumps(record, ensure_ascii=False))
    console.print(f"Прогон: python -m evalkit.cli run -d {record['dataset_id']}")


def _parse_changes(items: list[str]) -> dict:
    changes = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"ожидается KEY=VALUE, получено {item!r}")
        key, raw = item.split("=", 1)
        changes[key.strip()] = parse_value(raw.strip())
    return changes


def _backend() -> str:
    return f"agent-core {config.AGENT_PLATFORM_URL}"


def _run_with_progress(run: pl.Run) -> None:
    with Progress(
        TextColumn("[bold]{task.fields[step]:<10}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("", total=1.0, step="")
        current = {"step": None}

        def on_event(ev: pl.Event) -> None:
            if ev.step != current["step"]:
                current["step"] = ev.step
                progress.update(task, completed=0)
            style = {"warn": "yellow", "error": "red", "done": "green"}.get(ev.level)
            if ev.level != "info" or ev.progress is None:
                progress.console.print(f"[{style or 'dim'}][{ev.step}] {ev.message}")
            progress.update(
                task,
                step=ev.step,
                description=ev.message[:90],
                completed=ev.progress if ev.progress is not None else None,
            )

        try:
            pl.run_pipeline(run, on_event=on_event)
        except Exception as e:  # noqa: BLE001 — report, point to resume, exit 1
            console.print(f"[red]Пайплайн упал: {type(e).__name__}: {e}")
            console.print(f"Продолжить: [bold]python -m evalkit.cli resume {run.id}")
            raise typer.Exit(1)


@app.command()
def configs(
    presets_file: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="JSON с пресетами.")
    ] = None,
) -> None:
    """Готовые конфиги и прогноз, где каждый покажет себя лучше."""
    table = Table(title="Пресеты", show_lines=True)
    for col in ("name", "id", "конфигурация", "прогноз"):
        table.add_column(col, overflow="fold")
    for rec, preset in zip(
        preset_configs(path=presets_file), available_presets(presets_file)
    ):
        table.add_row(
            preset["name"],
            rec["id"],
            describe(rec["config"]),
            preset.get("forecast", ""),
        )
    console.print(table)


@app.command()
def run(
    preset: Annotated[
        list[str] | None,
        typer.Option(
            "--preset",
            "-p",
            help="Имя пресета (можно несколько). По умолчанию все пресеты agent-core.",
        ),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset", "-d", help="Ключ из EVAL_DATASETS. По умолчанию все."
        ),
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Изменить все выбранные конфиги: KEY=VALUE."),
    ] = None,
    real: Annotated[
        bool,
        typer.Option(
            "--real", help="Настоящий LLM для ответа и судьи (на нескольких вопросах)."
        ),
    ] = False,
    no_generation: Annotated[
        bool, typer.Option("--no-generation", help="Только retrieval.")
    ] = False,
    max_questions: Annotated[
        int | None,
        typer.Option(help="Ограничить число вопросов на датасет (быстрый прогон)."),
    ] = None,
    sample: Annotated[
        int | None, typer.Option(help="Вопросов на конфиг в этапе генерации.")
    ] = None,
    top_n: Annotated[
        int, typer.Option(help="Сколько лучших конфигов идёт в генерацию.")
    ] = config.PIPELINE_TOP_N_GENERATION,
    run_id: Annotated[
        str | None, typer.Option(help="Имя запуска (по умолчанию дата-время).")
    ] = None,
    presets_file: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="JSON с пресетами.")
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="UTF-8 файл системного промпта RAG."
        ),
    ] = None,
) -> None:
    """Запустить подбор: индексация → retrieval → генерация → выбор лучшего."""
    datasets = {
        k: v for k, v in available_datasets().items() if not dataset or k in dataset
    }
    if not datasets:
        raise typer.BadParameter(
            f"нет датасетов {dataset}; есть {list(available_datasets())}"
        )
    cfgs = preset_configs(preset or None, path=presets_file)
    changes = _parse_changes(set_ or [])
    if prompt_file is not None:
        changes["generation.system_prompt"] = prompt_file.read_text(encoding="utf-8")
    if changes:
        cfgs = [derive_config(c, changes) for c in cfgs]
    opts = pl.RunOptions(
        mode="real",
        generation=not no_generation,
        max_questions=max_questions,
        generation_sample=sample,
        top_n_generation=top_n,
    )
    r = pl.new_run(datasets, cfgs, opts, run_id)
    console.print(
        f"Запуск [bold]{r.id}[/] · {len(cfgs)} конфигов · датасеты {list(datasets)} · "
        f"режим {opts.mode} · бэкенд {_backend()}"
    )
    _run_with_progress(r)
    show(r.id)


@app.command()
def resume(run_id: str) -> None:
    """Продолжить прерванный запуск с того же места."""
    r = pl.load_run(run_id)
    _run_with_progress(r)
    show(r.id)


@app.command()
def show(
    run_id: Annotated[
        str | None, typer.Argument(help="Запуск (по умолчанию последний).")
    ] = None,
) -> None:
    """Таблица результатов, разбивка по типам вопросов, лучший конфиг."""
    runs = pl.list_runs()
    if not runs:
        console.print("Запусков ещё нет: python -m evalkit.cli run")
        raise typer.Exit(1)
    r = pl.load_run(run_id or runs[0])
    metric = r.options.select_metric
    summary = pl.summarize(r)
    if summary.empty:
        console.print(f"[yellow]В запуске {r.id} ещё нет результатов retrieval")
        raise typer.Exit(1)

    table = Table(title=f"Retrieval · запуск {r.id} · ранжирование по dev {metric}")
    for col in (
        "#",
        "id",
        "name",
        "конфигурация",
        f"dev {metric}",
        f"test {metric}",
        "dev recall@5",
        "dev mrr",
    ):
        table.add_column(col)
    best_id = (r.state.get("best") or {}).get("config_id")
    for cid, row in summary.iterrows():
        mark = "★ " if cid == best_id else ""
        table.add_row(
            str(row["rank"]),
            cid,
            mark + row["name"],
            row["setup"],
            _f(row.get(f"dev_{metric}")),
            _f(row.get(f"test_{metric}")),
            _f(row.get("dev_recall@5")),
            _f(row.get("dev_mrr")),
        )
    console.print(table)

    by_type = r.path / "summary_by_type.csv"
    if by_type.exists():
        import pandas as pd

        bt = pd.read_csv(by_type, index_col=0)
        t2 = Table(title=f"dev {metric} по типам вопросов (проверка прогнозов)")
        t2.add_column("name")
        types = [c for c in bt.columns if c != "name"]
        for c in types:
            t2.add_column(c)
        for _, row in bt.loc[[i for i in summary.index if i in bt.index]].iterrows():
            best_in = {c: bt[c].max() for c in types}
            t2.add_row(
                row["name"],
                *[
                    f"[bold green]{_f(row[c])}" if row[c] == best_in[c] else _f(row[c])
                    for c in types
                ],
            )
        console.print(t2)

    gen = pl.generation_summary(r)
    if not gen.empty:
        t3 = Table(title=f"Генерация · режим {r.options.mode}")
        cols = [
            "name",
            "n",
            "errors",
            *pl.GEN_METRICS,
            "false_decline_rate",
            "unanswerable_decline_rate",
            "mean_total_tokens",
            "mean_answer_seconds",
            "usage_answers",
            "usage_missing",
        ]
        for c in cols:
            t3.add_column(c)
        for _, row in gen.iterrows():
            t3.add_row(
                *[
                    _f(row[c]) if c not in ("name", "n", "errors") else str(row[c])
                    for c in cols
                ]
            )
        console.print(t3)

    for cid, reason in r.state.get("skipped", {}).items():
        console.print(f"[yellow]Пропущен {r.config(cid)['name']}: {reason}")
    best = r.state.get("best")
    if best:
        console.print(
            f"\n[bold green]Лучший:[/] {best['name']} · {best['setup']}\n  {best['reason']}"
        )
        console.print(f"  {r.path / 'best_config.json'}")
    console.print(
        f"\nИзменить конфиг и сравнить: [bold]python -m evalkit.cli tweak {r.id} <name|id> KEY=VALUE[/]\n"
        f"Поговорить с конфигом:      [bold]python -m evalkit.cli chat {r.id}[/]"
    )


def _f(v) -> str:
    try:
        return "—" if v is None or math.isnan(float(v)) else f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


@app.command()
def tweak(
    run_id: str,
    config_ref: Annotated[str, typer.Argument(help="name или id конфига-родителя")],
    changes: Annotated[
        list[str] | None,
        typer.Argument(help="KEY=VALUE, напр. chunking.chunk_size=384"),
    ] = None,
    prompt_file: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="UTF-8 файл нового промпта RAG."
        ),
    ] = None,
) -> None:
    """Изменить конфиг, прогнать только его и показать рядом с остальными."""
    r = pl.load_run(run_id)
    updates = _parse_changes(changes or [])
    if prompt_file is not None:
        updates["generation.system_prompt"] = prompt_file.read_text(encoding="utf-8")
    if not updates:
        raise typer.BadParameter("Укажите KEY=VALUE или --prompt-file")
    new = derive_config(r.config(config_ref), updates)
    added = pl.add_configs(r, [new])
    if not added:
        console.print(f"Такой конфиг уже есть в запуске: {new['id']}")
    else:
        console.print(f"Новый конфиг {new['id']}: {describe(new['config'])}")
        _run_with_progress(r)
    show(r.id)


@app.command("chat")
def chat_cmd(
    run_id: Annotated[str | None, typer.Argument()] = None,
    config_ref: Annotated[
        str | None,
        typer.Option("--config", "-c", help="name или id (по умолчанию лучший)."),
    ] = None,
    show_sources: Annotated[
        int, typer.Option(help="Сколько найденных чанков показывать.")
    ] = 3,
) -> None:
    """Чат с готовым RAG: вопрос → ответ + найденные чанки. Выход: пустая строка."""
    from evalkit.client import make_client

    r = pl.load_run(run_id or pl.list_runs(pl.current_backend())[0])
    cfg = r.config(config_ref or r.state["best"]["config_id"])
    mode = "real"
    client = make_client()
    console.print(
        f"Чат с [bold]{cfg['name']}[/] ({describe(cfg['config'])}), ответы: {mode}"
    )
    while True:
        question = console.input("[bold cyan]вы> ").strip()
        if not question:
            break
        res = pl.chat(r, cfg["id"], question, client=client)
        console.print(f"[bold green]rag>[/] {res['answer']}")
        for i, chunk in enumerate(res["retrieved_chunks"][:show_sources], 1):
            meta = chunk.get("metadata", {})
            console.print(
                f"  [dim]{i}. doc {meta.get('eval_doc_id')} [{meta.get('char_start')}:{meta.get('char_end')}] "
                f"score {chunk.get('score', 0):.3f}: {chunk.get('content', '')[:160]!r}"
            )


@app.command()
def deploy(
    run_id: str,
    name: Annotated[str, typer.Option("--name", help="Имя коллекции на платформе.")],
    config_ref: Annotated[
        str | None, typer.Option("--config", "-c", help="По умолчанию лучший.")
    ] = None,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Без подтверждения.")
    ] = False,
) -> None:
    """Выложить конфиг: коллекция + документы + агент."""
    r = pl.load_run(run_id)
    cfg = r.config(config_ref or r.state["best"]["config_id"])
    console.print(
        f"Выкладка [bold]{cfg['name']}[/] ({describe(cfg['config'])}) → {name} на {_backend()}"
    )
    if not yes and not typer.confirm("Продолжить?"):
        raise typer.Exit(1)
    record = pl.deploy(r, name, config_id=cfg["id"], on_event=pl.print_event)
    console.print_json(json.dumps(record, ensure_ascii=False))


@app.command()
def gui(port: int = 7860, share: bool = False) -> None:
    """Графическая оболочка (Gradio)."""
    from evalkit.gui import launch

    launch(server_port=port, share=share)


if __name__ == "__main__":
    app()
