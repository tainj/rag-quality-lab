"""
Config search pipeline: index every config, evaluate retrieval, run the
generation check on the best few, pick the winner, optionally deploy it.

Steps (each one resumable, state in ``runs/<run_id>/run.json``):
    prepare     load datasets, split questions into dev/test
    index       one collection per (config, dataset); reused across runs
    retrieval   recall/precision/ndcg/mrr per question (eval_runner)
    generation  top-N configs by dev retrieval: agent-core answer + LLM judge
    select      best config -> best_config.json

Used by evalkit.cli (terminal) and evalkit.gui (Gradio); both receive
progress through ``on_event(Event)``. The run directory holds only ids and
metrics (question/answer texts stay in generation/*.jsonl).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
from pathlib import Path
import random
import threading
import time
from typing import Any
import uuid

import pandas as pd

from config import (
    EVAL_COLLECTION_NAME_PREFIX,
    K_SET,
    PIPELINE_DEV_FRACTION,
    PIPELINE_GENERATION_SAMPLE,
    PIPELINE_GENERATION_TOP_K,
    PIPELINE_REAL_SAMPLE,
    PIPELINE_SELECT_METRIC,
    PIPELINE_TOP_N_GENERATION,
    RELEVANCE_METRIC,
    RELEVANCE_THRESHOLD,
    RUNS_DIR,
)
from evalkit.rag_config import (
    agent_settings,
    collection_config,
    collection_id,
    describe,
    retrieval_id,
    system_prompt,
)
from evalkit.usage import normalize_usage

STEPS = ("prepare", "index", "retrieval", "generation", "select")
METRIC_COLS = [
    f"{m}@{k}"
    for k in K_SET
    for m in ("recall", "precision", "ndcg", "evidence_precision", "f1")
] + ["mrr"]
GEN_METRICS = (
    "answer_correctness",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)
REGISTRY = RUNS_DIR / "index_registry.json"


@dataclass
class Event:
    step: str
    message: str
    progress: float | None = None  # 0..1 within the current step
    level: str = "info"  # info | warn | error | done
    ts: float = field(default_factory=time.time)


EventFn = Callable[[Event], None]


class StopRequested(Exception):
    pass


@dataclass
class RunOptions:
    mode: str = "real"  # agent-core answers and a real LLM judge
    dev_fraction: float = PIPELINE_DEV_FRACTION
    select_metric: str = PIPELINE_SELECT_METRIC
    top_n_generation: int = PIPELINE_TOP_N_GENERATION
    generation_sample: int | None = None  # None -> by mode
    generation: bool = True
    generation_full: bool = False
    continue_on_error: bool = True
    retry_errors: bool = False
    max_questions: int | None = None  # per dataset, for quick runs
    seed: int = 42

    def __post_init__(self):
        if self.mode != "real":
            raise ValueError("Only real agent-core evaluation is supported")
        if not 0 < self.dev_fraction < 1:
            raise ValueError("dev_fraction must be between 0 and 1")
        if self.select_metric not in METRIC_COLS:
            raise ValueError(f"Unknown selection metric: {self.select_metric}")
        for name in ("max_questions", "generation_sample", "top_n_generation"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    def sample_size(self) -> int:
        if self.generation_sample:
            return self.generation_sample
        return (
            PIPELINE_REAL_SAMPLE if self.mode == "real" else PIPELINE_GENERATION_SAMPLE
        )


# ── run state ─────────────────────────────────────────────────


class Run:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.state = json.loads((self.path / "run.json").read_text(encoding="utf-8"))
        self._lock = threading.Lock()
        self._data: dict[str, tuple[dict, list]] = {}

    @property
    def id(self) -> str:
        return self.state["id"]

    @property
    def backend(self) -> str:
        """Where the run's collections live; old runs: inferred from names."""
        if "backend" in self.state:
            return self.state["backend"]
        names = [c["name"] for c in self.configs]
        return (
            "stand"
            if names and all(n.startswith("stand_") for n in names)
            else "unsupported"
        )

    def check_backend(self) -> None:
        if self.backend != current_backend():
            raise RuntimeError(
                f"Запуск {self.id} сделан на бэкенде «{self.backend}», а сейчас "
                f"«{current_backend()}»: его коллекций здесь нет. "
                "Выберите запуск agent-core."
            )

    @property
    def options(self) -> RunOptions:
        return RunOptions(**self.state["options"])

    @property
    def configs(self) -> list[dict]:
        return self.state["configs"]

    def config(self, config_id: str) -> dict:
        for c in self.configs:
            if c["id"] == config_id or c["name"] == config_id:
                return c
        raise KeyError(f"config {config_id!r} not in run {self.id}")

    def save(self) -> None:
        with self._lock:
            tmp = self.path / "run.json.tmp"
            tmp.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            tmp.replace(self.path / "run.json")

    def set_status(self, key: str, value: Any) -> None:
        self.state.setdefault("status", {})[key] = value
        self.save()

    def status(self, key: str, default: Any = "pending") -> Any:
        return self.state.get("status", {}).get(key, default)

    def collection_name(self, config_id: str, ds_key: str) -> str:
        cfg = self.config(config_id)
        # Old run snapshots retain their original collection names.
        cid = (
            collection_id(cfg["config"]) if "generation" in cfg["config"] else cfg["id"]
        )
        prefix = self.state.get("collection_prefix", EVAL_COLLECTION_NAME_PREFIX)
        return f"{prefix}_{ds_key}_{cid}"

    def ensure_agent(
        self, client, cfg: dict, ds_key: str, collection: str | None = None
    ) -> str:
        coll = collection or self.collection_name(cfg["id"], ds_key)
        if "generation" not in cfg["config"]:
            return client.ensure_eval_agent(coll)
        return client.ensure_eval_agent(
            coll,
            system_prompt=system_prompt(cfg["config"]),
            agent_id=f"{coll}_{cfg['id']}_agent",
            **{
                k: v
                for k, v in agent_settings(cfg["config"]).items()
                if k in ("model", "description")
            }
            if "agent" in cfg["config"]
            else {},
        )

    def file(self, *parts: str) -> Path:
        p = self.path.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def dataset(self, ds_key: str) -> tuple[dict[str, str], list[dict]]:
        """(corpus, qa with "split"), cached in memory."""
        if ds_key not in self._data:
            from evalkit.dataset_loaders import load_dataset

            corpus, qa = load_dataset(ds_key, self.state["datasets"][ds_key])
            opts = self.options
            for row in qa:
                row["split"] = split_of(str(row["qid"]), opts.dev_fraction)
            if opts.max_questions:
                rng = random.Random(opts.seed)
                qa = rng.sample(qa, min(opts.max_questions, len(qa)))
            self._data[ds_key] = (corpus, qa)
        return self._data[ds_key]


def split_of(qid: str, dev_fraction: float) -> str:
    bucket = int(hashlib.sha1(qid.encode("utf-8")).hexdigest(), 16) % 1000
    return "dev" if bucket < dev_fraction * 1000 else "test"


def new_run(
    datasets: dict[str, dict],
    configs: list[dict],
    options: RunOptions | None = None,
    run_id: str | None = None,
) -> Run:
    if not datasets or not configs:
        raise ValueError("At least one dataset and configuration are required")
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a directory name")
    path = RUNS_DIR / run_id
    path.mkdir(parents=True, exist_ok=False)
    state = {
        "id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "datasets": datasets,
        "backend": current_backend(),
        "collection_prefix": EVAL_COLLECTION_NAME_PREFIX,
        "evaluation_version": "evidence-coverage-v2",
        "options": asdict(options or RunOptions()),
        "configs": _dedup(configs),
        "status": {},
    }
    (path / "run.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return Run(path)


def load_run(run_id: str | Path) -> Run:
    path = Path(run_id)
    if not (path / "run.json").exists():
        path = RUNS_DIR / str(run_id)
    if not (path / "run.json").exists():
        raise FileNotFoundError(f"run {run_id!r} not found in {RUNS_DIR}")
    return Run(path)


def current_backend() -> str:
    return "stand"


def list_runs(backend: str | None = None) -> list[str]:
    """Run ids, newest first; only runs made on ``backend`` if given."""
    if not RUNS_DIR.exists():
        return []
    runs = [p.name for p in RUNS_DIR.iterdir() if (p / "run.json").exists()]
    runs.sort(key=lambda r: (load_run(r).state.get("created", ""), r), reverse=True)
    if backend:
        runs = [r for r in runs if load_run(r).backend == backend]
    return runs


def add_configs(run: Run, configs: list[dict]) -> list[dict]:
    """Add configs to an existing run; select/generation are recomputed."""
    known = {c["id"] for c in run.configs}
    added = [c for c in _dedup(configs) if c["id"] not in known]
    run.state["configs"].extend(added)
    for step in ("generation", "select"):
        run.state.get("status", {}).pop(step, None)
    run.save()
    return added


def _dedup(configs: list[dict]) -> list[dict]:
    seen, out = set(), []
    for c in configs:
        if c["id"] not in seen:
            seen.add(c["id"])
            out.append(c)
    return out


# ── pipeline ──────────────────────────────────────────────────


def print_event(ev: Event) -> None:
    pct = f" [{ev.progress:4.0%}]" if ev.progress is not None else ""
    print(f"[{ev.step}]{pct} {ev.message}")


def run_pipeline(
    run: Run,
    client: Any = None,
    on_event: EventFn = print_event,
    stop: threading.Event | None = None,
) -> Run:
    with run.file("pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "Этот прогон уже запущен; откройте журнал вместо повторного запуска"
            ) from exc
        try:
            return _run_pipeline(run, client, on_event, stop)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_pipeline(run, client=None, on_event=print_event, stop=None):
    """Run (or resume) all steps. Steps already done are skipped."""
    from evalkit.client import make_client

    opts = run.options
    run.check_backend()
    client = client or make_client()
    log = run.file("events.log").open("a", encoding="utf-8")

    def emit(step, message, progress=None, level="info"):
        ev = Event(step, message, progress, level)
        log.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")
        log.flush()
        on_event(ev)
        if stop is not None and stop.is_set():
            raise StopRequested

    run.set_status("pipeline", "running")
    try:
        _step_prepare(run, emit)
        _step_index(run, client, emit)
        _step_retrieval(run, client, emit)
        if opts.generation:
            _step_generation(run, client, emit)
        else:
            run.set_status("generation", "skipped")
        _step_select(run, emit)
        partial = (
            bool(
                run.state.get("job_errors")
                or run.state.get("skipped")
                or run.state.get("generation_errors")
            )
            or run.status("select") != "done"
        )
        run.set_status("pipeline", "done_with_errors" if partial else "done")
        message = (
            f"Лучший конфиг: {run.state['best']['name']}"
            if run.state.get("best")
            else "Полных результатов для выбора нет"
        )
        emit(
            "select",
            ("Завершено с ошибками. " if partial else "Готово. ") + message,
            1.0,
            "warn" if partial else "done",
        )
    except StopRequested:
        run.set_status("pipeline", "stopped")
        on_event(Event("pipeline", "Остановлено пользователем", level="warn"))
    except Exception as e:
        run.set_status("pipeline", "failed")
        run.state.setdefault("errors", []).append(f"{type(e).__name__}: {e}")
        run.save()
        emit_safe = Event("pipeline", f"Ошибка: {type(e).__name__}: {e}", level="error")
        log.write(json.dumps(asdict(emit_safe), ensure_ascii=False) + "\n")
        on_event(emit_safe)
        raise
    finally:
        log.close()
    return run


def _step_prepare(run: Run, emit) -> None:
    stats = {}
    keys = list(run.state["datasets"])
    for i, ds_key in enumerate(keys):
        emit("prepare", f"Загрузка датасета {ds_key}", i / len(keys))
        corpus, qa = run.dataset(ds_key)
        fingerprint = hashlib.sha256(
            json.dumps([corpus, qa], sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        previous = run.state.get("data_fingerprints", {}).get(ds_key)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"{ds_key}: данные изменились после начала запуска; создайте новый запуск"
            )
        run.state.setdefault("data_fingerprints", {})[ds_key] = fingerprint
        answerable = [q for q in qa if q.get("gold_spans")]
        stats[ds_key] = {
            "docs": len(corpus),
            "questions": len(qa),
            "answerable": len(answerable),
            "dev": sum(q["split"] == "dev" for q in qa),
            "test": sum(q["split"] == "test" for q in qa),
            "corpus_hash": _corpus_hash(corpus),
        }
        emit(
            "prepare",
            f"{ds_key}: {len(corpus)} документов, {len(qa)} вопросов "
            f"(dev {stats[ds_key]['dev']} / test {stats[ds_key]['test']})",
        )
        if not answerable:
            emit(
                "prepare",
                f"{ds_key}: нет вопросов с gold_spans — retrieval не оценить",
                level="warn",
            )
    run.state["dataset_stats"] = stats
    run.set_status("prepare", "done")


def _corpus_hash(corpus: dict[str, str]) -> str:
    h = hashlib.sha1()
    for doc_id in sorted(corpus):
        h.update(
            json.dumps([doc_id, corpus[doc_id]], ensure_ascii=False).encode("utf-8")
        )
    return h.hexdigest()[:12]


def _load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}


def _save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _collection_ready(client, name: str) -> bool:
    import httpx

    exists = getattr(client, "collection_exists", None)
    try:
        return exists(name) if exists else True
    except (httpx.TimeoutException, httpx.NetworkError):
        # This helper is used only after a completed matching registry entry.
        # A transient health-check failure does not invalidate that index;
        # actual retrieval errors are recorded by the next pipeline step.
        return True


def _safe_upload_retry(result: str) -> bool:
    """Only retry failures known to precede sending the document body."""
    message = result.lower()
    return result.startswith("ERROR") and any(
        marker in message
        for marker in (
            "handshake",
            "connection refused",
            "all connection attempts failed",
            "name or service not known",
            "not submitted",
        )
    )


def _save_index_checkpoint(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _wait_tasks(
    client, task_ids: list[str], emit, step: str, timeout: float = 120
) -> list[str]:
    pending, failed = list(task_ids), []
    total = len(pending)
    started = time.monotonic()
    deadline = started + timeout
    absolute_deadline = started + 1800
    while pending:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Indexing stalled or exceeded its time limit: {len(pending)} tasks pending"
            )
        previous_pending = len(pending)
        still = []
        statuses = client.check_tasks(pending)
        if len(statuses) != len(pending):
            raise RuntimeError("Task status response has an unexpected length")
        for tid, st in zip(pending, statuses):
            state = st.get("state", "unknown")
            if state in ("failed", "FAILURE", "revoked", "REVOKED"):
                failed.append(tid)
            elif state != "completed":
                still.append(tid)
        pending = still
        if len(pending) < previous_pending:
            deadline = min(absolute_deadline, time.monotonic() + timeout)
        if pending:
            emit(
                step,
                f"Индексация на сервере: {total - len(pending)}/{total}",
                1 - len(pending) / total,
            )
            time.sleep(1)
    return failed


def _exclude_index(run, cfg, ds_key, message):
    collection = run.collection_name(cfg["id"], ds_key)
    for other in run.configs:
        if run.collection_name(other["id"], ds_key) == collection:
            run.state.setdefault("excluded_configs", {})[other["id"]] = message
    run.save()


def _step_index(run: Run, client, emit) -> None:
    reg = _load_registry()
    run.state["excluded_configs"] = {}
    run.state["job_errors"] = {}
    run.state["skipped"] = {}
    run.state["generation_errors"] = 0
    jobs = [(c, ds) for c in run.configs for ds in run.state["datasets"]]
    for n, (cfg, ds_key) in enumerate(jobs):
        if cfg["id"] in run.state["excluded_configs"]:
            continue
        try:
            _index_job(run, client, emit, reg, n, len(jobs), cfg, ds_key)
        except StopRequested:
            raise
        except Exception as exc:
            if not run.options.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            key = f"index:{cfg['id']}:{ds_key}"
            run.state["job_errors"][key] = message
            run.set_status(key, "failed")
            _exclude_index(run, cfg, ds_key, message)
            emit(
                "index",
                f"{cfg['name']}: ошибка, перехожу дальше — {message}",
                level="warn",
            )
    run.set_status("index", "partial" if run.state["excluded_configs"] else "done")


def _index_job(run, client, emit, reg, n, total, cfg, ds_key):
    from evalkit.api_client import UnsupportedConfigError

    key = f"index:{cfg['id']}:{ds_key}"
    name = run.collection_name(cfg["id"], ds_key)
    corpus, _ = run.dataset(ds_key)
    chash = run.state["dataset_stats"][ds_key]["corpus_hash"]
    entry = reg.get(name, {})
    base = n / total
    if (
        entry.get("corpus_hash") == chash
        and not entry.get("errors")
        and _collection_ready(client, name)
    ):
        emit("index", f"{cfg['name']} × {ds_key}: индекс уже есть, пропускаю", base)
        run.set_status(key, "done")
        return

    emit("index", f"{cfg['name']} × {ds_key}: {describe(cfg['config'])}", base)
    t0 = time.perf_counter()
    docs = [(doc_id, text, name) for doc_id, text in corpus.items()]
    checkpoint_path = run.file("index", f"{cfg['id']}__{ds_key}.json")
    saved = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {}
    exists = bool(
        getattr(client, "collection_exists", None)
    ) and client.collection_exists(name)
    resume_index = (
        exists
        and bool(saved.get("documents"))
        and saved.get("collection", name) == name
        and saved.get("corpus_hash", chash) == chash
        and [row["source_doc_id"] for row in saved["documents"]] == list(corpus)
    )
    checkpoint = {
        "collection": name,
        "corpus_hash": chash,
        "documents": saved["documents"]
        if resume_index
        else [
            {
                "source_doc_id": doc[0],
                "result": "ERROR: upload outcome unknown",
                "task_id": None,
            }
            for doc in docs
        ],
        "failed_tasks": [],
    }

    def upload_batch(positions, desc):
        _save_index_checkpoint(checkpoint_path, checkpoint)
        completed_results = 0

        def record(position, result):
            nonlocal completed_results
            completed_results += 1
            original = positions[position]
            checkpoint["documents"][original].update(
                result=result[0], task_id=result[1]
            )
            _save_index_checkpoint(checkpoint_path, checkpoint)
            if completed_results % 50 == 0:
                emit(
                    "index",
                    f"{cfg['name']}: сохранено {completed_results}/{len(positions)} результатов загрузки",
                    base,
                )

        return client.upload_many(
            [docs[i] for i in positions], desc=desc, on_result=record
        )

    if resume_index:
        results = [(row["result"], row.get("task_id")) for row in saved["documents"]]
        retry_positions = [
            i for i, (result, _) in enumerate(results) if _safe_upload_retry(result)
        ]
        emit(
            "index",
            f"Сохранённая загрузка: {sum(not d.startswith('ERROR') for d, _ in results)}/{len(docs)}; безопасных повторов: {len(retry_positions)}",
            base,
        )
        if retry_positions:
            repaired = upload_batch(retry_positions, f"{cfg['name']}:retry")
            if len(repaired) != len(retry_positions):
                raise RuntimeError("Upload retry response has an unexpected length")
            for i, result in zip(retry_positions, repaired):
                results[i] = result
    else:
        if exists:
            client.delete_collection(name)
        try:
            client.create_collection(
                collection_name=name,
                description=agent_settings(cfg["config"])["tool_description"],
                **collection_config(cfg["config"]),
            )
        except UnsupportedConfigError as e:
            emit("index", f"{cfg['name']}: пропущен — {e}", base, "warn")
            run.set_status(key, "skipped")
            run.state.setdefault("skipped", {})[cfg["id"]] = str(e)
            run.save()
            return
        results = upload_batch(list(range(len(docs))), f"{cfg['name']}:{ds_key}")
    checkpoint = {
        "collection": name,
        "corpus_hash": chash,
        "documents": [
            {"source_doc_id": source[0], "result": result[0], "task_id": result[1]}
            for source, result in zip(docs, results)
        ],
        "failed_tasks": [],
    }
    # Persist accepted documents before any network calls to poll their tasks.
    _save_index_checkpoint(checkpoint_path, checkpoint)
    upload_errors = [
        row for row in checkpoint["documents"] if row["result"].startswith("ERROR")
    ]
    if len(results) != len(docs) or upload_errors:
        run.set_status(key, "failed")
        raise RuntimeError(
            f"{name}: incomplete upload ({len(upload_errors)} errors); checkpoint saved: {upload_errors[:2]}"
        )
    failed = _wait_tasks(client, [t for _, t in results if t], emit, "index")
    details = [
        {"source_doc_id": source[0], "result": result[0], "task_id": result[1]}
        for source, result in zip(docs, results)
    ]
    failed_details = client.check_tasks(failed) if failed else []
    checkpoint["failed_tasks"] = failed_details
    _save_index_checkpoint(checkpoint_path, checkpoint)
    errors = sum(1 for d, _ in results if d.startswith("ERROR")) + len(failed)
    if len(results) != len(docs) or errors:
        run.set_status(key, "failed")
        raise RuntimeError(
            f"{name}: indexing failed ({errors} errors, {len(results)}/{len(docs)} results): "
            f"{[r for r in details if r['result'].startswith('ERROR')][:2]} {failed_details[:2]}"
        )
    reg[name] = {
        "config_id": cfg["id"],
        "dataset": ds_key,
        "corpus_hash": chash,
        "docs": len(corpus),
        "errors": errors,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    _save_registry(reg)
    run.state.setdefault("index_time", {})[f"{cfg['id']}:{ds_key}"] = reg[name][
        "seconds"
    ]
    run.set_status(key, "done")
    level = "warn" if errors else "info"
    emit(
        "index",
        f"{cfg['name']} × {ds_key}: готово за {reg[name]['seconds']} с, ошибок {errors}",
        base,
        level,
    )


def active_configs(run: Run) -> list[dict]:
    return [
        c
        for c in run.configs
        if c["id"] not in run.state.get("skipped", {})
        and c["id"] not in run.state.get("excluded_configs", {})
    ]


def _step_retrieval(run: Run, client, emit) -> None:
    jobs = [(c, ds) for c in active_configs(run) for ds in run.state["datasets"]]
    for n, (cfg, ds_key) in enumerate(jobs):
        if cfg["id"] in run.state.get("excluded_configs", {}):
            continue
        try:
            _retrieval_job(run, client, emit, n, len(jobs), cfg, ds_key)
        except StopRequested:
            raise
        except Exception as exc:
            if not run.options.continue_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            run.state.setdefault("job_errors", {})[
                f"retrieval:{cfg['id']}:{ds_key}"
            ] = message
            for other in run.configs:
                if retrieval_id(other["config"]) == retrieval_id(cfg["config"]):
                    run.state.setdefault("excluded_configs", {})[other["id"]] = message
            run.save()
            emit(
                "retrieval",
                f"{cfg['name']}: ошибка, перехожу дальше — {message}",
                level="warn",
            )
    summarize(run)
    run.set_status("retrieval", "partial" if run.state.get("job_errors") else "done")


def _retrieval_job(run, client, emit, n, total, cfg, ds_key):
    from evalkit.eval_runner import run_retrieval_grid

    out = run.file("retrieval", f"{cfg['id']}__{ds_key}.csv")
    previous_ok = pd.DataFrame()
    if out.exists():
        existing = pd.read_csv(out, dtype={"qid": str})
        if "error" in existing and existing["error"].notna().any():
            if not run.options.retry_errors:
                raise RuntimeError(
                    "Saved retrieval contains failed queries; excluded from selection"
                )
            previous_ok = existing[existing["error"].isna()]
        else:
            return
    # Prompt-only variants have identical retrieval. Reuse those scores.
    cached = next(
        (
            run.path / "retrieval" / f"{other['id']}__{ds_key}.csv"
            for other in active_configs(run)
            if other["id"] != cfg["id"]
            and retrieval_id(other["config"]) == retrieval_id(cfg["config"])
            and (run.path / "retrieval" / f"{other['id']}__{ds_key}.csv").exists()
        ),
        None,
    )
    if cached is not None:
        df = pd.read_csv(cached, dtype={"config_id": str, "qid": str})
        df["config_id"] = cfg["id"]
        df["collection"] = run.collection_name(cfg["id"], ds_key)
        tmp = out.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(out)
        return
    _, qa = run.dataset(ds_key)
    answerable = [q for q in qa if q.get("gold_spans")]
    if not previous_ok.empty:
        finished = set(previous_ok["qid"].astype(str))
        answerable = [q for q in answerable if str(q["qid"]) not in finished]
    if not answerable:
        return
    emit(
        "retrieval",
        f"{cfg['name']} × {ds_key}: {len(answerable)} вопросов",
        n / total,
    )
    df = run_retrieval_grid(
        qa_dataset=answerable,
        client=client,
        top_k=max(K_SET),
        k_set=K_SET,
        dataset_collections={ds_key: run.collection_name(cfg["id"], ds_key)},
        relevance_threshold=RELEVANCE_THRESHOLD,
        relevance_metric=RELEVANCE_METRIC,
        continue_on_error=run.options.continue_on_error,
    )
    split = {str(q["qid"]): q["split"] for q in answerable}
    df["split"] = df["qid"].astype(str).map(split)
    df["config_id"] = cfg["id"]
    if not previous_ok.empty:
        df = pd.concat([previous_ok, df], ignore_index=True)
    tmp = out.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    if "error" in df and df["error"].notna().any():
        raise RuntimeError(
            f"{int(df['error'].notna().sum())} retrieval queries failed; scores saved but excluded from selection"
        )
    m = df[df["split"] == "dev"][run.options.select_metric].mean()
    emit(
        "retrieval",
        f"{cfg['name']} × {ds_key}: dev {run.options.select_metric} = {m:.3f}",
        (n + 1) / total,
    )


def retrieval_rows(run: Run) -> pd.DataFrame:
    files = sorted((run.path / "retrieval").glob("*.csv"))
    if not files:
        return pd.DataFrame()
    rows = pd.concat(
        [pd.read_csv(f, dtype={"config_id": str, "qid": str}) for f in files],
        ignore_index=True,
    )
    eligible = {cfg["id"] for cfg in active_configs(run)}
    return rows[rows["config_id"].isin(eligible)]


def summarize(run: Run) -> pd.DataFrame:
    """Per config: dev/test means (macro over datasets), rank by dev metric."""
    rows = retrieval_rows(run)
    if rows.empty:
        run.file("summary.csv").unlink(missing_ok=True)
        run.file("summary_by_type.csv").unlink(missing_ok=True)
        return pd.DataFrame()
    metric = run.options.select_metric
    cols = [c for c in METRIC_COLS if c in rows.columns]
    per_ds = rows.groupby(["config_id", "dataset_source", "split"])[cols].mean()
    macro = per_ds.groupby(["config_id", "split"]).mean().unstack("split")
    names = {c["id"]: c for c in run.configs}
    table = pd.DataFrame(
        {
            "config_id": macro.index,
            "name": [names[i]["name"] for i in macro.index],
            "setup": [describe(names[i]["config"]) for i in macro.index],
        }
    ).set_index("config_id")
    for col in cols:
        for split in ("dev", "test"):
            if (col, split) in macro.columns:
                table[f"{split}_{col}"] = macro[(col, split)]
    if f"dev_{metric}" not in table or table[f"dev_{metric}"].isna().all():
        raise ValueError(
            "Нет dev-вопросов для выбора конфигурации; увеличьте max_questions или измените seed"
        )
    sort_cols = [f"dev_{metric}"]
    if metric.startswith("recall") and "dev_evidence_precision@5" in table:
        sort_cols.append("dev_evidence_precision@5")
    table = table.sort_values(sort_cols, ascending=False, kind="stable")
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(run.file("summary.csv"))

    by_type = (
        rows[rows["split"] == "dev"]
        .groupby(["config_id", "question_type"])[metric]
        .mean()
        .unstack("question_type")
    )
    by_type.insert(0, "name", [names[i]["name"] for i in by_type.index])
    by_type.to_csv(run.file("summary_by_type.csv"))
    return table


def generation_candidates(run: Run, table: pd.DataFrame) -> list[str]:
    """Take top-N retrieval setups and evaluate every prompt variant of each."""
    selected, setups = [], set()
    for cid in table.index:
        setup = retrieval_id(run.config(cid)["config"])
        if setup not in setups and len(setups) >= run.options.top_n_generation:
            continue
        setups.add(setup)
        selected.append(cid)
    return selected


def _step_generation(run: Run, client, emit) -> None:
    from evalkit.llm import make_judge

    opts = run.options
    table = summarize(run)
    if table.empty:
        run.set_status("generation", "skipped")
        return
    top = generation_candidates(run, table)

    rng = random.Random(opts.seed)
    sample: list[tuple[str, dict]] = []
    candidates = []
    for ds_key in run.state["datasets"]:
        _, qa = run.dataset(ds_key)
        dev = [
            q
            for q in qa
            if (opts.generation_full or q["split"] == "dev")
            and q.get("question_type") != "retrieval_only"
        ]
        candidates.extend((ds_key, q) for q in dev)
    sample = (
        candidates
        if opts.generation_full
        else rng.sample(candidates, min(opts.sample_size(), len(candidates)))
    )
    if not sample:
        emit(
            "generation",
            "Нет dev-вопросов с эталонными ответами для оценки генерации",
            level="warn",
        )
        run.set_status("generation", "skipped")
        return
    try:
        judge = make_judge(opts.mode)
    except Exception as exc:
        if not opts.continue_on_error:
            raise
        run.state.setdefault("job_errors", {})["generation:judge"] = (
            f"{type(exc).__name__}: {exc}"
        )
        run.set_status("generation", "skipped")
        emit(
            "generation", f"LLM-судья недоступен: {exc}; перехожу дальше", level="warn"
        )
        return

    run.set_status("generation", "running")
    kwargs = {}
    total = len(top) * len(sample)
    done = 0
    for cfg_id in top:
        cfg = run.config(cfg_id)
        out = run.file("generation", f"{cfg_id}.jsonl")
        if out.exists():
            existing = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not opts.retry_errors or not any(
                _trace_needs_retry(t) for t in existing
            ):
                done += len(sample)
                continue
        emit(
            "generation",
            f"{cfg['name']}: {len(sample)} вопросов, режим {opts.mode}",
            done / max(total, 1),
        )
        checkpoint = run.file("generation", f"{cfg_id}.checkpoint.json")
        saved = (
            json.loads(checkpoint.read_text(encoding="utf-8"))
            if checkpoint.exists()
            else {}
        )
        traces = []
        agents: dict[str, str] = {}

        def persist_trace(key, trace, saved=saved, checkpoint=checkpoint):
            saved[key] = trace
            temporary = checkpoint.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(checkpoint)

        for ds_key, q in sample:
            key = json.dumps([ds_key, str(q["qid"])])
            previous = saved.get(key)
            if (
                previous
                and previous.get("phase") == "complete"
                and not (opts.retry_errors and _trace_needs_retry(previous))
            ):
                traces.append(previous)
                done += 1
                continue
            unanswerable = not q.get("gold_spans")
            res, answer_seconds = {"answer": "", "retrieved_chunks": []}, None
            trace = {
                "config_id": cfg_id,
                "dataset": ds_key,
                "qid": q["qid"],
                "split": q["split"],
                "question_type": q.get("question_type"),
                "unanswerable": unanswerable,
                "question": q["question"],
                "reference_answer": q.get("reference_answer"),
                "system_prompt": system_prompt(cfg["config"]),
                "agent": agent_settings(cfg["config"]),
            }
            emit(
                "generation",
                f"{cfg['name']}: вопрос {len(traces) + 1}/{len(sample)} · " + ("сохранённый ответ" if previous and previous.get("answer") else "ответ агента"),
                done / max(total, 1),
            )
            try:
                if previous and previous.get("answer"):
                    res = {
                        k: previous.get(k)
                        for k in (
                            "answer",
                            "retrieved_chunks",
                            "usage",
                            "usage_complete",
                        )
                    }
                    answer_seconds = previous.get("answer_seconds")
                else:
                    if ds_key not in agents:
                        agents[ds_key] = run.ensure_agent(client, cfg, ds_key)
                    agent = agents[ds_key]
                    started = time.perf_counter()
                    res = client.agent_query(q["question"], agent_id=agent, **kwargs)
                    answer_seconds = time.perf_counter() - started
                trace.update(
                    answer=res.get("answer", ""),
                    retrieved_chunks=res.get("retrieved_chunks") or [],
                    usage=res.get("usage"),
                    usage_complete=res.get(
                        "usage_complete", res.get("usage") is not None
                    ),
                    answer_seconds=answer_seconds,
                    scores=previous.get("scores", {}) if previous else {},
                    error=None,
                    phase="answer",
                )
                persist_trace(key, trace)
                if not res.get("answer", "").strip():
                    raise RuntimeError("Agent returned an empty answer")
                emit(
                    "generation",
                    f"{cfg['name']}: вопрос {len(traces) + 1}/{len(sample)} · LLM-судья",
                    done / max(total, 1),
                )
                contexts = [
                    c.get("content", "") for c in res.get("retrieved_chunks", [])
                ]
                args = (
                    q["question"],
                    res["answer"],
                    q.get("reference_answer") or "",
                    contexts,
                    unanswerable,
                )
                if (
                    previous
                    and not unanswerable
                    and getattr(judge, "supports_metric_retry", False) is True
                ):
                    old_scores = previous.get("scores", {})
                    missing = [
                        name
                        for name in judge.metric_names
                        if old_scores.get(name) is None
                    ]
                    emit("generation", f"{cfg['name']}: дополняю оценки {', '.join(missing)} из сохранённого ответа", done / max(total, 1))
                    recovered = judge.score(*args, metrics=missing)
                    scores = {
                        key: value
                        for key, value in old_scores.items()
                        if key != "errors"
                    }
                    scores.update(
                        {
                            key: value
                            for key, value in recovered.items()
                            if key != "errors"
                        }
                    )
                    metric_errors = {
                        key: value
                        for key, value in old_scores.get("errors", {}).items()
                        if key not in missing
                    }
                    metric_errors.update(recovered.get("errors", {}))
                    if metric_errors:
                        scores["errors"] = metric_errors
                else:
                    scores = judge.score(*args)
                error = None
            except StopRequested:
                raise
            except Exception as exc:  # noqa: BLE001 — preserve provider usage even if judging fails
                scores = previous.get("scores", {}) if previous else {}
                error = f"{type(exc).__name__}: {exc}"
            trace.update(
                answer=res.get("answer", ""),
                retrieved_chunks=res.get("retrieved_chunks") or [],
                usage=res.get("usage"),
                usage_complete=res.get("usage_complete", res.get("usage") is not None),
                answer_seconds=answer_seconds,
                scores=scores,
                error=error,
                phase="complete",
            )
            persist_trace(key, trace)
            traces.append(trace)
            done += 1
            emit(
                "generation",
                f"{cfg['name']}: {len(traces)}/{len(sample)} ответов",
                done / max(total, 1),
            )
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        tmp.replace(out)
        n_err = sum(1 for t in traces if t["error"])
        if n_err:
            emit(
                "generation",
                f"{cfg['name']}: {n_err} ошибок, пример: {next(t['error'] for t in traces if t['error'])}",
                level="warn",
            )
        incomplete = [t for t in traces if not t.get("error") and _trace_needs_retry(t)]
        if incomplete:
            example = next(
                iter(incomplete[0].get("scores", {}).get("errors", {}).values()),
                "не получены все основные метрики",
            )
            emit(
                "generation",
                f"{cfg['name']}: неполная оценка судьи у {len(incomplete)}/{len(traces)} ответов. Пример: {example}. Продолжение повторит только недостающие оценки.",
                level="warn",
            )
    summary = generation_summary(run)
    run.state["generation_errors"] = (
        int(summary["errors"].sum() + summary["incomplete_scores"].sum())
        if not summary.empty
        else 0
    )
    run.set_status("generation", "done")


def _trace_needs_retry(trace: dict) -> bool:
    return bool(trace.get("error")) or (
        not trace.get("unanswerable", False)
        and any(
            trace.get("scores", {}).get(m) is None
            for m in ("answer_correctness", "faithfulness")
        )
    )


def generation_summary(run: Run) -> pd.DataFrame:
    rows = []
    for f in sorted((run.path / "generation").glob("*.jsonl")):
        with f.open(encoding="utf-8") as stream:
            traces = [json.loads(line) for line in stream if line.strip()]
        ok = [t for t in traces if not t["error"]]
        ans = [
            t for t in ok if not t["unanswerable"] and t.get("split", "dev") == "dev"
        ]
        una = [t for t in ok if t["unanswerable"] and t.get("split", "dev") == "dev"]
        row = {
            "config_id": f.stem,
            "name": run.config(f.stem)["name"],
            "n": len(traces),
            "errors": len(traces) - len(ok),
            "incomplete_scores": sum(
                not t["unanswerable"]
                and any(
                    t["scores"].get(m) is None
                    for m in ("answer_correctness", "faithfulness")
                )
                for t in ok
            ),
        }
        measured = [
            normalize_usage(t.get("usage"))
            for t in traces
            if t.get("usage_complete", False)
        ]
        measured = [u for u in measured if u is not None]
        row["usage_answers"] = len(measured)
        row["usage_missing"] = len(traces) - len(measured)
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            values = [u[key] for u in measured if key in u]
            row[f"sum_{key}"] = sum(values) if values else None
            row[f"mean_{key}"] = sum(values) / len(values) if values else None
        latencies = [
            t["answer_seconds"] for t in ok if t.get("answer_seconds") is not None
        ]
        row["mean_answer_seconds"] = (
            sum(latencies) / len(latencies) if latencies else None
        )
        for m in GEN_METRICS:
            vals = [t["scores"][m] for t in ans if t["scores"].get(m) is not None]
            row[m] = sum(vals) / len(vals) if vals else None
            test_vals = [
                t["scores"][m]
                for t in ok
                if not t["unanswerable"]
                and t.get("split") == "test"
                and t["scores"].get(m) is not None
            ]
            row[f"test_{m}"] = sum(test_vals) / len(test_vals) if test_vals else None
        row["false_decline_rate"] = (
            (sum(t["scores"].get("declined", 0) for t in ans) / len(ans))
            if ans
            else None
        )
        row["unanswerable_decline_rate"] = (
            (sum(t["scores"].get("declined", 0) for t in una) / len(una))
            if una
            else None
        )
        vals = [
            row[m] for m in ("answer_correctness", "faithfulness") if row[m] is not None
        ]
        row["gen_score"] = sum(vals) / 2 if len(vals) == 2 else None
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("gen_score", ascending=False, na_position="last")
        df.to_csv(run.file("generation_summary.csv"), index=False)
    return df


def _skip_selection(run, emit, reason):
    run.state.pop("best", None)
    run.file("best_config.json").unlink(missing_ok=True)
    run.state["selection_skipped_reason"] = reason
    run.set_status("select", "skipped")
    emit("select", reason, level="warn")


def _step_select(run: Run, emit) -> None:
    opts = run.options
    table = summarize(run)
    if table.empty:
        if not opts.continue_on_error:
            raise RuntimeError("Нет результатов retrieval — выбирать не из чего")
        _skip_selection(
            run, emit, "Нет полностью оценённых конфигураций; ошибки сохранены"
        )
        return
    gen = (
        generation_summary(run)
        if run.status("generation") == "done"
        else pd.DataFrame()
    )
    if not gen.empty:
        gen = gen[gen["config_id"].isin(generation_candidates(run, table))]

    if opts.generation and opts.continue_on_error:
        if not gen.empty:
            gen = gen[(gen["errors"] == 0) & gen["gen_score"].notna()]
            if "incomplete_scores" in gen:
                gen = gen[gen["incomplete_scores"] == 0]
        if gen.empty:
            _skip_selection(
                run, emit, "Нет полной оценки генерации; победитель не выбран"
            )
            return

    best_id = table.index[0]
    reason = f"лучший dev {opts.select_metric} по retrieval"
    if opts.mode == "real" and not gen.empty and gen["gen_score"].notna().any():
        best_id = gen.dropna(subset=["gen_score"]).iloc[0]["config_id"]
        reason = f"лучший gen_score (настоящий LLM) среди top-{opts.top_n_generation} по retrieval"
    elif opts.mode == "real" and not gen.empty:
        reason += " (LLM не ответил ни на один вопрос — генерация без оценок)"

    cfg = run.config(best_id)
    best = {
        "config_id": best_id,
        "name": cfg["name"],
        "setup": describe(cfg["config"]),
        "reason": reason,
        "retrieval": {
            k: _num(v)
            for k, v in table.loc[best_id].items()
            if k.startswith(("dev_", "test_"))
        },
        "generation": (
            {k: _num(v) for k, v in gen.set_index("config_id").loc[best_id].items()}
            if not gen.empty and best_id in set(gen["config_id"])
            else None
        ),
        "mode": opts.mode,
        "config": cfg["config"],
    }
    run.state["best"] = best
    run.save()
    run.file("best_config.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run.set_status("select", "done")
    emit("select", f"Лучший: {cfg['name']} ({best['setup']}) — {reason}", 1.0)


def _num(v):
    """JSON-safe scalar: NaN -> None, numpy numbers -> Python numbers."""
    if hasattr(v, "item"):
        v = v.item()
    try:
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return v


# ── after the run: chat and deploy ────────────────────────────


def chat(
    run: Run,
    config_id: str,
    question: str,
    ds_key: str | None = None,
    client=None,
    top_k: int = PIPELINE_GENERATION_TOP_K,
) -> dict:
    """Ask a question against one config's collection (answer + sources)."""
    from evalkit.client import make_client

    run.check_backend()
    cfg = run.config(config_id)
    ds_key = ds_key or next(iter(run.state["datasets"]))
    client = client or make_client()
    agent = run.ensure_agent(client, cfg, ds_key)
    kwargs = {}
    return client.agent_query(question, agent_id=agent, **kwargs)


def deploy(
    run: Run,
    name: str,
    client=None,
    on_event: EventFn = print_event,
    config_id: str | None = None,
) -> dict:
    """Create collection(s) ``name`` (``name_<dataset>`` for several datasets)
    with the best (or given) config, upload the corpus, create the agent."""
    from evalkit.client import make_client

    run.check_backend()
    cfg = run.config(config_id or run.state["best"]["config_id"])
    client = client or make_client()
    datasets = list(run.state["datasets"])
    deployed = []
    for ds_key in datasets:
        coll = name if len(datasets) == 1 else f"{name}_{ds_key}"
        on_event(
            Event("deploy", f"{coll}: создаю коллекцию ({describe(cfg['config'])})")
        )
        client.create_collection(
            collection_name=coll,
            description=agent_settings(cfg["config"])["tool_description"],
            **collection_config(cfg["config"]),
        )
        corpus, _ = run.dataset(ds_key)
        results = client.upload_many(
            [(d, t, coll) for d, t in corpus.items()], desc=coll
        )
        failed = _wait_tasks(
            client, [t for _, t in results if t], on_event_wrapper(on_event), "deploy"
        )
        errors = sum(d.startswith("ERROR") for d, _ in results) + len(failed)
        if errors or len(results) != len(corpus):
            raise RuntimeError(f"{coll}: document upload failed ({errors} errors)")
        agent = run.ensure_agent(client, cfg, ds_key, collection=coll)
        deployed.append(
            {
                "collection": coll,
                "agent_id": agent,
                "docs": len(corpus),
                "failed": len(failed),
            }
        )
        on_event(
            Event(
                "deploy",
                f"{coll}: загружено {len(corpus)} документов, агент {agent}",
                level="done",
            )
        )
    record = {
        "name": name,
        "config_id": cfg["id"],
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "targets": deployed,
    }
    run.state.setdefault("deployments", []).append(record)
    run.save()
    return record


def on_event_wrapper(on_event: EventFn):
    return lambda step, message, progress=None, level="info": on_event(
        Event(step, message, progress, level)
    )
