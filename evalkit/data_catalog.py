"""Configured and successfully generated datasets, available to CLI and GUI."""

from __future__ import annotations

import json
from pathlib import Path

import config


def generated_datasets() -> dict[str, dict]:
    """Completed generated manifests, newest first, read independently of GUI jobs."""
    records = {}
    paths = (config.RUNS_DIR / "datasets").glob("*/manifest.json")
    for path in sorted(paths, key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                continue
            if (
                record.get("status") != "done"
                or record["dataset_id"] != path.parent.name
            ):
                continue
            ds = record["dataset_config"]
            if not all(Path(ds[key]).is_file() for key in ("qa_path", "corpus_path")):
                continue
            records[record["dataset_id"]] = record
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records


def available_datasets() -> dict[str, dict]:
    datasets = dict(config.EVAL_DATASETS)
    datasets.update(
        {key: record["dataset_config"] for key, record in generated_datasets().items()}
    )
    return datasets
