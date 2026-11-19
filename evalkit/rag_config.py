"""
JSON-backed RAG experiment configs: retrieval settings and generation prompts.

Presets live in configs/presets.json; defaults in configs/settings.json.
Use collection_config() to build collection API payloads. The full config id
includes generation instructions; collection_id() ignores them so prompt variants
can share an index.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from config import (
    AGENT_DESCRIPTION,
    AGENT_MODEL,
    EMBEDDING_MODEL_PRESETS,
    EVAL_COLLECTION_CHUNKING,
    EVAL_COLLECTION_EMBEDDING,
    EVAL_COLLECTION_RERANKER,
    RAG_SYSTEM_PROMPT,
    RAG_TOOL_DESCRIPTION,
    REPO_ROOT,
)


def load_presets(path: str | Path | None = None) -> dict[str, list[dict[str, Any]]]:
    from evalkit.json_config import load_json_object

    path = path or os.environ.get(
        "RAG_EVAL_PRESETS", REPO_ROOT / "configs/presets.json"
    )
    data = load_json_object(path)
    if set(data) != {"presets"}:
        raise ValueError(f"{path}: expected a presets list")
    names = set()
    for backend, presets in data.items():
        if not isinstance(presets, list) or not presets:
            raise ValueError(f"{path}: {backend} must be a non-empty list")
        for preset in presets:
            if (
                not isinstance(preset, dict)
                or not isinstance(preset.get("name"), str)
                or not preset["name"].strip()
            ):
                raise ValueError(f"{path}: every preset needs a non-blank name")
            if preset["name"] in names:
                raise ValueError(f"{path}: duplicate preset name {preset['name']}")
            names.add(preset["name"])
            if not isinstance(preset.get("overrides"), dict):
                raise ValueError(f"{path}: {preset['name']} needs an overrides object")  # noqa: TRY004 — validation failures consistently use ValueError
            for key in ("title", "forecast"):
                if key in preset and not isinstance(preset[key], str):
                    raise ValueError(f"{path}: {preset['name']}.{key} must be text")
            validate_config(apply_overrides(base_config(), preset["overrides"]))
    return data


def available_presets(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Agent-core presets."""
    return load_presets(path)["presets"]


def base_config() -> dict[str, Any]:
    """Experiment template from JSON settings (deep copy)."""
    return {
        "embedding": copy.deepcopy(EVAL_COLLECTION_EMBEDDING),
        "chunking": copy.deepcopy(EVAL_COLLECTION_CHUNKING),
        "reranker": copy.deepcopy(EVAL_COLLECTION_RERANKER),
        "generation": {"system_prompt": RAG_SYSTEM_PROMPT},
        "agent": {
            "model": AGENT_MODEL,
            "description": AGENT_DESCRIPTION,
            "tool_description": RAG_TOOL_DESCRIPTION,
        },
    }


def parse_value(raw: str) -> Any:
    """Parse a CLI/GUI value: JSON if possible (2, false, "x"), else string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict:
    """Return a copy of ``config`` with dotted-path overrides applied.

    Choosing ``embedding.model_name`` also applies that model's preset
    (prefixes, vector size) unless those keys are overridden explicitly.
    """
    cfg = copy.deepcopy(config)
    model = overrides.get("embedding.model_name")
    if isinstance(model, str) and model in EMBEDDING_MODEL_PRESETS:
        for key, value in EMBEDDING_MODEL_PRESETS[model].items():
            cfg["embedding"][key] = value
    for path, value in overrides.items():
        if (
            not isinstance(path, str)
            or not path
            or any(not part for part in path.split("."))
        ):
            raise ValueError(f"Invalid override path: {path!r}")
        node = cfg
        *parents, leaf = path.split(".")
        for part in parents:
            if not isinstance(node.get(part, {}), dict):
                raise ValueError(f"Cannot set {path}: {part} is not an object")  # noqa: TRY004 — validation failures consistently use ValueError
            node = node.setdefault(part, {})
        node[leaf] = value
    validate_config(cfg)
    return cfg


def system_prompt(config: dict) -> str:
    """Old runs without a generation section use the JSON default."""
    return config.get("generation", {}).get("system_prompt", RAG_SYSTEM_PROMPT)


def collection_config(config: dict) -> dict:
    """Only retrieval settings belong in the collection API payload."""
    return {
        key: copy.deepcopy(config[key]) for key in ("embedding", "chunking", "reranker")
    }


def collection_id(config: dict) -> str:
    """Prompt variants share the same index, but have different config ids."""
    identity = collection_config(config)
    if "agent" in config:
        identity["tool_description"] = agent_settings(config)["tool_description"]
    return config_id(identity)


def retrieval_id(config: dict) -> str:
    return config_id(collection_config(config))


def agent_settings(config: dict) -> dict:
    return {
        "model": AGENT_MODEL,
        "description": AGENT_DESCRIPTION,
        "tool_description": RAG_TOOL_DESCRIPTION,
        **config.get("agent", {}),
    }


def validate_config(config: dict) -> None:
    if (config.get("reranker") or {}).get("enabled"):
        raise ValueError("Reranker is disabled for this project")
    allowed = {"embedding", "chunking", "reranker", "generation", "agent"}
    if set(config) - allowed:
        raise ValueError(
            f"Unknown configuration sections: {sorted(set(config) - allowed)}"
        )
    for key in ("embedding", "chunking", "reranker"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"{key} must be an object")  # noqa: TRY004 — validation failures consistently use ValueError
    if "generation" in config and (
        not isinstance(config["generation"], dict)
        or set(config["generation"]) != {"system_prompt"}
    ):
        raise ValueError("generation must contain only system_prompt")
    prompt = system_prompt(config)
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("generation.system_prompt must be non-blank text")
    if "agent" in config and (
        not isinstance(config["agent"], dict)
        or set(config["agent"]) - {"model", "description", "tool_description"}
    ):
        raise ValueError("agent supports model, description and tool_description")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in agent_settings(config).values()
    ):
        raise ValueError(
            "Agent model, description and tool_description must be non-blank text"
        )
    chunk = config["chunking"]
    size, overlap = chunk.get("chunk_size"), chunk.get("chunk_overlap")
    if (
        type(size) is not int
        or size <= 0
        or type(overlap) is not int
        or not 0 <= overlap < size
    ):
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    emb = config["embedding"]
    if not isinstance(emb.get("model_name"), str) or not emb["model_name"].strip():
        raise ValueError("embedding.model_name must be non-blank text")
    if type(emb.get("batch_size")) is not int or emb["batch_size"] <= 0:
        raise ValueError("embedding.batch_size must be a positive integer")
    sparse = emb.get("sparse") or {}
    if not isinstance(sparse, dict) or sparse.get("mode", "off") not in (
        "off",
        "lemmatized",
        "qdrant_bm25",
    ):
        raise ValueError("Unknown embedding.sparse.mode")
    weight = sparse.get("dense_rrf_weight", 1)
    if type(weight) not in (int, float) or not 0 <= weight < float("inf"):
        raise ValueError(
            "embedding.sparse.dense_rrf_weight must be finite and non-negative"
        )


def config_id(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def make_config(
    name: str,
    overrides: dict[str, Any],
    title: str = "",
    forecast: str = "",
    parent: str | None = None,
) -> dict[str, Any]:
    """Config record used by the pipeline: name, id, overrides, full config."""
    full = apply_overrides(base_config(), overrides)
    return {
        "id": config_id(full),
        "name": name,
        "title": title or name,
        "forecast": forecast,
        "parent": parent,
        "overrides": overrides,
        "config": full,
    }


def preset_configs(
    names: list[str] | None = None, path: str | Path | None = None
) -> list[dict[str, Any]]:
    """Config records for the given preset names (backend's presets by default)."""
    groups = load_presets(path)
    by_name = {p["name"]: p for group in groups.values() for p in group}
    unknown = [n for n in (names or []) if n not in by_name]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}. Known: {list(by_name)}")
    chosen = [by_name[n] for n in names] if names else groups["presets"]
    return [
        make_config(
            p["name"], p["overrides"], p.get("title", ""), p.get("forecast", "")
        )
        for p in chosen
    ]


def derive_config(
    parent: dict[str, Any], changes: dict[str, Any], name: str | None = None
) -> dict[str, Any]:
    """New config record = parent's overrides + ``changes``."""
    overrides = {**parent["overrides"], **changes}
    suffix = ",".join(
        f"prompt={hashlib.sha1(str(v).encode()).hexdigest()[:8]}"
        if k
        in ("generation.system_prompt", "agent.description", "agent.tool_description")
        else f"{k.split('.')[-1]}={v}"
        for k, v in changes.items()
    )
    # Derive from the saved snapshot, even after defaults/presets were edited.
    full = copy.deepcopy(parent["config"])
    full.setdefault("generation", {"system_prompt": RAG_SYSTEM_PROMPT})
    full = apply_overrides(full, changes)
    if full == parent["config"] and name is None:
        return copy.deepcopy(parent)
    return {
        "id": config_id(full),
        "name": name or f"{parent['name']}+{suffix}",
        "title": f"{parent.get('title', parent['name'])} | {suffix}",
        "forecast": parent.get("forecast", ""),
        "parent": parent["id"],
        "overrides": overrides,
        "config": full,
    }


def describe(config: dict[str, Any]) -> str:
    """One-line human summary of a full collection config."""
    emb, chunk = config["embedding"], config["chunking"]
    sparse = emb.get("sparse", {}) or {}
    mode = sparse.get("mode", "off")
    if mode == "off":
        search = "dense"
    elif sparse.get("dense_rrf_weight", 1) == 0:
        search = f"bm25({mode})"
    else:
        search = f"hybrid({mode}, dense×{sparse.get('dense_rrf_weight', 1)})"
    rerank = " +rerank" if (config.get("reranker") or {}).get("enabled") else ""
    model = "—" if search.startswith("bm25") else emb.get("model_name")
    return (
        f"{model} | {search}{rerank} | "
        f"chunk {chunk.get('chunk_size')}/{chunk.get('chunk_overlap')}"
    )
