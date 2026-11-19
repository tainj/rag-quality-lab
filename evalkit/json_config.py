"""Strict JSON loading shared by settings and RAG presets."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number: {value}")
    return number


def load_json_object(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_keys,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load JSON configuration {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")  # noqa: TRY004 — validation failures consistently use ValueError
    return data


def require_setting(data: dict, name: str, expected: type | tuple[type, ...]):
    if name not in data:
        raise ValueError(f"Missing setting: {name}")
    value = data[name]
    if not isinstance(value, expected) or (
        isinstance(value, bool) and expected is not bool
    ):
        raise ValueError(f"Invalid type for setting {name}")
    return value


def env_bool(name: str, default: bool, environ) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    if raw.lower() in ("1", "true", "yes", "on"):
        return True
    if raw.lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name}: expected 1/0 or true/false")
