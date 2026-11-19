"""Provider-reported token accounting. Missing usage is never treated as zero."""

from __future__ import annotations


def normalize_usage(raw) -> dict | None:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    if not isinstance(raw, dict):
        return None
    result = {}
    for target, names in {
        "input_tokens": ("input_tokens", "prompt_tokens"),
        "output_tokens": ("output_tokens", "completion_tokens"),
        "total_tokens": ("total_tokens",),
    }.items():
        for name in names:
            value = raw.get(name)
            if type(value) is int and value >= 0:
                result[target] = value
                break
    if (
        "total_tokens" not in result
        and {"input_tokens", "output_tokens"} <= result.keys()
    ):
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result if "total_tokens" in result else None


def agent_usage(body: dict, messages: list[dict]) -> tuple[dict | None, bool]:
    for raw in (
        body.get("usage"),
        (body.get("response_metadata") or {}).get("token_usage"),
    ):
        usage = normalize_usage(raw)
        if usage is not None:
            return usage, True
    usages, missing = [], 0
    for message in messages:
        if message.get("type") not in ("ai", "assistant"):
            continue
        meta = message.get("response_metadata") or {}
        usage = next(
            (
                u
                for raw in (
                    message.get("usage_metadata"),
                    meta.get("token_usage"),
                    meta.get("usage"),
                )
                if (u := normalize_usage(raw)) is not None
            ),
            None,
        )
        if usage is not None:
            usages.append(usage)
        elif message.get("content") or message.get("tool_calls"):
            missing += 1
    if not usages:
        return None, False
    keys = set.intersection(*(set(u) for u in usages))
    return {k: sum(u[k] for u in usages) for k in keys}, missing == 0
