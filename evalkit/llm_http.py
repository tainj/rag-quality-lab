"""
OpenAI-compatible clients for the configured LLM.

With LLM_VIA_PLATFORM=1 the LLM is reached through agent-core: its endpoint is
``/chat/completion`` (singular), while the OpenAI SDK calls
``/chat/completions``. The httpx clients below rewrite that path, so the
OpenAI SDK and deepeval work unchanged. Message contents sent as a list of
text parts (deepeval does that) are flattened to strings, which the platform
requires.
"""

from __future__ import annotations

import json

import httpx

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_VIA_PLATFORM
from evalkit.transport import forward_httpx, tunnel_port


def _flatten_content(body: dict) -> bool:
    """[{"type": "text", "text": ...}] -> "..." (the platform wants strings)."""
    changed = False
    for msg in body.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            msg["content"] = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
            changed = True
    return changed


def _rewrite(request: httpx.Request) -> None:
    path = request.url.path
    if not path.endswith("/chat/completions"):
        return
    request.url = request.url.copy_with(path=path[: -len("s")])
    try:
        body = json.loads(request.content)
    except (ValueError, httpx.RequestNotRead):
        return
    if _flatten_content(body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request.headers["Content-Length"] = str(len(data))
        request.stream = httpx.ByteStream(data)
        request._content = data


async def _arewrite(request: httpx.Request) -> None:
    _prepare(request)


def _prepare(request: httpx.Request) -> None:
    if LLM_VIA_PLATFORM:
        _rewrite(request)
    forward_httpx(request)


def http_clients(timeout: float = 600) -> dict:
    """``http_client`` / ``async_http_client`` kwargs (empty without the platform)."""
    if not LLM_VIA_PLATFORM and not tunnel_port():
        return {}
    return {
        "http_client": httpx.Client(
            timeout=timeout, event_hooks={"request": [_prepare]}, trust_env=False
        ),
        "async_http_client": httpx.AsyncClient(
            timeout=timeout, event_hooks={"request": [_arewrite]}, trust_env=False
        ),
    }


def openai_client(timeout: float = 600, max_retries: int | None = None):
    from openai import OpenAI

    client = (
        httpx.Client(
            timeout=timeout, event_hooks={"request": [_prepare]}, trust_env=False
        )
        if LLM_VIA_PLATFORM or tunnel_port()
        else None
    )
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        timeout=timeout,
        max_retries=LLM_MAX_RETRIES if max_retries is None else max_retries,
        **({"http_client": client} if client else {}),
    )
