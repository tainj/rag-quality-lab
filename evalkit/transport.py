"""HTTPS clients for agent-core with optional scoped SSH forwarding.

Host, TLS SNI and certificate verification always refer to the platform domain.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import httpx

import config


def tunnel_port() -> int | None:
    raw = os.environ.get("RAG_EVAL_TUNNEL_PORT")
    if not raw:
        return None
    port = int(raw)
    if not 1 <= port <= 65535:
        raise ValueError("RAG_EVAL_TUNNEL_PORT must be in 1..65535")
    return port


def forward_httpx(request: httpx.Request) -> None:
    port = tunnel_port()
    platform = urlsplit(config.AGENT_PLATFORM_URL)
    if port and request.url.host == platform.hostname:
        if request.url.scheme != "https":
            raise ValueError("SSH forwarding requires HTTPS")
        request.headers["Host"] = platform.netloc
        request.extensions["sni_hostname"] = platform.hostname
        request.url = request.url.copy_with(host="127.0.0.1", port=port)


def platform_session() -> httpx.Client:
    return httpx.Client(
        timeout=config.RAGAS_TIMEOUT,
        event_hooks={"request": [forward_httpx]},
        trust_env=not bool(tunnel_port()),
        transport=httpx.HTTPTransport(retries=2),
    )
