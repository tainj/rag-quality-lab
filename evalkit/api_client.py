"""
RagServiceClient — talks to agent-core for document upload, search, and
agent queries.

Interface:
    create_collection(collection_name, embedding=None, chunking=None,
                      reranker=None, description="") -> dict
        Create/update a global collection via the collections CRUD API
        (pre-creates the Qdrant vector layout and a ``{name}_search`` RAG tool).
    upload_document(doc_id, text, collection_name) -> (doc_id, task_id)
    upload_many(docs, max_workers, desc) -> list[(doc_id, task_id)]
    search(query, collection_name, top_k, **kwargs) -> List[dict]
    check_tasks(task_ids) -> list[dict]
    ensure_eval_agent(collection_name) -> str
        Create/update a test agent with the RAG tool for the given collection.
        Returns the agent_id to use with agent_query().
    agent_query(query, agent_id) -> dict
    delete_collection(collection_name) -> None
"""

from __future__ import annotations

from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeout,
    as_completed,
)
import json
import threading
import time
from typing import Any
import uuid

import httpx

from config import AGENT_MODEL, AGENT_TOOL_PROMPT, RAG_SYSTEM_PROMPT
from evalkit.usage import agent_usage

POLL_INTERVAL = 1.0
POLL_TIMEOUT = 120.0
SEARCH_RETRIES = 2


class UnsupportedConfigError(ValueError):
    pass


class RagServiceClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 120,
        agent_id: str = "rag_eval_agent",
    ):
        self.base_url = base_url.rstrip("/")
        self.headers: dict[str, str] = (
            {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        self.timeout = httpx.Timeout(timeout, connect=min(8, timeout))
        self.agent_id = agent_id
        self._sessions = threading.local()

    @property
    def session(self):
        if not hasattr(self._sessions, "client"):
            from evalkit.transport import platform_session

            self._sessions.client = platform_session()
        return self._sessions.client

    def upload_document(
        self, doc_id: str, text: str, collection_name: str = "eval_default"
    ) -> tuple[str, str | None]:
        """Upload a text document (no wait for indexing).

        Returns ``(server_doc_id, task_id)``.
        """
        if not doc_id or not str(doc_id).strip():
            raise ValueError("doc_id is empty or whitespace")
        if not text or not text.strip():
            raise ValueError(f"Text is empty for doc {doc_id}")
        if not collection_name or not collection_name.strip():
            raise ValueError(f"collection_name is empty for doc {doc_id}")

        doc_id = str(doc_id).strip()
        if len(doc_id) > 255:
            doc_id = doc_id[:255]

        doc_str = json.dumps({"name": doc_id, "metadata": {"eval_doc_id": doc_id}})
        target_str = json.dumps(
            {"collections": [collection_name], "user_id": None, "session_id": None}
        )

        for attempt in range(3):
            try:
                resp = self.session.post(
                    f"{self.base_url}/document/add",
                    data={
                        "text_content": text,
                        "doc_str": doc_str,
                        "target_str": target_str,
                    },
                    headers={**self.headers, "Accept": "application/json"},
                    timeout=self.timeout,
                )
                break
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                safe_to_retry = (
                    isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout))
                    or "handshake" in str(exc).lower()
                )
                if not safe_to_retry or attempt == 2:
                    raise
                time.sleep(attempt + 1)
        resp.raise_for_status()
        body = resp.json()
        return body.get("doc_id", doc_id), body.get("task_id")

    def upload_many(
        self,
        docs: list[tuple[str, str, str]],
        max_workers: int = 16,
        desc: str = "upload",
        on_result=None,
    ) -> list[tuple[str, str | None]]:
        """Upload documents in parallel.

        ``docs`` is a list of ``(doc_id, text, collection_name)``.
        Returns list of ``(server_doc_id, task_id)`` tuples (same order as input).
        """
        from tqdm import tqdm

        results: list[tuple[str, str | None]] = [("", None)] * len(docs)

        t0 = time.perf_counter()
        from config import AGENT_WORKER_POOL_SIZE

        with ThreadPoolExecutor(
            max_workers=min(max_workers, AGENT_WORKER_POOL_SIZE)
        ) as pool:
            futures = {
                pool.submit(self.upload_document, doc_id, text, coll): i
                for i, (doc_id, text, coll) in enumerate(docs)
            }
            pbar = tqdm(total=len(docs), desc=desc, unit="doc")
            consecutive_errors = 0
            aborted = False
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                    consecutive_errors = 0
                except CancelledError:
                    results[i] = ("ERROR: not submitted", None)
                except Exception as e:  # noqa: BLE001 — preserve every per-document failure
                    results[i] = (f"ERROR: {e}", None)
                    consecutive_errors += 1
                if on_result is not None:
                    on_result(i, results[i])
                if consecutive_errors >= 5 and not aborted:
                    aborted = True
                    print(
                        f"[{desc}] Five consecutive upload errors; cancelling queued documents",
                        flush=True,
                    )
                    for queued in futures:
                        queued.cancel()
                pbar.update(1)
            pbar.close()

        elapsed = time.perf_counter() - t0
        errors = [r for r in results if r[0].startswith("ERROR")]
        ok = len(results) - len(errors)
        print(
            f"[{desc}] {ok}/{len(docs)} ok in {elapsed:.1f}s "
            f"({ok / max(elapsed, 0.01):.1f} docs/s)"
        )
        if errors:
            print(f"[{desc}] WARN: {len(errors)} failed")

        return results

    def _wait_for_task(self, task_id: str) -> dict:
        deadline = time.monotonic() + POLL_TIMEOUT
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Task {task_id} did not finish within {POLL_TIMEOUT}s"
                )
            resp = self.session.get(
                f"{self.base_url}/document/task/{task_id}",
                headers=self.headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            state = body.get("state", "")
            if state == "completed":
                return body.get("result", {})
            if state == "failed":
                raise RuntimeError(
                    f"Indexing task {task_id} failed: {body.get('error', 'unknown')}"
                )
            time.sleep(POLL_INTERVAL)

    def check_tasks(self, task_ids: list[str]) -> list[dict]:
        """Check status of multiple Celery tasks in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _check_one(tid: str) -> dict:
            try:
                resp = self.session.get(
                    f"{self.base_url}/document/task/{tid}",
                    headers=self.headers,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPError, ValueError) as e:
                return {"task_id": tid, "error": str(e)}

        from config import AGENT_WORKER_POOL_SIZE

        with ThreadPoolExecutor(max_workers=AGENT_WORKER_POOL_SIZE) as pool:
            futures = {pool.submit(_check_one, tid): tid for tid in task_ids}
            results_by_tid: dict[str, dict] = {}
            try:
                for future in as_completed(futures, timeout=30):
                    tid = futures[future]
                    results_by_tid[tid] = future.result()
            except FuturesTimeout:
                for future in futures:
                    future.cancel()
                for tid in task_ids:
                    results_by_tid.setdefault(
                        tid, {"task_id": tid, "error": "Task check batch timed out"}
                    )
        return [results_by_tid[tid] for tid in task_ids]

    def search(
        self,
        query: str,
        collection_name: str = "eval_default",
        top_k: int = 10,
        **_kwargs: Any,
    ) -> list[dict]:
        """Direct vector search via /vdb/search.

        Returns list of {chunk_id, doc_id, text, metadata, score}.
        """
        body: dict[str, Any] = {
            "query": query,
            "collection_name": collection_name,
            "top_k": top_k,
            "threshold": 0.0,
        }
        # A slow search (e.g. with the reranker) must not kill a long run:
        # retry timeouts / connection errors / 5xx a few times.
        for attempt in range(SEARCH_RETRIES):
            try:
                resp = self.session.post(
                    f"{self.base_url}/vdb/search",
                    json=body,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                if resp.status_code < 500 and resp.status_code not in (403, 429):
                    break  # 403/429: seen transiently while the stand restarts
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == SEARCH_RETRIES - 1:
                    raise
            time.sleep(min(2**attempt, 30))
        resp.raise_for_status()
        raw = resp.json()

        results = []
        for item in raw:
            meta = item.get("metadata", {})
            doc_id = meta.get("eval_doc_id", meta.get("name", "unknown"))
            char_start = meta.get("char_start", 0)
            char_end = meta.get("char_end", 0)

            chunk_id = f"{doc_id}::{collection_name}::{char_start}-{char_end}"
            results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": item.get("text", ""),
                    "metadata": meta,
                    "score": item.get("score", 0.0),
                }
            )
        return results

    @staticmethod
    def _tool_name_for(collection_name: str) -> str:
        """RAG tool name convention: {collection_name}_search."""
        return f"{collection_name}_search"

    def create_collection(
        self,
        collection_name: str,
        embedding: dict[str, Any] | None = None,
        chunking: dict[str, Any] | None = None,
        reranker: dict[str, Any] | None = None,
        description: str = "",
    ) -> dict:
        """Create a global collection via POST /collections.

        The service persists the config, pre-creates the Qdrant collection with
        the matching vector layout (dense + sparse per
        ``embedding['sparse']['mode']``) and registers a ``{name}_search`` RAG
        tool. Passing ``{}``/``None`` for a section uses service defaults.

        Returns the server response body (``{"status": "created", ...}``).
        """
        payload: dict[str, Any] = {
            "name": collection_name,
            "embedding": embedding or {},
            "chunking": chunking or {},
            "reranker": reranker or {},
            "description": description,
        }
        resp = self.session.post(
            f"{self.base_url}/collections",
            json=payload,
            headers={**self.headers, "Accept": "application/json"},
            timeout=self.timeout,
        )
        if resp.status_code in (400, 422):
            raise UnsupportedConfigError(
                f"Collection rejected: HTTP {resp.status_code}: {resp.text[:500]}"
            )
        resp.raise_for_status()
        return resp.json()

    def describe_collection(self, collection_name: str) -> dict:
        response = self.session.get(
            f"{self.base_url}/collections/{collection_name}",
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def collection_exists(self, collection_name: str) -> bool:
        response = self.session.get(
            f"{self.base_url}/collections/{collection_name}",
            headers=self.headers,
            timeout=self.timeout,
        )
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    def ensure_eval_agent(
        self,
        collection_name: str,
        system_prompt: str | None = None,
        agent_id: str | None = None,
        model: str | None = None,
        description: str | None = None,
    ) -> str:
        """Create or update an eval agent with the RAG tool for *collection_name*.

        The CRUD collections API auto-creates a tool named
        ``{collection_name}_search`` when the collection is created.  This
        method registers (or replaces) an agent that exposes exactly that tool,
        so ``agent_query`` can be called against any collection.

        Returns the supplied agent_id (collection_name by default). Distinct
        ids let prompt variants use the same collection without overwriting
        each other's agent settings.
        """
        tool_name = self._tool_name_for(collection_name)
        agent_id = agent_id or collection_name
        prompt = RAG_SYSTEM_PROMPT if system_prompt is None else system_prompt
        system_prompt = (
            AGENT_TOOL_PROMPT.replace("{tool_name}", tool_name) + "\n\n" + prompt
        )
        # Platform AgentConfig: the system prompt goes to prompts.custom
        payload = {
            "prompts": {"custom": system_prompt},
            "model": model or AGENT_MODEL,
            "name": agent_id,
            "description": description
            or f"Eval agent for collection {collection_name}",
            "tools": [tool_name],
        }
        resp = self.session.post(
            f"{self.base_url}/agent/create",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return agent_id

    def agent_query(self, query: str, agent_id: str | None = None) -> dict:
        """Run the agent via /agent/run and parse the response.

        Handles the real agent-core message format:
          - human: user question
          - ai (finish_reason="tool_calls"): agent calls a tool
          - tool: tool result with retrieved_docs
          - ai (finish_reason="stop"): final answer text

        Returns {"answer", "tool_calls", "retrieved_chunks", "trace_id"}.
        """
        thread_id = str(uuid.uuid4())
        payload = {
            "messages": [{"role": "user", "content": query}],
            "agent_id": agent_id or self.agent_id,
            "thread_id": thread_id,
            "stream": False,
            "user_id": "rag_eval",
        }
        resp = self.session.post(
            f"{self.base_url}/agent/run",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        messages = body.get("messages", [])

        answer = ""
        tool_calls: list[dict] = []
        retrieved_chunks: list[dict] = []

        for msg in messages:
            msg_type = msg.get("type", "")
            meta = msg.get("response_metadata") or {}

            if msg_type == "ai":
                finish = meta.get("finish_reason", "")
                # Final answer: non-empty content with finish_reason="stop"
                if (
                    finish != "tool_calls"
                    and not msg.get("tool_calls")
                    and msg.get("content")
                ):
                    answer = msg["content"]
                # Tool calls (RAG or structured output)
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tool_calls.append(
                            {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "id": tc.get("id", ""),
                            }
                        )

            elif msg_type == "tool":
                # Extract retrieved docs from response_metadata or artifact
                docs = meta.get("retrieved_docs") or (msg.get("artifact") or {}).get(
                    "retrieved_docs"
                )
                if docs and isinstance(docs, list):
                    retrieved_chunks.extend(
                        {
                            **doc,
                            "content": doc.get("content")
                            or doc.get("text")
                            or doc.get("page_content")
                            or "",
                        }
                        for doc in docs
                        if isinstance(doc, dict)
                    )

        usage, complete = agent_usage(body, messages)
        return {
            "answer": answer,
            "tool_calls": tool_calls,
            "retrieved_chunks": retrieved_chunks,
            "trace_id": body.get("trace_id", ""),
            "usage": usage,
            "usage_complete": complete,
        }

    def delete_collection(self, collection_name: str) -> None:
        """Delete a global collection: its documents, then its config.

        The platform answers 500 on DELETE /collections/{name} while the
        collection still holds vectors, so the data is removed first via
        DELETE /document/collection (async task, waited for).
        """
        target = {"collections": [collection_name], "user_id": None, "session_id": None}
        resp = self.session.request(
            "DELETE",
            f"{self.base_url}/document/collection",
            data={"target_str": json.dumps(target)},
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        task_ids = [t["task_id"] for t in resp.json() if t.get("task_id")]
        deadline = time.time() + POLL_TIMEOUT
        while task_ids and time.time() < deadline:
            states = [s.get("state") for s in self.check_tasks(task_ids)]
            if all(s in ("completed", "failed") for s in states):
                break
            time.sleep(POLL_INTERVAL)

        resp = self.session.request(
            "DELETE",
            f"{self.base_url}/collections/{collection_name}",
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
