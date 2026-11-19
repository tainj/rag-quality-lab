"""
Synthetic QA generation grounded in RAW document spans (paragraph-level),
*independent* of any downstream chunking strategy -- this is the key design
choice that lets one synthetic dataset serve chunking ablation, embedding
ablation, and hybrid/reranker ablation without regeneration.

Uses the OpenAI-compatible API via LiteLLM proxy. Requires LLM_BASE_URL
and LLM_API_KEY in the environment (see config.py).
"""

from __future__ import annotations

import json
import re
import uuid

from openai import OpenAI

from config import (
    DEDUP_JUDGE_PROMPT,
    EXACT_MATCH_PROMPT,
    LLM_MAX_TOKENS,
    MULTI_HOP_PROMPT,
    SINGLE_HOP_PROMPT,
    UNANSWERABLE_PROMPT,
)

# ── Structured-output schemas ──────────────────
_QA_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_pair",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "reference_answer": {"type": "string"},
            },
            "required": ["question", "reference_answer"],
            "additionalProperties": False,
        },
    },
}

_DEDUP_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "dedup_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def split_into_paragraphs(doc_id: str, text: str, min_len: int = 200) -> list[dict]:
    """Logical generation units, decoupled from retrieval chunking. Merges
    tiny paragraphs forward so each unit has enough content for a real
    question."""
    units, buf, buf_start = [], "", 0
    cursor = 0
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            cursor += len(para) + 2
            continue
        start = text.find(para, cursor)
        start = start if start != -1 else cursor
        end = start + len(para)
        if not buf:
            buf, buf_start = para, start
        else:
            buf = buf + "\n\n" + para
        if len(buf) >= min_len:
            units.append(
                {"doc_id": doc_id, "start": buf_start, "end": end, "text": buf}
            )
            buf = ""
        cursor = end
    if buf:
        units.append({"doc_id": doc_id, "start": buf_start, "end": cursor, "text": buf})
    return units


def _call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    *,
    max_tokens: int = LLM_MAX_TOKENS,
    response_format: dict | None = None,
) -> dict | None:
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def generate_single_hop(client: OpenAI, model: str, unit: dict) -> dict | None:
    parsed = _call_llm(
        client,
        model,
        SINGLE_HOP_PROMPT.format(passage=unit["text"]),
        response_format=_QA_SCHEMA,
    )
    if not parsed:
        return None
    return {
        "qid": str(uuid.uuid4()),
        "question": parsed["question"],
        "reference_answer": parsed["reference_answer"],
        "question_type": "single_hop",
        "gold_spans": [
            {"doc_id": unit["doc_id"], "start": unit["start"], "end": unit["end"]}
        ],
    }


def generate_multi_hop(
    client: OpenAI, model: str, unit_a: dict, unit_b: dict
) -> dict | None:
    parsed = _call_llm(
        client,
        model,
        MULTI_HOP_PROMPT.format(passage_a=unit_a["text"], passage_b=unit_b["text"]),
        response_format=_QA_SCHEMA,
    )
    if not parsed:
        return None
    return {
        "qid": str(uuid.uuid4()),
        "question": parsed["question"],
        "reference_answer": parsed["reference_answer"],
        "question_type": "multi_hop",
        "gold_spans": [
            {
                "doc_id": unit_a["doc_id"],
                "start": unit_a["start"],
                "end": unit_a["end"],
            },
            {
                "doc_id": unit_b["doc_id"],
                "start": unit_b["start"],
                "end": unit_b["end"],
            },
        ],
    }


def generate_unanswerable(client: OpenAI, model: str, unit: dict) -> dict | None:
    parsed = _call_llm(
        client,
        model,
        UNANSWERABLE_PROMPT.format(passage=unit["text"]),
        response_format=_QA_SCHEMA,
    )
    if not parsed:
        return None
    return {
        "qid": str(uuid.uuid4()),
        "question": parsed["question"],
        "reference_answer": "NOT_ANSWERABLE",
        "question_type": "unanswerable",
        "gold_spans": [],
    }


def generate_exact_match(client: OpenAI, model: str, unit: dict) -> dict | None:
    parsed = _call_llm(
        client,
        model,
        EXACT_MATCH_PROMPT.format(passage=unit["text"]),
        response_format=_QA_SCHEMA,
    )
    if not parsed:
        return None
    return {
        "qid": str(uuid.uuid4()),
        "question": parsed["question"],
        "reference_answer": parsed["reference_answer"],
        "question_type": "exact_match",
        "gold_spans": [
            {"doc_id": unit["doc_id"], "start": unit["start"], "end": unit["end"]}
        ],
    }


def is_near_duplicate(client: OpenAI, model: str, q1: str, q2: str) -> bool:
    parsed = _call_llm(
        client,
        model,
        DEDUP_JUDGE_PROMPT.format(q1=q1, q2=q2),
        max_tokens=10,
        response_format=_DEDUP_SCHEMA,
    )
    if not parsed:
        return False
    return parsed["answer"].strip().upper().startswith("ДА")
