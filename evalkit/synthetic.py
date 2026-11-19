"""Generate answerable QA with short, verbatim, independently verifiable evidence."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import time
import uuid

import config
from evalkit.csv_loader import clean_and_filter
from evalkit.dataset_loaders import load_upload
from evalkit.llm_http import openai_client
from evalkit.usage import normalize_usage

QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "grounded_qa",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "reference_answer": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["source_id", "quote"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["question", "reference_answer", "evidence"],
            "additionalProperties": False,
        },
    },
}
QUESTION_TYPES = ("single_hop", "multi_hop", "exact_match")


def load_corpus(path: str | Path | None = None) -> dict[str, str]:
    import pandas as pd

    path = Path(path or config.CSV_PATH)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if {"IE_DETAIL_TEXT", "IE_XML_ID", "IE_NAME"} <= set(frame.columns):
            frame = clean_and_filter(frame)
            if frame["xml_id"].duplicated().any():
                raise ValueError("Source corpus contains duplicate doc ids")
            return dict(zip(frame["xml_id"], frame["clean_text"]))
    corpus, _ = load_upload(path, None)
    return corpus


def source_units(corpus: dict[str, str], max_chars: int) -> list[dict]:
    if max_chars < 100:
        raise ValueError("Source window must be at least 100 characters")
    units = []
    for doc_id, text in corpus.items():
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start + max_chars // 2, end),
                    text.rfind(". ", start + max_chars // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            if end - start >= 80:
                units.append(
                    {
                        "doc_id": str(doc_id),
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                    }
                )
            start = end
    return units


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    offset = text.find(quote)
    if offset >= 0:
        return offset, offset + len(quote)
    # Only whitespace may differ. Preserve a mapping back to ORIGINAL offsets.
    chars, starts, ends = [], [], []
    for match in re.finditer(r"\S+|\s+", text):
        if match.group().isspace():
            chars.append(" ")
            starts.append(match.start())
            ends.append(match.end())
        else:
            for n, char in enumerate(match.group()):
                chars.append(char)
                starts.append(match.start() + n)
                ends.append(match.start() + n + 1)
    normalized = re.sub(r"\s+", " ", quote).strip()
    offset = "".join(chars).find(normalized)
    if offset < 0 or not normalized:
        return None
    return starts[offset], ends[offset + len(normalized) - 1]


def validate_qa(
    parsed: dict, units: list[dict], kind: str, max_gold_chars: int
) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("Expected a QA object")  # noqa: TRY004 — validation failures consistently use ValueError
    question, answer = parsed.get("question"), parsed.get("reference_answer")
    if (
        not isinstance(question, str)
        or not question.strip()
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        raise ValueError("Question and reference answer must be non-blank")
    evidence = parsed.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Evidence quotes are required")
    spans, used, canonical_quotes = [], set(), []
    for piece in evidence:
        if not isinstance(piece, dict):
            raise ValueError("Invalid evidence object")  # noqa: TRY004 — validation failures consistently use ValueError
        quote = piece.get("quote")
        if not isinstance(quote, str) or not 8 <= len(quote) <= max_gold_chars:
            raise ValueError("Evidence quote has invalid length")
        located = False
        for n, unit in enumerate(units):
            if piece.get("source_id") is not None:
                if piece["source_id"] != unit.get("source_id"):
                    continue
            elif str(piece.get("doc_id")) != unit["doc_id"]:
                continue
            position = locate_quote(unit["text"], quote)
            if position is None:
                continue
            offset, end = position
            actual_quote = unit["text"][offset:end]
            if len(actual_quote) > max_gold_chars:
                continue
            span = {
                "doc_id": unit["doc_id"],
                "start": unit["start"] + offset,
                "end": unit["start"] + end,
            }
            if span not in spans:
                spans.append(span)
            used.add(n)
            canonical_quotes.append(
                {
                    "doc_id": unit["doc_id"],
                    "quote": actual_quote,
                    "source_id": unit.get("source_id"),
                }
            )
            located = True
            break
        if not located:
            raise ValueError("Quote is not an exact substring of the supplied source")
    if kind == "multi_hop" and len(used) < 2:
        raise ValueError("Multi-hop requires evidence from both supplied passages")
    return {
        "question": question.strip(),
        "reference_answer": answer.strip(),
        "question_type": kind,
        "gold_spans": spans,
        "span_granularity": "span",
        "evidence_quotes": canonical_quotes,
    }


def generate_synthetic(
    size: int = config.SYNTHESIS_DEFAULT_SIZE,
    name: str | None = None,
    corpus_path: str | Path | None = None,
    seed: int = 42,
    question_types: list[str] | None = None,
    on_event=None,
    stop=None,
    client=None,
    corpus: dict[str, str] | None = None,
) -> dict:
    if type(size) is not int or not 1 <= size <= 10000:
        raise ValueError("size must be in 1..10000")
    kinds = question_types or list(QUESTION_TYPES)
    if not kinds or set(kinds) - set(QUESTION_TYPES):
        raise ValueError("Unsupported question type")
    dataset_id = (
        name or f"synthetic-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:5]}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,36}", dataset_id):
        raise ValueError(
            "Dataset name must contain 1..36 ASCII letters, digits, underscore or hyphen"
        )
    if Path(dataset_id).name != dataset_id or dataset_id in (".", ".."):
        raise ValueError("Dataset name must be a simple directory name")
    output = config.RUNS_DIR / "datasets" / dataset_id
    output.mkdir(parents=True, exist_ok=False)
    emit = on_event or (lambda message, progress=None: None)
    rows, known, usages, errors = [], set(), [], []
    own_client = client is None
    provider = client or openai_client(timeout=config.RAGAS_TIMEOUT)
    attempts, network_errors = 0, 0
    try:
        emit("Загрузка корпуса", 0.0)
        corpus = load_corpus(corpus_path) if corpus is None else corpus
        if not corpus:
            raise ValueError("Corpus is empty")
        units = source_units(corpus, config.SYNTHESIS_MAX_SOURCE_CHARS)
        if not units:
            raise ValueError("No sufficiently long passages in corpus")
        by_doc = defaultdict(list)
        for unit in units:
            by_doc[unit["doc_id"]].append(unit)
        multi_docs = [doc for doc, parts in by_doc.items() if len(parts) >= 2]
        if "multi_hop" in kinds and not multi_docs:
            raise ValueError(
                "Multi-hop needs a document with at least two source passages"
            )
        corpus_file = output / "corpus.jsonl"
        corpus_file.write_text(
            "".join(
                json.dumps({"doc_id": d, "text": t}, ensure_ascii=False) + "\n"
                for d, t in corpus.items()
            ),
            encoding="utf-8",
        )
        rng = random.Random(seed)
        limit = size * config.SYNTHESIS_MAX_ATTEMPTS_FACTOR
        while len(rows) < size and attempts < limit:
            if stop is not None and stop.is_set():
                raise RuntimeError("Generation stopped")
            kind = kinds[len(rows) % len(kinds)]
            if kind == "multi_hop":
                parts = by_doc[rng.choice(multi_docs)]
                n = rng.randrange(len(parts) - 1)
                selected = [parts[n], parts[n + 1]]
            else:
                selected = [rng.choice(units)]
            selected = [
                {**unit, "source_id": f"source-{n + 1}"}
                for n, unit in enumerate(selected)
            ]
            attempts += 1
            prompt = json.dumps(
                {
                    "question_type": kind,
                    "sources": selected,
                    "max_quote_characters": config.SYNTHESIS_GOLD_MAX_CHARS,
                },
                ensure_ascii=False,
            )
            try:
                response = provider.chat.completions.create(
                    model=config.LLM_MODEL,
                    temperature=0.2,
                    max_tokens=config.LLM_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": config.SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format=QA_SCHEMA,
                    **(
                        {"reasoning_effort": config.LLM_REASONING_EFFORT}
                        if config.LLM_REASONING_EFFORT is not None
                        else {}
                    ),
                )
                usages.append(normalize_usage(response.usage))
                network_errors = 0
            except Exception as exc:
                network_errors += 1
                errors.append(f"{type(exc).__name__}: {str(exc)[:250]}")
                emit(
                    f"Ошибка LLM ({network_errors}/3): {type(exc).__name__}",
                    len(rows) / size,
                )
                if network_errors >= 3:
                    raise RuntimeError(
                        "LLM unavailable after three consecutive failures"
                    ) from exc
                continue
            try:
                parsed = json.loads(response.choices[0].message.content or "")
                row = validate_qa(
                    parsed, selected, kind, config.SYNTHESIS_GOLD_MAX_CHARS
                )
                canonical = re.sub(r"\s+", " ", row["question"]).casefold()
                if canonical in known:
                    raise ValueError("Duplicate question")
                known.add(canonical)
                row["qid"] = (
                    dataset_id
                    + "::"
                    + hashlib.sha256(canonical.encode()).hexdigest()[:16]
                )
                row["dataset_source"] = dataset_id
                rows.append(row)
                partial = output / "qa.partial.jsonl"
                partial.write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8",
                )
                emit(f"Готово {len(rows)}/{size} · {kind}", len(rows) / size)
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(str(exc))
                emit(f"Повтор: {exc}", len(rows) / size)
        if len(rows) != size:
            raise RuntimeError(
                f"Only {len(rows)}/{size} valid QA after {attempts} attempts"
            )
        qa_file = output / "qa.jsonl"
        (output / "qa.partial.jsonl").replace(qa_file)
        measured = [u for u in usages if u is not None]
        record = {
            "status": "done",
            "dataset_id": dataset_id,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dataset_config": {
                "type": "upload",
                "corpus_path": str(corpus_file.resolve()),
                "qa_path": str(qa_file.resolve()),
            },
            "questions": len(rows),
            "documents": len(corpus),
            "seed": seed,
            "model": config.LLM_MODEL,
            "attempts": attempts,
            "corpus_sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
            "qa_sha256": hashlib.sha256(qa_file.read_bytes()).hexdigest(),
            "max_source_chars": config.SYNTHESIS_MAX_SOURCE_CHARS,
            "max_gold_chars": config.SYNTHESIS_GOLD_MAX_CHARS,
            "system_prompt": config.SYNTHESIS_SYSTEM_PROMPT,
            "generation_total_tokens": sum(u["total_tokens"] for u in measured)
            if measured
            else None,
            "usage_attempts": len(measured),
            "usage_missing": attempts - len(measured),
            "rejections": errors,
        }
        (output / "manifest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        emit(f"Датасет {dataset_id} готов: {len(rows)} вопросов", 1.0)
        return record
    except BaseException as exc:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "dataset_id": dataset_id,
                    "questions": len(rows),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    finally:
        if own_client:
            provider.close()
