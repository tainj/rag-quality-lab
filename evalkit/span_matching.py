"""
Chunking-agnostic ground truth mapping.

Golden QA pairs are grounded in RAW document char spans (see synth_gen.py),
not in any particular chunker's chunk_id. This module maps golden spans onto
retrieved chunks via overlap, so the same synthetic dataset can be reused
across arbitrary collection configurations without regeneration.
"""

from __future__ import annotations

from typing import Any


def _merged(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def evidence_metrics(gold_spans: list[dict], results: list[dict], ks=(1, 3, 5)) -> dict:
    """Character coverage of ALL gold evidence, deduplicated by source offsets.

    Evidence precision uses the union of retrieved source intervals as its
    denominator, so redundant overlap and padding cannot manufacture useful text.
    nDCG is undefined here: unseen relevant chunk ids are not known.
    """
    gold = {}
    for span in gold_spans:
        if (
            not isinstance(span.get("start"), int)
            or not isinstance(span.get("end"), int)
            or span["end"] <= span["start"]
        ):
            raise ValueError("Gold spans require valid start/end character offsets")
        gold.setdefault(str(span["doc_id"]), []).append((span["start"], span["end"]))
    gold = {doc: _merged(spans) for doc, spans in gold.items()}
    gold_size = sum(end - start for spans in gold.values() for start, end in spans)
    output = {}
    for k in ks:
        retrieved, relevant, invalid_chars = {}, 0, 0
        for result in results[:k]:
            meta = result.get("metadata") or {}
            start, end = (
                meta.get("char_start", meta.get("start")),
                meta.get("char_end", meta.get("end")),
            )
            doc = str(result.get("doc_id"))
            if type(start) is not int or type(end) is not int or end <= start:
                invalid_chars += len(result.get("text") or result.get("content") or "")
                continue
            retrieved.setdefault(doc, []).append((start, end))
            if any(_overlap_len(start, end, a, b) for a, b in gold.get(doc, [])):
                relevant += 1
        retrieved = {doc: _merged(spans) for doc, spans in retrieved.items()}
        total = invalid_chars + sum(
            b - a for spans in retrieved.values() for a, b in spans
        )
        covered = sum(
            _overlap_len(a, b, c, d)
            for doc, spans in retrieved.items()
            for a, b in spans
            for c, d in gold.get(doc, [])
        )
        recall = covered / gold_size if gold_size else 0.0
        precision = covered / total if total else 0.0
        output.update(
            {
                f"recall@{k}": recall,
                f"precision@{k}": relevant / k,
                f"evidence_precision@{k}": precision,
                f"f1@{k}": 2 * recall * precision / (recall + precision)
                if recall + precision
                else 0.0,
                f"ndcg@{k}": None,
            }
        )
    output["mrr"] = 0.0
    for rank, result in enumerate(results, 1):
        meta = result.get("metadata") or {}
        start, end = meta.get("char_start"), meta.get("char_end")
        if (
            type(start) is int
            and type(end) is int
            and any(
                _overlap_len(start, end, a, b) > 0
                for a, b in gold.get(str(result.get("doc_id")), [])
            )
        ):
            output["mrr"] = 1 / rank
            break
    return output


def _overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = _overlap_len(a_start, a_end, b_start, b_end)
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def coverage_of_gold(
    gold_start: int, gold_end: int, chunk_start: int, chunk_end: int
) -> float:
    """What fraction of the GOLD span does this chunk cover?"""
    gold_len = gold_end - gold_start
    if gold_len <= 0:
        return 0.0
    return _overlap_len(gold_start, gold_end, chunk_start, chunk_end) / gold_len


def label_relevant_chunks_from_results(
    gold_spans: list[dict[str, Any]],
    results: list[dict[str, Any]],
    threshold: float = 0.3,
    metric: str = "coverage",
) -> dict[str, float]:
    """Label retrieved chunks as relevant based on char-span overlap.

    Each result dict must have keys: chunk_id, doc_id, and metadata
    containing char_start / char_end (as set by agent-core).

    Returns {chunk_id: relevance_score} for chunks overlapping any gold span.
    """
    relevant: dict[str, float] = {}
    for gs in gold_spans:
        for r in results:
            if r.get("doc_id") != gs["doc_id"]:
                continue
            meta = r.get("metadata", {})
            ch_start = meta.get("char_start", meta.get("start", 0))
            ch_end = meta.get("char_end", meta.get("end", 0))
            score = (
                coverage_of_gold(gs["start"], gs["end"], ch_start, ch_end)
                if metric == "coverage"
                else iou(gs["start"], gs["end"], ch_start, ch_end)
            )
            if score >= threshold:
                cid = r["chunk_id"]
                relevant[cid] = max(relevant.get(cid, 0.0), score)
    return relevant
