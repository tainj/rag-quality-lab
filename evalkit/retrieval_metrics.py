"""Pure retrieval arithmetic; no LLM clients or model downloads."""

import math


def _dcg_at_k(retrieved_ids: list[str], relevance: dict[str, float], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        rel = relevance.get(cid, 0.0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)
    return dcg


def _idcg_at_k(relevance: dict[str, float], k: int) -> float:
    scores = sorted(relevance.values(), reverse=True)[:k]
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))


def eval_single_query(
    retrieved_ids: list[str],
    relevant: dict[str, float],
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    """Compute recall@k, precision@k, nDCG@k and MRR for a single query.

    ``retrieved_ids`` is the ordered list of retrieved chunk/doc ids.
    ``relevant`` maps ground-truth relevant ids to a graded relevance score
    (0..1), e.g. the char-span coverage produced by span_matching.py.
    """
    relevant_set = set(relevant)
    metrics: dict[str, float] = {}

    for k in ks:
        if k <= 0:
            continue
        hits = retrieved_ids[:k]
        hits_relevant = sum(1 for cid in hits if cid in relevant_set)
        metrics[f"recall@{k}"] = (
            hits_relevant / len(relevant_set) if relevant_set else 0.0
        )
        metrics[f"precision@{k}"] = hits_relevant / k
        idcg = _idcg_at_k(relevant, k)
        metrics[f"ndcg@{k}"] = (
            _dcg_at_k(retrieved_ids, relevant, k) / idcg if idcg > 0 else 0.0
        )

    metrics["mrr"] = 0.0
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_set:
            metrics["mrr"] = 1.0 / (i + 1)
            break

    return metrics
