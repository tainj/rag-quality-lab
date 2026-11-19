"""
Dataset loaders for the RAG eval pipeline.

Each loader returns (corpus, qa_dataset) where:
  corpus     = {doc_id: text}
  qa_dataset = list of dicts with keys:
    qid, question, reference_answer, question_type, gold_spans,
    dataset_source, span_granularity
"""

from __future__ import annotations

import json
from pathlib import Path


def load_synthetic(path: str | Path) -> tuple[dict[str, str], list[dict]]:
    """Load a synthetic QA dataset produced by notebook 01.

    Returns a minimal corpus extracted from gold_spans and the QA rows.
    The corpus is built from the document texts referenced in gold_spans;
    for synthetic data the caller should also provide the full corpus
    separately for indexing.
    """
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            row["dataset_source"] = "synthetic_bpmsoft"
            row.setdefault("span_granularity", "span")
            rows.append(row)
    return {}, rows


def load_ir_dataset(
    hf_name: str,
    min_relevance: int = 1,
    dataset_source: str | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Load a dataset via ir_datasets (corpus + queries + qrels).

    Args:
        hf_name: Full ir_datasets path, e.g. "beir/nfcorpus/test" or "miracl/ru/test-a".
        dataset_source: Label used for `dataset_source` field in QA rows.
                       Defaults to `hf_name`.

    Returns (corpus, qa_dataset) with document-level gold spans.
    Requires: pip install ir_datasets
    """
    import ir_datasets

    ds = ir_datasets.load(hf_name)

    corpus = {doc.doc_id: doc.text for doc in ds.docs_iter()}
    queries = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    source_label = dataset_source if dataset_source is not None else hf_name
    rows: list[dict] = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        gold_spans = [
            {
                "doc_id": doc_id,
                "start": 0,
                "end": len(corpus.get(doc_id, "")),
            }
            for doc_id, score in rels.items()
            if score >= min_relevance and doc_id in corpus
        ]
        if not gold_spans:
            continue
        rows.append(
            {
                "qid": f"{source_label}::{qid}",
                "question": queries[qid],
                "reference_answer": None,
                "question_type": "retrieval_only",
                "gold_spans": gold_spans,
                "dataset_source": source_label,
                "span_granularity": "document",
            }
        )
    return corpus, rows


def load_rubq_retrieval(
    hf_name: str = "ai-forever/rubq-retrieval",
    min_relevance: int = 1,
    dataset_source: str | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Load RuBQ Retrieval via HF datasets (corpus + queries + qrels).

    The dataset has three subsets: corpus, queries, and default (qrels).
    Unlike ir_datasets-based loaders, this one uses HuggingFace datasets directly.

    Args:
        hf_name: HF dataset path, e.g. "ai-forever/rubq-retrieval".
        dataset_source: Label used for `dataset_source` field in QA rows.
                       Defaults to `hf_name`.

    Returns (corpus, qa_dataset) with document-level gold spans.
    Requires: pip install datasets
    """
    from datasets import load_dataset

    corpus_ds = load_dataset(hf_name, "corpus")
    corpus_key = next(iter(corpus_ds))
    corpus: dict[str, str] = {}
    for doc in corpus_ds[corpus_key]:
        title = doc.get("title", "").strip()
        text = doc.get("text", "")
        corpus[doc["_id"]] = f"{title}\n{text}".strip() if title else text

    queries_ds = load_dataset(hf_name, "queries")
    queries_key = next(iter(queries_ds))
    queries: dict[str, str] = {q["_id"]: q["text"] for q in queries_ds[queries_key]}

    qrels_ds = load_dataset(hf_name, split="test")
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        qrels.setdefault(qid, {})[doc_id] = row["score"]

    source_label = dataset_source if dataset_source is not None else hf_name
    rows: list[dict] = []
    for qid, rels in qrels.items():
        if qid not in queries:
            continue
        gold_spans = [
            {
                "doc_id": doc_id,
                "start": 0,
                "end": len(corpus.get(doc_id, "")),
            }
            for doc_id, score in rels.items()
            if score >= min_relevance and doc_id in corpus
        ]
        if not gold_spans:
            continue
        rows.append(
            {
                "qid": f"{source_label}::{qid}",
                "question": queries[qid],
                "reference_answer": None,
                "question_type": "retrieval_only",
                "gold_spans": gold_spans,
                "dataset_source": source_label,
                "span_granularity": "document",
            }
        )
    return corpus, rows


def load_hotpotqa(
    hf_name: str,
    max_examples: int | None = None,
    dataset_source: str | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Load HotpotQA with span-level evidence ground truth.

    Args:
        hf_name: HF datasets path, e.g. "hotpot_qa/distractor/validation".
        dataset_source: Label used for `dataset_source` field in QA rows.
                       Defaults to `hf_name`.

    Returns (corpus, qa_dataset) where gold_spans are real char offsets
    located via verbatim text search.
    Requires: pip install datasets
    """
    from datasets import load_dataset

    parts = hf_name.split("/")
    name = parts[0]
    subset = parts[1] if len(parts) > 1 else None
    split = parts[2] if len(parts) > 2 else None

    ds = load_dataset(name, subset, split=split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    source_label = dataset_source if dataset_source is not None else hf_name

    # Build corpus from context paragraphs
    corpus: dict[str, str] = {}
    for item in ds:
        for title, sents in zip(item["context"]["title"], item["context"]["sentences"]):
            full_text = "".join(sents)
            if title not in corpus or full_text not in corpus[title]:
                corpus[title] = full_text

    rows: list[dict] = []
    for item in ds:
        evidence_spans: list[dict] = []
        for title, sent_ids in zip(
            item["supporting_facts"]["title"],
            item["supporting_facts"]["sent_id"],
        ):
            doc_text = corpus.get(title)
            if doc_text is None:
                continue
            # Reconstruct the sentence text from context
            ctx_idx = item["context"]["title"].index(title)
            sent_text = item["context"]["sentences"][ctx_idx][sent_ids]
            start = doc_text.find(sent_text)
            if start == -1:
                continue
            evidence_spans.append(
                {"doc_id": title, "start": start, "end": start + len(sent_text)}
            )
        if not evidence_spans:
            continue
        rows.append(
            {
                "qid": f"{source_label}::{item['id']}",
                "question": item["question"],
                "reference_answer": item["answer"],
                "question_type": (
                    "multi_hop" if len(evidence_spans) > 1 else "single_hop"
                ),
                "gold_spans": evidence_spans,
                "dataset_source": source_label,
                "span_granularity": "span",
            }
        )
    return corpus, rows


def load_upload(
    corpus_path: str | Path, qa_path: str | Path | None
) -> tuple[dict[str, str], list[dict]]:
    """Load a user-uploaded dataset.

    Corpus: ``.jsonl`` with {"doc_id", "text"} per line, or ``.csv`` with
    columns doc_id,text. QA: ``.jsonl`` in the synthetic format
    (qid, question, reference_answer, question_type, gold_spans
    [{doc_id, start, end}]); optional — without it only chat/deploy work.
    """
    corpus_path = Path(corpus_path)
    corpus: dict[str, str] = {}
    if corpus_path.suffix == ".csv":
        import pandas as pd

        df = pd.read_csv(corpus_path)
        missing = {"doc_id", "text"} - set(df.columns)
        if missing:
            raise ValueError(f"{corpus_path.name}: нет колонок {sorted(missing)}")
        corpus = {str(d): str(t) for d, t in zip(df["doc_id"], df["text"])}
    else:
        with open(corpus_path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    if "doc_id" not in row or "text" not in row:
                        raise ValueError(
                            f"{corpus_path.name}:{n}: нужны поля doc_id и text"
                        )
                    corpus[str(row["doc_id"])] = str(row["text"])

    rows: list[dict] = []
    if qa_path:
        with open(qa_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row.setdefault("span_granularity", "span")
                    rows.append(row)
    return corpus, rows


def load_dataset(ds_key: str, ds_cfg: dict) -> tuple[dict[str, str], list[dict]]:
    """(corpus, qa) for one entry of EVAL_DATASETS; qa rows tagged with ds_key.

    Types: synthetic (BPMSoft CSV corpus + synthetic QA), beir, hotpotqa,
    rubq, upload (``corpus_path`` / ``qa_path``).
    """
    from config import CSV_PATH, DATA_DIR

    ds_type = ds_cfg["type"]
    if ds_type == "synthetic":
        import pandas as pd

        from evalkit.csv_loader import clean_and_filter

        df = clean_and_filter(pd.read_csv(str(CSV_PATH), sep=","))
        corpus = dict(zip(df["xml_id"], df["clean_text"]))
        _, qa = load_synthetic(DATA_DIR / ds_cfg["path"])
    elif ds_type == "beir":
        corpus, qa = load_ir_dataset(hf_name=ds_cfg["hf_name"], dataset_source=ds_key)
    elif ds_type == "hotpotqa":
        corpus, qa = load_hotpotqa(
            hf_name=ds_cfg["hf_name"],
            max_examples=ds_cfg.get("max_examples"),
            dataset_source=ds_key,
        )
    elif ds_type == "rubq":
        corpus, qa = load_rubq_retrieval(
            hf_name=ds_cfg["hf_name"], dataset_source=ds_key
        )
    elif ds_type == "upload":
        corpus, qa = load_upload(ds_cfg["corpus_path"], ds_cfg.get("qa_path"))
    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")
    for row in qa:
        row["dataset_source"] = ds_key
    return corpus, qa
