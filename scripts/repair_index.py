"""Reconcile missing uploads in an existing evaluation collection."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalkit import pipeline as pl
from evalkit.client import make_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("dataset")
    parser.add_argument("config_id")
    args = parser.parse_args()
    run = pl.load_run(args.run_id)
    # Validate the corpus against the immutable run snapshot before touching data.
    pl._step_prepare(run, lambda *a: None)
    corpus, _ = run.dataset(args.dataset)
    collection = run.collection_name(args.config_id, args.dataset)
    path = run.file("index", f"{args.config_id}__{args.dataset}.json")
    checkpoint = json.loads(path.read_text())
    client = make_client()

    def probe(source):
        response = client.session.post(
            client.base_url + "/vdb/search",
            headers=client.headers,
            json={
                "query": corpus[source][:200],
                "collection_name": collection,
                "top_k": 1,
                "threshold": 0,
                "qdrant_filters": {
                    "must": [{"key": "eval_doc_id", "match": {"value": source}}]
                },
            },
            timeout=15,
        )
        response.raise_for_status()
        hits = response.json()
        if not isinstance(hits, list) or any(
            str(h.get("metadata", {}).get("eval_doc_id")) != source for h in hits
        ):
            raise RuntimeError("Document filter did not return the expected source")
        return hits

    control = next(
        row for row in checkpoint["documents"] if not row["result"].startswith("ERROR")
    )
    if not probe(control["source_doc_id"]):
        raise RuntimeError(
            "Cannot verify document filter against a known accepted upload"
        )
    for row in checkpoint["documents"]:
        if not row["result"].startswith("ERROR"):
            continue
        source = row["source_doc_id"]
        if probe(source):
            raise RuntimeError(
                f"{source}: document exists; verify indexing before another upload"
            )
        # The exact source filter was verified against a known existing document.
        # No vectors for this source were found; only this missing source is sent.
        row["recovery"] = {
            "previous_error": row["result"],
            "confirmed_missing_at": time.time(),
        }
        try:
            result, task = client.upload_document(source, corpus[source], collection)
        except Exception as exc:
            row.update(result=f"ERROR: {exc}", task_id=None)
            pl._save_index_checkpoint(path, checkpoint)
            raise
        row.update(result=result, task_id=task)
        pl._save_index_checkpoint(path, checkpoint)
        print(f"Recovered {source}: task={task}", flush=True)
    print(
        "Upload checkpoint repaired; resume the run to verify all indexing tasks.",
        flush=True,
    )


if __name__ == "__main__":
    main()
