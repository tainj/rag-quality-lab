import fcntl
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx
import pandas as pd

from evalkit import pipeline as pl
from evalkit.eval_runner import run_retrieval_grid
from evalkit.rag_config import derive_config, preset_configs


class FailureContinuationTests(unittest.TestCase):
    def test_cached_complete_index_survives_health_check_timeout(self):
        client = Mock()
        client.collection_exists.side_effect = httpx.ConnectTimeout("offline")
        self.assertTrue(pl._collection_ready(client, "known-complete"))
        client.collection_exists.side_effect = None
        client.collection_exists.return_value = False
        self.assertFalse(pl._collection_ready(client, "deleted"))

    def test_running_experiment_cannot_be_started_twice(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run({"ds": {"type": "upload"}}, [cfg], run_id="locked")
            with run.file("pipeline.lock").open("a") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "уже запущен"):
                    pl.run_pipeline(run, client=Mock(), on_event=lambda e: None)

    def test_failed_query_is_recorded_and_next_question_runs(self):
        client = Mock()
        client.search.side_effect = [
            httpx.ConnectTimeout("unreachable"),
            [{"doc_id": "doc", "metadata": {"char_start": 0, "char_end": 3}}],
        ]
        qa = [
            {
                "qid": str(i),
                "question": "query",
                "dataset_source": "ds",
                "gold_spans": [{"doc_id": "doc", "start": 0, "end": 3}],
            }
            for i in range(2)
        ]
        frame = run_retrieval_grid(qa, client, continue_on_error=True)
        self.assertEqual(client.search.call_count, 2)
        self.assertIn("ConnectTimeout", frame.iloc[0]["error"])
        self.assertTrue(pd.isna(frame.iloc[0]["recall@5"]))
        self.assertEqual(frame.iloc[1]["recall@5"], 1)

    def test_failed_index_skips_shared_prompt_and_runs_next_setup(self):
        first, second = preset_configs(["stand_dense_512", "stand_dense_1024"])
        prompt = derive_config(first, {"generation.system_prompt": "another prompt"})
        client = Mock()
        client.collection_exists.return_value = False

        def create(**kwargs):
            if kwargs["chunking"]["chunk_size"] == 512:
                raise httpx.ConnectTimeout("offline")

        client.create_collection.side_effect = create
        client.upload_many.return_value = [("doc", None)]
        with (
            TemporaryDirectory() as tmp,
            patch.object(pl, "RUNS_DIR", Path(tmp)),
            patch.object(pl, "REGISTRY", Path(tmp) / "registry.json"),
        ):
            run = pl.new_run(
                {"ds": {"type": "upload"}}, [first, prompt, second], run_id="skip"
            )
            run._data["ds"] = ({"doc": "text"}, [])
            run.state["dataset_stats"] = {"ds": {"corpus_hash": "hash"}}
            pl._step_index(run, client, lambda *a, **kw: None)
            self.assertEqual([c["id"] for c in pl.active_configs(run)], [second["id"]])
            self.assertEqual(run.status(f"index:{second['id']}:ds"), "done")
            self.assertEqual(run.status("index"), "partial")
            self.assertEqual(client.create_collection.call_count, 2)

    def test_partial_generation_cannot_beat_complete_generation(self):
        first, second = preset_configs(["stand_dense_512", "stand_dense_1024"])
        table = pd.DataFrame(
            {"dev_recall@5": [1.0, 0.5]}, index=[first["id"], second["id"]]
        )
        generation = pd.DataFrame(
            [
                {"config_id": first["id"], "errors": 1, "gen_score": 1.0},
                {"config_id": second["id"], "errors": 0, "gen_score": 0.5},
            ]
        )
        with (
            TemporaryDirectory() as tmp,
            patch.object(pl, "RUNS_DIR", Path(tmp)),
            patch.object(pl, "summarize", return_value=table),
            patch.object(pl, "generation_summary", return_value=generation),
        ):
            run = pl.new_run(
                {"ds": {"type": "upload"}}, [first, second], run_id="selection"
            )
            run.set_status("generation", "done")
            pl._step_select(run, lambda *a, **kw: None)
            self.assertEqual(run.state["best"]["config_id"], second["id"])

    def test_all_failed_jobs_finish_without_inventing_a_winner(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        client = Mock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = httpx.ConnectTimeout("offline")
        with (
            TemporaryDirectory() as tmp,
            patch.object(pl, "RUNS_DIR", Path(tmp)),
            patch.object(pl, "REGISTRY", Path(tmp) / "registry.json"),
            patch(
                "evalkit.dataset_loaders.load_dataset",
                return_value=({"doc": "text"}, []),
            ),
        ):
            run = pl.new_run({"ds": {"type": "upload"}}, [cfg], run_id="all-failed")
            pl.run_pipeline(run, client=client, on_event=lambda e: None)
            saved = json.loads((run.path / "run.json").read_text())
            self.assertEqual(saved["status"]["pipeline"], "done_with_errors")
            self.assertEqual(saved["status"]["select"], "skipped")
            self.assertNotIn("best", saved)
            self.assertFalse((run.path / "best_config.json").exists())


if __name__ == "__main__":
    unittest.main()
