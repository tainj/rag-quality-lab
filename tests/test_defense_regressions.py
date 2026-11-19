import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from evalkit import pipeline as pl
from evalkit.llm_judge import is_declined_verdict
from evalkit.rag_config import preset_configs


class DefenseRegressionTests(unittest.TestCase):
    def test_negative_rejection_reads_boolean_field(self):
        self.assertFalse(is_declined_verdict({"declined": False, "reason": "answered"}))
        self.assertFalse(is_declined_verdict({"reason": "unknown"}))
        self.assertFalse(is_declined_verdict({"declined": "false"}))
        self.assertTrue(is_declined_verdict({"declined": True}))
        self.assertTrue(is_declined_verdict(True))

    def test_retrieval_import_does_not_initialize_llm_framework(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import evalkit.eval_runner; assert 'deepeval' not in sys.modules",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retry_only_failed_retrieval_questions(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        qa = [
            {
                "qid": qid,
                "question": qid,
                "split": "dev",
                "gold_spans": [{"doc_id": "doc", "start": 0, "end": 3}],
            }
            for qid in ("good", "bad")
        ]
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run(
                {"ds": {"type": "upload"}},
                [cfg],
                pl.RunOptions(retry_errors=True),
                "retry-search",
            )
            run._data["ds"] = ({"doc": "text"}, qa)
            path = run.file("retrieval", f"{cfg['id']}__ds.csv")
            old = pd.DataFrame(
                [
                    {
                        "qid": "good",
                        "error": None,
                        "split": "dev",
                        "config_id": cfg["id"],
                        "recall@5": 0.75,
                    },
                    {
                        "qid": "bad",
                        "error": "timeout",
                        "split": "dev",
                        "config_id": cfg["id"],
                        "recall@5": None,
                    },
                ]
            )
            old.to_csv(path, index=False)
            repaired = pd.DataFrame([{"qid": "bad", "error": None, "recall@5": 1.0}])
            with patch(
                "evalkit.eval_runner.run_retrieval_grid", return_value=repaired
            ) as search:
                pl._retrieval_job(run, Mock(), lambda *a, **k: None, 0, 1, cfg, "ds")
            self.assertEqual(
                [q["qid"] for q in search.call_args.kwargs["qa_dataset"]], ["bad"]
            )
            rows = pd.read_csv(path).set_index("qid")
            self.assertEqual(set(rows.index), {"good", "bad"})
            self.assertEqual(rows.loc["good", "recall@5"], 0.75)
            self.assertEqual(rows.loc["bad", "recall@5"], 1.0)

    def test_judge_retry_uses_saved_answer_without_agent_request(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        qa = [
            {
                "qid": "q",
                "question": "question",
                "split": "dev",
                "reference_answer": "answer",
                "gold_spans": [{"doc_id": "doc", "start": 0, "end": 3}],
            }
        ]
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run(
                {"ds": {"type": "upload"}},
                [cfg],
                pl.RunOptions(retry_errors=True),
                "retry-judge",
            )
            run._data["ds"] = ({"doc": "answer"}, qa)
            trace = {
                "qid": "q",
                "dataset": "ds",
                "config_id": cfg["id"],
                "split": "dev",
                "answer": "saved answer",
                "retrieved_chunks": [{"content": "answer"}],
                "usage": {"total_tokens": 17},
                "usage_complete": True,
                "answer_seconds": 2.0,
                "unanswerable": False,
                "phase": "complete",
                "error": "judge timeout",
                "scores": {},
            }
            run.file("generation", f"{cfg['id']}.jsonl").write_text(
                json.dumps(trace) + "\n"
            )
            key = json.dumps(["ds", "q"])
            run.file("generation", f"{cfg['id']}.checkpoint.json").write_text(
                json.dumps({key: trace})
            )
            judge = Mock()
            judge.score.return_value = {"answer_correctness": 1.0, "faithfulness": 1.0}
            client = Mock()
            with (
                patch.object(
                    pl,
                    "summarize",
                    return_value=pd.DataFrame(
                        {"dev_recall@5": [1.0]}, index=[cfg["id"]]
                    ),
                ),
                patch("evalkit.llm.make_judge", return_value=judge),
            ):
                pl._step_generation(run, client, lambda *a, **k: None)
            client.agent_query.assert_not_called()
            client.ensure_eval_agent.assert_not_called()
            judge.score.assert_called_once()
            saved = json.loads(run.file("generation", f"{cfg['id']}.jsonl").read_text())
            self.assertEqual(saved["answer"], "saved answer")
            self.assertEqual(saved["usage"]["total_tokens"], 17)
            self.assertIsNone(saved["error"])


if __name__ == "__main__":
    unittest.main()
