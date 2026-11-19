import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from evalkit import gui, pipeline as pl
from evalkit.llm import DeepEvalJudge
from evalkit.metrics import AnswerCorrectnessFacts, RagasStyleAnswerCorrectnessMetric
from evalkit.rag_config import preset_configs


class JudgeRecoveryTests(unittest.TestCase):
    def test_score_only_judge_disables_auxiliary_explanations(self):
        judge = object.__new__(DeepEvalJudge)
        judge.model = Mock()
        metric = Mock(score=0.75, error=None)
        with (
            patch("evalkit.metrics.make_metric", return_value=metric),
            patch("evalkit.llm.JUDGE_INCLUDE_REASON", False),
        ):
            self.assertEqual(judge._measure("faithfulness", Mock()), (0.75, None))
        self.assertFalse(metric.include_reason)
        metric.measure.assert_called_once()

    def test_correctness_without_explanation_preserves_fact_counts(self):
        metric = object.__new__(RagasStyleAnswerCorrectnessMetric)
        metric.include_reason = False
        metric._build_prompt = Mock(return_value="compare facts")

        def generate(**kwargs):
            self.assertIs(kwargs["schema_cls"], AnswerCorrectnessFacts)
            self.assertNotIn("reason", kwargs["schema_cls"].model_fields)
            return kwargs["extract_schema"](
                AnswerCorrectnessFacts(tp=["a", "b"], fp=["c"], fn=["d"])
            )

        with patch(
            "evalkit.metrics.generate_with_schema_and_extract", side_effect=generate
        ):
            result = metric._generate_analysis(Mock())
        self.assertAlmostEqual(
            metric._f1(len(result.tp), len(result.fp), len(result.fn)), 2 / 3
        )
        self.assertEqual(result.reason, "")

    def test_judge_has_a_separate_output_budget_and_timeout(self):
        with (
            patch("evalkit.llm.LLM_MAX_TOKENS", 100),
            patch("evalkit.llm.JUDGE_MAX_TOKENS", 8000),
            patch("evalkit.llm.JUDGE_TIMEOUT", 200),
            patch("deepeval.models.llms.GPTModel") as model,
            patch("evalkit.metrics.patch_model", side_effect=lambda m: m),
            patch("evalkit.llm_http.http_clients", return_value={}),
        ):
            DeepEvalJudge(["faithfulness"])
            self.assertEqual(
                model.call_args.kwargs["generation_kwargs"]["max_tokens"], 8000
            )
            self.assertEqual(model.call_args.kwargs["timeout"], 200)

    def test_judge_measures_only_requested_missing_metric(self):
        judge = object.__new__(DeepEvalJudge)
        judge.metric_names = ["answer_correctness", "faithfulness"]
        judge._measure = Mock(return_value=(0.8, None))
        scores = judge.score(
            "question", "answer", "reference", ["context"], metrics=["faithfulness"]
        )
        self.assertEqual(judge._measure.call_count, 1)
        self.assertEqual(judge._measure.call_args.args[0], "faithfulness")
        self.assertEqual(scores["faithfulness"], 0.8)
        self.assertNotIn("answer_correctness", scores)

    def test_retry_preserves_answer_usage_and_successful_metric(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        q = {
            "qid": "q",
            "question": "question",
            "reference_answer": "reference",
            "split": "dev",
            "gold_spans": [{"doc_id": "doc", "start": 0, "end": 3}],
        }
        judge = SimpleNamespace(
            supports_metric_retry=True,
            metric_names=["answer_correctness", "faithfulness"],
            score=Mock(return_value={"faithfulness": 0.8}),
        )
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run(
                {"ds": {"type": "upload"}},
                [cfg],
                pl.RunOptions(retry_errors=True),
                "metric-retry",
            )
            run._data["ds"] = ({"doc": "answer"}, [q])
            trace = {
                "qid": "q",
                "dataset": "ds",
                "config_id": cfg["id"],
                "split": "dev",
                "answer": "saved answer",
                "retrieved_chunks": [{"content": "source"}],
                "usage": {"total_tokens": 17},
                "usage_complete": True,
                "answer_seconds": 2.0,
                "unanswerable": False,
                "phase": "complete",
                "error": None,
                "scores": {
                    "answer_correctness": 0.6,
                    "faithfulness": None,
                    "errors": {"faithfulness": "length limit"},
                },
            }
            run.file("generation", f"{cfg['id']}.jsonl").write_text(
                json.dumps(trace) + "\n"
            )
            run.file("generation", f"{cfg['id']}.checkpoint.json").write_text(
                json.dumps({json.dumps(["ds", "q"]): trace})
            )
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
            self.assertEqual(judge.score.call_args.kwargs["metrics"], ["faithfulness"])
            saved = json.loads(run.file("generation", f"{cfg['id']}.jsonl").read_text())
            self.assertEqual(saved["answer"], trace["answer"])
            self.assertEqual(saved["usage"], trace["usage"])
            self.assertEqual(saved["scores"]["answer_correctness"], 0.6)
            self.assertEqual(saved["scores"]["faithfulness"], 0.8)
            self.assertNotIn("errors", saved["scores"])

    def test_gui_resume_enables_failed_measurement_retry(self):
        run = Mock(state={"options": {"retry_errors": False}})
        with (
            patch.object(gui, "_busy", return_value=None),
            patch.object(gui, "_require_current_backend"),
            patch.object(gui.pl, "load_run", return_value=run),
            patch.object(gui, "_start"),
        ):
            self.assertEqual(gui.resume_run("run"), "run")
            self.assertTrue(run.state["options"]["retry_errors"])
            run.save.assert_called_once()

    def test_resuming_active_run_is_idempotent(self):
        busy = SimpleNamespace(kind="pipeline", run_id="active")
        with (
            patch.object(gui, "_busy", return_value=busy),
            patch.object(gui, "_start") as start,
            patch.object(gui.gr, "Info"),
        ):
            self.assertEqual(gui.resume_run("active"), "active")
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
