import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx

from evalkit import pipeline as pl
from evalkit.api_client import RagServiceClient
from evalkit.llm import LLMAnswerer
from evalkit.rag_config import (
    collection_id,
    derive_config,
    preset_configs,
    retrieval_id,
)
from evalkit.span_matching import evidence_metrics
from evalkit.usage import agent_usage, normalize_usage
from scripts import repair_index


def hit(doc, start, end):
    return {
        "doc_id": doc,
        "chunk_id": f"{doc}:{start}:{end}",
        "metadata": {"char_start": start, "char_end": end},
    }


class EvidenceTests(unittest.TestCase):
    def test_repair_probes_missing_source_and_keeps_existing_documents(self):
        for exists in (False, True):
            with self.subTest(exists=exists), TemporaryDirectory() as tmp:
                path = Path(tmp) / "index.json"
                path.write_text(
                    json.dumps(
                        {
                            "documents": [
                                {
                                    "source_doc_id": "known",
                                    "result": "accepted",
                                    "task_id": "old",
                                },
                                {
                                    "source_doc_id": "missing",
                                    "result": "ERROR: response timeout",
                                    "task_id": None,
                                },
                            ]
                        }
                    )
                )
                run = Mock()
                run.dataset.return_value = ({"known": "one", "missing": "two"}, [])
                run.collection_name.return_value = "test-collection"
                run.file.return_value = path
                client = Mock()
                client.base_url = "https://example.invalid"

                def hit(source):
                    return {"metadata": {"eval_doc_id": source}}

                client.session.post.side_effect = [
                    Mock(json=lambda: [hit("known")]),
                    Mock(json=lambda exists=exists: [hit("missing")] if exists else []),
                ]
                client.upload_document.return_value = ("recovered", "new-task")
                with (
                    patch.object(repair_index.pl, "load_run", return_value=run),
                    patch.object(repair_index.pl, "_step_prepare"),
                    patch.object(repair_index, "make_client", return_value=client),
                    patch("sys.argv", ["repair_index.py", "run", "ds", "cfg"]),
                ):
                    if exists:
                        with self.assertRaisesRegex(RuntimeError, "document exists"):
                            repair_index.main()
                        client.upload_document.assert_not_called()
                    else:
                        repair_index.main()
                        client.upload_document.assert_called_once_with(
                            "missing", "two", "test-collection"
                        )
                        saved = json.loads(path.read_text())["documents"]
                        self.assertEqual(saved[0]["result"], "accepted")
                        self.assertEqual(saved[1]["result"], "recovered")

    def test_partial_index_resume_preserves_documents_and_ambiguous_uploads(self):
        cfg = preset_configs(["stand_dense_512"])[0]
        corpus = {"accepted": "one", "handshake": "two", "ambiguous": "three"}
        client = Mock()
        client.collection_exists.return_value = True
        client.upload_many.return_value = [("repaired", "task-new")]
        with (
            TemporaryDirectory() as tmp,
            patch.object(pl, "RUNS_DIR", Path(tmp)),
            patch.object(pl, "REGISTRY", Path(tmp) / "registry.json"),
        ):
            run = pl.new_run(
                {"x": {"type": "upload"}},
                [cfg],
                pl.RunOptions(continue_on_error=False),
                run_id="partial",
            )
            run._data["x"] = (corpus, [])
            run.state["dataset_stats"] = {"x": {"corpus_hash": "hash"}}
            path = run.file("index", f"{cfg['id']}__x.json")
            path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "source_doc_id": "accepted",
                                "result": "accepted-id",
                                "task_id": "task-old",
                            },
                            {
                                "source_doc_id": "handshake",
                                "result": "ERROR: TLS handshake timed out",
                                "task_id": None,
                            },
                            {
                                "source_doc_id": "ambiguous",
                                "result": "ERROR: The read operation timed out",
                                "task_id": None,
                            },
                        ]
                    }
                )
            )
            with (
                patch.object(
                    pl, "_wait_tasks", side_effect=ConnectionError("poll disconnected")
                ),
                self.assertRaisesRegex(RuntimeError, "incomplete upload"),
            ):
                pl._step_index(run, client, lambda *a: None)
            client.delete_collection.assert_not_called()
            client.create_collection.assert_not_called()
            self.assertEqual(
                client.upload_many.call_args.args[0],
                [("handshake", "two", run.collection_name(cfg["id"], "x"))],
            )
            saved = json.loads(path.read_text())
            self.assertEqual(saved["documents"][0]["result"], "accepted-id")
            self.assertEqual(saved["documents"][1]["result"], "repaired")
            self.assertIn("read operation timed out", saved["documents"][2]["result"])
            client.upload_many.reset_mock()
            with (
                patch.object(pl, "_wait_tasks", return_value=[]),
                self.assertRaisesRegex(RuntimeError, "incomplete upload"),
            ):
                pl._step_index(run, client, lambda *a: None)
            client.upload_many.assert_not_called()
            client.delete_collection.assert_not_called()

    def test_upload_retries_tls_handshake_but_not_ambiguous_read_timeout(self):
        client = RagServiceClient("https://example.invalid")
        response = Mock(json=lambda: {"doc_id": "doc", "task_id": "task"})
        with (
            patch.object(
                client.session,
                "post",
                side_effect=[httpx.ConnectTimeout("handshake timed out"), response],
            ) as post,
            patch("evalkit.api_client.time.sleep"),
        ):
            self.assertEqual(
                client.upload_document("doc", "text", "collection"), ("doc", "task")
            )
            self.assertEqual(post.call_count, 2)
        with patch.object(
            client.session, "post", side_effect=httpx.ReadTimeout("response timed out")
        ) as post:
            with self.assertRaises(httpx.ReadTimeout):
                client.upload_document("doc", "text", "collection")
            self.assertEqual(post.call_count, 1)

    def test_unretrieved_gold_stays_in_denominator(self):
        gold = [
            {"doc_id": "a", "start": 0, "end": 100},
            {"doc_id": "b", "start": 0, "end": 100},
        ]
        values = evidence_metrics(gold, [hit("a", 0, 100)])
        self.assertEqual(values["recall@5"], 0.5)
        self.assertIsNone(values["ndcg@5"])

    def test_overlap_does_not_double_count(self):
        gold = [
            {"doc_id": "a", "start": 0, "end": 100},
            {"doc_id": "a", "start": 50, "end": 150},
        ]
        values = evidence_metrics(gold, [hit("a", 0, 75), hit("a", 25, 75)])
        self.assertEqual(values["recall@5"], 0.5)
        self.assertEqual(values["evidence_precision@5"], 1)

    def test_large_chunks_pay_for_padding(self):
        gold = [{"doc_id": "a", "start": 100, "end": 150}]
        small = evidence_metrics(gold, [hit("a", 100, 150)])
        large = evidence_metrics(gold, [hit("a", 0, 1000)])
        self.assertEqual(small["recall@5"], large["recall@5"])
        self.assertGreater(small["evidence_precision@5"], large["evidence_precision@5"])
        self.assertGreater(small["f1@5"], large["f1@5"])

    def test_empty_results_and_bad_metadata(self):
        gold = [{"doc_id": "a", "start": 0, "end": 10}]
        for results in ([], [{"doc_id": "a", "text": "unmapped"}]):
            values = evidence_metrics(gold, results)
            self.assertEqual(values["recall@5"], 0)
            self.assertEqual(values["f1@5"], 0)


class UsageTests(unittest.TestCase):
    def test_normalizes_usage_without_inventing_zero(self):
        self.assertIsNone(normalize_usage(None))
        self.assertIsNone(normalize_usage({"input_tokens": 10}))
        self.assertEqual(
            normalize_usage({"prompt_tokens": 10, "completion_tokens": 5}),
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        self.assertEqual(normalize_usage({"total_tokens": 0}), {"total_tokens": 0})

    def test_sums_agent_steps_and_detects_missing_usage(self):
        messages = [
            {
                "type": "ai",
                "tool_calls": [{}],
                "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
            },
            {"type": "tool"},
            {
                "type": "ai",
                "content": "answer",
                "response_metadata": {
                    "token_usage": {"prompt_tokens": 20, "completion_tokens": 10}
                },
            },
        ]
        usage, complete = agent_usage({}, messages)
        self.assertTrue(complete)
        self.assertEqual(usage["total_tokens"], 45)
        usage, complete = agent_usage({"usage": {"total_tokens": 45}}, messages)
        self.assertEqual(usage["total_tokens"], 45)
        messages[-1].pop("response_metadata")
        self.assertFalse(agent_usage({}, messages)[1])

    def test_average_excludes_unknown_usage_and_keeps_billed_errors(self):
        cfg = preset_configs(["stand_hybrid_512"])[0]
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run({"x": {"type": "upload"}}, [cfg], run_id="tokens")
            traces = [
                {
                    "error": None,
                    "unanswerable": False,
                    "scores": {"answer_correctness": 1, "faithfulness": 1},
                    "usage": {"total_tokens": 100},
                    "usage_complete": True,
                    "answer_seconds": 2,
                },
                {
                    "error": None,
                    "unanswerable": False,
                    "scores": {},
                    "usage": None,
                    "usage_complete": False,
                },
                {
                    "error": "judge failed",
                    "unanswerable": False,
                    "scores": {},
                    "usage": {"total_tokens": 200},
                    "usage_complete": True,
                },
            ]
            run.file("generation", cfg["id"] + ".jsonl").write_text(
                "".join(json.dumps(t) + "\n" for t in traces)
            )
            result = pl.generation_summary(run).iloc[0]
            self.assertEqual(result["mean_total_tokens"], 150)
            self.assertEqual(result["sum_total_tokens"], 300)
            self.assertEqual(result["usage_answers"], 2)
            self.assertEqual(result["usage_missing"], 1)

    def test_llm_usage_and_per_variant_model(self):
        answerer = object.__new__(LLMAnswerer)
        response = Mock()
        response.choices = [Mock(message=Mock(content="answer"))]
        response.usage = Mock(
            model_dump=lambda: {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            }
        )
        answerer._completion = Mock(return_value=response)
        result = answerer.generate(
            "q", ["context"], system_prompt="p", model="variant-model"
        )
        self.assertEqual(result["usage"]["total_tokens"], 13)
        self.assertEqual(
            answerer._completion.call_args.kwargs["model"], "variant-model"
        )

    def test_platform_returns_all_step_usage(self):
        client = RagServiceClient("https://example.invalid")
        body = {
            "messages": [
                {
                    "type": "ai",
                    "content": "answer",
                    "response_metadata": {"finish_reason": "stop"},
                    "usage_metadata": {"input_tokens": 10, "output_tokens": 3},
                }
            ]
        }
        with patch.object(client.session, "post", return_value=Mock(json=lambda: body)):
            result = client.agent_query("q")
        self.assertEqual(result["usage"]["total_tokens"], 13)
        self.assertTrue(result["usage_complete"])


class AgentSettingsTests(unittest.TestCase):
    def test_scientific_notation_config_ids_remain_strings(self):
        import pandas as pd

        cfg = preset_configs(["stand_hybrid_512"])[0]
        cfg["id"] = "3e463250"
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run({"x": {"type": "upload"}}, [cfg], run_id="csv-ids")
            pd.DataFrame(
                [
                    {
                        "config_id": cfg["id"],
                        "qid": "0001",
                        "dataset_source": "x",
                        "split": "dev",
                        "question_type": "single_hop",
                        "recall@5": 0.5,
                    }
                ]
            ).to_csv(run.file("retrieval", "ids.csv"), index=False)
            rows = pl.retrieval_rows(run)
            self.assertEqual(rows.iloc[0]["config_id"], "3e463250")
            self.assertEqual(rows.iloc[0]["qid"], "0001")
            self.assertIn("3e463250", pl.summarize(run).index)

    def test_tool_description_isolated_but_same_retrieval(self):
        first = preset_configs(["stand_hybrid_512"])[0]
        second = derive_config(
            first, {"agent.tool_description": "Different search description"}
        )
        self.assertNotEqual(
            collection_id(first["config"]), collection_id(second["config"])
        )
        self.assertEqual(retrieval_id(first["config"]), retrieval_id(second["config"]))
        client = RagServiceClient("https://example.invalid")
        with patch.object(client.session, "post") as post:
            client.ensure_eval_agent(
                "collection",
                "prompt",
                "agent",
                model="new-model",
                description="new description",
            )
        self.assertEqual(post.call_args.kwargs["json"]["model"], "new-model")
        self.assertEqual(
            post.call_args.kwargs["json"]["description"], "new description"
        )

    def test_tunnel_httpx_keeps_tls_hostname(self):
        from evalkit.transport import forward_httpx

        with (
            patch("config.AGENT_PLATFORM_URL", "https://stand.example"),
            patch.dict("os.environ", {"RAG_EVAL_TUNNEL_PORT": "18443"}),
        ):
            request = httpx.Request("GET", "https://stand.example/collections")
            forward_httpx(request)
        self.assertEqual(request.url.host, "127.0.0.1")
        self.assertEqual(request.extensions["sni_hostname"], "stand.example")
        self.assertEqual(request.headers["Host"], "stand.example")


if __name__ == "__main__":
    unittest.main()
