"""Offline regression tests for JSON migration and prompt propagation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from typer.testing import CliRunner

import config
from evalkit import pipeline as pl
from evalkit.api_client import RagServiceClient
from evalkit.cli import app
from evalkit.json_config import load_json_object
from evalkit.llm import LLMAnswerer
from evalkit.rag_config import (
    apply_overrides,
    base_config,
    collection_id,
    derive_config,
    load_presets,
    preset_configs,
    system_prompt,
)


class JsonConfigTests(unittest.TestCase):
    def test_all_presets_are_json_and_valid(self):
        groups = load_presets()
        self.assertEqual(set(groups), {"presets"})
        self.assertEqual(len(groups["presets"]), 4)
        for group in groups.values():
            for preset in group:
                full = apply_overrides(base_config(), preset["overrides"])
                self.assertTrue(system_prompt(full).strip())

    def test_invalid_json_and_duplicate_presets_fail_with_filename(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            for raw in (
                '{"presets":[],"presets":[]}',
                '{"value": NaN}',
                '{"value": 1e999}',
                "[]",
                "{",
            ):
                path.write_text(raw)
                with (
                    self.subTest(raw=raw),
                    self.assertRaisesRegex(ValueError, "bad.json"),
                ):
                    load_json_object(path)
            presets = load_presets()
            presets["presets"].append(deepcopy(presets["presets"][0]))
            path.write_text(json.dumps(presets))
            with self.assertRaisesRegex(ValueError, "duplicate preset"):
                load_presets(path)

    def test_alternative_settings_and_env_precedence(self):
        with TemporaryDirectory() as tmp:
            settings = load_json_object(config.REPO_ROOT / "configs/settings.json")
            settings["LLM_MODEL"] = "json-model"
            settings["RAG_SYSTEM_PROMPT"] = "Промпт из JSON"
            settings["RUNS_DIR"] = "custom-runs"
            path = Path(tmp) / "settings.json"
            path.write_text(json.dumps(settings))
            env = {
                **os.environ,
                "RAG_EVAL_CONFIG": str(path),
                "RAG_EVAL_ROOT": tmp,
                "LLM_MODEL": "env-model",
            }
            script = "import config; print(config.LLM_MODEL); print(config.RAG_SYSTEM_PROMPT); print(config.RUNS_DIR)"
            result = subprocess.run(
                [sys.executable, "-c", script],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                result.stdout.splitlines(),
                ["env-model", "Промпт из JSON", str(Path(tmp) / "custom-runs")],
            )
            settings["LLM_VIA_PLATFORM"] = "false"
            path.write_text(json.dumps(settings))
            result = subprocess.run(
                [sys.executable, "-c", "import config"],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid type for setting LLM_VIA_PLATFORM", result.stderr)

    def test_prompt_ids_share_index_and_derive_uses_saved_snapshot(self):
        parent = preset_configs(["stand_hybrid_512"])[0]
        original = deepcopy(parent)
        with patch(
            "evalkit.rag_config.EVAL_COLLECTION_CHUNKING",
            {"chunk_size": 999, "chunk_overlap": 5},
        ):
            child = derive_config(
                parent, {"generation.system_prompt": "Другой\nпромпт {literal}"}
            )
        self.assertNotEqual(parent["id"], child["id"])
        self.assertEqual(
            collection_id(parent["config"]), collection_id(child["config"])
        )
        self.assertEqual(parent["config"]["chunking"], child["config"]["chunking"])
        self.assertEqual(parent, original)
        self.assertNotIn("\n", child["name"])

    def test_invalid_prompt_and_overrides_rejected(self):
        for changes in (
            {"generation.system_prompt": ""},
            {"generation.system_prompt": 7},
            {"generation.other": "typo"},
            {"chunking.chunk_size": -1},
            {"embedding.sparse.dense_rrf_weight": -2},
            {"": 1},
            {"unknown.key": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                apply_overrides(base_config(), changes)

    def test_custom_presets_cli(self):
        with TemporaryDirectory() as tmp:
            groups = load_presets()
            groups["presets"][0]["name"] = "custom_json"
            path = Path(tmp) / "presets.json"
            path.write_text(json.dumps(groups))
            result = CliRunner().invoke(app, ["configs", "--presets-file", str(path)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("custom_json", result.output)

    def test_cli_prompt_file_run_and_tweak(self):
        with TemporaryDirectory() as tmp:
            prompt = Path(tmp) / "prompt.txt"
            prompt.write_text("Проверочный промпт\nс переносом")
            fake_run = Mock(id="test")
            with (
                patch("evalkit.cli.pl.new_run", return_value=fake_run) as new_run,
                patch("evalkit.cli._run_with_progress"),
                patch("evalkit.cli.show"),
            ):
                result = CliRunner().invoke(
                    app, ["run", "-p", "stand_hybrid_512", "--prompt-file", str(prompt)]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(
                    system_prompt(new_run.call_args.args[1][0]["config"]),
                    prompt.read_text(),
                )
            fake_run.config.return_value = preset_configs(["stand_hybrid_512"])[0]
            with (
                patch("evalkit.cli.pl.load_run", return_value=fake_run),
                patch("evalkit.cli.pl.add_configs", return_value=[{}]) as added,
                patch("evalkit.cli._run_with_progress"),
                patch("evalkit.cli.show"),
            ):
                result = CliRunner().invoke(
                    app,
                    ["tweak", "test", "stand_hybrid_512", "--prompt-file", str(prompt)],
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(
                    system_prompt(added.call_args.args[1][0]["config"]),
                    prompt.read_text(),
                )


class PromptRoutingTests(unittest.TestCase):
    def test_gui_saves_and_edits_prompt(self):
        from evalkit import gui

        parent = preset_configs(["stand_hybrid_512"])[0]
        saved_run = Mock(id="gui-run")
        saved_run.config.return_value = parent
        with (
            TemporaryDirectory() as tmp,
            patch("config.RUNS_DIR", Path(tmp)),
            patch.object(gui, "_busy", return_value=None),
            patch.object(gui, "_start"),
            patch.object(gui, "_available_runs", return_value=[]),
            patch.object(gui.pl, "new_run", return_value=saved_run) as new_run,
        ):
            gui.start_run(
                [next(iter(config.EVAL_DATASETS))],
                None,
                None,
                None,
                ["stand_hybrid_512"],
                "",
                "real",
                0,
                True,
                1,
                sample=0,
                prompt_text="GUI prompt",
                generation_full=True,
            )
            self.assertEqual(
                system_prompt(new_run.call_args.args[1][0]["config"]), "GUI prompt"
            )
            self.assertTrue(new_run.call_args.args[2].generation_full)
            self.assertIsNone(new_run.call_args.args[2].max_questions)
            self.assertIsNone(new_run.call_args.args[2].generation_sample)
        with (
            patch.object(gui, "_busy", return_value=None),
            patch.object(gui, "_start"),
            patch.object(gui, "_require_current_backend"),
            patch.object(gui.pl, "load_run", return_value=saved_run),
            patch.object(gui.pl, "add_configs", return_value=[{}]) as added,
        ):
            gui.tweak("gui-run", parent["id"], "", "Edited GUI prompt")
            self.assertEqual(
                system_prompt(added.call_args.args[1][0]["config"]), "Edited GUI prompt"
            )

    def test_gui_builds_with_two_prompt_editors(self):
        from evalkit.gui import build_app

        with (
            TemporaryDirectory() as tmp,
            patch("config.RUNS_DIR", Path(tmp)),
            patch.object(pl, "RUNS_DIR", Path(tmp)),
        ):
            app = build_app()
            gui_config = app.get_config_file()
            for block in app.blocks.values():
                if getattr(block, "label", None) in (
                    "Вопросов для поиска",
                    "Вопросов для ответов",
                ):
                    self.assertEqual(block.preprocess(0), 0)
                    self.assertIsNone(block.preprocess(None))
        labels = [
            c["props"].get("label", "")
            for c in gui_config["components"]
            if c["type"] == "textbox"
        ]
        self.assertEqual(
            sum("промпт" in label for label in labels if isinstance(label, str)), 2
        )

    def test_llm_message_uses_custom_prompt_verbatim(self):
        answerer = object.__new__(LLMAnswerer)
        answerer._chat = Mock(return_value="answer")
        prompt = 'Отвечай JSON: {"answer": "..."}\nТолько по источникам.'
        self.assertEqual(
            answerer.answer("question", ["context"], system_prompt=prompt), "answer"
        )
        messages = answerer._chat.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": prompt})
        self.assertIn("context", messages[1]["content"])
        self.assertIn("question", messages[1]["content"])

    def test_platform_agent_contains_prompt_and_tool(self):
        client = RagServiceClient("https://example.invalid")
        with patch.object(client.session, "post") as post:
            agent = client.ensure_eval_agent(
                "shared_collection", "Custom {braces}", "prompt_variant"
            )
        self.assertEqual(agent, "prompt_variant")
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tools"], ["shared_collection_search"])
        self.assertIn("shared_collection_search", payload["prompts"]["custom"])
        self.assertTrue(payload["prompts"]["custom"].endswith("Custom {braces}"))


class PipelinePromptTests(unittest.TestCase):
    def test_legacy_run_retains_collection_and_agent_interface(self):
        from evalkit.rag_config import config_id

        cfg = preset_configs(["stand_hybrid_512"])[0]
        cfg["config"].pop("generation")
        cfg["id"] = config_id(cfg["config"])
        with TemporaryDirectory() as tmp, patch.object(pl, "RUNS_DIR", Path(tmp)):
            run = pl.new_run({"legacy": {"type": "upload"}}, [cfg], run_id="old")
            coll = run.collection_name(cfg["id"], "legacy")
            self.assertTrue(coll.endswith("_legacy_" + cfg["id"]))
            client = Mock()
            run.ensure_agent(client, cfg, "legacy")
            client.ensure_eval_agent.assert_called_once_with(coll)

    def test_prompt_variants_pipeline_resume_chat_and_deploy(self):
        class Client:
            def __init__(self):
                self.collections = set()
                self.created = []
                self.agents = {}

            def collection_exists(self, name):
                return name in self.collections

            def create_collection(self, collection_name, **kwargs):
                self.collections.add(collection_name)
                self.created.append(kwargs)
                if "generation" in kwargs:
                    raise AssertionError("Generation settings leaked to collection API")

            def upload_many(self, docs, **kwargs):
                return [(d, None) for d, _, _ in docs]

            def ensure_eval_agent(
                self, collection_name, system_prompt=None, agent_id=None, **kwargs
            ):
                aid = agent_id or collection_name
                self.agents[aid] = (collection_name, system_prompt)
                return aid

            def agent_query(self, question, agent_id, **kwargs):
                return {
                    "answer": self.agents[agent_id][1],
                    "retrieved_chunks": [{"content": "answer"}],
                }

        class Judge:
            def score(self, question, answer, *args):
                score = 1.0 if answer == "better" else 0.1
                return {"answer_correctness": score, "faithfulness": score}

        first = derive_config(
            preset_configs(["stand_hybrid_512"])[0],
            {"generation.system_prompt": "first"},
        )
        better = derive_config(first, {"generation.system_prompt": "better"})
        qid = next(str(n) for n in range(100) if pl.split_of(str(n), 0.7) == "dev")
        corpus = {"doc": "answer context"}
        qa = [
            {
                "qid": qid,
                "question": "question",
                "reference_answer": "answer",
                "question_type": "single_hop",
                "dataset_source": "test",
                "gold_spans": [{"doc_id": "doc", "start": 0, "end": 6}],
            }
        ]
        frame = pd.DataFrame(
            [
                {
                    "qid": qid,
                    "dataset_source": "test",
                    "question_type": "single_hop",
                    **{m: 1.0 for m in pl.METRIC_COLS},
                }
            ]
        )
        client = Client()
        with (
            TemporaryDirectory() as tmp,
            patch.object(pl, "RUNS_DIR", Path(tmp)),
            patch.object(pl, "REGISTRY", Path(tmp) / "registry.json"),
            patch(
                "evalkit.dataset_loaders.load_dataset",
                side_effect=lambda *a: (deepcopy(corpus), deepcopy(qa)),
            ),
            patch(
                "evalkit.eval_runner.run_retrieval_grid",
                side_effect=lambda **k: frame.copy(),
            ) as retrieval,
            patch("evalkit.llm.make_judge", return_value=Judge()),
        ):
            run = pl.new_run(
                {"test": {"type": "upload"}},
                [first, better],
                pl.RunOptions(mode="real", top_n_generation=1),
                "prompts",
            )
            pl.run_pipeline(run, client=client, on_event=lambda e: None)
            self.assertEqual(run.state["best"]["config_id"], better["id"])
            self.assertEqual(len(client.created), 1)
            self.assertEqual(retrieval.call_count, 1)
            self.assertEqual(
                run.collection_name(first["id"], "test"),
                run.collection_name(better["id"], "test"),
            )
            self.assertEqual(len(client.agents), 2)
            self.assertEqual(
                pl.chat(run, first["id"], "question", client=client)["answer"], "first"
            )
            self.assertEqual(
                pl.chat(run, better["id"], "question", client=client)["answer"],
                "better",
            )
            pl.run_pipeline(
                pl.load_run("prompts"), client=client, on_event=lambda e: None
            )
            self.assertEqual(len(client.created), 1)
            self.assertEqual(retrieval.call_count, 1)
            traces = [
                json.loads(line)
                for line in (run.path / "generation" / f"{better['id']}.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(traces[0]["system_prompt"], "better")
            record = pl.deploy(run, "deployed", client=client, on_event=lambda e: None)
            aid = record["targets"][0]["agent_id"]
            self.assertEqual(client.agents[aid], ("deployed", "better"))
            corpus["doc"] = "changed content"
            with self.assertRaisesRegex(ValueError, "данные изменились"):
                pl.run_pipeline(
                    pl.load_run("prompts"), client=client, on_event=lambda e: None
                )


if __name__ == "__main__":
    unittest.main()
