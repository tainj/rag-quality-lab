import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from evalkit.data_catalog import available_datasets
from evalkit.synthetic import generate_synthetic, source_units, validate_qa


class SyntheticTests(unittest.TestCase):
    def test_whitespace_mapping_preserves_verbatim_gold_and_source_identity(self):
        units = [
            {
                "source_id": "source-1",
                "doc_id": "a",
                "start": 10,
                "text": "Feature\n  enabled now.",
            },
            {
                "source_id": "source-2",
                "doc_id": "a",
                "start": 100,
                "text": "Different content here.",
            },
        ]
        parsed = {
            "question": "How?",
            "reference_answer": "Enabled.",
            "evidence": [{"source_id": "source-1", "quote": "Feature enabled now."}],
        }
        row = validate_qa(parsed, units, "single_hop", 400)
        self.assertEqual(row["evidence_quotes"][0]["quote"], units[0]["text"])
        self.assertEqual(
            row["gold_spans"][0],
            {"doc_id": "a", "start": 10, "end": 10 + len(units[0]["text"])},
        )
        parsed["evidence"][0]["source_id"] = "source-2"
        with self.assertRaises(ValueError):
            validate_qa(parsed, units, "single_hop", 400)
        parsed["evidence"][0] = {
            "source_id": "source-1",
            "quote": "Feature disabled now.",
        }
        with self.assertRaises(ValueError):
            validate_qa(parsed, units, "single_hop", 400)

    def test_units_preserve_original_offsets(self):
        text = "First sentence.\nSecond line with a fact.\n" * 30
        units = source_units({"doc": text}, 200)
        self.assertGreater(len(units), 1)
        for unit in units:
            self.assertEqual(unit["text"], text[unit["start"] : unit["end"]])
            self.assertLessEqual(len(unit["text"]), 200)

    def test_gold_must_be_short_and_verbatim(self):
        units = [
            {
                "doc_id": "a",
                "start": 20,
                "text": "The actual feature is enabled by a configuration flag.",
            }
        ]
        parsed = {
            "question": "How?",
            "reference_answer": "A flag.",
            "evidence": [
                {"doc_id": "a", "quote": "feature is enabled by a configuration flag"}
            ],
        }
        row = validate_qa(parsed, units, "single_hop", 400)
        self.assertEqual(row["gold_spans"][0]["start"], 31)
        for bad in ("invented evidence phrase", "x" * 401):
            parsed["evidence"][0]["quote"] = bad
            with self.assertRaises(ValueError):
                validate_qa(parsed, units, "single_hop", 400)

    def test_multi_hop_requires_both_passages(self):
        units = [
            {"doc_id": "a", "start": 0, "text": "First source evidence phrase."},
            {"doc_id": "a", "start": 100, "text": "Second source evidence phrase."},
        ]
        parsed = {
            "question": "Combine?",
            "reference_answer": "Both.",
            "evidence": [{"doc_id": "a", "quote": "First source evidence phrase"}],
        }
        with self.assertRaises(ValueError):
            validate_qa(parsed, units, "multi_hop", 400)
        parsed["evidence"].append(
            {"doc_id": "a", "quote": "Second source evidence phrase"}
        )
        self.assertEqual(
            len(validate_qa(parsed, units, "multi_hop", 400)["gold_spans"]), 2
        )

    def test_generation_registers_only_finished_dataset_and_records_usage(self):
        source = "The actual feature is enabled by a configuration flag. " + (
            "Additional documented information. " * 10
        )
        corpus = {"doc": source}
        response = Mock()
        response.usage = Mock(
            model_dump=lambda: {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        )
        response.choices = [
            Mock(
                message=Mock(
                    content=json.dumps(
                        {
                            "question": "How do I enable the feature?",
                            "reference_answer": "Use a configuration flag.",
                            "evidence": [
                                {
                                    "doc_id": "doc",
                                    "quote": "The actual feature is enabled by a configuration flag.",
                                }
                            ],
                        }
                    )
                )
            )
        ]
        provider = Mock()
        provider.chat.completions.create.return_value = response
        with TemporaryDirectory() as tmp, patch("config.RUNS_DIR", Path(tmp)):
            record = generate_synthetic(
                size=1,
                name="generated-test",
                corpus=corpus,
                question_types=["single_hop"],
                client=provider,
            )
            self.assertEqual(record["questions"], 1)
            self.assertEqual(record["generation_total_tokens"], 150)
            self.assertIn("generated-test", available_datasets())
            qa = json.loads(Path(record["dataset_config"]["qa_path"]).read_text())
            span = qa["gold_spans"][0]
            self.assertEqual(
                corpus["doc"][span["start"] : span["end"]],
                qa["evidence_quotes"][0]["quote"],
            )
            record["status"] = "failed"
            manifest = (
                Path(record["dataset_config"]["qa_path"]).parent / "manifest.json"
            )
            manifest.write_text(json.dumps(record))
            self.assertNotIn("generated-test", available_datasets())

    def test_provider_failure_does_not_publish_incomplete_dataset(self):
        provider = Mock()
        provider.chat.completions.create.side_effect = ConnectionError("unavailable")
        with TemporaryDirectory() as tmp, patch("config.RUNS_DIR", Path(tmp)):
            with self.assertRaisesRegex(RuntimeError, "three consecutive"):
                generate_synthetic(
                    size=1,
                    name="failed-test",
                    corpus={"doc": "Documented source content. " * 20},
                    question_types=["single_hop"],
                    client=provider,
                )
            self.assertNotIn("failed-test", available_datasets())
            manifest = json.loads(
                (Path(tmp) / "datasets/failed-test/manifest.json").read_text()
            )
            self.assertEqual(manifest["status"], "failed")


if __name__ == "__main__":
    unittest.main()
