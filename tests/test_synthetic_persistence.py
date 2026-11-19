import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evalkit import gui


class SyntheticPersistenceTests(unittest.TestCase):
    def write_dataset(self, root, name):
        folder = root / "datasets" / name
        folder.mkdir(parents=True)
        qa = {
            "question_type": "single_hop",
            "question": "Saved question?",
            "reference_answer": "Saved answer.",
            "gold_spans": [{"doc_id": "doc", "start": 0, "end": 3}],
        }
        (folder / "qa.jsonl").write_text(json.dumps(qa) + "\n")
        (folder / "corpus.jsonl").write_text('{"doc_id":"doc","text":"Saved source"}\n')
        record = {
            "dataset_id": name,
            "status": "done",
            "questions": 1,
            "dataset_config": {
                "type": "upload",
                "qa_path": str(folder / "qa.jsonl"),
                "corpus_path": str(folder / "corpus.jsonl"),
            },
        }
        (folder / "manifest.json").write_text(json.dumps(record))
        return folder

    def test_saved_preview_and_use_survive_empty_gui_job_memory(self):
        with (
            TemporaryDirectory() as tmp,
            patch("config.RUNS_DIR", Path(tmp)),
            patch("config.EVAL_DATASETS", {}),
            patch.dict(gui.JOBS, {}, clear=True),
        ):
            folder = self.write_dataset(Path(tmp), "saved")
            before = (folder / "qa.jsonl").read_bytes()
            panel = gui.restore_synthetic_panel()
            self.assertEqual(panel[0], "saved")
            self.assertIn("загружен с диска", panel[2])
            self.assertEqual(panel[5].iloc[0]["Вопрос"], "Saved question?")
            self.assertEqual(panel[6]["value"], ["saved"])
            self.assertEqual(gui.use_synthetic("saved")[0]["value"], ["saved"])
            self.assertEqual((folder / "qa.jsonl").read_bytes(), before)

    def test_catalog_refresh_picks_up_new_dataset_without_changing_selection(self):
        with (
            TemporaryDirectory() as tmp,
            patch("config.RUNS_DIR", Path(tmp)),
            patch("config.EVAL_DATASETS", {}),
            patch.dict(gui.JOBS, {}, clear=True),
        ):
            self.write_dataset(Path(tmp), "first")
            self.assertEqual(gui.restore_synthetic_panel()[0], "first")
            self.write_dataset(Path(tmp), "second")
            panel = gui.refresh_synthetic_panel("first")
            self.assertEqual(panel[0], "first")
            self.assertNotIn("value", panel[6])
            self.assertIn("second", panel[6]["choices"])
            self.assertIn("second", [value for _, value in panel[1]["choices"]])

    def test_generation_error_stays_visible_instead_of_switching_to_old_success(self):
        failed = SimpleNamespace(
            kind="synthetic",
            alive=False,
            error="LLM unavailable",
            result=None,
            events=[],
        )
        with (
            TemporaryDirectory() as tmp,
            patch("config.RUNS_DIR", Path(tmp)),
            patch("config.EVAL_DATASETS", {}),
            patch.dict(gui.JOBS, {"failed": failed}, clear=True),
        ):
            self.write_dataset(Path(tmp), "old-success")
            panel = gui.refresh_synthetic_panel("failed")
            self.assertEqual(panel[0], "failed")
            self.assertIn("LLM unavailable", panel[2])
            self.assertEqual(panel[4], [])


if __name__ == "__main__":
    unittest.main()
