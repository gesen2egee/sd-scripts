import tempfile
import unittest
from pathlib import Path

from ui.backend.dataset_inspector import detect_image_dirs, inspect_dataset, normalize_path


class DatasetInspectorTests(unittest.TestCase):
    def test_normalize_path_trims_spaces(self):
        p = normalize_path("  C:/tmp/test  ")
        self.assertTrue(p.endswith("tmp\\test") or p.endswith("tmp/test"))

    def test_detect_image_dirs_prefers_repeat_subdirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "data"
            base.mkdir()
            (base / "10_a").mkdir()
            (base / "20_b").mkdir()
            (base / "misc").mkdir()

            dirs = detect_image_dirs(str(base))
            self.assertEqual(len(dirs), 2)
            self.assertTrue(all(d.endswith(("10_a", "20_b")) for d in dirs))

    def test_inspect_dataset_caption_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "data"
            base.mkdir()
            (base / "a.png").write_bytes(b"x")
            (base / "a.txt").write_text("tag1, tag2", encoding="utf-8")
            (base / "b.png").write_bytes(b"x")
            (base / "b.txt").write_text("   ", encoding="utf-8")

            result = inspect_dataset(str(base))
            self.assertTrue(result["exists"])
            self.assertEqual(result["image_count"], 2)
            self.assertEqual(result["caption_valid_count"], 1)
            self.assertEqual(result["caption_invalid_count"], 1)


if __name__ == "__main__":
    unittest.main()
