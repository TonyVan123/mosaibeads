import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from beadsketch.manual_editor import BUILTIN_PROJECTS, builtin_project_path, load_project, save_project, smoke_test


class ManualEditorTests(unittest.TestCase):
    def test_builtins_are_valid_and_distinct(self) -> None:
        loaded = [load_project(builtin_project_path(label)) for label in BUILTIN_PROJECTS]
        for item in loaded:
            self.assertEqual(item.result.indices.shape, (57, 58))
            self.assertEqual(len(item.result.palette), 19)
            self.assertEqual(sum(n for _color, n in item.result.counts()), 3306)
        self.assertGreater(np.count_nonzero(loaded[0].result.indices != loaded[1].result.indices), 0)

    def test_project_round_trip_preserves_manual_cell(self) -> None:
        loaded = load_project(builtin_project_path(next(iter(BUILTIN_PROJECTS))))
        result = loaded.result
        original = int(result.indices[4, 7])
        replacement = (original + 1) % len(result.palette)
        result.indices[4, 7] = replacement
        with tempfile.TemporaryDirectory() as folder:
            output = save_project(Path(folder) / "edited.json", result, loaded.raw)
            restored = load_project(output)
            self.assertEqual(int(restored.result.indices[4, 7]), replacement)
            raw = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(raw["app"], "MOSAIBeads Manual Editor")
            self.assertEqual(raw["grid"], [58, 57])

    def test_manual_editor_smoke(self) -> None:
        smoke_test()


if __name__ == "__main__":
    unittest.main()
