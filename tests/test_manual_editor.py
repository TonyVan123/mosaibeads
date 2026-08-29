import json
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

import numpy as np

from beadsketch.manual_editor import BUILTIN_PROJECTS, ManualEditor, builtin_project_path, load_project, save_project, smoke_test


class ManualEditorTests(unittest.TestCase):
    def test_builtins_are_valid_and_distinct(self) -> None:
        loaded = [load_project(builtin_project_path(label)) for label in BUILTIN_PROJECTS]
        for item in loaded:
            self.assertEqual(item.result.indices.shape, (57, 58))
            self.assertGreaterEqual(len(item.result.palette), 291)
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

    def test_full_palette_row_fill_and_undo(self) -> None:
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        root.withdraw()
        try:
            editor = ManualEditor(root)
            self.assertGreaterEqual(len(editor.result.palette), 291)
            before = editor.result.indices[0].copy()
            target = next(i for i in range(len(editor.result.palette)) if np.any(before != i))
            editor.selected_palette = target
            editor.selection = (0, 0, editor.result.width - 1, 0)
            editor.fill_selection()
            self.assertTrue(np.all(editor.result.indices[0] == target))
            editor.undo()
            np.testing.assert_array_equal(editor.result.indices[0], before)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
