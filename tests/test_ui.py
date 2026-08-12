from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from beadsketch.app import BeadSketchApp
from beadsketch.engine import ConvertOptions, convert_image
from beadsketch.palettes import load_palette


def colorful_sample() -> Image.Image:
    image = Image.new("RGB", (72, 72), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 34, 66), fill=(225, 42, 55))
    draw.ellipse((24, 8, 68, 54), fill=(28, 102, 220))
    draw.polygon(((10, 62), (42, 30), (68, 68)), fill=(248, 190, 35))
    return image


class V3UiTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.app = BeadSketchApp()
        except tk.TclError as exc:
            self.skipTest(f"Tk is unavailable: {exc}")
        self.app.withdraw()
        self.app.update_idletasks()

    def tearDown(self) -> None:
        if hasattr(self, "app"):
            self.app.destroy()

    @staticmethod
    def descendants(widget):
        for child in widget.winfo_children():
            yield child
            yield from V3UiTests.descendants(child)

    def test_v3_layout_moves_sections_three_and_four_to_top(self) -> None:
        self.app.geometry("1360x820+0+0")
        self.app.deiconify()
        self.app.update()
        left_sections = [str(widget.cget("text")) for widget in self.app.left.winfo_children()
                         if widget.winfo_class() == "TLabel" and
                         str(widget.cget("text")).startswith(("1 ·", "2 ·", "3 ·", "4 ·"))]
        top_sections = [str(widget.cget("text")) for widget in self.app.top_controls.winfo_children()
                        if widget.winfo_class() == "TLabelframe"]
        all_labels = [str(widget.cget("text")) for widget in self.descendants(self.app)
                      if "text" in widget.keys()]

        self.assertEqual(left_sections, ["1 · 尺寸与色板", "2 · 传神程度"])
        self.assertEqual(top_sections, ["3 · 智能方案", "4 · 预览与精修"])
        self.assertEqual(self.app.title(), "MOSAIBeads 3.0.1 · 智能拼豆图纸")
        self.assertFalse(any("用更少豆粒" in text for text in all_labels))
        self.assertEqual(self.app.right.winfo_width(), 330)
        self.assertGreaterEqual(self.app.center.winfo_width(), 650)
        self.app.withdraw()

    def test_palette_selection_survives_paint_undo_and_redo(self) -> None:
        result = convert_image(colorful_sample(), load_palette("MARD 291"),
                               ConvertOptions(width=16, max_colors=6))
        self.assertGreaterEqual(len(result.palette), 2)
        self.app._apply_result(result)
        self.assertEqual(len(self.app.swatch_buttons), len(result.palette))

        chosen = 1
        self.app._set_selected_color(chosen)
        candidates = np.argwhere(result.indices != chosen)
        self.assertGreater(len(candidates), 0)
        y, x = (int(value) for value in candidates[0])
        old = int(result.indices[y, x])

        self.app.display_info = (0.0, 0.0, 1.0)
        self.app._paint_event(SimpleNamespace(x=x + 0.25, y=y + 0.25))
        self.assertEqual(int(result.indices[y, x]), chosen)
        self.assertEqual(self.app.selected_color, chosen)
        self.assertEqual(self.app.color_tree.selection(), (str(chosen),))

        self.app.undo()
        self.assertEqual(int(result.indices[y, x]), old)
        self.assertEqual(self.app.selected_color, chosen)
        self.app.redo()
        self.assertEqual(int(result.indices[y, x]), chosen)
        self.assertEqual(self.app.selected_color, chosen)


if __name__ == "__main__":
    unittest.main()
