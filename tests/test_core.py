from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from beadsketch.color import delta_e_ciede2000, rgb_to_lab
from beadsketch.autotune import (SCHEME_BALANCED, SCHEME_CRAFT, SCHEME_LIKENESS,
                                 _joint_candidates, auto_tune)
from beadsketch.engine import ConvertOptions, convert_image, recommend_bead_colors
from beadsketch.exporter import export_bundle
from beadsketch.palettes import BeadColor, available_palettes, load_palette


def synthetic_portrait() -> Image.Image:
    image = Image.new("RGB", (240, 240), (226, 214, 197))
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 20, 192, 177), fill=(105, 63, 42))
    draw.ellipse((62, 39, 178, 181), fill=(231, 176, 139))
    draw.ellipse((88, 90, 104, 101), fill=(32, 28, 27))
    draw.ellipse((136, 90, 152, 101), fill=(32, 28, 27))
    draw.line((120, 101, 115, 128, 125, 130), fill=(143, 83, 69), width=5)
    draw.arc((96, 115, 146, 153), 15, 165, fill=(126, 45, 48), width=5)
    draw.polygon(((64, 169), (176, 169), (216, 240), (24, 240)), fill=(44, 83, 119))
    return image


class ColorTests(unittest.TestCase):
    def test_lab_reference_points(self) -> None:
        lab = rgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
        np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.02)
        np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.02)
        np.testing.assert_allclose(lab[2], [53.2408, 80.0925, 67.2032], atol=0.03)

    def test_ciede2000_published_pair(self) -> None:
        # Sharma et al. supplementary test pair 1.
        a = np.array([50.0000, 2.6772, -79.7751])
        b = np.array([50.0000, 0.0000, -82.7485])
        self.assertAlmostEqual(float(delta_e_ciede2000(a, b)), 2.0425, places=4)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.palette = load_palette("MARD 291")

    def test_all_packaged_palettes_load(self) -> None:
        for name in available_palettes():
            with self.subTest(name=name):
                self.assertGreater(len(load_palette(name)), 50)

    def test_convert_respects_limits(self) -> None:
        result = convert_image(synthetic_portrait(), self.palette,
                               ConvertOptions(width=32, max_colors=12, profile="插画/动漫"))
        self.assertEqual(result.width, 32)
        self.assertEqual(result.height, 32)
        self.assertLessEqual(len(result.counts()), 12)
        self.assertEqual(sum(n for _, n in result.counts()), 32 * 32)
        self.assertEqual(result.rgb.shape, (32, 32, 3))

    def test_thin_salient_feature_is_rescued(self) -> None:
        image = Image.new("RGB", (240, 240), "white")
        ImageDraw.Draw(image).line((10, 230, 230, 10), fill="black", width=3)
        result = convert_image(image, self.palette,
                               ConvertOptions(width=24, max_colors=8, profile="插画/动漫",
                                              detail=0.9, cleanup=0.25))
        dark_cells = int(np.sum(np.mean(result.rgb, axis=2) < 105))
        # A 3 px line is below 1/3 of a 10 px output cell and disappears with box averaging.
        self.assertGreaterEqual(dark_cells, 16)

    def test_export_bundle_is_complete(self) -> None:
        result = convert_image(synthetic_portrait(), self.palette,
                               ConvertOptions(width=24, max_colors=9, profile="插画/动漫"))
        with tempfile.TemporaryDirectory() as tmp:
            files = export_bundle(result, tmp, "portrait.png")
            self.assertEqual(len(files), 6)
            self.assertTrue(all(Path(p).exists() and Path(p).stat().st_size > 20 for p in files))
            self.assertTrue(any(p.suffix == ".pdf" for p in files))

    def test_fixed_palette_is_an_exact_allow_list(self) -> None:
        chosen = [self.palette[i] for i in (0, 12, 35, 70)]
        result = convert_image(synthetic_portrait(), chosen,
                               ConvertOptions(width=18, max_colors=2, fixed_palette=True))
        self.assertEqual(result.palette, chosen)
        self.assertTrue(result.metadata["fixed_palette"])
        allowed = {color.rgb for color in chosen}
        self.assertTrue({tuple(rgb) for rgb in result.rgb.reshape(-1, 3)} <= allowed)

    def test_click_recommendation_ranks_exact_color_first(self) -> None:
        palette = [BeadColor("R", "red", (220, 30, 40)),
                   BeadColor("B", "blue", (20, 40, 220)),
                   BeadColor("W", "white", (250, 250, 250))]
        image = Image.new("RGB", (20, 20), (220, 30, 40))
        ranked = recommend_bead_colors(image, 10, 10, palette, radius=2, top_k=3)
        self.assertEqual(ranked[0][0].code, "R")
        self.assertLess(ranked[0][1], 0.01)

    def test_auto_tune_returns_three_switchable_intents(self) -> None:
        image = synthetic_portrait().resize((96, 96))
        bundle = auto_tune(image, self.palette, ConvertOptions(width=16, max_colors=8))
        self.assertEqual(set(bundle.results), {SCHEME_LIKENESS, SCHEME_BALANCED, SCHEME_CRAFT})
        self.assertTrue(all(result.width == 16 for result in bundle.results.values()))
        self.assertTrue(all(options.max_colors == 8 for options in bundle.options.values()))
        recipes = {(round(o.detail, 2), round(o.cleanup, 2), o.max_colors)
                   for o in bundle.options.values()}
        self.assertEqual(len(recipes), 3)
        self.assertTrue(all(np.isfinite(score.likeness) and np.isfinite(score.craft)
                            for score in bundle.scores.values()))

    def test_joint_tuner_varies_every_requested_parameter_group(self) -> None:
        base = ConvertOptions(width=48, max_colors=24)
        designs = _joint_candidates(base, ["A", "B", "C", "D"], "A")
        self.assertGreaterEqual(len(designs), 16)
        self.assertEqual({name for name, _ in designs}, {"A", "B", "C", "D"})
        self.assertGreaterEqual(len({o.width for _, o in designs}), 4)
        self.assertGreaterEqual(len({o.max_colors for _, o in designs}), 4)
        self.assertEqual({o.profile for _, o in designs}, {"自动", "人像/宠物", "插画/动漫", "照片"})
        self.assertEqual({o.background for _, o in designs}, {"白色", "黑色", "透明区域留白"})
        visual = {(o.detail, o.cleanup, o.saturation, o.contrast, o.dither) for _, o in designs}
        self.assertGreaterEqual(len(visual), 6)

    def test_joint_tuner_never_changes_locked_parameters(self) -> None:
        base = ConvertOptions(width=53, max_colors=17, profile="照片", background="黑色",
                              detail=0.77, cleanup=0.41, saturation=1.13,
                              contrast=0.93, dither="明显")
        locked = {"palette", "background", "width", "max_colors", "profile",
                  "cleanup", "saturation", "contrast", "dither"}
        designs = _joint_candidates(base, ["A", "B", "C", "D"], "B", locked)
        self.assertGreaterEqual(len(designs), 3)
        self.assertTrue(all(name == "B" for name, _ in designs))
        self.assertTrue(all(o.width == 53 and o.max_colors == 17 for _, o in designs))
        self.assertTrue(all(o.profile == "照片" and o.background == "黑色" for _, o in designs))
        self.assertTrue(all(o.cleanup == 0.41 and o.saturation == 1.13 and
                            o.contrast == 0.93 and o.dither == "明显" for _, o in designs))
        self.assertGreaterEqual(len({o.detail for _, o in designs}), 3)

        all_locked = set(locked) | {"detail"}
        one = _joint_candidates(base, ["A", "B", "C", "D"], "B", all_locked)
        self.assertEqual(len(one), 1)


if __name__ == "__main__":
    unittest.main()
