from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Protocol

import cv2
import numpy as np
from PIL import Image

from .color import delta_e_ciede2000, rgb_to_lab
from .engine import ConvertOptions, PatternResult, convert_image
from .palettes import BeadColor


ProgressFn = Callable[[int, str], None]


class SemanticBackend(Protocol):
    name: str

    def score(self, source_rgb: np.ndarray, candidates_rgb: list[np.ndarray]) -> list[float]: ...


@dataclass(frozen=True)
class CandidateScore:
    likeness: float
    craft: float
    balanced: float
    ssim: float
    edge: float
    color: float
    semantic: float
    singleton_ratio: float
    fragmentation: float
    used_colors: int


@dataclass
class AutoTuneBundle:
    results: dict[str, PatternResult]
    options: dict[str, ConvertOptions]
    scores: dict[str, CandidateScore]
    provider: str
    elapsed_seconds: float
    searched_candidates: int
    palette_names: dict[str, str] = field(default_factory=dict)


SCHEME_LIKENESS = "最传神"
SCHEME_BALANCED = "综合平衡"
SCHEME_CRAFT = "最易制作"


def _notify(cb: ProgressFn | None, value: int, text: str) -> None:
    if cb:
        cb(int(value), text)


def _small_rgb(rgb: np.ndarray, limit: int = 320) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, limit / max(h, w))
    if scale >= 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _reconstruct(result: PatternResult, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(result.rgb, size, interpolation=cv2.INTER_NEAREST)


def _ssim(source: np.ndarray, candidate: np.ndarray) -> float:
    a = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1, c2 = 6.5025, 58.5225
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    var_a = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    var_b = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    value = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2) + 1e-8)
    return float(np.clip(np.mean(value), 0, 1))


def _edge_f1(source: np.ndarray, candidate: np.ndarray) -> float:
    a = cv2.Canny(cv2.cvtColor(source, cv2.COLOR_RGB2GRAY), 55, 135) > 0
    b = cv2.Canny(cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY), 55, 135) > 0
    # Pixel art edges move by a few source pixels; a small dilation makes the metric tolerant.
    kernel = np.ones((3, 3), np.uint8)
    ad = cv2.dilate(a.astype(np.uint8), kernel) > 0
    bd = cv2.dilate(b.astype(np.uint8), kernel) > 0
    precision = float(np.sum(b & ad)) / max(1, int(np.sum(b)))
    recall = float(np.sum(a & bd)) / max(1, int(np.sum(a)))
    return 2 * precision * recall / max(1e-8, precision + recall)


def _color_score(source: np.ndarray, candidate: np.ndarray) -> float:
    a = cv2.resize(source, (72, 72), interpolation=cv2.INTER_AREA)
    b = cv2.resize(candidate, (72, 72), interpolation=cv2.INTER_AREA)
    de = delta_e_ciede2000(rgb_to_lab(a), rgb_to_lab(b))
    # Delta-E around 22 is visibly poor; exponential scaling preserves useful separation.
    return float(np.exp(-float(np.mean(de)) / 20.0))


def _craft_metrics(result: PatternResult, color_budget: int) -> tuple[float, float, float]:
    labels = result.indices
    h, w = labels.shape
    singleton = 0
    components = 0
    for label in np.unique(labels):
        mask = (labels == label).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 4)
        if count > 1:
            sizes = stats[1:, cv2.CC_STAT_AREA]
            singleton += int(np.sum(sizes <= 1))
            components += int(count - 1)
    singleton_ratio = singleton / max(1, h * w)
    fragmentation = min(1.0, components / max(1.0, h * w * 0.20))
    used = len(result.counts())
    color_load = min(1.0, used / max(2, color_budget))
    craft = 1.0 - np.clip(2.8 * singleton_ratio + 0.48 * fragmentation + 0.20 * color_load, 0, 1)
    return float(craft), float(singleton_ratio), float(fragmentation)


def evaluate_result(result: PatternResult, semantic: float = 0.5,
                    color_budget: int | None = None) -> CandidateScore:
    source = _small_rgb(result.source_rgb)
    candidate = _reconstruct(result, (source.shape[1], source.shape[0]))
    ssim = _ssim(source, candidate)
    edge = _edge_f1(source, candidate)
    color = _color_score(source, candidate)
    craft, singleton, fragmentation = _craft_metrics(result, color_budget or len(result.palette))
    # Without a model, semantic=0.5 is neutral and its weight is redistributed in practice.
    likeness = 0.38 * ssim + 0.30 * edge + 0.25 * color + 0.07 * semantic
    balanced = 0.72 * likeness + 0.28 * craft
    return CandidateScore(
        likeness=float(likeness), craft=craft, balanced=float(balanced),
        ssim=ssim, edge=edge, color=color, semantic=float(semantic),
        singleton_ratio=singleton, fragmentation=fragmentation,
        used_colors=len(result.counts()),
    )


def _candidate_options(base: ConvertOptions) -> list[ConvertOptions]:
    # Physical/material choices are constraints, not tuning knobs.  Smart tuning
    # only searches the controls that change visual likeness.
    recipes = [
        (0.94, 0.16, 1.03, 1.08, "关闭"),
        (0.88, 0.30, 1.08, 1.06, "轻微"),
        (0.82, 0.42, 1.03, 1.08, "关闭"),
        (0.76, 0.54, 1.06, 1.04, "关闭"),
        (0.70, 0.66, 1.02, 1.05, "关闭"),
        (0.64, 0.78, 1.00, 1.03, "关闭"),
        (0.58, 0.90, 0.98, 1.02, "关闭"),
        (0.86, 0.48, 1.12, 1.10, "轻微"),
        (0.72, 0.72, 1.08, 1.08, "关闭"),
    ]
    return [replace(base, detail=d, cleanup=c, saturation=s, contrast=k, dither=di)
            for d, c, s, k, di in recipes]


def _pareto_indices(scores: list[CandidateScore]) -> list[int]:
    keep: list[int] = []
    for i, score in enumerate(scores):
        dominated = any(
            j != i and other.likeness >= score.likeness and other.craft >= score.craft
            and (other.likeness > score.likeness or other.craft > score.craft)
            for j, other in enumerate(scores)
        )
        if not dominated:
            keep.append(i)
    return keep or list(range(len(scores)))


def _choose_three(scores: list[CandidateScore]) -> dict[str, int]:
    if len(scores) == 1:
        return {SCHEME_LIKENESS: 0, SCHEME_BALANCED: 0, SCHEME_CRAFT: 0}
    if len(scores) == 2:
        likeness = max(range(2), key=lambda i: scores[i].likeness)
        craft = 1 - likeness
        balanced = max(range(2), key=lambda i: scores[i].balanced)
        return {SCHEME_LIKENESS: likeness, SCHEME_BALANCED: balanced, SCHEME_CRAFT: craft}
    frontier = _pareto_indices(scores)
    craft = max(frontier, key=lambda i: scores[i].craft)
    remaining = [i for i in frontier if i != craft] or [i for i in range(len(scores)) if i != craft]
    likeness = max(remaining, key=lambda i: scores[i].likeness)
    remaining2 = [i for i in frontier if i not in (likeness, craft)] or [
        i for i in range(len(scores)) if i not in (likeness, craft)]
    balanced = max(remaining2, key=lambda i: scores[i].balanced) if remaining2 else likeness
    return {SCHEME_LIKENESS: likeness, SCHEME_BALANCED: balanced, SCHEME_CRAFT: craft}


def auto_tune(image: Image.Image, bead_palette: list[BeadColor], base: ConvertOptions,
              semantic_backend: SemanticBackend | None = None,
              progress: ProgressFn | None = None) -> AutoTuneBundle:
    """Deterministic, lightweight multi-objective parameter search.

    Width is treated as a physical constraint. Search renders use a smaller grid and image,
    then the three winners are rendered once at the requested final width.
    """
    started = time.perf_counter()
    candidates = _candidate_options(base)
    preview_image = image.copy()
    preview_image.thumbnail((520, 520), Image.Resampling.LANCZOS)
    preview_width = min(int(base.width), 32)
    previews: list[PatternResult] = []
    for i, opts in enumerate(candidates):
        _notify(progress, 4 + round(48 * i / len(candidates)),
                f"智能搜索 {i + 1}/{len(candidates)}")
        previews.append(convert_image(preview_image, bead_palette,
                                      replace(opts, width=preview_width), None))

    semantics = [0.5] * len(previews)
    provider = "轻量视觉评分（无需模型）"
    if semantic_backend is not None:
        _notify(progress, 54, "AI 语义相似度批量评分")
        try:
            semantics = semantic_backend.score(
                previews[0].source_rgb, [p.rgb for p in previews])
            if len(semantics) != len(previews) or not all(math.isfinite(v) for v in semantics):
                raise ValueError("AI 返回了无效评分")
            provider = semantic_backend.name
        except Exception as exc:
            provider = f"AI 不可用，已回退轻量评分（{type(exc).__name__}）"
            semantics = [0.5] * len(previews)

    scores = [evaluate_result(result, sem, opts.max_colors)
              for result, sem, opts in zip(previews, semantics, candidates)]
    chosen = _choose_three(scores)
    results: dict[str, PatternResult] = {}
    options: dict[str, ConvertOptions] = {}
    final_scores: dict[str, CandidateScore] = {}
    for order, (scheme, index) in enumerate(chosen.items()):
        _notify(progress, 62 + order * 12, f"精算方案：{scheme}")
        opts = replace(candidates[index], width=base.width)
        result = convert_image(image, bead_palette, opts, None)
        score = evaluate_result(result, semantics[index], opts.max_colors)
        result.metadata["auto_tune_scheme"] = scheme
        result.metadata["auto_tune_provider"] = provider
        result.metadata["auto_tune_score"] = asdict(score)
        result.metadata["auto_tune_options"] = asdict(opts)
        results[scheme] = result
        options[scheme] = opts
        final_scores[scheme] = score
    elapsed = time.perf_counter() - started
    _notify(progress, 100, f"智能调参完成（{elapsed:.1f} 秒）")
    return AutoTuneBundle(results, options, final_scores, provider, elapsed, len(candidates))


def _joint_candidates(base: ConvertOptions, palette_names: list[str],
                      base_palette_name: str,
                      locked: set[str] | None = None) -> list[tuple[str, ConvertOptions]]:
    """Space-filling deterministic designs for all user-visible generation knobs."""
    locked = set(locked or ())
    names = ([base_palette_name] if "palette" in locked else
             [base_palette_name] + [name for name in palette_names if name != base_palette_name])
    visual_recipes = _candidate_options(base)
    width_factors = (0.72, 0.86, 1.0, 1.14, 1.28)
    color_factors = (0.60, 0.76, 0.92, 1.08, 1.22)
    profiles = ("自动", "人像/宠物", "插画/动漫", "照片")
    backgrounds = ("白色", "黑色", "透明区域留白")
    count = max(16, len(names) * 4)
    designs: list[tuple[str, ConvertOptions]] = [(base_palette_name, replace(base))]
    for i in range(1, count):
        recipe = visual_recipes[(i * 5 + i // 3) % len(visual_recipes)]
        width = (base.width if "width" in locked else
                 int(np.clip(round(base.width * width_factors[(i * 2 + i // 4) % 5]), 16, 120)))
        colors = (base.max_colors if "max_colors" in locked else
                  int(np.clip(round(base.max_colors * color_factors[(i * 3 + i // 5) % 5]), 4, 48)))
        opts = replace(
            recipe,
            width=width,
            max_colors=colors,
            profile=base.profile if "profile" in locked else profiles[(i * 3 + i // 4) % len(profiles)],
            background=base.background if "background" in locked else backgrounds[(i * 2 + i // 3) % len(backgrounds)],
            detail=base.detail if "detail" in locked else recipe.detail,
            cleanup=base.cleanup if "cleanup" in locked else recipe.cleanup,
            saturation=base.saturation if "saturation" in locked else recipe.saturation,
            contrast=base.contrast if "contrast" in locked else recipe.contrast,
            dither=base.dither if "dither" in locked else recipe.dither,
        )
        designs.append((names[i % len(names)], opts))
    unique: list[tuple[str, ConvertOptions]] = []
    seen: set[tuple] = set()
    for palette_name, opts in designs:
        key = (palette_name, opts.width, opts.max_colors, opts.profile, opts.background,
               opts.detail, opts.cleanup, opts.saturation, opts.contrast, opts.dither)
        if key not in seen:
            seen.add(key)
            unique.append((palette_name, opts))
    return unique


def _budget_adjusted(score: CandidateScore, opts: ConvertOptions,
                     min_width: int, max_width: int,
                     min_colors: int, max_colors: int) -> CandidateScore:
    width_ease = 1.0 - (opts.width - min_width) / max(1, max_width - min_width)
    color_ease = 1.0 - (opts.max_colors - min_colors) / max(1, max_colors - min_colors)
    craft = float(np.clip(0.62 * score.craft + 0.24 * width_ease + 0.14 * color_ease, 0, 1))
    return replace(score, craft=craft, balanced=0.74 * score.likeness + 0.26 * craft)


def auto_tune_all(image: Image.Image, palettes: dict[str, list[BeadColor]],
                  base: ConvertOptions, base_palette_name: str,
                  semantic_backend: SemanticBackend | None = None,
                  progress: ProgressFn | None = None,
                  locked: set[str] | None = None) -> AutoTuneBundle:
    """Jointly tune brand, grid/color budget, scene/background and visual controls."""
    if not palettes:
        raise ValueError("没有可用于智能搜索的品牌色板")
    started = time.perf_counter()
    designs = _joint_candidates(base, list(palettes), base_palette_name, locked)
    min_width = min(opts.width for _, opts in designs)
    max_width = max(opts.width for _, opts in designs)
    min_colors = min(opts.max_colors for _, opts in designs)
    max_colors = max(opts.max_colors for _, opts in designs)
    preview_image = image.copy()
    preview_image.thumbnail((520, 520), Image.Resampling.LANCZOS)
    preview_scale = min(1.0, 40 / max_width)
    previews: list[PatternResult] = []
    preview_options: list[ConvertOptions] = []
    for i, (palette_name, opts) in enumerate(designs):
        _notify(progress, 3 + round(51 * i / len(designs)),
                f"全参数搜索 {i + 1}/{len(designs)} · {palette_name}")
        search_opts = replace(opts, width=max(12, round(opts.width * preview_scale)))
        previews.append(convert_image(preview_image, palettes[palette_name], search_opts, None))
        preview_options.append(search_opts)

    semantics = [0.5] * len(previews)
    provider = "轻量视觉评分（无需模型）"
    if semantic_backend is not None:
        _notify(progress, 56, "AI 批量比较全部候选方案")
        try:
            semantics = semantic_backend.score(previews[0].source_rgb, [p.rgb for p in previews])
            if len(semantics) != len(previews) or not all(math.isfinite(v) for v in semantics):
                raise ValueError("AI 返回了无效评分")
            provider = semantic_backend.name
        except Exception as exc:
            provider = f"AI 不可用，已回退轻量评分（{type(exc).__name__}）"
            semantics = [0.5] * len(previews)

    scores = []
    for result, semantic, (_, original_opts), search_opts in zip(
            previews, semantics, designs, preview_options):
        raw = evaluate_result(result, semantic, search_opts.max_colors)
        scores.append(_budget_adjusted(raw, original_opts, min_width, max_width,
                                       min_colors, max_colors))
    chosen = _choose_three(scores)
    results: dict[str, PatternResult] = {}
    options: dict[str, ConvertOptions] = {}
    final_scores: dict[str, CandidateScore] = {}
    selected_palettes: dict[str, str] = {}
    for order, (scheme, index) in enumerate(chosen.items()):
        palette_name, opts = designs[index]
        _notify(progress, 64 + order * 11, f"精算：{scheme} · {palette_name}")
        result = convert_image(image, palettes[palette_name], opts, None)
        raw = evaluate_result(result, semantics[index], opts.max_colors)
        score = _budget_adjusted(raw, opts, min_width, max_width, min_colors, max_colors)
        result.metadata["auto_tune_scheme"] = scheme
        result.metadata["auto_tune_provider"] = provider
        result.metadata["auto_tune_palette"] = palette_name
        result.metadata["auto_tune_score"] = asdict(score)
        result.metadata["auto_tune_options"] = asdict(opts)
        result.metadata["auto_tune_locked"] = sorted(locked or ())
        results[scheme] = result
        options[scheme] = opts
        final_scores[scheme] = score
        selected_palettes[scheme] = palette_name
    elapsed = time.perf_counter() - started
    _notify(progress, 100, f"全参数智能调参完成（{elapsed:.1f} 秒）")
    return AutoTuneBundle(results, options, final_scores, provider, elapsed,
                          len(designs), selected_palettes)
