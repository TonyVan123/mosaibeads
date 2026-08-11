from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .color import delta_e_ciede2000, lab_to_rgb, pairwise_delta_e, rgb_to_lab
from .palettes import BeadColor, palette_arrays


ProgressFn = Callable[[int, str], None]


@dataclass
class ConvertOptions:
    width: int = 48
    max_colors: int = 24
    profile: str = "自动"
    detail: float = 0.72
    cleanup: float = 0.54
    saturation: float = 1.06
    contrast: float = 1.05
    dither: str = "关闭"
    background: str = "白色"
    # When enabled, bead_palette is an exact allow-list rather than a brand palette
    # from which the engine may choose a smaller subset.
    fixed_palette: bool = False


@dataclass
class PatternResult:
    indices: np.ndarray
    palette: list[BeadColor]
    selected_source_indices: np.ndarray
    source_rgb: np.ndarray
    sampled_rgb: np.ndarray
    saliency: np.ndarray
    profile: str
    metadata: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])

    @property
    def height(self) -> int:
        return int(self.indices.shape[0])

    @property
    def rgb(self) -> np.ndarray:
        colors = np.asarray([c.rgb for c in self.palette], dtype=np.uint8)
        return colors[self.indices]

    def counts(self) -> list[tuple[BeadColor, int]]:
        values, counts = np.unique(self.indices, return_counts=True)
        rows = [(self.palette[int(v)], int(n)) for v, n in zip(values, counts)]
        return sorted(rows, key=lambda x: x[1], reverse=True)


def _notify(cb: ProgressFn | None, value: int, message: str) -> None:
    if cb:
        cb(value, message)


def _normalize01(arr: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, percentile))
    if hi <= lo + 1e-8:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def _pil_to_rgb(image: Image.Image, background: str) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        if background == "透明区域留白":
            bg = (255, 255, 255, 255)
        elif background == "黑色":
            bg = (18, 18, 20, 255)
        else:
            bg = (255, 255, 255, 255)
        canvas = Image.new("RGBA", rgba.size, bg)
        canvas.alpha_composite(rgba)
        image = canvas.convert("RGB")
    else:
        image = image.convert("RGB")
    return np.asarray(image)


def _limit_size(rgb: np.ndarray, limit: int = 1100) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, limit / max(h, w))
    if scale == 1:
        return rgb
    return cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    try:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        found = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5,
                                         minSize=(max(20, rgb.shape[1] // 16),) * 2)
        return [tuple(int(v) for v in box) for box in found]
    except Exception:
        return []


def _auto_profile(rgb: np.ndarray, requested: str, faces: list[tuple[int, int, int, int]]) -> str:
    if requested != "自动":
        return requested
    if faces:
        return "人像/宠物"
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edge_density = float(np.mean(cv2.Canny(gray, 70, 150) > 0))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    unique_hint = len(np.unique(cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3), axis=0))
    return "插画/动漫" if edge_density > 0.105 or unique_hint < 900 else "照片"


def _preprocess(rgb: np.ndarray, profile: str, opts: ConvertOptions) -> np.ndarray:
    pil = Image.fromarray(rgb)
    pil = ImageEnhance.Contrast(pil).enhance(opts.contrast)
    pil = ImageEnhance.Color(pil).enhance(opts.saturation)
    arr = np.asarray(pil)
    if profile == "插画/动漫":
        return cv2.bilateralFilter(arr, 5, 32, 26)
    if profile == "人像/宠物":
        smooth = cv2.bilateralFilter(arr, 7, 38, 34)
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.45, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    return cv2.bilateralFilter(arr, 7, 30, 30)


def _spectral_saliency(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    spectrum = np.fft.fft2(small)
    log_amp = np.log(np.abs(spectrum) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (5, 5))
    recon = np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))
    sal = cv2.GaussianBlur(np.abs(recon) ** 2, (9, 9), 0)
    return _normalize01(cv2.resize(sal, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC))


def _feature_map(rgb: np.ndarray, faces: list[tuple[int, int, int, int]], profile: str) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = _normalize01(cv2.magnitude(gx, gy), 98.5)
    local = _normalize01(cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3.0)), 98.5)
    spectral = _spectral_saliency(gray)
    sal = 0.47 * edge + 0.25 * local + 0.28 * spectral
    if faces:
        boost = np.zeros(gray.shape, np.float32)
        for x, y, w, h in faces:
            center = (x + w // 2, y + int(h * 0.48))
            cv2.ellipse(boost, center, (max(2, int(w * 0.54)), max(2, int(h * 0.61))), 0, 0, 360, 1.0, -1)
            # Eyes, nose and mouth occupy the central facial band; protect its contrast.
            cv2.ellipse(boost, (x + w // 2, y + int(h * 0.55)),
                        (max(2, int(w * 0.42)), max(2, int(h * 0.32))), 0, 0, 360, 0.65, -1)
        boost = cv2.GaussianBlur(boost, (0, 0), max(2, min(rgb.shape[:2]) / 160))
        sal = np.maximum(sal, 0.48 * boost + 0.62 * edge * boost)
    if profile == "插画/动漫":
        sal = np.maximum(sal, edge * 0.9)
    return _normalize01(sal, 99), edge


def _weighted_representative(patch_lab: np.ndarray, patch_sal: np.ndarray,
                             detail: float) -> tuple[np.ndarray, float]:
    ph, pw = patch_lab.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw]
    sx, sy = max(1.0, pw * 0.42), max(1.0, ph * 0.42)
    center = np.exp(-0.5 * (((xx - (pw - 1) / 2) / sx) ** 2 + ((yy - (ph - 1) / 2) / sy) ** 2))
    weight = center * (1.0 + (0.7 + 3.0 * detail) * patch_sal)
    flat = patch_lab.reshape(-1, 3)
    wf = weight.ravel()
    # Lab histogram supplies a stable dominant-color candidate without averaging across boundaries.
    bins = np.stack((np.clip(flat[:, 0] // 7, 0, 14),
                     np.clip((flat[:, 1] + 128) // 10, 0, 25),
                     np.clip((flat[:, 2] + 128) // 10, 0, 25)), axis=1).astype(np.int32)
    keys = bins[:, 0] * 26 * 26 + bins[:, 1] * 26 + bins[:, 2]
    scores = np.bincount(keys, weights=wf, minlength=15 * 26 * 26)
    winning = int(np.argmax(scores))
    mask = keys == winning
    dominant = np.average(flat[mask], axis=0, weights=wf[mask]) if np.any(mask) else np.average(flat, axis=0, weights=wf)
    mean = np.average(flat, axis=0, weights=wf)
    # A thin eye, whisker, outline or antenna can occupy too little area to become the
    # dominant bin. Among plausible bins, reward coherent minority colors that contrast
    # with the cell mean. Saliency gates this rescue so texture noise is not promoted.
    nonzero = np.flatnonzero(scores > 0)
    top = nonzero[np.argsort(scores[nonzero])[-min(10, len(nonzero)):]]
    total_score = float(np.sum(scores)) + 1e-9
    accent = dominant
    accent_strength = 0.0
    best_quality = -1.0
    for key in top:
        kmask = keys == int(key)
        share = float(scores[int(key)] / total_score)
        if share < 0.018:
            continue
        center_lab = np.average(flat[kmask], axis=0, weights=wf[kmask])
        contrast = float(delta_e_ciede2000(center_lab, mean))
        quality = share ** 0.62 * (1.0 + contrast / 17.0)
        if quality > best_quality:
            best_quality, accent = quality, center_lab
            accent_strength = contrast
    edge_strength = float(np.average(patch_sal, weights=center))
    blend = np.clip(0.18 + detail * 0.24 + edge_strength * 0.30, 0.18, 0.70)
    peak_saliency = float(np.percentile(patch_sal, 88))
    rescue = np.clip((peak_saliency - 0.42) / 0.42, 0, 1) * np.clip((accent_strength - 14) / 35, 0, 1) * detail
    structural_color = dominant * (1 - rescue) + accent * rescue
    candidate = mean * (1 - blend) + structural_color * blend
    # Snap to a real source color medoid for crisp boundaries, while keeping some tonal stability.
    stride = max(1, len(flat) // 100)
    candidates = flat[::stride]
    dist = np.sum((candidates - candidate) ** 2 * np.array([1.0, 0.72, 0.72]), axis=1)
    medoid = candidates[int(np.argmin(dist))]
    return 0.48 * medoid + 0.52 * candidate, float(rescue)


def _adaptive_downsample(rgb: np.ndarray, saliency: np.ndarray, out_w: int, out_h: int,
                         detail: float) -> tuple[np.ndarray, np.ndarray]:
    lab = rgb_to_lab(rgb)
    h, w = rgb.shape[:2]
    output = np.empty((out_h, out_w, 3), dtype=np.float64)
    sal_grid = cv2.resize(saliency, (out_w, out_h), interpolation=cv2.INTER_AREA)
    # The area reference is tonally stable. Structure-aware representatives replace it
    # progressively only where the image contains information worth spending a bead on.
    reference = rgb_to_lab(cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA))
    for oy in range(out_h):
        y0, y1 = oy * h / out_h, (oy + 1) * h / out_h
        for ox in range(out_w):
            x0, x1 = ox * w / out_w, (ox + 1) * w / out_w
            pad_x, pad_y = (x1 - x0) * 0.16, (y1 - y0) * 0.16
            xa, xb = max(0, int(math.floor(x0 - pad_x))), min(w, int(math.ceil(x1 + pad_x)))
            ya, yb = max(0, int(math.floor(y0 - pad_y))), min(h, int(math.ceil(y1 + pad_y)))
            representative, rescued = _weighted_representative(lab[ya:yb, xa:xb], saliency[ya:yb, xa:xb], detail)
            local_top = float(np.percentile(saliency[ya:yb, xa:xb], 85))
            structure_signal = max(float(sal_grid[oy, ox]), local_top * (0.55 + 0.25 * rescued))
            structure_mix = float(np.clip(0.08 + (0.28 + 0.58 * detail) * structure_signal ** 0.76
                                                + 0.18 * rescued, 0.08, 0.92))
            output[oy, ox] = reference[oy, ox] * (1 - structure_mix) + representative * structure_mix
    return output, sal_grid


def _select_palette(sample_lab: np.ndarray, source_palette_lab: np.ndarray, max_colors: int,
                    importance: np.ndarray) -> np.ndarray:
    points = sample_lab.reshape(-1, 3)
    weights = (0.55 + 1.8 * importance.ravel()).astype(np.float64)
    full_dist = pairwise_delta_e(points, source_palette_lab)
    nearest = np.argmin(full_dist, axis=1)
    usage = np.bincount(nearest, weights=weights, minlength=len(source_palette_lab))
    max_colors = max(2, min(max_colors, len(source_palette_lab), len(points)))
    # Start with high-usage real colors, then greedily add colors that reduce weighted perceptual error.
    first = int(np.argmax(usage))
    selected = [first]
    best = full_dist[:, first].copy()
    candidate_order = np.argsort(usage)[::-1]
    pool = candidate_order[:min(len(candidate_order), max(max_colors * 8, 48))]
    while len(selected) < max_colors:
        gains = []
        for p in pool:
            if int(p) in selected:
                gains.append(-1.0)
                continue
            gains.append(float(np.sum(weights * np.maximum(0, best - full_dist[:, p]))))
        pick = int(pool[int(np.argmax(gains))])
        if max(gains) <= 1e-7:
            remaining = [int(x) for x in candidate_order if int(x) not in selected]
            if not remaining:
                break
            pick = remaining[0]
        selected.append(pick)
        best = np.minimum(best, full_dist[:, pick])
    return np.asarray(selected, dtype=np.int32)


def _dither_labs(sample_lab: np.ndarray, palette_lab: np.ndarray, strength: float,
                 saliency: np.ndarray) -> np.ndarray:
    work = sample_lab.copy()
    h, w = work.shape[:2]
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in xs:
            d = delta_e_ciede2000(work[y, x][None, :], palette_lab)
            chosen = palette_lab[int(np.argmin(d))]
            err = (work[y, x] - chosen) * strength * (1 - 0.78 * saliency[y, x])
            for dx, dy, factor in ((direction, 0, 7/16), (-direction, 1, 3/16), (0, 1, 5/16), (direction, 1, 1/16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and ny < h:
                    work[ny, nx] += err * factor
    return work


def _regularize(labels: np.ndarray, unary: np.ndarray, saliency: np.ndarray,
                edge: np.ndarray, cleanup: float) -> np.ndarray:
    if cleanup <= 0.02:
        return labels
    h, w = labels.shape
    result = labels.copy()
    k = unary.shape[2]
    for _ in range(3):
        changed = 0
        for y in range(h):
            for x in range(w):
                neighbors = []
                boundary_edges = []
                for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if 0 <= nx < w and 0 <= ny < h:
                        neighbors.append(int(result[ny, nx]))
                        boundary_edges.append(max(float(edge[y, x]), float(edge[ny, nx])))
                candidates = set(np.argpartition(unary[y, x], min(3, k-1))[:min(4, k)].tolist())
                candidates.update(neighbors)
                best_label, best_energy = int(result[y, x]), float("inf")
                for label in candidates:
                    energy = float(unary[y, x, label]) * (0.72 + 1.65 * saliency[y, x])
                    for nlabel, e in zip(neighbors, boundary_edges):
                        if label != nlabel:
                            energy += cleanup * 7.0 * math.exp(-4.2 * e) * (1 - 0.55 * saliency[y, x])
                    if energy < best_energy:
                        best_energy, best_label = energy, int(label)
                if best_label != result[y, x]:
                    result[y, x] = best_label
                    changed += 1
        if not changed:
            break
    return result


def convert_image(image: Image.Image, bead_palette: list[BeadColor], opts: ConvertOptions,
                  progress: ProgressFn | None = None) -> PatternResult:
    _notify(progress, 3, "读取并校正图片")
    original = _limit_size(_pil_to_rgb(image, opts.background))
    faces = _detect_faces(original)
    profile = _auto_profile(original, opts.profile, faces)
    _notify(progress, 13, f"识别场景：{profile}")
    prepared = _preprocess(original, profile, opts)
    h, w = prepared.shape[:2]
    out_w = max(8, min(160, int(opts.width)))
    out_h = max(8, min(160, int(round(out_w * h / w))))
    _notify(progress, 25, "分析轮廓、显著区域与人脸")
    saliency, edge = _feature_map(prepared, faces, profile)
    _notify(progress, 42, "结构感知降采样")
    sampled_lab, sal_grid = _adaptive_downsample(prepared, saliency, out_w, out_h, opts.detail)
    edge_grid = cv2.resize(edge, (out_w, out_h), interpolation=cv2.INTER_AREA)
    palette_rgb, palette_lab = palette_arrays(bead_palette)
    _notify(progress, 62, "选择最有效的真实豆色")
    if opts.fixed_palette:
        if not bead_palette:
            raise ValueError("固定色板至少需要 1 种拼豆颜色")
        selected_source = np.arange(len(bead_palette), dtype=np.int32)
    else:
        selected_source = _select_palette(sampled_lab, palette_lab, opts.max_colors, sal_grid)
    chosen_colors = [bead_palette[int(i)] for i in selected_source]
    chosen_lab = palette_lab[selected_source]
    if opts.dither == "轻微":
        sampled_for_map = _dither_labs(sampled_lab, chosen_lab, 0.34, sal_grid)
    elif opts.dither == "明显":
        sampled_for_map = _dither_labs(sampled_lab, chosen_lab, 0.58, sal_grid)
    else:
        sampled_for_map = sampled_lab
    _notify(progress, 77, "CIEDE2000 感知色差匹配")
    unary = pairwise_delta_e(sampled_for_map.reshape(-1, 3), chosen_lab).reshape(out_h, out_w, -1)
    labels = np.argmin(unary, axis=2).astype(np.int32)
    _notify(progress, 88, "保护边缘并清理孤立杂色")
    labels = _regularize(labels, unary, sal_grid, edge_grid, opts.cleanup)
    sampled_rgb = lab_to_rgb(sampled_lab)
    _notify(progress, 100, "完成")
    return PatternResult(
        indices=labels,
        palette=chosen_colors,
        selected_source_indices=selected_source,
        source_rgb=original,
        sampled_rgb=sampled_rgb,
        saliency=sal_grid,
        profile=profile,
        metadata={
            "faces_detected": len(faces),
            "source_size": [int(original.shape[1]), int(original.shape[0])],
            "grid_size": [out_w, out_h],
            "total_beads": out_w * out_h,
            "boards_29x29": [math.ceil(out_w / 29), math.ceil(out_h / 29)],
            "physical_mm_5mm": [out_w * 5, out_h * 5],
            "fixed_palette": bool(opts.fixed_palette),
        },
    )


def recommend_bead_colors(image: Image.Image, x: int, y: int,
                          bead_palette: list[BeadColor], radius: int = 4,
                          top_k: int = 6) -> list[tuple[BeadColor, float]]:
    """Rank real bead colors for a clicked source-image neighbourhood.

    The representative is a robust Lab median.  It is less easily pulled across an
    outline or highlight than a plain RGB average, which is important at pixel-art
    boundaries.
    """
    if not bead_palette:
        return []
    rgb = _pil_to_rgb(image, "白色")
    h, w = rgb.shape[:2]
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    r = max(0, int(radius))
    patch = rgb[max(0, y-r):min(h, y+r+1), max(0, x-r):min(w, x+r+1)]
    patch_lab = rgb_to_lab(patch.reshape(-1, 3))
    representative = np.median(patch_lab, axis=0)
    _, palette_lab = palette_arrays(bead_palette)
    distances = delta_e_ciede2000(representative[None, :], palette_lab).reshape(-1)
    order = np.argsort(distances)[:max(1, min(int(top_k), len(bead_palette)))]
    return [(bead_palette[int(i)], float(distances[int(i)])) for i in order]
