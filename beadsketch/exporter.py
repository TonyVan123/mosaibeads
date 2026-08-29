from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import PatternResult


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("msyhbd.ttc" if bold else "msyh.ttc"),
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("arialbd.ttf" if bold else "arial.ttf"),
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_clean(result: PatternResult, scale: int = 20, bead_holes: bool = False) -> Image.Image:
    rgb = result.rgb
    image = Image.fromarray(rgb, "RGB").resize((result.width * scale, result.height * scale), Image.Resampling.NEAREST)
    if bead_holes and scale >= 8:
        draw = ImageDraw.Draw(image)
        radius = max(1, round(scale * 0.14))
        for y in range(result.height):
            for x in range(result.width):
                cx, cy = x * scale + scale // 2, y * scale + scale // 2
                color = tuple(int(v * 0.45) for v in rgb[y, x])
                draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=color)
    return image


def _text_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return (24, 25, 30) if lum > 150 else (250, 250, 252)


def render_chart(result: PatternResult, cell: int = 42) -> Image.Image:
    cell = max(28, int(cell))
    margin_left, margin_top = 68, 104
    legend_w = 310
    page_w = margin_left + result.width * cell + legend_w + 34
    min_h = margin_top + result.height * cell + 60
    legend_h = 155 + len(result.counts()) * 34
    page_h = max(min_h, legend_h)
    canvas = Image.new("RGB", (page_w, page_h), (249, 248, 245))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font = _font(30, True), _font(max(11, int(cell * 0.28)), True)
    small_font, legend_font = _font(14), _font(16)
    draw.text((32, 24), "MOSAIBeads 拼豆图纸", fill=(31, 35, 41), font=title_font)
    subtitle = f"{result.width} × {result.height}  ·  {result.width * result.height} 颗  ·  {len(result.counts())} 色"
    draw.text((34, 66), subtitle, fill=(93, 99, 109), font=small_font)
    rgb = result.rgb
    for y in range(result.height):
        for x in range(result.width):
            x0, y0 = margin_left + x * cell, margin_top + y * cell
            color = tuple(int(v) for v in rgb[y, x])
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color)
            code = result.palette[int(result.indices[y, x])].code
            box = draw.textbbox((0, 0), code, font=label_font)
            tw, th = box[2] - box[0], box[3] - box[1]
            if tw < cell - 3:
                draw.text((x0 + (cell - tw) / 2, y0 + (cell - th) / 2 - 1), code,
                          fill=_text_color(color), font=label_font)
    # Fine, 5-cell, and physical 29x29 board lines use three visual weights.
    for x in range(result.width + 1):
        px = margin_left + x * cell
        if x % 29 == 0:
            fill, width = (40, 48, 57), 4
        elif x % 5 == 0:
            fill, width = (74, 80, 88), 2
        else:
            fill, width = (115, 120, 126), 1
        draw.line((px, margin_top, px, margin_top + result.height * cell), fill=fill, width=width)
    for y in range(result.height + 1):
        py = margin_top + y * cell
        if y % 29 == 0:
            fill, width = (40, 48, 57), 4
        elif y % 5 == 0:
            fill, width = (74, 80, 88), 2
        else:
            fill, width = (115, 120, 126), 1
        draw.line((margin_left, py, margin_left + result.width * cell, py), fill=fill, width=width)
    for x in range(0, result.width, 5):
        draw.text((margin_left + x * cell + 4, margin_top - 24), str(x + 1), fill=(72, 77, 84), font=small_font)
    for y in range(0, result.height, 5):
        draw.text((margin_left - 38, margin_top + y * cell + 6), str(y + 1), fill=(72, 77, 84), font=small_font)

    lx = margin_left + result.width * cell + 30
    draw.text((lx, 28), "用量清单", fill=(31, 35, 41), font=_font(23, True))
    draw.text((lx, 66), "色号 / 名称 / 数量", fill=(93, 99, 109), font=small_font)
    ly = 105
    for color, count in result.counts():
        draw.rounded_rectangle((lx, ly, lx + 24, ly + 24), radius=5, fill=color.rgb, outline=(75, 78, 82))
        name = color.name if color.name != color.code else ""
        label = f"{color.code}  {name}".strip()
        if len(label) > 21:
            label = label[:20] + "…"
        draw.text((lx + 34, ly + 1), label, fill=(42, 45, 50), font=legend_font)
        count_text = str(count)
        box = draw.textbbox((0, 0), count_text, font=legend_font)
        draw.text((page_w - 36 - (box[2] - box[0]), ly + 1), count_text, fill=(42, 45, 50), font=legend_font)
        ly += 34
    return canvas


def export_bundle(result: PatternResult, folder: str | Path, source_name: str = "image") -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(source_name).stem) or "pattern"
    outputs: list[Path] = []
    clean = folder / f"{safe}_beads.png"
    render_clean(result, max(8, min(30, 1200 // max(result.width, result.height))), bead_holes=True).save(clean)
    outputs.append(clean)
    pixels = folder / f"{safe}_pixel_art.png"
    render_clean(result, max(1, min(24, 1200 // max(result.width, result.height))), bead_holes=False).save(pixels)
    outputs.append(pixels)
    chart_image = render_chart(result)
    chart = folder / f"{safe}_chart.png"
    chart_image.save(chart, optimize=True)
    outputs.append(chart)
    pdf = folder / f"{safe}_chart.pdf"
    chart_image.save(pdf, "PDF", resolution=180.0)
    outputs.append(pdf)
    materials = folder / f"{safe}_materials.csv"
    with materials.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["色号", "名称", "RGB", "数量"])
        for color, count in result.counts():
            writer.writerow([color.code, color.name, "#%02X%02X%02X" % color.rgb, count])
    outputs.append(materials)
    project = folder / f"{safe}_project.json"
    data = {
        "app": "MOSAIBeads",
        "grid": [result.width, result.height],
        "profile": result.profile,
        "metadata": result.metadata,
        "palette": [{"code": c.code, "name": c.name, "rgb": list(c.rgb)} for c in result.palette],
        "indices": result.indices.tolist(),
    }
    project.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.append(project)
    from .excel_io import export_pattern_xlsx
    workbook = folder / f"{safe}_editable.xlsx"
    export_pattern_xlsx(result, workbook)
    outputs.append(workbook)
    return outputs
