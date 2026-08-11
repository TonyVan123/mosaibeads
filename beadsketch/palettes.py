from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .color import rgb_to_lab


@dataclass(frozen=True)
class BeadColor:
    code: str
    name: str
    rgb: tuple[int, int, int]
    source: str = "BeadColors"


PALETTE_FILES = {
    "MARD 291": "mard.csv",
    "Perler": "perler.csv",
    "Hama Midi": "hama.csv",
    "Artkal S 5mm": "artkal_s.csv",
}


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def available_palettes() -> list[str]:
    return list(PALETTE_FILES)


def load_palette(name: str) -> list[BeadColor]:
    filename = PALETTE_FILES.get(name, PALETTE_FILES["MARD 291"])
    path = resource_root() / "assets" / "palettes" / filename
    colors: list[BeadColor] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 5:
                continue
            try:
                rgb = tuple(max(0, min(255, int(v))) for v in row[2:5])
            except ValueError:
                continue
            colors.append(BeadColor(row[0].strip(), row[1].strip() or row[0].strip(), rgb))
    if not colors:
        raise ValueError(f"色板为空：{path}")
    return colors


def palette_arrays(colors: list[BeadColor]) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray([c.rgb for c in colors], dtype=np.uint8)
    return rgb, rgb_to_lab(rgb)

