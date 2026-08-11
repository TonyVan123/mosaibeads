from __future__ import annotations

import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float sRGB (..., 3) to CIE L*a*b* (D65)."""
    arr = np.asarray(rgb, dtype=np.float64) / 255.0
    arr = np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    xyz = arr @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    return np.stack((116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* (..., 3) to uint8 sRGB."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.stack((fx, fy, fz), axis=-1)
    xyz = np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kappa)
    xyz *= np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    linear = xyz @ np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]], dtype=np.float64
    ).T
    linear = np.clip(linear, 0.0, 1.0)
    srgb = np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)
    return np.clip(np.rint(srgb * 255), 0, 255).astype(np.uint8)


def delta_e_ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorized CIEDE2000. Inputs broadcast on all dimensions except last."""
    x = np.asarray(lab1, dtype=np.float64)
    y = np.asarray(lab2, dtype=np.float64)
    l1, a1, b1 = np.moveaxis(x, -1, 0)
    l2, a2, b2 = np.moveaxis(y, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    cbar = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cbar ** 7 / (cbar ** 7 + 25 ** 7 + 1e-20)))
    ap1, ap2 = (1 + g) * a1, (1 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360)
    hp1 = np.where((ap1 == 0) & (b1 == 0), 0, hp1)
    hp2 = np.where((ap2 == 0) & (b2 == 0), 0, hp2)

    dl = l2 - l1
    dc = cp2 - cp1
    dh_raw = hp2 - hp1
    dh = np.where(cp1 * cp2 == 0, 0, dh_raw)
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    d_h = 2 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh / 2))

    lbar = (l1 + l2) / 2
    cpbar = (cp1 + cp2) / 2
    hp_sum = hp1 + hp2
    hpbar = np.where(cp1 * cp2 == 0, hp_sum, hp_sum / 2)
    hpbar = np.where((cp1 * cp2 != 0) & (np.abs(hp1 - hp2) > 180) & (hp_sum < 360),
                     (hp_sum + 360) / 2, hpbar)
    hpbar = np.where((cp1 * cp2 != 0) & (np.abs(hp1 - hp2) > 180) & (hp_sum >= 360),
                     (hp_sum - 360) / 2, hpbar)

    t = (1 - 0.17 * np.cos(np.radians(hpbar - 30))
         + 0.24 * np.cos(np.radians(2 * hpbar))
         + 0.32 * np.cos(np.radians(3 * hpbar + 6))
         - 0.20 * np.cos(np.radians(4 * hpbar - 63)))
    sl = 1 + 0.015 * (lbar - 50) ** 2 / np.sqrt(20 + (lbar - 50) ** 2)
    sc = 1 + 0.045 * cpbar
    sh = 1 + 0.015 * cpbar * t
    rt = (-2 * np.sqrt(cpbar ** 7 / (cpbar ** 7 + 25 ** 7 + 1e-20))
          * np.sin(np.radians(60 * np.exp(-((hpbar - 275) / 25) ** 2))))
    vl, vc, vh = dl / sl, dc / sc, d_h / sh
    return np.sqrt(np.maximum(0, vl * vl + vc * vc + vh * vh + rt * vc * vh))


def pairwise_delta_e(samples: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return delta_e_ciede2000(samples[:, None, :], palette[None, :, :])

