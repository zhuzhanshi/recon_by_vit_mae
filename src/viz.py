from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw


def _normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx - mn < 1e-6:
        return np.zeros_like(a)
    return (a - mn) / (mx - mn)


def save_overlay_heatmap(
    h_global: np.ndarray,
    out_path: str,
    base_gray: Optional[np.ndarray] = None,
):
    h, w = h_global.shape
    if base_gray is None:
        base_gray = np.zeros((h, w), dtype=np.float32)
    base = _normalize(base_gray)
    heat = _normalize(h_global)

    rgb = np.stack([base, base, base], axis=-1)
    rgb[..., 0] = np.maximum(rgb[..., 0], heat)
    img = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(out_path)


def save_overlay_bboxes(
    h_global: np.ndarray,
    bboxes: List[dict],
    out_path: str,
    base_gray: Optional[np.ndarray] = None,
):
    h, w = h_global.shape
    if base_gray is None:
        base_gray = np.zeros((h, w), dtype=np.float32)
    base = _normalize(base_gray)
    rgb = np.stack([base, base, base], axis=-1)
    img = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for b in bboxes:
        x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
        draw.rectangle([x, y, x + bw, y + bh], outline=(255, 0, 0), width=2)
    pil.save(out_path)
