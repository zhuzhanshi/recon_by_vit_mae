from typing import List, Tuple

import numpy as np


def _disk_offsets(radius: int) -> List[Tuple[int, int]]:
    if radius <= 0:
        return [(0, 0)]
    r = radius
    offsets = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                offsets.append((dy, dx))
    return offsets


def _binary_erode(mask: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for y in range(h):
        for x in range(w):
            ok = True
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if yy < 0 or yy >= h or xx < 0 or xx >= w or not mask[yy, xx]:
                    ok = False
                    break
            out[y, x] = ok
    return out


def _binary_dilate(mask: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for y in range(h):
        for x in range(w):
            val = False
            for dy, dx in offsets:
                yy = y + dy
                xx = x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                    val = True
                    break
            out[y, x] = val
    return out


def _binary_opening(mask: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
    return _binary_dilate(_binary_erode(mask, offsets), offsets)


def _binary_closing(mask: np.ndarray, offsets: List[Tuple[int, int]]) -> np.ndarray:
    return _binary_erode(_binary_dilate(mask, offsets), offsets)


def _connected_components(mask: np.ndarray, intensity: np.ndarray) -> List[dict]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_y = max_y = y
            min_x = max_x = x
            area = 0
            intensity_sum = 0.0
            while stack:
                cy, cx = stack.pop()
                area += 1
                intensity_sum += float(intensity[cy, cx])
                if cy < min_y:
                    min_y = cy
                if cy > max_y:
                    max_y = cy
                if cx < min_x:
                    min_x = cx
                if cx > max_x:
                    max_x = cx
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(
                {
                    "min_row": min_y,
                    "min_col": min_x,
                    "max_row": max_y + 1,
                    "max_col": max_x + 1,
                    "area": area,
                    "mean_intensity": intensity_sum / max(area, 1),
                }
            )
    return comps


def heatmap_to_bboxes(
    h_global: np.ndarray,
    quantile: float = 0.90,
    min_area: int = 200,
    morph_radius: int = 2,
) -> List[dict]:
    """Threshold + morphology + CC to produce bboxes."""
    thresh = np.quantile(h_global, quantile)
    mask = h_global >= thresh

    if morph_radius > 0:
        offsets = _disk_offsets(morph_radius)
        mask = _binary_opening(mask, offsets)
        mask = _binary_closing(mask, offsets)

    comps = _connected_components(mask, h_global)
    bboxes = []
    for c in comps:
        if c["area"] < min_area:
            continue
        bboxes.append(
            {
                "x": int(c["min_col"]),
                "y": int(c["min_row"]),
                "w": int(c["max_col"] - c["min_col"]),
                "h": int(c["max_row"] - c["min_row"]),
                "score": float(c["mean_intensity"]),
                "area": int(c["area"]),
            }
        )
    return bboxes
