import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def _bar_profile(h: int) -> np.ndarray:
    ys = np.arange(h, dtype=np.float32)
    profile = np.zeros((h, 1), dtype=np.float32)
    for y0 in range(60, h, 140):
        profile[:, 0] += np.exp(-((ys - y0) ** 2) / (2 * 8 * 8)) * 12
    return profile


def _bolt_centers(h: int, w_total: int) -> List[Tuple[int, int]]:
    centers = []
    for y0 in range(80, h, 200):
        for x0 in range(120, w_total, 220):
            centers.append((y0, x0))
    return centers


def _render_block(
    h: int,
    w_block: int,
    x_offset: int,
    w_total: int,
    rng: np.random.Generator,
    bar_profile: np.ndarray,
    bolt_centers: List[Tuple[int, int]],
    anomalies: List[Dict[str, int]],
) -> np.ndarray:
    ys = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xs_global = (x_offset + np.arange(w_block, dtype=np.float32))[None, :]
    xs_norm = xs_global / float(w_total)

    base = 80 + 60 * xs_norm + 40 * ys
    base += 10 * np.sin(2 * np.pi * ys * 4.0)
    base += 8 * np.sin(2 * np.pi * xs_norm * 6.0)
    base += rng.normal(0, 4.0, size=(h, w_block)).astype(np.float32)

    base += bar_profile

    ys_idx = np.arange(h)[:, None]
    xs_idx = np.arange(w_block)[None, :]
    for cy, cx in bolt_centers:
        if cx < x_offset - 40 or cx > x_offset + w_block + 40:
            continue
        local_cx = cx - x_offset
        rr = (ys_idx - cy) ** 2 + (xs_idx - local_cx) ** 2
        base += np.exp(-rr / (2 * 10 * 10)) * 25

    for a in anomalies:
        x1, y1, x2, y2 = a["x1"], a["y1"], a["x2"], a["y2"]
        ix1 = max(x1, x_offset)
        ix2 = min(x2, x_offset + w_block)
        if ix2 <= ix1:
            continue
        lx1 = ix1 - x_offset
        lx2 = ix2 - x_offset
        base[y1:y2, lx1:lx2] -= 90

    return np.clip(base, 0, 255)


def _anomaly_boxes(w_block: int) -> List[Dict[str, int]]:
    bboxes = []

    # Single-series anomaly
    s_anom = 5
    x_local, y_local = 200, 300
    w_rect, h_rect = 120, 80
    x1 = (s_anom - 1) * w_block + x_local
    y1 = y_local
    x2 = x1 + w_rect
    y2 = y1 + h_rect
    bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Boundary-spanning anomaly between series 6 and 7
    s_left = 6
    span = 100
    x1b = (s_left - 1) * w_block + (w_block - span)
    x2b = s_left * w_block + span
    y1b = 520
    y2b = y1b + 60
    bboxes.append({"x1": x1b, "y1": y1b, "x2": x2b, "y2": y2b})

    return bboxes


def _write_group(
    root: str,
    vehicle_id: int,
    camera_id: int,
    s_max: int,
    anomaly_group: Tuple[int, int],
    seed: int,
) -> Dict[str, object]:
    if camera_id == 1:
        h, w_block = 840, 1228
    else:
        h, w_block = 800, 819
    w_total = s_max * w_block

    rng = np.random.default_rng(seed + vehicle_id * 100 + camera_id * 10)
    bar_profile = _bar_profile(h)
    bolt_centers = _bolt_centers(h, w_total)

    bboxes = []
    anomalies = []
    if (vehicle_id, camera_id) == anomaly_group:
        bboxes = _anomaly_boxes(w_block)
        anomalies = bboxes

    for s in range(1, s_max + 1):
        x0 = (s - 1) * w_block
        block = _render_block(
            h=h,
            w_block=w_block,
            x_offset=x0,
            w_total=w_total,
            rng=rng,
            bar_profile=bar_profile,
            bolt_centers=bolt_centers,
            anomalies=anomalies,
        )
        name = f"{vehicle_id}_10X{camera_id}{s:03d}.jpg"
        path = os.path.join(root, name)
        Image.fromarray(block.astype(np.uint8), mode="L").save(path, quality=95)

    return {
        "vehicle_id": vehicle_id,
        "camera_id": camera_id,
        "w_block": w_block,
        "s_max": s_max,
        "w_total": w_total,
        "bboxes": bboxes,
    }


def generate_synth_data(
    root: str,
    s_max: int = 12,
    vehicles: Tuple[int, ...] = (1, 2),
    cameras: Tuple[int, ...] = (1, 2),
    runs: int = 2,
    anomaly_group: Tuple[int, int] = (1, 1),
    seed: int = 123,
) -> str:
    os.makedirs(root, exist_ok=True)

    roots_txt = os.path.join(root, "roots.txt")
    run_dirs = []
    gt = {"groups": []}

    for r in range(1, runs + 1):
        run_dir = os.path.join(root, f"run_{r:04d}")
        os.makedirs(run_dir, exist_ok=True)
        run_dirs.append(run_dir)

        for vehicle_id in vehicles:
            for camera_id in cameras:
                group = _write_group(
                    root=run_dir,
                    vehicle_id=vehicle_id,
                    camera_id=camera_id,
                    s_max=s_max,
                    anomaly_group=anomaly_group,
                    seed=seed + r * 1000,
                )
                group["dir_path"] = run_dir
                gt["groups"].append(group)

    with open(roots_txt, "w", encoding="utf-8") as f:
        for rd in run_dirs:
            f.write(rd + "\n")

    with open(os.path.join(root, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    return root


def main():
    p = argparse.ArgumentParser(description="Generate synthetic line-scan dataset")
    p.add_argument("--root", default="synthetic_root")
    p.add_argument("--series", type=int, default=12)
    p.add_argument("--runs", type=int, default=2)
    args = p.parse_args()

    generate_synth_data(root=args.root, s_max=args.series, runs=args.runs)
    print(f"Synthetic data written to {args.root}")


if __name__ == "__main__":
    main()
