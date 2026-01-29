import json
import os
import subprocess
import sys

from scripts.gen_synth_data import generate_synth_data


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def main():
    root = "synthetic_root"
    outputs = "outputs"

    generate_synth_data(root=root, s_max=12, runs=2)
    roots_txt = os.path.join(root, "roots.txt")

    cmd = [
        sys.executable,
        "-m",
        "src.runner",
        "demo-infer",
        "--roots-txt",
        roots_txt,
        "--dir-path",
        os.path.join(root, "run_0001"),
        "--vehicle",
        "1",
        "--camera",
        "1",
        "--output",
        outputs,
    ]
    subprocess.run(cmd, check=True)

    with open(os.path.join(root, "gt.json"), "r", encoding="utf-8") as f:
        gt = json.load(f)
    with open(os.path.join(outputs, "bboxes.json"), "r", encoding="utf-8") as f:
        preds = json.load(f)

    gt_boxes = []
    for g in gt["groups"]:
        if g["dir_path"].endswith("run_0001") and g["vehicle_id"] == 1 and g["camera_id"] == 1:
            gt_boxes = g["bboxes"]
            break

    pred_boxes = [
        (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]) for b in preds
    ]

    if not pred_boxes:
        print("No predicted boxes")
        return

    for i, g in enumerate(gt_boxes):
        g_box = (g["x1"], g["y1"], g["x2"], g["y2"])
        best = 0.0
        for p in pred_boxes:
            best = max(best, _iou(g_box, p))
        print(f"GT box {i}: best IoU={best:.3f}")


if __name__ == "__main__":
    main()
