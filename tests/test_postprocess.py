import numpy as np

from src.postprocess import heatmap_to_bboxes


def test_postprocess_bboxes():
    h = np.zeros((10, 10), dtype=np.float32)
    h[1:4, 2:5] = 1.0
    h[6:9, 6:9] = 0.9

    bboxes = heatmap_to_bboxes(h, quantile=0.9, min_area=4, morph_radius=0)
    assert len(bboxes) == 2
    xs = sorted([b["x"] for b in bboxes])
    ys = sorted([b["y"] for b in bboxes])
    assert xs[0] == 2
    assert ys[0] == 1
