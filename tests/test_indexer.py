import numpy as np
from PIL import Image

from src.indexer import build_index


def _write_gray(path, h=819, w=800):
    arr = (np.random.rand(h, w) * 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def test_build_index(tmp_path):
    _write_gray(tmp_path / "1_10X1001.jpg")
    _write_gray(tmp_path / "1_10X1002.jpg")

    entries, stats = build_index([str(tmp_path)])
    assert len(entries) == 2
    key = (str(tmp_path), 1, 1)
    assert key in stats
    s = stats[key]
    assert s.s_max == 2
    assert s.w_total == s.w_block * 2
