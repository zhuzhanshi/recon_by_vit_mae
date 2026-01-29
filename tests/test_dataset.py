import numpy as np
from PIL import Image

from src.dataset import LineScanDataset
from src.indexer import build_index


def _write_gray(path, h=819, w=800):
    arr = (np.random.rand(h, w) * 255).astype("uint8")
    Image.fromarray(arr, mode="L").save(path)


def test_dataset_crop_alignment(tmp_path):
    _write_gray(tmp_path / "1_10X1001.jpg")
    entries, stats = build_index([str(tmp_path)])

    ds = LineScanDataset(entries, stats, patch_size=16, crop_size=512)
    x, meta = ds[0]
    assert x.shape == (3, 512, 512)
    assert meta["dx"] % 16 == 0
    assert meta["dy"] % 16 == 0
