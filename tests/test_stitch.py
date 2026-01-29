import numpy as np

from src.stitch import stitch_blocks


def test_stitch_blocks():
    b1 = np.ones((4, 3), dtype=np.float32)
    b2 = np.zeros((4, 3), dtype=np.float32)
    h = stitch_blocks({1: b1, 2: b2}, w_block=3, h_block=4, s_max=2)
    assert h.shape == (4, 6)
    assert np.allclose(h[:, :3], 1.0)
    assert np.allclose(h[:, 3:], 0.0)
