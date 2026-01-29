from typing import Dict, Tuple

import numpy as np


def stitch_blocks(
    block_maps: Dict[int, np.ndarray],
    w_block: int,
    h_block: int,
    s_max: int,
) -> np.ndarray:
    """Stitch per-series block maps into full-vehicle heatmap."""
    w_total = s_max * w_block
    h_global = np.zeros((h_block, w_total), dtype=np.float32)
    for series, block in block_maps.items():
        if block.shape != (h_block, w_block):
            raise ValueError("Block shape mismatch")
        x0 = (series - 1) * w_block
        h_global[:, x0 : x0 + w_block] = block
    return h_global
