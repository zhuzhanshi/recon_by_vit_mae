from typing import Dict

import numpy as np
import torch

from .pos_encoding import linear_3ch


def _weight_window(h: int, w: int) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False)
    wx = torch.hann_window(w, periodic=False)
    w2d = wy[:, None] * wx[None, :]
    return w2d


def infer_block(
    model: torch.nn.Module,
    gray_block: np.ndarray,
    series: int,
    stats: Dict[str, int],
    crop_size: int = 512,
    stride: int = 256,
    residual_mode: str = "gray",
    device: str = "cpu",
) -> np.ndarray:
    """Sliding-window inference for one block with weighted fusion."""
    h_block, w_block = gray_block.shape
    model.eval()

    weight = _weight_window(crop_size, crop_size).to(device)
    acc = torch.zeros((h_block, w_block), device=device)
    acc_w = torch.zeros((h_block, w_block), device=device)

    for y in range(0, h_block - crop_size + 1, stride):
        for x in range(0, w_block - crop_size + 1, stride):
            crop = gray_block[y : y + crop_size, x : x + crop_size]
            gray_t = torch.from_numpy(crop).float().unsqueeze(0)
            x_enc = linear_3ch(
                gray_t,
                series=series,
                dx=x,
                dy=y,
                w_total=stats["w_total"],
                h_block=stats["h_block"],
                w_block=stats["w_block"],
            ).unsqueeze(0)

            with torch.no_grad():
                recon = model(x_enc.to(device))

            if residual_mode == "gray":
                recon_gray = recon[:, 0:1]
                heat = (gray_t.unsqueeze(0).to(device) - recon_gray).abs()[0, 0]
            elif residual_mode == "feature":
                raise NotImplementedError("feature residual placeholder")
            else:
                raise ValueError(f"Unknown residual_mode: {residual_mode}")

            acc[y : y + crop_size, x : x + crop_size] += heat * weight
            acc_w[y : y + crop_size, x : x + crop_size] += weight

    acc_w = torch.clamp(acc_w, min=1e-6)
    fused = (acc / acc_w).clamp(0.0, 1.0)
    return fused.cpu().numpy()
