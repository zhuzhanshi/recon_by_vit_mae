import torch
from torch import nn


class StubMAE(nn.Module):
    """Stub MAE-style model that reconstructs the gray channel."""

    def __init__(self, kernel_size: int = 9):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> recon gray: (B, 1, H, W)
        gray = x[:, 0:1]
        recon = self.pool(gray)
        return recon.clamp(0.0, 1.0)
