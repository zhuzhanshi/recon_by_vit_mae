import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .indexer import GroupStats, IndexEntry
from .pos_encoding import linear_3ch


class LineScanDataset(Dataset):
    def __init__(
        self,
        entries: List[IndexEntry],
        group_stats: Dict[Tuple[int, int], GroupStats],
        patch_size: int = 16,
        crop_size: int = 512,
    ):
        self.entries = entries
        self.group_stats = group_stats
        self.patch_size = patch_size
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.entries)

    def _aligned_rand(self, max_start: int) -> int:
        if max_start <= 0:
            return 0
        aligned_max = max_start // self.patch_size
        return random.randint(0, aligned_max) * self.patch_size

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        key = (entry.dir_path, entry.vehicle_id, entry.camera_id)
        stats = self.group_stats[key]

        with Image.open(entry.path) as im:
            im = im.convert("L")
            gray = np.array(im, dtype=np.float32) / 255.0

        h_block, w_block = gray.shape
        if h_block < self.crop_size or w_block < self.crop_size:
            raise ValueError("Block smaller than crop size")

        max_dx = w_block - self.crop_size
        max_dy = h_block - self.crop_size
        dx = self._aligned_rand(max_dx)
        dy = self._aligned_rand(max_dy)

        crop = gray[dy : dy + self.crop_size, dx : dx + self.crop_size]
        gray_t = torch.from_numpy(crop).unsqueeze(0)

        x = linear_3ch(
            gray_t,
            series=entry.series,
            dx=dx,
            dy=dy,
            w_total=stats.w_total,
            h_block=stats.h_block,
            w_block=stats.w_block,
        )

        meta = {
            "dir_path": entry.dir_path,
            "path": entry.path,
            "vehicle_id": entry.vehicle_id,
            "camera_id": entry.camera_id,
            "series": entry.series,
            "dx": dx,
            "dy": dy,
            "w_total": stats.w_total,
            "h_block": stats.h_block,
            "w_block": stats.w_block,
        }
        return x, meta
