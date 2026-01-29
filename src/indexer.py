import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

from .parsing import parse_filename


@dataclass
class IndexEntry:
    dir_path: str
    path: str
    vehicle_id: int
    camera_id: int
    series: int
    width: int
    height: int


@dataclass
class GroupStats:
    dir_path: str
    vehicle_id: int
    camera_id: int
    s_max: int
    w_block: int
    h_block: int
    w_total: int


def scan_root(
    root: str,
    vehicle_allowlist: Optional[List[int]] = None,
    camera_allowlist: Optional[List[int]] = None,
) -> List[IndexEntry]:
    entries: List[IndexEntry] = []
    for name in sorted(os.listdir(root)):
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(root, name)
        vehicle_id, camera_id, series = parse_filename(name)
        if vehicle_allowlist is not None and vehicle_id not in vehicle_allowlist:
            continue
        if camera_allowlist is not None and camera_id not in camera_allowlist:
            continue
        with Image.open(path) as im:
            width, height = im.size
        entries.append(
            IndexEntry(
                dir_path=root,
                path=path,
                vehicle_id=vehicle_id,
                camera_id=camera_id,
                series=series,
                width=width,
                height=height,
            )
        )
    return entries


def build_group_stats(entries: List[IndexEntry]) -> Dict[Tuple[str, int, int], GroupStats]:
    grouped: Dict[Tuple[str, int, int], List[IndexEntry]] = defaultdict(list)
    for e in entries:
        grouped[(e.dir_path, e.vehicle_id, e.camera_id)].append(e)

    stats: Dict[Tuple[str, int, int], GroupStats] = {}
    for key, items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x.series)
        s_max = max(i.series for i in items_sorted)
        w_block = items_sorted[0].width
        h_block = items_sorted[0].height
        for i in items_sorted[1:]:
            if i.width != w_block:
                raise ValueError(f"Width mismatch for group {key}")
            if i.height != h_block:
                raise ValueError(f"Height mismatch for group {key}")
        w_total = s_max * w_block
        stats[key] = GroupStats(
            dir_path=key[0],
            vehicle_id=key[1],
            camera_id=key[2],
            s_max=s_max,
            w_block=w_block,
            h_block=h_block,
            w_total=w_total,
        )
    return stats


def read_roots_txt(roots_txt: str) -> List[str]:
    roots: List[str] = []
    with open(roots_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            roots.append(line)
    return roots


def build_index(
    roots: Iterable[str],
    vehicle_allowlist: Optional[List[int]] = None,
    camera_allowlist: Optional[List[int]] = None,
):
    entries: List[IndexEntry] = []
    for root in roots:
        entries.extend(scan_root(root, vehicle_allowlist, camera_allowlist))
    stats = build_group_stats(entries)
    return entries, stats
