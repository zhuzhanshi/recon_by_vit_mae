import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from .config import ConfigView, load_config
from .indexer import build_index, read_roots_txt
from .infer_block import infer_block
from .mae import MAE, unpatchify_gray
from .model_stub import StubMAE
from .postprocess import heatmap_to_bboxes
from .stitch import stitch_blocks
from .viz import save_overlay_bboxes, save_overlay_heatmap


def _load_config(path: str) -> ConfigView:
    return ConfigView(load_config(path))


def _stitch_gray(entries, stats_key: Tuple[str, int, int]) -> np.ndarray:
    block_entries = [
        e for e in entries if (e.dir_path, e.vehicle_id, e.camera_id) == stats_key
    ]
    block_entries = sorted(block_entries, key=lambda x: x.series)
    if not block_entries:
        raise ValueError("No entries for group")
    h_block = block_entries[0].height
    w_block = block_entries[0].width
    s_max = max(e.series for e in block_entries)
    w_total = s_max * w_block
    full = np.zeros((h_block, w_total), dtype=np.float32)
    for e in block_entries:
        with Image.open(e.path) as im:
            im = im.convert("L")
            gray = np.array(im, dtype=np.float32) / 255.0
        x0 = (e.series - 1) * w_block
        full[:, x0 : x0 + w_block] = gray
    return full


def _resolve_roots(args, cfg: ConfigView) -> list:
    roots_txt = args.roots_txt or cfg.get("data.roots_txt", None)
    if roots_txt:
        return read_roots_txt(roots_txt)
    if args.root_opt or args.root_pos:
        return [args.root_opt or args.root_pos]
    raise ValueError("roots_txt or root is required")


def _allowlists(cfg: ConfigView):
    return cfg.get("data.vehicle_allowlist", None), cfg.get("data.camera_allowlist", None)


def _build_model(args, cfg: ConfigView) -> torch.nn.Module:
    device = cfg.get("infer.device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mode == "mae":
        preset = cfg.get("model.encoder_preset", "tiny")
        mae = MAE.from_preset(
            preset,
            img_size=cfg.get("infer.crop_size", 512),
            patch_size=cfg.get("model.patch_size", 16),
            mask_ratio=0.0,
        ).to(device)
        ckpt = torch.load(args.ckpt, map_location="cpu")
        mae.load_state_dict(ckpt["model"], strict=True)
        mae.eval()

        class _MaeRecon(torch.nn.Module):
            def __init__(self, model: MAE):
                super().__init__()
                self.model = model

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                pred, _ = self.model(x)
                recon = unpatchify_gray(
                    pred,
                    patch_size=self.model.patch_size,
                    h=self.model.img_size,
                    w=self.model.img_size,
                )
                return recon.clamp(0.0, 1.0)

        return _MaeRecon(mae).to(device)
    return StubMAE().to(device)


def _model_device(model: torch.nn.Module) -> torch.device:
    param = next(model.parameters(), None)
    if param is not None:
        return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_build_index(args):
    cfg = _load_config(args.config)
    roots = _resolve_roots(args, cfg)
    vehicle_allowlist, camera_allowlist = _allowlists(cfg)
    entries, stats = build_index(roots, vehicle_allowlist, camera_allowlist)
    print(f"Found {len(entries)} files")
    for key, s in stats.items():
        print(
            f"Group {key}: s_max={s.s_max}, w_block={s.w_block}, h_block={s.h_block}, w_total={s.w_total}"
        )


def cmd_demo_infer(args):
    cfg = _load_config(args.config)
    if getattr(args, "residual_mode", None):
        cfg.cfg.setdefault("postprocess", {})["residual_mode"] = args.residual_mode
    roots = _resolve_roots(args, cfg)
    vehicle_allowlist, camera_allowlist = _allowlists(cfg)
    entries, stats = build_index(roots, vehicle_allowlist, camera_allowlist)
    if not stats:
        raise ValueError("No groups found")

    if args.vehicle is not None and args.camera is not None:
        if args.dir_path is None:
            candidates = [
                k for k in stats.keys() if k[1] == args.vehicle and k[2] == args.camera
            ]
            if len(candidates) != 1:
                raise ValueError("Multiple or zero groups match; specify --dir-path")
            group_key = candidates[0]
        else:
            group_key = (args.dir_path, args.vehicle, args.camera)
            if group_key not in stats:
                raise ValueError(f"Group {group_key} not found")
    else:
        group_key = sorted(stats.keys())[0]
    group_stats = stats[group_key]

    block_entries = [
        e for e in entries if (e.dir_path, e.vehicle_id, e.camera_id) == group_key
    ]
    block_entries = sorted(block_entries, key=lambda x: x.series)

    model = _build_model(args, cfg)
    device = _model_device(model)

    block_maps: Dict[int, np.ndarray] = {}
    for e in block_entries:
        with Image.open(e.path) as im:
            im = im.convert("L")
            gray = np.array(im, dtype=np.float32) / 255.0
        block_map = infer_block(
            model,
            gray,
            series=e.series,
            stats={
                "w_total": group_stats.w_total,
                "h_block": group_stats.h_block,
                "w_block": group_stats.w_block,
            },
            crop_size=cfg.get("infer.crop_size", 512),
            stride=cfg.get("infer.stride", 256),
            residual_mode=cfg.get("postprocess.residual_mode", "gray"),
            device=str(device),
        )
        block_maps[e.series] = block_map

    h_global = stitch_blocks(
        block_maps,
        w_block=group_stats.w_block,
        h_block=group_stats.h_block,
        s_max=group_stats.s_max,
    )

    bboxes = heatmap_to_bboxes(
        h_global,
        quantile=cfg.get("postprocess.quantile", 0.995),
        min_area=cfg.get("postprocess.min_area", 200),
        morph_radius=cfg.get("postprocess.morph_radius", 2),
    )

    os.makedirs(args.output, exist_ok=True)
    np.save(os.path.join(args.output, "H_global.npy"), h_global)
    with open(os.path.join(args.output, "bboxes.json"), "w", encoding="utf-8") as f:
        json.dump(bboxes, f, indent=2)

    base_gray = _stitch_gray(entries, group_key)
    save_overlay_heatmap(
        h_global, os.path.join(args.output, "overlay_heatmap.jpg"), base_gray=base_gray
    )
    save_overlay_bboxes(
        h_global,
        bboxes,
        os.path.join(args.output, "overlay_bboxes.jpg"),
        base_gray=base_gray,
    )

    print(f"Saved outputs to {args.output}")


def cmd_dataset_check(args):
    cfg = _load_config(args.config)
    roots = _resolve_roots(args, cfg)
    vehicle_allowlist, camera_allowlist = _allowlists(cfg)
    entries, stats = build_index(roots, vehicle_allowlist, camera_allowlist)
    if not stats:
        raise ValueError("No groups found")
    print(f"Files: {len(entries)}")
    print(f"Groups: {len(stats)}")
    for key, s in stats.items():
        print(f"Group {key}: W_total={s.w_total}")


def cmd_infer(args):
    cfg = _load_config(args.config)
    if getattr(args, "residual_mode", None):
        cfg.cfg.setdefault("postprocess", {})["residual_mode"] = args.residual_mode
    roots = _resolve_roots(args, cfg)
    vehicle_allowlist, camera_allowlist = _allowlists(cfg)
    entries, stats = build_index(roots, vehicle_allowlist, camera_allowlist)
    if not stats:
        raise ValueError("No groups found")

    model = _build_model(args, cfg)
    device = _model_device(model)

    os.makedirs(args.output, exist_ok=True)
    for group_key, group_stats in stats.items():
        block_entries = [
            e for e in entries if (e.dir_path, e.vehicle_id, e.camera_id) == group_key
        ]
        block_entries = sorted(block_entries, key=lambda x: x.series)

        block_maps: Dict[int, np.ndarray] = {}
        for e in block_entries:
            with Image.open(e.path) as im:
                im = im.convert("L")
                gray = np.array(im, dtype=np.float32) / 255.0
            block_map = infer_block(
                model,
                gray,
                series=e.series,
                stats={
                    "w_total": group_stats.w_total,
                    "h_block": group_stats.h_block,
                    "w_block": group_stats.w_block,
                },
                crop_size=cfg.get("infer.crop_size", 512),
                stride=cfg.get("infer.stride", 256),
                residual_mode=cfg.get("postprocess.residual_mode", "gray"),
                device=str(device),
            )
            block_maps[e.series] = block_map

        h_global = stitch_blocks(
            block_maps,
            w_block=group_stats.w_block,
            h_block=group_stats.h_block,
            s_max=group_stats.s_max,
        )
        bboxes = heatmap_to_bboxes(
            h_global,
            quantile=cfg.get("postprocess.quantile", 0.995),
            min_area=cfg.get("postprocess.min_area", 200),
            morph_radius=cfg.get("postprocess.morph_radius", 2),
        )

        safe_dir = os.path.basename(group_key[0].rstrip(os.sep)) or "root"
        out_dir = os.path.join(
            args.output, f"{safe_dir}_v{group_key[1]}_c{group_key[2]}"
        )
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "H_global.npy"), h_global)
        with open(os.path.join(out_dir, "bboxes.json"), "w", encoding="utf-8") as f:
            json.dump(bboxes, f, indent=2)
        base_gray = _stitch_gray(entries, group_key)
        save_overlay_heatmap(
            h_global, os.path.join(out_dir, "overlay_heatmap.jpg"), base_gray=base_gray
        )
        save_overlay_bboxes(
            h_global, bboxes, os.path.join(out_dir, "overlay_bboxes.jpg"), base_gray=base_gray
        )
        print(f"Saved outputs to {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description="Unified line-scan pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("build-index", help="Scan roots and print group stats")
    p_idx.add_argument("root_pos", nargs="?", default=None)
    p_idx.add_argument("--root", dest="root_opt", default=None)
    p_idx.add_argument("--roots-txt", dest="roots_txt", default=None)
    p_idx.add_argument("--config", default="configs/default.yaml")
    p_idx.set_defaults(func=cmd_build_index)

    p_demo = sub.add_parser("demo-infer", help="Run demo inference")
    p_demo.add_argument("root_pos", nargs="?", default=None)
    p_demo.add_argument("--root", dest="root_opt", default=None)
    p_demo.add_argument("--roots-txt", dest="roots_txt", default=None)
    p_demo.add_argument("--dir-path", default=None)
    p_demo.add_argument("--vehicle", type=int, default=None)
    p_demo.add_argument("--camera", type=int, default=None)
    p_demo.add_argument("--config", default="configs/default.yaml")
    p_demo.add_argument("--output", default="outputs")
    p_demo.add_argument("--residual-mode", default=None)
    p_demo.add_argument("--mode", choices=["stub", "mae"], default="stub")
    p_demo.add_argument("--ckpt", default="outputs/mae_finetuned.pth")
    p_demo.set_defaults(func=cmd_demo_infer)

    p_check = sub.add_parser("dataset-check", help="Validate dataset and grouping")
    p_check.add_argument("root_pos", nargs="?", default=None)
    p_check.add_argument("--root", dest="root_opt", default=None)
    p_check.add_argument("--roots-txt", dest="roots_txt", default=None)
    p_check.add_argument("--config", default="configs/default.yaml")
    p_check.set_defaults(func=cmd_dataset_check)

    p_inf = sub.add_parser("infer", help="Run inference for all groups")
    p_inf.add_argument("root_pos", nargs="?", default=None)
    p_inf.add_argument("--root", dest="root_opt", default=None)
    p_inf.add_argument("--roots-txt", dest="roots_txt", default=None)
    p_inf.add_argument("--config", default="configs/default.yaml")
    p_inf.add_argument("--output", default="outputs")
    p_inf.add_argument("--residual-mode", default=None)
    p_inf.add_argument("--mode", choices=["stub", "mae"], default="stub")
    p_inf.add_argument("--ckpt", default="outputs/mae_finetuned.pth")
    p_inf.set_defaults(func=cmd_infer)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
