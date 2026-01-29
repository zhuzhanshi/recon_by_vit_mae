import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ConfigView, load_config, resolve_split_roots
from src.indexer import build_index, read_roots_txt
from src.mae import MAE, unpatchify_gray
from src.pos_encoding import linear_3ch


def _load_gt(roots_txt: str | None, root: str | None) -> Dict[str, dict]:
    if roots_txt:
        base = os.path.dirname(os.path.abspath(roots_txt))
        path = os.path.join(base, "gt.json")
    elif root:
        path = os.path.join(root, "gt.json")
    else:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("groups", {})


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


class MaePretrainDataset(Dataset):
    def __init__(
        self,
        entries,
        group_stats,
        gt_groups: Dict[str, dict],
        patch_size: int = 16,
        crop_size: int = 512,
        max_tries: int = 50,
    ):
        self.entries = entries
        self.group_stats = group_stats
        self.gt_groups = gt_groups
        self.patch_size = patch_size
        self.crop_size = crop_size
        self.max_tries = max_tries

    def __len__(self) -> int:
        return len(self.entries)

    def _aligned_rand(self, max_start: int) -> int:
        if max_start <= 0:
            return 0
        aligned_max = max_start // self.patch_size
        return np.random.randint(0, aligned_max + 1) * self.patch_size

    def _get_bboxes(self, entry) -> List[dict]:
        if not self.gt_groups:
            return []
        if isinstance(self.gt_groups, list):
            for g in self.gt_groups:
                if (
                    g.get("dir_path") == entry.dir_path
                    and g.get("vehicle_id") == entry.vehicle_id
                    and g.get("camera_id") == entry.camera_id
                ):
                    return g.get("bboxes", [])
            return []
        key = f"{entry.vehicle_id}_{entry.camera_id}"
        return self.gt_groups.get(key, {}).get("bboxes", [])

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        key = (entry.dir_path, entry.vehicle_id, entry.camera_id)
        stats = self.group_stats[key]

        bboxes = self._get_bboxes(entry)

        with Image.open(entry.path) as im:
            im = im.convert("L")
            gray = np.array(im, dtype=np.float32) / 255.0

        h_block, w_block = gray.shape
        if h_block < self.crop_size or w_block < self.crop_size:
            raise ValueError("Block smaller than crop size")

        max_dx = w_block - self.crop_size
        max_dy = h_block - self.crop_size

        dx = dy = 0
        for _ in range(self.max_tries):
            dx = int(self._aligned_rand(max_dx))
            dy = int(self._aligned_rand(max_dy))
            x0_block = (entry.series - 1) * w_block
            crop_box = (x0_block + dx, dy, x0_block + dx + self.crop_size, dy + self.crop_size)
            if not any(
                _overlaps(
                    crop_box,
                    (b["x1"], b["y1"], b["x2"], b["y2"]),
                )
                for b in bboxes
            ):
                break

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
        return x


def _resolve_roots(args, cfg: ConfigView) -> Tuple[list, list]:
    train_txt, val_txt, _ = resolve_split_roots(args, cfg)
    train_roots = read_roots_txt(train_txt) if train_txt else [args.root]
    val_roots = read_roots_txt(val_txt) if val_txt else []
    return train_roots, val_roots


def _build_loader(entries, stats, gt_groups, cfg: ConfigView, batch_size: int, shuffle: bool):
    dataset = MaePretrainDataset(
        entries=entries,
        group_stats=stats,
        gt_groups=gt_groups,
        patch_size=cfg.get("model.patch_size", 16),
        crop_size=cfg.get("infer.crop_size", 512),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.get("train.num_workers", 0),
    )
    return loader


def _eval_epoch(model: MAE, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            pred, mask = model(x)
            loss = model.forward_loss(x, pred, mask)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def _residual_quantiles(
    model: MAE,
    loader: DataLoader,
    device: str,
    quantiles: List[float],
    residual_mode: str,
    max_pixels_per_batch: int = 200000,
) -> Dict[str, float]:
    # For efficiency, subsample residual pixels per batch when large.
    if residual_mode != "gray":
        raise ValueError(f"Unsupported residual_mode for logging: {residual_mode}")

    model.eval()
    samples = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            pred, _ = model(x)
            recon = unpatchify_gray(pred, model.patch_size, model.img_size, model.img_size)
            resid = (x[:, 0:1] - recon).abs().flatten()
            if resid.numel() > max_pixels_per_batch:
                idx = torch.randperm(resid.numel(), device=resid.device)[:max_pixels_per_batch]
                resid = resid[idx]
            samples.append(resid.cpu().numpy())

    if not samples:
        return {}
    values = np.concatenate(samples, axis=0)
    return {f"residual_q{int(q * 1000)}": float(np.quantile(values, q)) for q in quantiles}


def _save_viz(
    model: MAE,
    loader: DataLoader,
    device: str,
    out_dir: str,
    epoch: int,
    viz_samples: int,
    writer: SummaryWriter,
):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    samples = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            pred, _ = model(x)
            recon = unpatchify_gray(pred, model.patch_size, model.img_size, model.img_size)
            gray = x[:, 0:1]
            resid = (gray - recon).abs()
            for i in range(x.shape[0]):
                if len(samples) >= viz_samples:
                    break
                row = torch.cat([gray[i], recon[i], resid[i]], dim=2)
                samples.append(row)
            if len(samples) >= viz_samples:
                break

    if not samples:
        return

    grid = make_grid(samples, nrow=1, normalize=True)
    grid_path = os.path.join(out_dir, f"epoch_{epoch:04d}_grid.jpg")
    grid_img = (grid.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(grid_img.squeeze()).save(grid_path)
    writer.add_image("val/recon_grid", grid, epoch)


def _extract_state_dict(ckpt: dict) -> dict:
    for key in ("model", "state_dict", "encoder"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


def _strip_prefix(state: dict, prefix: str) -> dict:
    if not prefix:
        return state
    return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}


def _interpolate_pos_embed(pos_embed: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    # pos_embed: (1, N, C) or (N, C); may include extra tokens.
    pos = pos_embed if pos_embed.dim() == 3 else pos_embed.unsqueeze(0)
    n = pos.shape[1]
    c = pos.shape[2]

    chosen_extra = None
    gs = None
    for extra_tokens in (1, 2, 0):
        p = n - extra_tokens
        if p <= 0:
            continue
        s = int(p**0.5)
        if s * s == p:
            chosen_extra = extra_tokens
            gs = s
            break

    print(
        f"[pos_embed] shape={tuple(pos.shape)} N={n} extra_tokens={chosen_extra} gs={gs} target=({grid_h},{grid_w})"
    )

    if chosen_extra is None or gs is None:
        raise ValueError(
            f"Cannot infer grid size for pos_embed interpolation: shape={tuple(pos.shape)} N={n}"
        )

    special = pos[:, :chosen_extra, :]
    patch = pos[:, chosen_extra:, :]
    patch = patch.reshape(1, gs, gs, c).permute(0, 3, 1, 2)
    patch = torch.nn.functional.interpolate(
        patch, size=(grid_h, grid_w), mode="bicubic", align_corners=False
    )
    patch = patch.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, c)
    out = torch.cat([special, patch], dim=1)
    return out


def _adapt_patch_embed(weight: torch.Tensor, patch_size: int, in_chans: int) -> torch.Tensor:
    if weight.ndim == 4:
        # Conv patch embed: (embed_dim, in_chans, p, p)
        src_chans = weight.shape[1]
        if src_chans == in_chans:
            return weight
        if src_chans > in_chans:
            return weight[:, :in_chans, :, :]
        reps = (in_chans + src_chans - 1) // src_chans
        w = weight.repeat(1, reps, 1, 1)[:, :in_chans, :, :]
        return w

    # Linear patch embed: (embed_dim, p*p*in_chans)
    target_dim = patch_size * patch_size * in_chans
    if weight.shape[1] == target_dim:
        return weight
    if weight.shape[1] % (patch_size * patch_size) != 0:
        raise ValueError("patch_embed shape mismatch and cannot infer in_chans")
    src_chans = weight.shape[1] // (patch_size * patch_size)
    w = weight.reshape(weight.shape[0], patch_size * patch_size, src_chans)
    if src_chans == in_chans:
        return weight
    if src_chans > in_chans:
        w = w[:, :, :in_chans]
    else:
        reps = (in_chans + src_chans - 1) // src_chans
        w = w.repeat(1, 1, reps)[:, :, :in_chans]
    return w.reshape(weight.shape[0], target_dim)


def _load_pretrained_encoder(
    model: MAE, cfg: ConfigView, exp_dir: str, writer: SummaryWriter, preset: str
):
    init_from_pretrained = cfg.get("mae.init_from_pretrained", False)
    report = {
        "init_from_pretrained": bool(init_from_pretrained),
        "pretrained_ckpt": cfg.get("mae.pretrained_ckpt", None),
        "load_encoder_only": cfg.get("mae.load_encoder_only", True),
        "pos_embed_strategy": cfg.get("mae.pos_embed_strategy", "interpolate"),
        "patch_embed_strategy": cfg.get("mae.patch_embed_strategy", "strict"),
        "resolved_encoder_preset": preset,
        "encoder_dims": {
            "embed_dim": model.embed_dim,
            "depth": model.depth,
            "heads": model.num_heads,
        },
    }
    if not init_from_pretrained:
        report["note"] = "init_from_pretrained=false; using random init"
        report_path = os.path.join(exp_dir, "init_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        writer.add_text("init/report", json.dumps(report, indent=2), 0)
        print(report)
        return

    ckpt_path = cfg.get("mae.pretrained_ckpt", None)
    if not ckpt_path:
        raise ValueError("mae.pretrained_ckpt is required when init_from_pretrained=true")

    raw = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(raw)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Prefer encoder-only keys if present
    encoder_state = {}
    for k, v in state.items():
        if k.startswith("encoder."):
            encoder_state[k] = v
        elif k.startswith("model.encoder."):
            encoder_state[k[len("model.") :]] = v
        elif k.startswith("patch_embed.") or k.startswith("pos_embed") or k.startswith("cls_token"):
            encoder_state[k] = v
        elif k.startswith("model.patch_embed.") or k.startswith("model.pos_embed") or k.startswith("model.cls_token"):
            encoder_state[k[len("model.") :]] = v
        elif k.startswith("blocks.") or k.startswith("norm."):
            encoder_state[k] = v

    if not encoder_state:
        encoder_state = state

    # Handle pos_embed
    pos_strategy = cfg.get("mae.pos_embed_strategy", "interpolate")
    if "pos_embed" in encoder_state:
        pos = encoder_state["pos_embed"]
        target_n = model.num_patches
        pos_n = pos.shape[1] if pos.dim() == 3 else pos.shape[0]
        try:
            if pos_n != target_n:
                if pos_strategy == "interpolate":
                    encoder_state["pos_embed"] = _interpolate_pos_embed(
                        pos, model.grid_h, model.grid_w
                    )
                    report["pos_embed_action"] = "interpolated"
                elif pos_strategy == "skip":
                    encoder_state.pop("pos_embed", None)
                    report["pos_embed_action"] = "skipped"
                else:
                    raise ValueError(f"Unknown pos_embed_strategy: {pos_strategy}")
            else:
                report["pos_embed_action"] = "loaded"
        except ValueError as e:
            if pos_strategy == "skip":
                print(f"[pos_embed] skip due to error: {e}")
                encoder_state.pop("pos_embed", None)
                report["pos_embed_action"] = "skipped_on_error"
            else:
                raise

    # Handle patch_embed
    patch_strategy = cfg.get("mae.patch_embed_strategy", "strict")
    if "patch_embed.proj.weight" in encoder_state and "patch_embed.weight" not in encoder_state:
        encoder_state["patch_embed.proj.weight"] = encoder_state["patch_embed.proj.weight"]
    if "patch_embed.proj.weight" in encoder_state:
        w = encoder_state["patch_embed.proj.weight"]
        if w.shape != model.patch_embed.proj.weight.shape:
            if patch_strategy == "adapt_in_chans":
                encoder_state["patch_embed.proj.weight"] = _adapt_patch_embed(
                    w, model.patch_size, 3
                )
                report["patch_embed_action"] = "adapted_in_chans"
            else:
                raise ValueError("patch_embed weight shape mismatch (strict)")

    # Filter to encoder-only if configured
    if cfg.get("mae.load_encoder_only", True):
        filtered = {}
        for k, v in encoder_state.items():
            if k.startswith("encoder.") or k.startswith("patch_embed.") or k.startswith("pos_embed") or k.startswith("cls_token") or k.startswith("blocks.") or k.startswith("norm."):
                filtered[k] = v
        encoder_state = filtered

    model_state = model.state_dict()
    filtered = {}
    skipped_mismatch = []
    for k, v in encoder_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
        elif k in model_state:
            skipped_mismatch.append(k)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    report["num_loaded_keys"] = len(filtered)
    report["skipped_shape_mismatch"] = len(skipped_mismatch)
    report["skipped_shape_mismatch_sample"] = skipped_mismatch[:20]
    report["num_missing_keys"] = len(missing)
    report["num_unexpected_keys"] = len(unexpected)
    report["missing_keys_sample"] = missing[:10]
    report["unexpected_keys_sample"] = unexpected[:10]

    # Critical check: require some encoder attention and MLP weights loaded.
    if preset == "vitb":
        critical_ok = any("blocks.0.attn.qkv" in k for k in filtered.keys()) and any(
            "blocks.0.mlp.fc1" in k for k in filtered.keys()
        )
    else:
        critical_ok = any("encoder.layers.0.self_attn" in k for k in filtered.keys()) and any(
            "encoder.layers.0.linear1" in k for k in filtered.keys()
        )
    if not critical_ok:
        raise RuntimeError("Critical encoder weights missing after pretrained init")

    # Sanity check vs fresh init
    fresh = MAE.from_preset(
        preset,
        img_size=model.img_size,
        patch_size=model.patch_size,
        mask_ratio=cfg.get("mae.mask_ratio_pretrain", 0.75),
    )
    norms = {}
    if preset == "vitb":
        names = ("patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "blocks.0.mlp.fc1.weight")
    else:
        names = ("patch_embed.proj.weight", "encoder.layers.0.self_attn.in_proj_weight", "encoder.layers.0.linear1.weight")
    for name in names:
        if name in model.state_dict() and name in fresh.state_dict():
            n_loaded = torch.norm(model.state_dict()[name]).item()
            n_fresh = torch.norm(fresh.state_dict()[name]).item()
            norms[name] = {"loaded": n_loaded, "fresh": n_fresh}
    report["encoder_norms"] = norms

    report_path = os.path.join(exp_dir, "init_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    writer.add_text("init/report", json.dumps(report, indent=2), 0)
    print(report)


def train(args):
    cfg = ConfigView(load_config(args.config))

    vehicle_allowlist = args.vehicle_allowlist
    camera_allowlist = args.camera_allowlist
    if vehicle_allowlist is None:
        vehicle_allowlist = cfg.get("data.vehicle_allowlist", None)
    if camera_allowlist is None:
        camera_allowlist = cfg.get("data.camera_allowlist", None)

    train_roots, val_roots = _resolve_roots(args, cfg)
    entries, stats = build_index(train_roots, vehicle_allowlist, camera_allowlist)
    gt_groups = _load_gt(args.roots_txt, args.root)

    batch_size = args.batch_size or cfg.get("train.batch_size", 8)
    train_loader = _build_loader(entries, stats, gt_groups, cfg, batch_size, True)

    val_loader = None
    if val_roots:
        val_entries, val_stats = build_index(val_roots, vehicle_allowlist, camera_allowlist)
        val_loader = _build_loader(val_entries, val_stats, gt_groups, cfg, batch_size, False)

    seed = cfg.get("train.seed", None)
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = cfg.get("model.encoder_preset", "tiny")
    init_pretrained = cfg.get("mae.init_from_pretrained", False)
    if init_pretrained and preset != "vitb":
        print("[preset] init_from_pretrained=true -> forcing encoder_preset=vitb")
        preset = "vitb"
    model = MAE.from_preset(
        preset,
        img_size=cfg.get("infer.crop_size", 512),
        patch_size=cfg.get("model.patch_size", 16),
        mask_ratio=cfg.get("mae.mask_ratio_pretrain", 0.75),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr or cfg.get("train.lr", 1e-4),
        weight_decay=cfg.get("train.weight_decay", 0.05),
    )

    output_dir = args.output or cfg.get("train.output_dir", "outputs")
    base_exp_name = cfg.get("train.exp_name", "mae_pretrain")
    use_suffix = cfg.get("train.exp_name_time_suffix", False)
    time_fmt = cfg.get("train.exp_name_time_format", "%Y%m%d_%H%M%S")
    resolved_exp_name = (
        f"{base_exp_name}_{datetime.now().strftime(time_fmt)}" if use_suffix else base_exp_name
    )
    exp_dir = os.path.join(output_dir, resolved_exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "resolved_exp_name.txt"), "w", encoding="utf-8") as f:
        f.write(f"base_exp_name: {base_exp_name}\\n")
        f.write(f"resolved_exp_name: {resolved_exp_name}\\n")

    log_dir = cfg.get("train.log_dir", "runs")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, resolved_exp_name))

    _load_pretrained_encoder(model, cfg, exp_dir, writer, preset)

    epochs = args.epochs or cfg.get("train.epochs", 10)
    eval_interval = cfg.get("train.eval_interval", 1)
    viz_interval = cfg.get("train.viz_interval", 1)
    viz_samples = cfg.get("train.viz_samples", 4)
    save_interval = cfg.get("train.save_interval", 1)
    save_best = cfg.get("train.save_best", True)
    log_residual_q = cfg.get("postprocess.log_residual_quantiles", False)
    quantiles = cfg.get("postprocess.quantile_candidates", [0.95, 0.99, 0.995])
    residual_mode = cfg.get("postprocess.residual_mode", "gray")

    best_val = None
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total = 0.0
        count = 0
        for x in pbar:
            x = x.to(device)
            pred, mask = model(x)
            loss = model.forward_loss(x, pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
            pbar.set_postfix({"loss": total / max(count, 1)})

        train_loss = total / max(count, 1)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        val_loss = None
        if val_loader and epoch % eval_interval == 0:
            val_loss = _eval_epoch(model, val_loader, device)
            writer.add_scalar("val/loss", val_loss, epoch)
            if log_residual_q:
                q_stats = _residual_quantiles(
                    model, val_loader, device, quantiles, residual_mode
                )
                for key, value in q_stats.items():
                    writer.add_scalar(f"val/{key}", value, epoch)
                stats_dir = os.path.join(exp_dir, "val_stats")
                os.makedirs(stats_dir, exist_ok=True)
                stats_path = os.path.join(stats_dir, f"epoch_{epoch:04d}_residual_quantiles.json")
                with open(stats_path, "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch, **q_stats}, f, indent=2)

        if val_loader and epoch % viz_interval == 0:
            viz_dir = os.path.join(exp_dir, "val_viz")
            _save_viz(model, val_loader, device, viz_dir, epoch, viz_samples, writer)

        if epoch % save_interval == 0:
            torch.save({"model": model.state_dict(), "epochs": epoch}, os.path.join(exp_dir, "last.pth"))

        if save_best and val_loss is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "epochs": epoch}, os.path.join(exp_dir, "best.pth"))

    out_path = os.path.join(output_dir, "mae_pretrained.pth")
    torch.save({"model": model.state_dict(), "epochs": epochs}, out_path)
    print(f"Saved checkpoint to {out_path}")


def main():
    p = argparse.ArgumentParser(description="MAE pretraining on synthetic data")
    p.add_argument("--root", default="synthetic_root/run_0001")
    p.add_argument("--roots-txt", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--vehicle-allowlist", nargs="+", type=int, default=None)
    p.add_argument("--camera-allowlist", nargs="+", type=int, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
