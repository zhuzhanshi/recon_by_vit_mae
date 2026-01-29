import argparse
import os
import sys
from datetime import datetime

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.train_mae import MaePretrainDataset, _load_gt
from src.config import ConfigView, load_config, resolve_split_roots
from src.indexer import build_index, read_roots_txt
from src.mae import MAE, unpatchify_gray


def _freeze_encoder(model: MAE):
    for p in model.patch_embed.parameters():
        p.requires_grad = False
    for p in model.encoder.parameters():
        p.requires_grad = False


def _assert_frozen_encoder(model: MAE):
    for name, p in model.named_parameters():
        if name.startswith("patch_embed") or name.startswith("encoder"):
            if p.requires_grad:
                raise AssertionError(f"Encoder param not frozen: {name}")


def _resolve_roots(args, cfg: ConfigView) -> tuple[list, list]:
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
    quantiles: list[float],
    residual_mode: str,
    max_pixels_per_batch: int = 200000,
) -> dict[str, float]:
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
    from PIL import Image

    Image.fromarray(grid_img.squeeze()).save(grid_path)
    writer.add_image("val/recon_grid", grid, epoch)


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
    model = MAE.from_preset(
        preset,
        img_size=cfg.get("infer.crop_size", 512),
        patch_size=cfg.get("model.patch_size", 16),
        mask_ratio=cfg.get("mae.mask_ratio_finetune", 0.15),
    ).to(device)
    ckpt = torch.load(args.pretrained, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    if cfg.get("finetune.freeze_encoder", True):
        _freeze_encoder(model)
        _assert_frozen_encoder(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr or cfg.get("train.lr", 1e-4),
        weight_decay=cfg.get("train.weight_decay", 0.05),
    )

    output_dir = args.output or cfg.get("train.output_dir", "outputs")
    base_exp_name = cfg.get("train.exp_name", "mae_finetune")
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

    for name, p in model.named_parameters():
        if name.startswith("patch_embed") or name.startswith("encoder"):
            if p.grad is not None and torch.any(p.grad != 0):
                raise AssertionError(f"Encoder received gradient: {name}")

    out_path = os.path.join(output_dir, "mae_finetuned.pth")
    torch.save({"model": model.state_dict(), "epochs": epochs}, out_path)
    print(f"Saved checkpoint to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune MAE decoder only")
    p.add_argument("--root", default="synthetic_root/run_0001")
    p.add_argument("--roots-txt", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pretrained", default="outputs/mae_pretrained.pth")
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
