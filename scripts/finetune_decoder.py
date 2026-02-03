import argparse
import json
import os
import sys
from datetime import datetime

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
from src.mae import MAE


class CNNDecoder(torch.nn.Module):
    """Simple, stable decoder: one upsample + small conv stack."""

    def __init__(self, in_chans: int, out_chans: int = 1):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(in_chans, 32, 1)
        # self.conv1 = torch.nn.Conv2d(32, 32, 3, padding=1)
        # self.conv2 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.out = torch.nn.Conv2d(32, out_chans, 3, padding=1)
        self.act = torch.nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        x = self.conv0(x)
        x = torch.nn.functional.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        # x = self.act(self.conv1(x))
        # x = self.act(self.conv2(x))
        return self.out(x)


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


def _ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    filt = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size * window_size)
    mu_x = torch.nn.functional.conv2d(x, filt, padding=pad)
    mu_y = torch.nn.functional.conv2d(y, filt, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = torch.nn.functional.conv2d(x * x, filt, padding=pad) - mu_x2
    sigma_y2 = torch.nn.functional.conv2d(y * y, filt, padding=pad) - mu_y2
    sigma_xy = torch.nn.functional.conv2d(x * y, filt, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return ssim_map.mean()


def _recon_loss(recon: torch.Tensor, target: torch.Tensor, alpha: float, ssim_window: int) -> torch.Tensor:
    pixel = (recon - target).abs().mean()
    ssim_val = _ssim(recon, target, window_size=ssim_window)
    return alpha * pixel + (1 - alpha) * (1 - ssim_val)


def _eval_epoch(model: MAE, decoder: CNNDecoder, loader: DataLoader, device: str, target_hw) -> float:
    model.eval()
    decoder.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            feat = model.encode_tokens(x)
            b, n, c = feat.shape
            h = int(model.grid_h)
            w = int(model.grid_w)
            feat_map = feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
            recon = decoder(feat_map, target_hw)
            gt = torch.nn.functional.interpolate(x[:, 0:1], size=target_hw, mode="bilinear", align_corners=False)
            loss = _recon_loss(recon, gt, model.loss_alpha, model.loss_ssim_window)
            total += loss.item()
            count += 1
    return total / max(count, 1)


def _save_viz(model: MAE, decoder: CNNDecoder, loader: DataLoader, device: str, out_dir: str, epoch: int, viz_samples: int, writer: SummaryWriter, target_hw):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    decoder.eval()
    samples = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            feat = model.encode_tokens(x)
            b, n, c = feat.shape
            h = int(model.grid_h)
            w = int(model.grid_w)
            feat_map = feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
            recon_low = decoder(feat_map, target_hw)
            recon = torch.nn.functional.interpolate(recon_low, size=x.shape[-2:], mode="bilinear", align_corners=False)
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

    train_roots, val_roots = _resolve_roots(args, cfg)
    entries, stats = build_index(
        train_roots,
        cfg.get("data.vehicle_allowlist", None),
        cfg.get("data.camera_allowlist", None),
    )
    gt_groups = _load_gt(args.roots_txt, args.root)

    batch_size = args.batch_size or cfg.get("train.batch_size", 8)
    train_loader = _build_loader(entries, stats, gt_groups, cfg, batch_size, True)

    val_loader = None
    if val_roots:
        val_entries, val_stats = build_index(
            val_roots,
            cfg.get("data.vehicle_allowlist", None),
            cfg.get("data.camera_allowlist", None),
        )
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
        mask_ratio=0.0,
    ).to(device)

    ckpt = torch.load(args.pretrained, map_location="cpu")
    encoder_state = ckpt.get("model", ckpt)
    model.load_state_dict(encoder_state, strict=False)

    if cfg.get("finetune.freeze_encoder", True):
        for p in model.parameters():
            p.requires_grad = False

    recon_scale = cfg.get("finetune.recon_scale", 0.5)
    if recon_scale not in (0.5, 0.25):
        recon_scale = 0.5
    target_hw = (int(model.img_size * recon_scale), int(model.img_size * recon_scale))

    decoder = CNNDecoder(model.embed_dim).to(device)

    model.loss_alpha = cfg.get("finetune.pixel_loss_weight", 0.3)
    model.loss_ssim_window = cfg.get("finetune.loss_ssim_window", 7)

    enc_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dec_trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f"encoder_trainable={enc_trainable}")
    print(f"decoder_trainable={dec_trainable}")

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
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
        f.write(f"base_exp_name: {base_exp_name}\n")
        f.write(f"resolved_exp_name: {resolved_exp_name}\n")

    log_dir = cfg.get("train.log_dir", "runs")
    writer = SummaryWriter(log_dir=os.path.join(log_dir, resolved_exp_name))

    epochs = args.epochs or cfg.get("train.epochs", 10)
    eval_interval = cfg.get("train.eval_interval", 1)
    viz_interval = cfg.get("train.viz_interval", 1)
    viz_samples = cfg.get("train.viz_samples", 4)
    save_interval = cfg.get("train.save_interval", 1)
    save_best = cfg.get("train.save_best", True)

    best_val = None
    for epoch in range(1, epochs + 1):
        decoder.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        total = 0.0
        count = 0
        for x in pbar:
            x = x.to(device)
            feat = model.encode_tokens(x)
            b, n, c = feat.shape
            h = int(model.grid_h)
            w = int(model.grid_w)
            feat_map = feat.reshape(b, h, w, c).permute(0, 3, 1, 2)
            recon = decoder(feat_map, target_hw)
            gt = torch.nn.functional.interpolate(x[:, 0:1], size=target_hw, mode="bilinear", align_corners=False)
            loss = _recon_loss(recon, gt, model.loss_alpha, model.loss_ssim_window)
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
            val_loss = _eval_epoch(model, decoder, val_loader, device, target_hw)
            writer.add_scalar("val/loss", val_loss, epoch)

        if val_loader and epoch % viz_interval == 0:
            viz_dir = os.path.join(exp_dir, "val_viz")
            _save_viz(model, decoder, val_loader, device, viz_dir, epoch, viz_samples, writer, target_hw)

        if epoch % save_interval == 0:
            torch.save({"encoder": model.state_dict(), "cnn_decoder": decoder.state_dict()}, os.path.join(exp_dir, "last.pth"))

        if save_best and val_loss is not None:
            if best_val is None or val_loss < best_val:
                best_val = val_loss
                torch.save({"encoder": model.state_dict(), "cnn_decoder": decoder.state_dict()}, os.path.join(exp_dir, "best.pth"))

    out_path = os.path.join(output_dir, "mae_finetuned.pth")
    torch.save({"encoder": model.state_dict(), "cnn_decoder": decoder.state_dict(), "decoder_type": "cnn", "epochs": epochs}, out_path)
    print(f"Saved checkpoint to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Fine-tune CNN decoder only")
    p.add_argument("--root", default="synthetic_root/run_0001")
    p.add_argument("--roots-txt", default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--pretrained", default="outputs/mae_pretrained.pth")
    p.add_argument("--output", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
