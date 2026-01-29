import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import ConfigView, load_config
from src.mae import MAE


def _param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    p = argparse.ArgumentParser(description="Inspect MAE model presets")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--preset", choices=["tiny", "vitb"], required=True)
    p.add_argument("--print", dest="do_print", action="store_true", default=True)
    p.add_argument("--no-print", dest="do_print", action="store_false")
    p.add_argument("--params", dest="do_params", action="store_true", default=True)
    p.add_argument("--no-params", dest="do_params", action="store_false")
    p.add_argument("--torchinfo", dest="do_torchinfo", action="store_true", default=True)
    p.add_argument("--no-torchinfo", dest="do_torchinfo", action="store_false")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = ConfigView(load_config(args.config))
    img_size = cfg.get("infer.crop_size", 512)
    patch_size = cfg.get("model.patch_size", 16)

    model = MAE.from_preset(
        args.preset,
        img_size=img_size,
        patch_size=patch_size,
        mask_ratio=0.0,
    )

    lines = []
    lines.append(f"preset: {args.preset}")
    lines.append(
        f"dims: embed_dim={model.embed_dim}, depth={model.depth}, heads={model.num_heads}, patch_size={model.patch_size}"
    )

    if args.do_params:
        total, trainable = _param_counts(model)
        lines.append(f"params_total: {total}")
        lines.append(f"params_trainable: {trainable}")

    if args.do_print:
        lines.append("model_repr:")
        lines.append(str(model))

    if args.do_torchinfo:
        try:
            from torchinfo import summary

            lines.append("torchinfo_summary:")
            info = summary(model, input_size=(1, 3, img_size, img_size), depth=4, verbose=0)
            lines.append(str(info))
        except Exception as e:
            lines.append(f"torchinfo not available or failed: {e}")

    output = "\n".join(lines)
    print(output)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        out_path = os.path.join(args.out, f"inspect_{args.preset}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(output)


if __name__ == "__main__":
    main()
