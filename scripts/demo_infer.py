import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.runner import cmd_demo_infer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=None)
    p.add_argument("--root", dest="root_opt", default=None)
    p.add_argument("--roots-txt", dest="roots_txt", default=None)
    p.add_argument("--dir-path", default=None)
    p.add_argument("--vehicle", type=int, default=None)
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output", default="outputs")
    p.add_argument("--residual-mode", default=None)
    p.add_argument("--mode", choices=["stub", "mae"], default="stub")
    p.add_argument("--ckpt", default="outputs/mae_finetuned.pth")
    args = p.parse_args()
    args.root_pos = args.root
    cmd_demo_infer(args)


if __name__ == "__main__":
    main()
