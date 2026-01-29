from types import SimpleNamespace

from src.config import ConfigView, resolve_split_roots


def test_resolve_split_roots_cli_overrides():
    cfg = ConfigView({"data": {"train_roots_txt": "a.txt", "val_roots_txt": "b.txt"}})
    args = SimpleNamespace(roots_txt="cli.txt")
    train, val, test = resolve_split_roots(args, cfg)
    assert train == "cli.txt"
    assert val is None
    assert test is None


def test_resolve_split_roots_cfg():
    cfg = ConfigView({"data": {"train_roots_txt": "a.txt", "val_roots_txt": "b.txt"}})
    args = SimpleNamespace(roots_txt=None)
    train, val, test = resolve_split_roots(args, cfg)
    assert train == "a.txt"
    assert val == "b.txt"
    assert test is None
