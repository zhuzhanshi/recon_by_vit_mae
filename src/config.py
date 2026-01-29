from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class ConfigView:
    cfg: Dict[str, Any]

    def get(self, path: str, default: Any) -> Any:
        return _get(self.cfg, path, default)

    def with_overrides(self, overrides: Dict[str, Any]) -> "ConfigView":
        merged = dict(self.cfg)
        for key, value in overrides.items():
            if value is None:
                continue
            merged[key] = value
        return ConfigView(merged)


def resolve_split_roots(args, cfg: ConfigView) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if getattr(args, "roots_txt", None):
        return args.roots_txt, None, None
    train = cfg.get("data.train_roots_txt", None)
    val = cfg.get("data.val_roots_txt", None)
    test = cfg.get("data.test_roots_txt", None)
    return train, val, test
