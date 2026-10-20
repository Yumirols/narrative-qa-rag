# -*- coding: utf-8 -*-
"""配置加载与路径解析。

所有相对路径以 project/ 根目录为基准解析,保证从任意工作目录启动都一致。
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, root: Path = PROJECT_ROOT) -> dict:
    """把 cfg['paths'] 里的相对路径解析为绝对 Path。"""
    return {k: (root / v if not Path(v).is_absolute() else Path(v))
            for k, v in cfg.get("paths", {}).items()}
