"""Project configuration: locate the repository and load config/config.yaml."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_RELPATH = Path("config") / "config.yaml"


def project_root() -> Path:
    """Repository root: $WMH_ROOT if set, else the first parent of the working directory holding
    config/config.yaml, else the folder above ``src/`` (editable install)."""
    if env := os.environ.get("WMH_ROOT"):
        return Path(env).resolve()
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / CONFIG_RELPATH).is_file():
            return folder
    return Path(__file__).resolve().parents[2]


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the YAML configuration. ``cfg["root"]`` is set to the repository root."""
    root = project_root()
    path = Path(path) if path else root / CONFIG_RELPATH
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["root"] = root
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path of ``cfg["paths"][key]`` (paths in the config are relative to the root)."""
    return cfg["root"] / cfg["paths"][key]
