"""Utility helpers for reproducibility and logging."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class MetaInfo:
    git_hash: str
    config: Dict[str, Any]
    seed: int
    device: str
    torch_version: str
    platform: str


def write_meta(save_dir: Path, meta: MetaInfo) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)


def collect_meta(config: Dict[str, Any], seed: int, save_dir: Path) -> MetaInfo:
    git_hash = "unknown"
    try:
        import subprocess

        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd())
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return MetaInfo(
        git_hash=git_hash,
        config=config,
        seed=seed,
        device=device,
        torch_version=torch.__version__,
        platform=platform.platform(),
    )
