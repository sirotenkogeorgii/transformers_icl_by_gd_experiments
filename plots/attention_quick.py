"""Minimal attention heatmap generator for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


def iter_attention_files(attn_dir: Path) -> Iterable[Path]:
    if not attn_dir.exists():
        return []
    return sorted(attn_dir.glob("layer*_head*.npy"))


def plot_attention(path: Path, out_dir: Path) -> None:
    tensor = np.load(path)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(tensor, cmap="viridis")
    ax.set_title(path.stem)
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    fig.colorbar(im, ax=ax, shrink=0.8)
    out_path = out_dir / f"{path.stem}.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render attention heatmaps from saved arrays")
    parser.add_argument("--run", type=Path, required=True, help="Run directory containing attn/*.npy")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write figures (defaults to <run>/figs)",
    )
    args = parser.parse_args()

    run_dir: Path = args.run.resolve()
    attn_dir = run_dir / "attn"
    out_dir = args.out.resolve() if args.out else run_dir / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_attention_files(attn_dir))
    if not files:
        raise FileNotFoundError(
            f"No attention files found in {attn_dir}. Run with --save_attention first."
        )
    for path in files:
        plot_attention(path, out_dir)

    print(f"Generated {len(files)} figures in {out_dir}")


if __name__ == "__main__":
    main()

# /Users/georgiisirotenko/transformers_icl_by_gd_experiments/runs/smoke_copy_attn