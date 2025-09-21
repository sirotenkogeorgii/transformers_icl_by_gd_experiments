"""Plot GD baseline and transformer layerwise curves on one figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_curves(gd_metrics: dict, tr_metrics: dict, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    gd_sem = gd_metrics.get("sem") or [0.0] * len(gd_metrics["steps"])
    plt.plot(gd_metrics["steps"], gd_metrics["mse"], label="GD Steps", color="green")
    plt.fill_between(
        gd_metrics["steps"],
        [m - s for m, s in zip(gd_metrics["mse"], gd_sem)],
        [m + s for m, s in zip(gd_metrics["mse"], gd_sem)],
        color="green",
        alpha=0.2,
    )

    tr_sem = tr_metrics.get("sem") or [0.0] * len(tr_metrics["layers"])
    plt.plot(
        tr_metrics["layers"],
        tr_metrics["mse"],
        label="Transformer Layers",
        color="purple",
        marker="o",
    )
    plt.fill_between(
        tr_metrics["layers"],
        [m - s for m, s in zip(tr_metrics["mse"], tr_sem)],
        [m + s for m, s in zip(tr_metrics["mse"], tr_sem)],
        color="purple",
        alpha=0.2,
    )

    plt.xlabel("Steps / Layers")
    plt.ylabel("Query MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gd", required=True, help="Path to GD baseline JSON")
    parser.add_argument("--tr", required=True, help="Path to transformer layerwise JSON")
    parser.add_argument("--out", required=True, help="Output path (png)")
    args = parser.parse_args(argv)

    gd_metrics = load_metrics(Path(args.gd))
    tr_metrics = load_metrics(Path(args.tr))
    plot_curves(gd_metrics, tr_metrics, Path(args.out))
    print(f"Saved comparison plot to {args.out}")


if __name__ == "__main__":
    main()
