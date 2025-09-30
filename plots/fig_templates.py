"""Generate canonical figures from aggregated experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ORDERING_ORDER = ["random", "easy2hard", "hard2easy", "smallnorm2largenorm"]


def _load_results(path: Path) -> List[Dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]
    for row in rows:
        row["p"] = float(row["p"])
        row["sigma"] = float(row["sigma"])
        row["seed"] = int(row["seed"])
        row["test_mse"] = float(row["test_mse"])
        row["align_cos"] = float(row["align_cos"])
        row["align_l2"] = float(row["align_l2"])
        row["auroc"] = float(row["auroc"])
        row["ood_alpha3"] = float(row["ood_alpha3"])
    return rows


def _load_attention_stats(run_dir: Path) -> Dict:
    stats_path = run_dir / "analysis" / "attention_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing attention stats at {stats_path}")
    with stats_path.open() as f:
        return json.load(f)


def _load_ood_curve(run_dir: Path):
    path = run_dir / "analysis" / "ood.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = [(float(r["alpha"]), float(r["test_mse"])) for r in reader]
    rows.sort()
    return rows


def plot_order_sensitivity(rows: List[Dict], out_dir: Path) -> None:
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["noise_mode"] != "clean":
            continue
        grouped[row["model"]][row["ordering"]].append(row["test_mse"])

    fig, ax = _new_fig()
    handles: List = []
    labels: List[str] = []
    for model, ordering_dict in grouped.items():
        xs = []
        ys = []
        for ordering in ORDERING_ORDER:
            if ordering in ordering_dict:
                xs.append(ordering)
                ys.append(np.mean(ordering_dict[ordering]))
        if xs:
            (handle,) = ax.plot(xs, ys, marker="o", label=model)
            handles.append(handle)
            labels.append(model)
    ax.set_xlabel("Ordering")
    ax.set_ylabel("Test MSE")
    ax.set_title("Order sensitivity")
    if handles:
        ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(out_dir / "order_sensitivity.png")
    plt.close(fig)


def plot_noise_robustness(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = _new_fig()
    placements = defaultdict(list)
    for row in rows:
        if row["noise_mode"] == "clean":
            continue
        placements[row["placement"]].append((row["p"], row["test_mse"]))
    handles = []
    labels = []
    for placement, values in placements.items():
        values = sorted(values)
        if not values:
            continue
        xs, ys = zip(*values)
        (handle,) = ax.plot(xs, ys, marker="o", label=placement)
        handles.append(handle)
        labels.append(placement)
    ax.set_xlabel("Noise rate p")
    ax.set_ylabel("Test MSE")
    ax.set_title("Noise robustness")
    if handles:
        ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(out_dir / "noise_robustness.png")
    plt.close(fig)


def plot_attention_separation(rows: List[Dict], out_dir: Path) -> None:
    clean_vals = []
    noisy_vals = []
    aurocs = []
    for row in rows:
        run_dir = Path(row["exp"])
        stats = _load_attention_stats(run_dir)
        aurocs.append(stats["auroc"])
        clean_vals.append(stats["mean_clean"])
        noisy_vals.append(stats["mean_noisy"])
    def _finite_mean(values):
        if not values:
            return float("nan")
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.mean(arr))

    mean_clean = _finite_mean(clean_vals)
    mean_noisy = _finite_mean(noisy_vals)
    mean_auroc = _finite_mean(aurocs)

    fig, ax = _new_fig(width=4, height=3)
    ax.bar(["clean", "noisy"], [mean_clean, mean_noisy], color=["#377eb8", "#ff7f00"])
    ax.set_ylabel("Mean attention")
    ax.set_title(f"Attention separation (AUROC≈{mean_auroc:.3f})")
    fig.tight_layout()
    fig.savefig(out_dir / "attention_separation.png")
    plt.close(fig)


def plot_alignment(rows: List[Dict], out_dir: Path) -> None:
    fig, ax = _new_fig(width=4, height=3)
    for row in rows:
        ax.scatter(row["align_cos"], row["align_l2"], label=row["model"], alpha=0.7)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("L2 pred diff")
    ax.set_title("Alignment vs GD")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    fig.tight_layout()
    fig.savefig(out_dir / "alignment_metrics.png")
    plt.close(fig)


def plot_ood(rows: List[Dict], out_dir: Path) -> None:
    seen = set()
    fig, ax = _new_fig()
    handles = []
    labels = []
    for row in rows:
        model = row["model"]
        if model in seen:
            continue
        run_dir = Path(row["exp"])
        curve = _load_ood_curve(run_dir)
        if not curve:
            continue
        xs, ys = zip(*curve)
        (handle,) = ax.plot(xs, ys, marker="o", label=model)
        handles.append(handle)
        labels.append(model)
        seen.add(model)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Test MSE")
    ax.set_title("OOD scaling")
    if handles:
        ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(out_dir / "ood_curve.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figure templates from results.csv")
    parser.add_argument("--from", dest="results", type=Path, required=True, help="Results CSV path")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for figures")
    args = parser.parse_args()

    rows = _load_results(args.results.resolve())
    args.out.mkdir(parents=True, exist_ok=True)

    plot_order_sensitivity(rows, args.out)
    plot_noise_robustness(rows, args.out)
    plot_attention_separation(rows, args.out)
    plot_alignment(rows, args.out)
    plot_ood(rows, args.out)

    print(f"Generated figures in {args.out}")


def _new_fig(width: float = 5, height: float = 3):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax
    
if __name__ == "__main__":
    main()
