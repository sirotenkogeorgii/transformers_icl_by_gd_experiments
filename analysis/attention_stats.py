"""Compute attention statistics (clean vs noisy) and AUROC for a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_raw_attention(attn_dir: Path) -> List[np.ndarray]:
    layers: List[np.ndarray] = []
    for path in sorted(attn_dir.glob("layer*.npz")):
        with np.load(path) as data:
            layers.append(data["data"])
    if not layers:
        raise FileNotFoundError(f"No raw attention files found inside {attn_dir}")
    return layers


def _binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    idx = 0
    while idx < scores.size:
        j = idx
        while j + 1 < scores.size and scores[order[j + 1]] == scores[order[idx]]:
            j += 1
        average_rank = (idx + j + 2) / 2.0
        ranks[order[idx : j + 1]] = average_rank
        idx = j + 1
    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def compute_attention_stats(run_dir: Path) -> Dict[str, float]:
    attn_dir = run_dir / "attn"
    analysis_dir = run_dir / "analysis"
    layers = _load_raw_attention(attn_dir)

    noise_mask_path = analysis_dir / "noise_mask.npy"
    if not noise_mask_path.exists():
        raise FileNotFoundError(f"Missing noise mask at {noise_mask_path}")
    noise_mask = np.load(noise_mask_path)
    if noise_mask.ndim == 1:
        noise_mask = noise_mask[None, :]

    context_len = noise_mask.shape[-1]
    clean_vals: List[float] = []
    noisy_vals: List[float] = []
    layer_stats: List[Dict[str, float]] = []

    for layer_idx, layer in enumerate(layers):
        # layer shape: [batch, heads, query, key]
        query_attn = layer[:, :, -1, :context_len]
        mask = noise_mask[:, None, :]
        clean = query_attn[~mask]
        noisy = query_attn[mask]
        mean_clean = float(clean.mean()) if clean.size else float("nan")
        mean_noisy = float(noisy.mean()) if noisy.size else float("nan")
        scores = -query_attn.reshape(-1)
        labels = noise_mask[:, None, :].reshape(-1)
        auroc = _binary_auroc(scores, labels)
        layer_stats.append(
            {
                "layer": layer_idx,
                "mean_clean": mean_clean,
                "mean_noisy": mean_noisy,
                "auroc": auroc,
            }
        )
        clean_vals.extend(clean.tolist())
        noisy_vals.extend(noisy.tolist())

    scores_all = np.concatenate([-np.array(clean_vals), -np.array(noisy_vals)])
    labels_all = np.concatenate([
        np.zeros(len(clean_vals), dtype=np.int32),
        np.ones(len(noisy_vals), dtype=np.int32),
    ])
    global_auroc = _binary_auroc(scores_all, labels_all) if scores_all.size else float("nan")
    mean_clean_all = float(np.mean(clean_vals)) if clean_vals else float("nan")
    mean_noisy_all = float(np.mean(noisy_vals)) if noisy_vals else float("nan")

    stats = {
        "mean_clean": mean_clean_all,
        "mean_noisy": mean_noisy_all,
        "auroc": global_auroc,
        "layer_stats": layer_stats,
    }

    stats_path = analysis_dir / "attention_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(["clean", "noisy"], [mean_clean_all, mean_noisy_all], color=["#4daf4a", "#e41a1c"])
    ax.set_ylabel("Mean attention (query→context)")
    ax.set_title(f"Mean attention vs noise (AUROC={global_auroc:.3f})")
    fig.tight_layout()
    fig.savefig(analysis_dir / "attention_stats.png")
    plt.close(fig)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention statistics for a saved run")
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    args = parser.parse_args()
    compute_attention_stats(args.run.resolve())


if __name__ == "__main__":
    main()
