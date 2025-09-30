"""Evaluate OOD robustness by scaling input features."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from src import train as train_utils
from src.config import config


def _load_metadata(run_dir: Path) -> None:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing run metadata at {metadata_path}")
    with metadata_path.open() as f:
        metadata = json.load(f)
    cfg = metadata.get("config", {})
    for key, value in cfg.items():
        try:
            setattr(config, key, value)
        except Exception:
            pass


def _load_params(run_dir: Path):
    params_path = run_dir / "params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters at {params_path}")
    with params_path.open("rb") as f:
        params_np = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params_np)


def _load_batch(run_dir: Path):
    batch_path = run_dir / "analysis" / "analysis_batch.npz"
    with np.load(batch_path) as data:
        seq = np.array(data["seq"], dtype=np.float32)
        target = np.array(data["target"], dtype=np.float32)
    return seq, target


def _predict(params, seq: np.ndarray) -> np.ndarray:
    rng = jax.random.PRNGKey(int(config.seed))
    preds = train_utils.predict.apply(params, rng, jnp.asarray(seq), False)
    return np.array(preds[:, -1, -1] * (-1.0))


def evaluate_ood(run_dir: Path, alphas: Sequence[float]) -> Path:
    _load_metadata(run_dir)
    params = _load_params(run_dir)
    seq, target = _load_batch(run_dir)

    results = []
    for alpha in alphas:
        scaled = seq.copy()
        scaled[..., :-1] = scaled[..., :-1] * alpha
        pred = _predict(params, scaled)
        mse = float(np.mean((pred - target[:, -1]) ** 2))
        results.append((alpha, mse))

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "ood.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "test_mse"])
        writer.writerows(results)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD scaling evaluation")
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    parser.add_argument("--alphas", type=float, nargs="+", required=True, help="Scaling factors")
    args = parser.parse_args()
    evaluate_ood(args.run.resolve(), args.alphas)


if __name__ == "__main__":
    main()
