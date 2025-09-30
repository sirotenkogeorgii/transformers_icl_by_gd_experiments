"""Alignment metrics between the trained transformer and GD baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from baselines.gd_step import gd_predict, gd_sensitivity
from src import train as train_utils
from src.config import config


def _load_metadata(run_dir: Path) -> Dict:
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
    return metadata


def _load_params(run_dir: Path):
    params_path = run_dir / "params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters at {params_path}")
    with params_path.open("rb") as f:
        params_np = pickle.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params_np)
    return params


def _load_analysis_batch(run_dir: Path):
    batch_path = run_dir / "analysis" / "analysis_batch.npz"
    if not batch_path.exists():
        raise FileNotFoundError(f"Missing analysis batch at {batch_path}")
    with np.load(batch_path) as data:
        seq = jnp.asarray(data["seq"])
        target = jnp.asarray(data["target"])
        weights = jnp.asarray(data["weights"])
    return seq, target, weights


def _transformer_predictions(params, seq_batch):
    rng = jax.random.PRNGKey(int(config.seed))
    preds = train_utils.predict.apply(params, rng, seq_batch, False)
    return preds[:, -1, -1] * (-1.0)


def _transformer_sensitivities(params, seq_batch):
    rng = jax.random.PRNGKey(int(config.seed))

    def _predict_single(tokens):
        preds = train_utils.predict.apply(params, rng, tokens[None, ...], False)
        return preds[0, -1, -1] * (-1.0)

    grad_fn = jax.grad(_predict_single)
    grads = jax.vmap(grad_fn)(seq_batch)
    if getattr(config, "classic_token_const", False):
        return grads[:, -1, :]
    return grads[:, -1, :-1]


def _prepare_sequences_for_baseline(seq_batch: np.ndarray) -> np.ndarray:
    if getattr(config, "classic_token_const", False):
        context = seq_batch[:, :-1, :]
        if context.shape[1] % 2 != 0:
            raise ValueError("Alternating token layout expects an even number of context tokens.")
        x_tokens = context[:, ::2, :]
        y_tokens = context[:, 1::2, :]
        y_values = y_tokens[..., -1]
        query = seq_batch[:, -1:, :]
        query_pad = np.zeros_like(query[..., :1])
        context_with_targets = np.concatenate([x_tokens, y_values[..., None]], axis=-1)
        query_with_slot = np.concatenate([query, query_pad], axis=-1)
        return np.concatenate([context_with_targets, query_with_slot], axis=1)
    return seq_batch


def _sha1_hex(array: np.ndarray) -> str:
    return hashlib.sha1(array.tobytes()).hexdigest()


def _evaluate_eta(
    eta: float,
    preds_tf: np.ndarray,
    sens_tf: np.ndarray,
    baseline_seq: np.ndarray,
) -> Dict[str, float]:
    preds_gd = gd_predict(baseline_seq, eta=eta)
    sens_gd = gd_sensitivity(baseline_seq, eta=eta)
    sens_gd = np.array(sens_gd)
    if sens_gd.ndim == 1:
        sens_gd = sens_gd[None, :]

    pred_diff = preds_tf - preds_gd
    l2_pred_diff = float(math.sqrt(float(np.mean(pred_diff ** 2))))

    dot = np.sum(sens_tf * sens_gd, axis=1)
    denom = (np.linalg.norm(sens_tf, axis=1) * np.linalg.norm(sens_gd, axis=1)) + 1e-8
    cosine = dot / denom
    align_cos = float(np.mean(cosine))

    return {
        "l2_pred_diff": l2_pred_diff,
        "align_cos": align_cos,
    }


def compute_alignment(
    run_dir: Path,
    *,
    eta: float | None = None,
    eta_grid: Sequence[float] | None = None,
    pe_scale: float = 1.0,
) -> Dict[str, object]:
    _load_metadata(run_dir)

    prev_scale = config.pos_enc_scale_eval
    prev_zero = config.zero_pos_enc
    try:
        config.pos_enc_scale_eval = float(pe_scale)
        if pe_scale == 0.0:
            config.zero_pos_enc = True
        params = _load_params(run_dir)
        seq_batch, target_batch, _ = _load_analysis_batch(run_dir)

        preds_tf = np.array(_transformer_predictions(params, seq_batch))
        sens_tf = np.array(_transformer_sensitivities(params, seq_batch))

        baseline_seq = _prepare_sequences_for_baseline(np.array(seq_batch))

        candidate_etas: List[float] = []
        if eta is not None:
            candidate_etas.append(float(eta))
        if eta_grid is not None:
            candidate_etas.extend(float(x) for x in eta_grid)
        if not candidate_etas:
            raise ValueError("Specify --eta and/or --eta-grid with at least one value.")

        seen = set()
        per_eta: List[Dict[str, object]] = []
        for eta_value in candidate_etas:
            if eta_value in seen:
                continue
            seen.add(eta_value)
            metrics = _evaluate_eta(eta_value, preds_tf, sens_tf, baseline_seq)
            per_eta.append({"eta": eta_value, **metrics})

        def _score(entry: Dict[str, object]) -> tuple[float, float]:
            return (entry["l2_pred_diff"], -entry["align_cos"])

        best = min(per_eta, key=_score)

        stats = {
            "eta": best["eta"],
            "pe_scale": pe_scale,
            "l2_pred_diff": best["l2_pred_diff"],
            "align_cos": best["align_cos"],
            "eta_candidates": sorted(seen),
            "per_eta": per_eta,
            "seq_sha1": _sha1_hex(np.array(seq_batch)),
            "target_sha1": _sha1_hex(np.array(target_batch)),
        }
    finally:
        config.pos_enc_scale_eval = prev_scale
        config.zero_pos_enc = prev_zero

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "alignment.json").write_text(json.dumps(stats, indent=2, sort_keys=True))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Alignment to GD baseline")
    parser.add_argument("--run", type=Path, required=True, help="Run directory")
    parser.add_argument("--eta", type=float, help="Fixed GD step size")
    parser.add_argument(
        "--eta-grid",
        type=float,
        nargs="*",
        help="Optional grid of GD step sizes to sweep; defaults to []",
    )
    parser.add_argument(
        "--pe-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to positional encodings during alignment",
    )
    args = parser.parse_args()
    compute_alignment(args.run.resolve(), eta=args.eta, eta_grid=args.eta_grid, pe_scale=args.pe_scale)


if __name__ == "__main__":
    main()
