"""Shared helpers for runner scripts.

The original project exposes most of the training logic through global
functions in ``src.train``. These utilities wrap that functionality into a
repeatable, file-backed experiment pipeline that keeps the authors' modules
unchanged while adding logging, hooks, and canonical defaults.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

try:  # JAX imports fail gracefully when not available in the host env.
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - runtime guard for missing JAX
    jax = None  # type: ignore
    jnp = None  # type: ignore

from src.config import config
from src import data as data_utils
from src import train as train_utils
from utils_attention import save_attention_stack
from utils_log import JsonlLogger


RUNS_DIR = Path("runs")
CANON_CLIP_RANGE = (-10.0, 10.0)

ORDERING_CHOICES = ("random", "smallnorm2largenorm", "easy2hard", "hard2easy")
NOISE_MODE_CHOICES = ("clean", "label_noise", "random_pairs")
PLACEMENT_CHOICES = ("mixed", "clean_first", "noisy_first")


@dataclass
class RunArgs:
    """Normalized arguments used internally during a run."""

    seed: int
    steps: int
    eval_every: int
    run_dir: Path
    save_attention: bool
    ordering: str
    noise_mode: str
    noise_p: float
    noise_sigma: float
    noise_placement: str
    train_ordering: str = "auto"
    eval_ordering: str = "auto"
    train_noise_mode: Optional[str] = None
    train_noise_p: Optional[float] = None
    train_noise_sigma: Optional[float] = None
    train_noise_placement: Optional[str] = None
    eval_noise_mode: Optional[str] = None
    eval_noise_p: Optional[float] = None
    eval_noise_sigma: Optional[float] = None
    eval_noise_placement: Optional[str] = None


def positive_int(value: str) -> int:
    """Argparse helper that ensures strictly positive integers."""

    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return ivalue


def build_common_parser(description: str) -> argparse.ArgumentParser:
    """Return an ``ArgumentParser`` pre-populated with shared CLI flags."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the run")
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=None,
        help="Override the number of training steps (defaults to config.training_steps)",
    )
    parser.add_argument(
        "--eval-every",
        type=positive_int,
        default=500,
        help="Evaluation / logging interval in training steps",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Run directory (defaults to runs/<script_name>_<timestamp>)",
    )
    parser.add_argument(
        "--save_attention",
        action="store_true",
        help="Enable attention capture hooks during evaluation",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        default="random",
        choices=ORDERING_CHOICES,
        help="Ordering applied to context shots (default training & eval unless overridden)",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="clean",
        choices=NOISE_MODE_CHOICES,
        help="Noise injection strategy for context shots",
    )
    parser.add_argument(
        "--noise_p",
        type=float,
        default=0.0,
        help="Probability of corrupting a context shot when noise is enabled",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.0,
        help="Std-dev for label noise when applicable",
    )
    parser.add_argument(
        "--noise_placement",
        type=str,
        default="mixed",
        choices=PLACEMENT_CHOICES,
        help="Placement of noisy shots after corruption",
    )
    parser.add_argument(
        "--train_ordering",
        type=str,
        default="auto",
        choices=ORDERING_CHOICES + ("auto",),
        help="Override ordering used for training batches (auto => use --ordering)",
    )
    parser.add_argument(
        "--eval_ordering",
        type=str,
        default="auto",
        choices=ORDERING_CHOICES + ("auto",),
        help="Override ordering used for evaluation/analysis batches (auto => use --ordering)",
    )
    parser.add_argument(
        "--train_noise_mode",
        type=str,
        default="auto",
        choices=NOISE_MODE_CHOICES + ("auto",),
        help="Override noise mode during training batches",
    )
    parser.add_argument(
        "--train_noise_p",
        type=float,
        default=None,
        help="Override training noise probability",
    )
    parser.add_argument(
        "--train_noise_sigma",
        type=float,
        default=None,
        help="Override training noise sigma",
    )
    parser.add_argument(
        "--train_noise_placement",
        type=str,
        default="auto",
        choices=PLACEMENT_CHOICES + ("auto",),
        help="Override training noise placement",
    )
    parser.add_argument(
        "--eval_noise_mode",
        type=str,
        default="auto",
        choices=NOISE_MODE_CHOICES + ("auto",),
        help="Override noise mode for evaluation batches",
    )
    parser.add_argument(
        "--eval_noise_p",
        type=float,
        default=None,
        help="Override evaluation noise probability",
    )
    parser.add_argument(
        "--eval_noise_sigma",
        type=float,
        default=None,
        help="Override evaluation noise sigma",
    )
    parser.add_argument(
        "--eval_noise_placement",
        type=str,
        default="auto",
        choices=PLACEMENT_CHOICES + ("auto",),
        help="Override evaluation noise placement",
    )
    return parser


def normalize_run_args(
    args: argparse.Namespace, *, default_run_name: str
) -> RunArgs:
    """Convert CLI namespace into :class:`RunArgs` with derived defaults."""

    run_dir = Path(args.out) if args.out else build_default_run_dir(default_run_name)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    steps = args.steps if args.steps is not None else int(config.training_steps)

    def _norm_mode(value):
        return None if value in (None, "auto") else str(value)

    def _norm_placement(value):
        return None if value in (None, "auto") else str(value)

    return RunArgs(
        seed=int(args.seed),
        steps=steps,
        eval_every=int(args.eval_every),
        run_dir=run_dir,
        save_attention=bool(args.save_attention),
        ordering=str(args.ordering),
        noise_mode=str(args.noise_mode),
        noise_p=float(args.noise_p),
        noise_sigma=float(args.noise_sigma),
        noise_placement=str(args.noise_placement),
        train_ordering=str(args.train_ordering),
        eval_ordering=str(args.eval_ordering),
        train_noise_mode=_norm_mode(args.train_noise_mode),
        train_noise_p=args.train_noise_p,
        train_noise_sigma=args.train_noise_sigma,
        train_noise_placement=_norm_placement(args.train_noise_placement),
        eval_noise_mode=_norm_mode(args.eval_noise_mode),
        eval_noise_p=args.eval_noise_p,
        eval_noise_sigma=args.eval_noise_sigma,
        eval_noise_placement=_norm_placement(args.eval_noise_placement),
    )


def build_default_run_dir(stem: str) -> Path:
    """Create a monotonic run directory inside ``runs/``."""

    RUNS_DIR.mkdir(exist_ok=True)
    index = 0
    while True:
        candidate = RUNS_DIR / (stem if index == 0 else f"{stem}_{index}")
        if not candidate.exists():
            return candidate
        index += 1


def set_pythonhashseed(seed: int) -> None:
    """Ensure deterministic hashing across Python libraries."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def set_global_seeds(seed: int) -> None:
    """Seed libraries that expose RNGs (numpy / python / jax)."""

    import random

    random.seed(seed)
    np.random.seed(seed)
    if jax is not None:
        jax.random.PRNGKey(seed)


def summarize_environment() -> Dict[str, Any]:
    """Return a dictionary with basic interpreter / accelerator information."""

    import platform
    import sys

    summary: Dict[str, Any] = {
        "python_version": sys.version.split(" ")[0],
        "platform": platform.platform(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
    }

    try:
        import torch

        summary["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            summary["cuda_available"] = True
            summary["cuda_device"] = torch.cuda.get_device_name(0)
        else:
            summary["cuda_available"] = False
    except Exception:
        summary["torch_version"] = None
        summary["cuda_available"] = False

    for module_name in ("jax", "jaxlib", "numpy", "optax", "haiku", "ml_collections"):
        try:
            module = __import__(module_name)
            summary[f"{module_name}_version"] = getattr(module, "__version__", None)
        except Exception:
            summary[f"{module_name}_version"] = None
    return summary


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def clamp_tensor(x: Any, clip_range: Tuple[float, float]) -> Any:
    if jnp is None:
        return x
    lo, hi = clip_range
    return jnp.clip(x, lo, hi)


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if jnp is not None and isinstance(array, jnp.ndarray):
        return np.array(array)
    return np.asarray(array)


def _save_array(path: Path, array: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, _to_numpy(array))


def canonical_preprocess(batch: Tuple[Any, Any, Any]) -> Tuple[Any, Any, Any]:
    """Apply canonical token/value clipping to a batch."""

    seq, target, weights = batch
    seq = clamp_tensor(seq, CANON_CLIP_RANGE)
    target = clamp_tensor(target, CANON_CLIP_RANGE)
    return seq, target, weights


def _apply_noise_placement(
    context_tokens: jnp.ndarray,
    ordering_idx: jnp.ndarray,
    noise_mask: jnp.ndarray,
    *,
    placement: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if placement == "mixed":
        return context_tokens, ordering_idx, noise_mask
    if placement not in ("clean_first", "noisy_first"):
        raise ValueError(f"Unsupported noise placement: {placement}")

    pair_span = 2 if getattr(config, "classic_token_const", False) else 1
    prefix_len = context_tokens.shape[1] % pair_span if pair_span > 1 else 0

    prefix_tokens = context_tokens[:, :prefix_len, :]
    prefix_idx = ordering_idx[:, :prefix_len]
    prefix_mask = noise_mask[:, :prefix_len]

    pair_tokens = context_tokens[:, prefix_len:, :]
    pair_idx = ordering_idx[:, prefix_len:]
    pair_mask = noise_mask[:, prefix_len:]

    if pair_span > 1 and pair_tokens.shape[1] % pair_span != 0:
        raise ValueError("Context tokens cannot be evenly grouped into pairs; check tokenization setup.")

    if pair_span > 1:
        batch_size = pair_tokens.shape[0]
        pair_count = pair_tokens.shape[1] // pair_span
        pair_tokens = pair_tokens.reshape(batch_size, pair_count, pair_span, -1)
        pair_idx = pair_idx.reshape(batch_size, pair_count, pair_span)
        pair_mask = pair_mask.reshape(batch_size, pair_count, pair_span)
        pair_mask_flag = jnp.any(pair_mask.astype(jnp.bool_), axis=-1)
    else:
        pair_count = pair_tokens.shape[1]
        pair_tokens = pair_tokens[:, :, None, :]
        pair_idx = pair_idx[:, :, None]
        pair_mask = pair_mask[:, :, None]
        pair_mask_flag = pair_mask[:, :, 0].astype(jnp.bool_)

    if placement == "clean_first":
        pair_perm = jnp.argsort(pair_mask_flag.astype(jnp.int32), axis=-1)
    else:
        pair_perm = jnp.argsort(-pair_mask_flag.astype(jnp.int32), axis=-1)

    pair_tokens = jnp.take_along_axis(pair_tokens, pair_perm[..., None, None], axis=1)
    pair_idx = jnp.take_along_axis(pair_idx, pair_perm[..., None], axis=1)
    pair_mask = jnp.take_along_axis(pair_mask, pair_perm[..., None], axis=1)

    if pair_span > 1:
        pair_tokens_flat = pair_tokens.reshape(pair_tokens.shape[0], pair_count * pair_span, -1)
        pair_idx_flat = pair_idx.reshape(pair_idx.shape[0], pair_count * pair_span)
        pair_mask_flat = pair_mask.reshape(pair_mask.shape[0], pair_count * pair_span)
    else:
        pair_tokens_flat = pair_tokens[:, :, 0, :]
        pair_idx_flat = pair_idx[:, :, 0]
        pair_mask_flat = pair_mask[:, :, 0]

    context_tokens_new = jnp.concatenate([prefix_tokens, pair_tokens_flat], axis=1)
    ordering_idx_new = jnp.concatenate([prefix_idx, pair_idx_flat], axis=1)
    noise_mask_new = jnp.concatenate([prefix_mask, pair_mask_flat], axis=1)
    return context_tokens_new, ordering_idx_new, noise_mask_new


def apply_transforms(
    batch: Tuple[Any, Any, Any],
    *,
    ordering: str,
    noise_mode: str,
    noise_p: float,
    noise_sigma: float,
    noise_placement: str,
    rng: Optional[Any],
) -> Tuple[Tuple[Any, Any, Any], Dict[str, Any]]:
    """Apply ordering and noise transforms defined in ``src.data``."""

    if jnp is None:
        raise RuntimeError("JAX is required to apply ordering/noise transforms.")

    desired_placement = noise_placement
    noisy_batch, noise_meta = data_utils._inject_noise_core(
        batch,
        mode=noise_mode,
        p=noise_p,
        sigma=noise_sigma,
        placement="mixed",
        rng=rng,
        apply_placement=False,
    )

    metadata: Dict[str, Any] = {}
    weights = noisy_batch[2]
    ordered_batch, ordering_meta = data_utils.apply_ordering(
        noisy_batch, mode=ordering, teacher_W=weights
    )
    metadata.update(ordering_meta)

    ordering_idx = ordering_meta.get("ordering_idx")
    noise_mask = noise_meta["noise_mask"]
    if ordering_idx is not None:
        if noise_mask.ndim == ordering_idx.ndim:
            noise_mask = jnp.take_along_axis(noise_mask, ordering_idx, axis=1)

    context_tokens = ordered_batch[0][:, :-1, :]
    if desired_placement != "mixed":
        context_tokens, ordering_idx_new, noise_mask = _apply_noise_placement(
            context_tokens,
            ordering_idx if ordering_idx is not None else jnp.broadcast_to(
                jnp.arange(context_tokens.shape[1])[None, :], context_tokens.shape[:2]
            ),
            noise_mask,
            placement=desired_placement,
        )
        ordered_batch = (
            jnp.concatenate([context_tokens, ordered_batch[0][:, -1:, :]], axis=1),
            ordered_batch[1],
            ordered_batch[2],
        )
        metadata["ordering_idx"] = ordering_idx_new
        ordering_idx = ordering_idx_new
    else:
        ordered_batch = (
            jnp.concatenate([context_tokens, ordered_batch[0][:, -1:, :]], axis=1),
            ordered_batch[1],
            ordered_batch[2],
        )

    noise_meta["noise_mask"] = noise_mask
    noise_meta["noise_placement"] = desired_placement
    metadata.update(noise_meta)

    final_batch = canonical_preprocess(ordered_batch)
    return final_batch, metadata


def resolve_ordering(base: str, override: str) -> str:
    if override and override != "auto":
        return override
    return base


def resolve_noise(
    base_mode: str,
    base_p: float,
    base_sigma: float,
    base_placement: str,
    override_mode: Optional[str],
    override_p: Optional[float],
    override_sigma: Optional[float],
    override_placement: Optional[str],
) -> Tuple[str, float, float, str]:
    if override_mode == "auto":
        override_mode = None
    if override_placement == "auto":
        override_placement = None
    mode = override_mode if override_mode is not None else base_mode
    p = override_p if override_p is not None else base_p
    sigma = override_sigma if override_sigma is not None else base_sigma
    placement = override_placement if override_placement is not None else base_placement
    return mode, float(p), float(sigma), placement


def sample_training_batch(
    rng: Any,
    *,
    batch_size: int,
    ordering: str,
    noise_mode: str,
    noise_p: float,
    noise_sigma: float,
    noise_placement: str,
) -> Tuple[Tuple[Any, Any, Any], Dict[str, Any], Any]:
    """Draw a batch using the requested transforms."""

    if jax is None:
        raise RuntimeError("JAX is required to sample training batches.")
    split_rng, next_rng = jax.random.split(rng)
    batch = train_utils.data_creator(
        jax.random.split(split_rng, num=batch_size),
        config.input_size,
        config.dataset_size,
        config.size_distract,
        config.input_range,
        config.weight_scale,
    )
    transformed, metadata = apply_transforms(
        batch,
        ordering=ordering,
        noise_mode=noise_mode,
        noise_p=noise_p,
        noise_sigma=noise_sigma,
        noise_placement=noise_placement,
        rng=next_rng,
    )
    return transformed, metadata, next_rng


def build_eval_dataset(
    rng: Any,
    *,
    batch_size: int,
    ordering: str,
    noise_mode: str,
    noise_p: float,
    noise_sigma: float,
    noise_placement: str,
) -> Tuple[Tuple[Any, Any, Any], Dict[str, Any]]:
    """Sample a held-out dataset for periodic evaluation."""

    batch = train_utils.data_creator(
        jax.random.split(rng, num=batch_size),
        config.input_size,
        config.dataset_size,
        config.size_distract,
        config.input_range,
        config.weight_scale,
    )
    eval_batch, metadata = apply_transforms(
        batch,
        ordering=ordering,
        noise_mode=noise_mode,
        noise_p=noise_p,
        noise_sigma=noise_sigma,
        noise_placement=noise_placement,
        rng=None,
    )
    return eval_batch, metadata


def export_run_metadata(run_args: RunArgs) -> None:
    """Persist metadata describing the run configuration."""

    metadata = asdict(run_args)
    metadata["environment"] = summarize_environment()
    metadata["run_dir"] = str(metadata["run_dir"])
    try:
        metadata["config"] = config.to_dict()
    except AttributeError:
        metadata["config"] = {
            key: getattr(config, key)
            for key in dir(config)
            if not key.startswith("_") and not callable(getattr(config, key))
        }
    write_json(run_args.run_dir / "run_metadata.json", metadata)


def configure_training_defaults(depth: int, *, pos_enc_size: int) -> None:
    """Apply canonical hyper-parameter defaults described in PLAN.md."""

    config.bs = 2048
    config.grad_clip_value = 10.0
    config.grad_clip_value_gd = 10.0
    config.dropout_rate = 0.0
    config.training_steps = int(config.training_steps)

    if depth < 3:
        config.lr = 1e-3
    else:
        config.lr = 5e-4

    config.pos_enc = True
    config.concat_pos_enc = True
    config.pos_enc_size = pos_enc_size
    config.zero_pos_enc = False

    config.save_attention = False


def prepare_training(run_args: RunArgs) -> Tuple[Any, Any, Any, Any]:
    """Initialise optimiser, train/test states, and RNG for a run."""

    if jax is None:
        raise RuntimeError("JAX is required to run training loops.")

    config.seed = run_args.seed
    set_pythonhashseed(run_args.seed)
    set_global_seeds(run_args.seed)

    train_utils.change_dataloader()

    optimiser, train_state, test_state, rng = train_utils.init()
    return optimiser, train_state, test_state, rng



def evaluate_model(train_state: Any, eval_batch: Tuple[Any, Any, Any], rng: Any) -> Dict[str, float]:
    """Run ``predict_test`` and return scalar evaluation metrics."""

    loss, _, _ = train_utils.predict_test.apply(train_state.params, rng, eval_batch, False)
    return {"test_loss": float(loss)}


def train_loop(run_args: RunArgs) -> Dict[str, Any]:
    """Execute the training loop and return summary metrics."""

    optimiser, train_state, _test_state, rng = prepare_training(run_args)
    rng, data_rng, eval_rng = jax.random.split(rng, 3)

    train_ordering = resolve_ordering(run_args.ordering, run_args.train_ordering)
    eval_ordering = resolve_ordering(run_args.ordering, run_args.eval_ordering)

    train_noise_mode, train_noise_p, train_noise_sigma, train_noise_placement = resolve_noise(
        run_args.noise_mode,
        run_args.noise_p,
        run_args.noise_sigma,
        run_args.noise_placement,
        run_args.train_noise_mode,
        run_args.train_noise_p,
        run_args.train_noise_sigma,
        run_args.train_noise_placement,
    )

    eval_noise_mode, eval_noise_p, eval_noise_sigma, eval_noise_placement = resolve_noise(
        run_args.noise_mode,
        run_args.noise_p,
        run_args.noise_sigma,
        run_args.noise_placement,
        run_args.eval_noise_mode,
        run_args.eval_noise_p,
        run_args.eval_noise_sigma,
        run_args.eval_noise_placement,
    )

    eval_batch, _eval_metadata = build_eval_dataset(
        eval_rng,
        batch_size=config.bs,
        ordering=eval_ordering,
        noise_mode=eval_noise_mode,
        noise_p=eval_noise_p,
        noise_sigma=eval_noise_sigma,
        noise_placement=eval_noise_placement,
    )
    metrics_history = []
    logger = JsonlLogger(run_args.run_dir / "metrics.jsonl")

    try:
        for step in range(run_args.steps):
            batch, _metadata, data_rng = sample_training_batch(
                data_rng,
                batch_size=config.bs,
                ordering=train_ordering,
                noise_mode=train_noise_mode,
                noise_p=train_noise_p,
                noise_sigma=train_noise_sigma,
                noise_placement=train_noise_placement,
            )
            train_state, metrics = train_utils.update(train_state, batch, optimiser)
            if (step % run_args.eval_every == 0) or (step == run_args.steps - 1):
                eval_metrics = evaluate_model(train_state, eval_batch, eval_rng)
                record = {
                    "step": int(metrics["step"]),
                    "train_loss": float(metrics["train_loss"]),
                    **eval_metrics,
                }
                metrics_history.append(record)
                logger.log(**record)
    finally:
        logger.close()

    _train_batch, train_metadata, data_rng = sample_training_batch(
        data_rng,
        batch_size=config.bs,
        ordering=train_ordering,
        noise_mode=train_noise_mode,
        noise_p=train_noise_p,
        noise_sigma=train_noise_sigma,
        noise_placement=train_noise_placement,
    )

    analysis_batch, analysis_metadata, data_rng = sample_training_batch(
        data_rng,
        batch_size=config.bs,
        ordering=eval_ordering,
        noise_mode=eval_noise_mode,
        noise_p=eval_noise_p,
        noise_sigma=eval_noise_sigma,
        noise_placement=eval_noise_placement,
    )

    analysis_dir = run_args.run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        analysis_dir / "analysis_batch.npz",
        seq=_to_numpy(analysis_batch[0]),
        target=_to_numpy(analysis_batch[1]),
        weights=_to_numpy(analysis_batch[2]),
    )

    metadata_json: Dict[str, Any] = {
        "train_config": {
            "ordering": train_ordering,
            "noise_mode": train_noise_mode,
            "noise_p": train_noise_p,
            "noise_sigma": train_noise_sigma,
            "noise_placement": train_noise_placement,
        },
        "eval_config": {
            "ordering": eval_ordering,
            "noise_mode": eval_noise_mode,
            "noise_p": eval_noise_p,
            "noise_sigma": eval_noise_sigma,
            "noise_placement": eval_noise_placement,
        },
        "train_dataset_metadata": {},
        "eval_dataset_metadata": {},
    }
    if "ordering_idx" in train_metadata:
        _save_array(analysis_dir / "train_ordering_idx.npy", train_metadata["ordering_idx"])
        metadata_json["train_dataset_metadata"]["ordering_idx_shape"] = list(
            _to_numpy(train_metadata["ordering_idx"]).shape
        )
    if "noise_mask" in train_metadata:
        _save_array(analysis_dir / "train_noise_mask.npy", train_metadata["noise_mask"])
        metadata_json["train_dataset_metadata"]["noise_mask_shape"] = list(
            _to_numpy(train_metadata["noise_mask"]).shape
        )
    if "ordering_idx" in analysis_metadata:
        ordering_array = analysis_metadata["ordering_idx"]
        _save_array(analysis_dir / "ordering_idx.npy", ordering_array)
        _save_array(run_args.run_dir / "ordering_idx.npy", ordering_array)
        metadata_json["eval_dataset_metadata"]["ordering_idx_shape"] = list(
            _to_numpy(analysis_metadata["ordering_idx"]).shape
        )
    if "noise_mask" in analysis_metadata:
        noise_array = analysis_metadata["noise_mask"]
        _save_array(analysis_dir / "noise_mask.npy", noise_array)
        _save_array(run_args.run_dir / "noise_mask.npy", noise_array)
        metadata_json["eval_dataset_metadata"]["noise_mask_shape"] = list(
            _to_numpy(analysis_metadata["noise_mask"]).shape
        )
    if "noise_mode" in analysis_metadata:
        metadata_json["eval_dataset_metadata"]["noise_mode"] = analysis_metadata["noise_mode"]
    if "noise_placement" in analysis_metadata:
        metadata_json["eval_dataset_metadata"]["noise_placement"] = analysis_metadata["noise_placement"]
    metadata_json["eval_dataset_metadata"]["ordering_mode"] = eval_ordering
    write_json(analysis_dir / "analysis_metadata.json", metadata_json)

    summary = metrics_history[-1] if metrics_history else {}
    summary["num_steps"] = run_args.steps
    summary["ordering"] = run_args.ordering
    summary["noise_mode"] = run_args.noise_mode
    summary["noise_p"] = run_args.noise_p
    summary["noise_sigma"] = run_args.noise_sigma
    summary["noise_placement"] = run_args.noise_placement
    summary["save_attention"] = run_args.save_attention
    summary["train_ordering"] = train_ordering
    summary["eval_ordering"] = eval_ordering
    summary["train_noise_mode"] = train_noise_mode
    summary["eval_noise_mode"] = eval_noise_mode
    summary["train_noise_p"] = train_noise_p
    summary["train_noise_sigma"] = train_noise_sigma
    summary["train_noise_placement"] = train_noise_placement
    summary["eval_noise_p"] = eval_noise_p
    summary["eval_noise_sigma"] = eval_noise_sigma
    summary["eval_noise_placement"] = eval_noise_placement

    params_path = run_args.run_dir / "params.pkl"
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with params_path.open("wb") as f:
        pickle.dump(params_np, f)

    if run_args.save_attention:
        attn_dir = run_args.run_dir / "attn"
        attn_stack = train_utils.predict_attn.apply(
            train_state.params, eval_rng, analysis_batch[0], False
        )
        save_attention_stack(
            list(attn_stack), attn_dir, summary_path=attn_dir / "summary.json"
        )

    write_json(run_args.run_dir / "metrics_summary.json", summary)
    write_json(run_args.run_dir / "metrics_history.json", {"history": metrics_history})
    export_run_metadata(run_args)
    return summary
