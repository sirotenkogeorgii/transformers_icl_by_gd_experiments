"""Utilities for capturing and persisting attention weights during runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

try:  # Support runtime without JAX by falling back to numpy arrays.
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None  # type: ignore


def _to_numpy(array) -> np.ndarray:
    if array is None:
        return np.array([])
    if jnp is not None and isinstance(array, jnp.ndarray):
        return np.array(array)
    return np.array(array)


def save_attention_stack(
    attn_stack: Sequence[np.ndarray],
    out_dir: Path,
    *,
    summary_path: Path | None = None,
) -> None:
    """Store per-layer, per-head attention maps to ``out_dir``.

    The tensors are averaged over the batch dimension and saved under the
    naming scheme ``layer{idx}_head{idx}.npy``. A JSON summary describing the
    shapes is written to ``summary_path`` when provided.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    summary: List[dict] = []
    for layer_idx, layer_tensor in enumerate(attn_stack):
        layer_np = _to_numpy(layer_tensor)
        if layer_np.size == 0:
            continue
        if layer_np.ndim < 4:
            raise ValueError(
                "Expected attention tensor with shape [batch, heads, query, key]"
            )
        raw_path = out_dir / f"layer{layer_idx}.npz"
        np.savez_compressed(raw_path, data=layer_np)

        batch_mean = layer_np.mean(axis=0)
        num_heads = batch_mean.shape[0]
        for head_idx in range(num_heads):
            head_attn = batch_mean[head_idx]
            file_path = out_dir / f"layer{layer_idx}_head{head_idx}.npy"
            np.save(file_path, head_attn)
            summary.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "path": file_path.name,
                    "shape": list(head_attn.shape),
                    "raw_path": raw_path.name,
                }
            )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
