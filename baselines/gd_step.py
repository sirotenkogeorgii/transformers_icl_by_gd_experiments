"""Closed-form one-step gradient descent baseline for linear regression."""

from __future__ import annotations

import numpy as np


def _extract_components(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split concatenated tokens into context features, context labels, query."""

    if seq.ndim != 2:
        raise ValueError("Expected shape [tokens, dim] for a single sequence.")
    context = seq[:-1, :]
    query = seq[-1, :]
    x_context = context[:, :-1]
    y_context = context[:, -1]
    x_query = query[:-1]
    return x_context, y_context, x_query


def _batchify(seq: np.ndarray) -> np.ndarray:
    if seq.ndim == 2:
        return seq[None, ...]
    if seq.ndim == 3:
        return seq
    raise ValueError("Expected sequence tensor with rank 2 or 3.")


def gd_weights(seq: np.ndarray, eta: float) -> np.ndarray:
    """Return the weight vector after one GD step from zero initialisation."""

    batched = _batchify(seq)
    x_context = batched[:, :-1, :-1]
    y_context = batched[:, :-1, -1]
    num_shots = x_context.shape[1]
    xt_y = np.einsum("bnd,bn->bd", x_context, y_context)
    weights = (eta / max(num_shots, 1)) * xt_y
    return weights.squeeze() if seq.ndim == 2 else weights


def gd_predict(seq: np.ndarray, eta: float) -> np.ndarray:
    """Predict the query label after one GD step from zero initialisation."""

    batched = _batchify(seq)
    weights = gd_weights(batched, eta)
    x_query = batched[:, -1, :-1]
    preds = np.einsum("bd,bd->b", x_query, weights)
    return preds.squeeze()


def gd_sensitivity(seq: np.ndarray, eta: float) -> np.ndarray:
    """Compute ∂ŷ/∂x_test for the GD baseline."""

    weights = gd_weights(seq, eta)
    return weights
