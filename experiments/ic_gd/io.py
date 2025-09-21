"""Packing utilities to interface linear tasks with the transformer."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def pack_sequence(
    Xc: torch.Tensor, yc: torch.Tensor, Xq: torch.Tensor, yq: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Pack context/query pairs into a single transformer-friendly sequence."""

    if Xc.dim() != 2 or Xq.dim() != 2:
        raise ValueError("Xc and Xq must be 2D tensors")
    if yc.dim() != 1:
        raise ValueError("yc must be a 1D tensor")
    if Xc.shape[0] != yc.shape[0]:
        raise ValueError("Context inputs/targets mismatch")
    if Xc.shape[1] != Xq.shape[1]:
        raise ValueError("Context/query feature dimension mismatch")

    n_context = Xc.shape[0]
    n_query = Xq.shape[0]
    d = Xc.shape[1]
    seq_len = n_context + n_query

    tokens = torch.zeros(seq_len, d + 2, dtype=Xc.dtype, device=Xc.device)
    tokens[:n_context, :d] = Xc
    tokens[:n_context, d] = yc
    tokens[n_context:, :d] = Xq
    tokens[n_context:, d] = 0.0
    tokens[n_context:, d + 1] = 1.0  # query indicator

    attention_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=Xc.device)
    # Disallow attending to future queries
    for i in range(n_query):
        q_idx = n_context + i
        attention_mask[q_idx, n_context + i + 1 :] = False

    type_ids = torch.zeros(seq_len, dtype=torch.long, device=Xc.device)
    type_ids[n_context:] = 1
    position_ids = torch.arange(seq_len, device=Xc.device)
    query_mask = torch.zeros(seq_len, dtype=torch.bool, device=Xc.device)
    query_mask[n_context:] = True

    targets = torch.cat(
        [
            yc,
            yq if yq is not None else torch.zeros(n_query, dtype=yc.dtype, device=yc.device),
        ]
    )

    return {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "type_ids": type_ids,
        "position_ids": position_ids,
        "query_mask": query_mask,
        "targets": targets,
    }


def compute_query_loss(
    preds: torch.Tensor, targets: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    """Compute MSE on query positions only."""

    if preds.shape != targets.shape:
        raise ValueError("Predictions and targets shape mismatch")
    if query_mask.dim() == 1:
        mask = query_mask
    elif query_mask.dim() == preds.dim():
        mask = query_mask
    else:
        mask = query_mask.unsqueeze(0).expand_as(preds)

    masked = preds[mask]
    masked_targets = targets[mask]
    if masked.numel() == 0:
        raise ValueError("Query mask selects no elements")
    return torch.mean((masked - masked_targets) ** 2)
