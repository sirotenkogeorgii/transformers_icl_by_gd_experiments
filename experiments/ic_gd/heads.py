"""Shared readout heads and layerwise evaluation helpers."""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from .io import compute_query_loss


class LinearReadout(nn.Module):
    """Linear projection applied to hidden states of query tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden).squeeze(-1)


class LayerwisePredictor(nn.Module):
    """Evaluate a transformer using a shared readout on each layer."""

    def __init__(self, transformer: nn.Module, readout: LinearReadout) -> None:
        super().__init__()
        self.transformer = transformer
        self.readout = readout

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        outputs = self.transformer(
            batch["tokens"],
            attention_mask=batch.get("attention_mask"),
            type_ids=batch.get("type_ids"),
            position_ids=batch.get("position_ids"),
            return_layer_states=True,
        )
        layer_states: List[torch.Tensor] = outputs["layer_states"]
        preds = [self.readout(h) for h in layer_states]
        return {"preds": preds, "layer_states": layer_states}

    @torch.no_grad()
    def evaluate(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return per-layer MSE on query tokens."""

        result = self.forward(batch)
        preds = result["preds"]
        losses = []
        for pred in preds:
            loss = compute_query_loss(
                pred,
                batch["targets"],
                batch["query_mask"],
            )
            losses.append(loss.detach())
        return torch.stack(losses)
