"""Evaluate layerwise losses of a trained transformer checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data import LinearRegressionTaskset
from .heads import LayerwisePredictor, LinearReadout
from .train_transformer import ICGDTransformer, _collate


def load_meta(meta_path: Path) -> Dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_per_task_losses(pred: torch.Tensor, targets: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    if query_mask.dim() == 1:
        mask = query_mask.bool().unsqueeze(0).expand_as(pred)
    else:
        mask = query_mask.bool()
    masked_errors = (pred - targets) ** 2 * mask
    denom = mask.sum(dim=-1).clamp_min(1)
    return masked_errors.sum(dim=-1) / denom


def evaluate_layerwise(
    predictor: LayerwisePredictor,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, List[float]]:
    per_layer_losses: List[List[float]] = []
    predictor.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = predictor.forward(batch)
            preds = outputs["preds"]
            for layer_idx, pred in enumerate(preds):
                losses = compute_per_task_losses(pred, batch["targets"], batch["query_mask"])
                if layer_idx >= len(per_layer_losses):
                    per_layer_losses.append([])
                per_layer_losses[layer_idx].extend(losses.cpu().tolist())
    predictor.train()

    mse = [float(torch.tensor(losses).mean()) for losses in per_layer_losses]
    sem = [
        float(torch.tensor(losses).std(unbiased=False) / (len(losses) ** 0.5))
        for losses in per_layer_losses
    ]
    return {
        "layers": list(range(1, len(per_layer_losses) + 1)),
        "mse": mse,
        "sem": sem,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (best.pt)")
    parser.add_argument("--nsamples", type=int, default=5000)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args(argv)

    ckpt_path = Path(args.ckpt)
    save_dir = Path(args.save_dir)
    meta = load_meta(ckpt_path.parent / "meta.json")
    config = meta["config"]

    dataset_cfg = config["dataset"].copy()
    dataset_cfg["n_tasks"] = args.nsamples
    dataset_cfg["seed"] = config.get("seed", 0) + 123

    dataset = LinearRegressionTaskset(**dataset_cfg)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_len = dataset_cfg["n_context"] + dataset_cfg["n_query"]
    model_cfg = config["model"]
    model = ICGDTransformer(
        input_dim=dataset_cfg["d"] + 2,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        depth=model_cfg["depth"],
        ff_mult=model_cfg.get("ff_mult", 4),
        dropout=model_cfg.get("dropout", 0.0),
        seq_len=seq_len,
    ).to(device)
    readout = LinearReadout(model_cfg["d_model"]).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    readout.load_state_dict(state["readout"])

    predictor = LayerwisePredictor(model, readout)
    metrics = evaluate_layerwise(predictor, loader, device)

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "layerwise_metrics.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved layerwise metrics to {out_file}")


if __name__ == "__main__":
    main()
