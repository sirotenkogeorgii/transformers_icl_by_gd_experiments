"""Training entry point for the in-context GD replication experiment."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from .data import LinearRegressionTaskset
from .heads import LinearReadout
from .io import compute_query_loss, pack_sequence
from .utils import collect_meta, set_seed, write_meta


class ICGDTransformer(nn.Module):
    """Simple Transformer encoder that returns layer-wise hidden states."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        n_heads: int,
        depth: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
        seq_len: int = 0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * ff_mult,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        if seq_len > 0:
            self.positional = nn.Parameter(torch.zeros(1, seq_len, d_model))
            self.position_embedding = nn.Embedding(seq_len, d_model)
        else:
            self.register_parameter("positional", None)
            self.position_embedding = None
        self.type_embedding = nn.Embedding(2, d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        return_layer_states: bool = False,
    ) -> Dict[str, List[torch.Tensor]] | torch.Tensor:
        x = self.input_proj(tokens)
        if self.positional is not None:
            x = x + self.positional
        if type_ids is not None:
            x = x + self.type_embedding(type_ids)
        if position_ids is not None and self.position_embedding is not None:
            x = x + self.position_embedding(position_ids)

        src_mask = None
        if attention_mask is not None:
            src_mask = ~attention_mask.bool()

        layer_states = []
        for layer in self.layers:
            x = layer(x, src_mask=src_mask)
            layer_states.append(x)
        x = self.norm(x)
        layer_states[-1] = x
        if return_layer_states:
            return {"layer_states": layer_states, "final": x}
        return x


def _cycle(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    packed = [
        pack_sequence(sample["Xc"], sample["yc"], sample["Xq"], sample["yq"]) for sample in batch
    ]
    tokens = torch.stack([item["tokens"] for item in packed])
    attention_mask = packed[0]["attention_mask"]
    type_ids = torch.stack([item["type_ids"] for item in packed])
    position_ids = torch.stack([item["position_ids"] for item in packed])
    targets = torch.stack([item["targets"] for item in packed])
    query_mask = packed[0]["query_mask"].unsqueeze(0).expand(len(batch), -1)
    return {
        "tokens": tokens,
        "attention_mask": attention_mask,
        "type_ids": type_ids,
        "position_ids": position_ids,
        "targets": targets,
        "query_mask": query_mask,
    }


def _parse_overrides(unknown: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in unknown:
        if "=" not in item:
            raise ValueError(f"Cannot parse override '{item}'")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _apply_overrides(config: Dict, overrides: Dict[str, str]) -> Dict:
    mapping = {
        "d": ("dataset", "d"),
        "n_context": ("dataset", "n_context"),
        "n_query": ("dataset", "n_query"),
        "sigma_w": ("dataset", "sigma_w"),
        "sigma_eps": ("dataset", "sigma_eps"),
        "n_tasks": ("dataset", "n_tasks"),
        "d_model": ("model", "d_model"),
        "n_heads": ("model", "n_heads"),
        "ffw_mult": ("model", "ff_mult"),
        "ff_mult": ("model", "ff_mult"),
        "L": ("model", "depth"),
        "depth": ("model", "depth"),
        "dropout": ("model", "dropout"),
        "batch_size": ("training", "batch_size"),
        "lr": ("training", "lr"),
        "wd": ("training", "weight_decay"),
        "weight_decay": ("training", "weight_decay"),
        "steps": ("training", "steps"),
        "warmup": ("training", "warmup"),
        "grad_clip": ("training", "grad_clip"),
        "eval_interval": ("training", "eval_interval"),
        "save_dir": (None, "save_dir"),
        "seed": (None, "seed"),
    }
    for key, value in overrides.items():
        if key not in mapping:
            raise ValueError(f"Unknown override '{key}'")
        group, name = mapping[key]
        target = config if group is None else config.setdefault(group, {})
        if name in {"d", "n_context", "n_query", "n_tasks", "depth", "n_heads", "batch_size", "steps", "warmup", "eval_interval", "seed"}:
            cast_value = int(value)
        elif name in {"sigma_w", "sigma_eps", "lr", "weight_decay", "grad_clip", "dropout"}:
            cast_value = float(value)
        else:
            cast_value = value
        target[name] = cast_value
    return config


def _build_optimizer(params, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


def evaluate(
    model: ICGDTransformer,
    readout: LinearReadout,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    readout.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                batch["tokens"],
                attention_mask=batch["attention_mask"],
                type_ids=batch["type_ids"],
                position_ids=batch["position_ids"],
                return_layer_states=True,
            )
            preds = readout(outputs["layer_states"][-1])
            loss = compute_query_loss(preds, batch["targets"], batch["query_mask"])
            losses.append(loss.item())
    model.train()
    readout.train()
    return float(sum(losses) / max(1, len(losses)))


def train(config: Dict[str, Dict]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.get("seed", 0))

    dataset_cfg = config["dataset"]
    dataset = LinearRegressionTaskset(**dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=_collate,
    )
    iterator = _cycle(dataloader)

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

    optimizer = _build_optimizer(
        itertools.chain(model.parameters(), readout.parameters()),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _lr_lambda(step, config["training"].get("warmup", 0)),
    )

    eval_dataset = LinearRegressionTaskset(
        **{**dataset_cfg, "n_tasks": config.get("eval", {}).get("nsamples", 1024), "seed": dataset_cfg["seed"] + 1}
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    save_dir = Path(config.get("save_dir", "runs/ic_gd/default"))
    save_dir.mkdir(parents=True, exist_ok=True)

    meta = collect_meta(config, config.get("seed", 0), save_dir)
    write_meta(save_dir, meta)

    best_loss = float("inf")
    best_state = None
    total_steps = config["training"]["steps"]
    eval_interval = config["training"].get("eval_interval", 1000)

    for step in range(1, total_steps + 1):
        batch = next(iterator)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            batch["tokens"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            position_ids=batch["position_ids"],
            return_layer_states=True,
        )
        preds = readout(outputs["layer_states"][-1])
        loss = compute_query_loss(preds, batch["targets"], batch["query_mask"])

        optimizer.zero_grad()
        loss.backward()
        if config["training"].get("grad_clip", 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(model.parameters(), readout.parameters()),
                config["training"]["grad_clip"],
            )
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == total_steps:
            val_loss = evaluate(model, readout, eval_loader, device)
            print(f"Step {step}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    "model": model.state_dict(),
                    "readout": readout.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(best_state, save_dir / "best.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "readout": readout.state_dict(),
            "step": total_steps,
            "val_loss": best_loss,
        },
        save_dir / "last.pt",
    )


def load_config(args: Iterable[str]) -> Dict:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--help", action="help")
    parsed, unknown = parser.parse_known_args(args)

    if parsed.config:
        with open(parsed.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        with open(Path(__file__).parent / "configs" / "linreg.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    overrides = _parse_overrides(unknown)
    config = _apply_overrides(config, overrides)
    return config


def main(argv: Iterable[str] | None = None) -> None:
    config = load_config(argv)
    train(config)


if __name__ == "__main__":
    main()
