"""Exact gradient-descent baseline for linear regression tasks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from .data import LinearRegressionTaskset


def _ridge_init(
    Xc: torch.Tensor, yc: torch.Tensor, ridge: float = 0.0
) -> torch.Tensor:
    d = Xc.shape[-1]
    eye = torch.eye(d, dtype=Xc.dtype, device=Xc.device)
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    return torch.linalg.solve(XtX + ridge * eye, Xty)


def gd_predict(
    Xc: torch.Tensor,
    yc: torch.Tensor,
    Xq: torch.Tensor,
    lr: float,
    steps: int,
    init: str = "zero",
    ridge: float = 0.0,
) -> List[torch.Tensor]:
    """Return predictions on ``Xq`` after each GD step."""

    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("learning rate must be positive")

    Xc = Xc.float()
    yc = yc.float()
    Xq = Xq.float()
    n_context = Xc.shape[0]
    d = Xc.shape[1]

    if init == "zero":
        w = torch.zeros(d, dtype=Xc.dtype, device=Xc.device)
    elif init == "ridge":
        w = _ridge_init(Xc, yc, ridge=ridge)
    else:
        raise ValueError(f"Unknown init '{init}'")

    preds: List[torch.Tensor] = []
    for _ in range(steps):
        grad = (Xc.T @ (Xc @ w - yc)) / n_context
        if ridge > 0:
            grad = grad + ridge * w
        w = w - lr * grad
        preds.append(Xq @ w)
    return preds


def _safe_lr(Xc: torch.Tensor) -> float:
    cov = (Xc.T @ Xc) / Xc.shape[0]
    eigvals = torch.linalg.eigvalsh(cov)
    max_eig = eigvals.max().item()
    if max_eig <= 0:
        return 1.0
    return 1.0 / (max_eig + 1e-12)


def gd_curve(
    taskset: Sequence[dict],
    lr: float,
    steps: int,
    lr_search: bool = False,
    init: str = "zero",
    ridge: float = 0.0,
) -> dict:
    """Compute mean/sem query MSE across tasks for GD baseline."""

    per_task_losses: List[torch.Tensor] = []
    for task in taskset:
        Xc = task["Xc"].to(torch.float32)
        yc = task["yc"].to(torch.float32)
        Xq = task["Xq"].to(torch.float32)
        yq = task["yq"].to(torch.float32)
        task_lr = _safe_lr(Xc) if lr_search else lr
        preds = gd_predict(Xc, yc, Xq, task_lr, steps, init=init, ridge=ridge)
        losses = [torch.mean((p - yq) ** 2) for p in preds]
        per_task_losses.append(torch.stack(losses))

    loss_matrix = torch.stack(per_task_losses)
    mean = loss_matrix.mean(dim=0)
    sem = loss_matrix.std(dim=0, unbiased=False) / math.sqrt(len(taskset))
    return {
        "mse": mean.tolist(),
        "sem": sem.tolist(),
        "steps": list(range(1, steps + 1)),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--n_context", type=int, required=True)
    parser.add_argument("--n_query", type=int, required=True)
    parser.add_argument("--sigma_w", type=float, required=True)
    parser.add_argument("--sigma_eps", type=float, required=True)
    parser.add_argument("--n_tasks", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_search", action="store_true")
    parser.add_argument("--init", choices=["zero", "ridge"], default="zero")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, required=True)
    return parser


def main(args: Iterable[str] | None = None) -> None:
    parser = _build_arg_parser()
    parsed = parser.parse_args(args=args)

    taskset = LinearRegressionTaskset(
        d=parsed.d,
        n_context=parsed.n_context,
        n_query=parsed.n_query,
        sigma_w=parsed.sigma_w,
        sigma_eps=parsed.sigma_eps,
        n_tasks=parsed.n_tasks,
        seed=parsed.seed,
    )
    metrics = gd_curve(
        taskset,
        lr=parsed.lr,
        steps=parsed.steps,
        lr_search=parsed.lr_search,
        init=parsed.init,
        ridge=parsed.ridge,
    )

    save_dir = Path(parsed.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / "gd_metrics.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved GD metrics to {out_file}")


if __name__ == "__main__":
    main()
