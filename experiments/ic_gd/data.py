"""Task sampling utilities for in-context GD experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LinearRegressionTasksetConfig:
    """Configuration for :class:`LinearRegressionTaskset`."""

    d: int
    n_context: int
    n_query: int
    sigma_w: float
    sigma_eps: float
    n_tasks: int
    seed: int = 0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


class LinearRegressionTaskset(Dataset):
    """Dataset of synthetic linear regression tasks with Gaussian data."""

    def __init__(
        self,
        d: int,
        n_context: int,
        n_query: int,
        sigma_w: float,
        sigma_eps: float,
        n_tasks: int,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("Feature dimension must be positive.")
        if n_context <= 0 or n_query <= 0:
            raise ValueError("Context and query counts must be positive.")
        if sigma_w < 0 or sigma_eps < 0:
            raise ValueError("Scales must be non-negative.")
        if n_tasks <= 0:
            raise ValueError("Number of tasks must be positive.")

        self.d = d
        self.n_context = n_context
        self.n_query = n_query
        self.sigma_w = float(sigma_w)
        self.sigma_eps = float(sigma_eps)
        self.n_tasks = n_tasks
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        self._w_star = self._sample_w(generator)
        self._Xc, eps_c = self._sample_inputs(generator, n_context)
        self._Xq, eps_q = self._sample_inputs(generator, n_query)

        self._yc = (self._Xc @ self._w_star.unsqueeze(-1)).squeeze(-1) + eps_c
        self._yq = (self._Xq @ self._w_star.unsqueeze(-1)).squeeze(-1) + eps_q

        self._move_to_device()

    def _sample_w(self, generator: torch.Generator) -> torch.Tensor:
        w = torch.randn(self.n_tasks, self.d, generator=generator, dtype=self.dtype)
        return w * self.sigma_w

    def _sample_inputs(
        self, generator: torch.Generator, n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(
            self.n_tasks, n_points, self.d, generator=generator, dtype=self.dtype
        )
        noise = torch.randn(
            self.n_tasks, n_points, generator=generator, dtype=self.dtype
        ) * self.sigma_eps
        return x, noise

    def _move_to_device(self) -> None:
        if self.device is None:
            return
        self._w_star = self._w_star.to(self.device)
        self._Xc = self._Xc.to(self.device)
        self._Xq = self._Xq.to(self.device)
        self._yc = self._yc.to(self.device)
        self._yq = self._yq.to(self.device)

    def __len__(self) -> int:
        return self.n_tasks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.n_tasks:
            raise IndexError(idx)
        return {
            "Xc": self._Xc[idx],
            "yc": self._yc[idx],
            "Xq": self._Xq[idx],
            "yq": self._yq[idx],
            "w_star": self._w_star[idx],
        }

    def ridge_solution(self, lam: float = 0.0) -> torch.Tensor:
        """Return the closed-form ridge estimator for each task."""

        eye = torch.eye(self.d, dtype=self.dtype, device=self.device)
        XtX = torch.matmul(self._Xc.transpose(-1, -2), self._Xc)
        Xty = torch.matmul(self._Xc.transpose(-1, -2), self._yc.unsqueeze(-1))
        lam_eye = (lam + 1e-6) * eye
        w_hat = torch.linalg.solve(XtX + lam_eye, Xty)
        return w_hat.squeeze(-1)

    def ridge_mse(self, lam: float = 0.0) -> torch.Tensor:
        """Return query MSE of the ridge solution averaged over tasks."""

        w_hat = self.ridge_solution(lam)
        preds = torch.einsum("tqd,td->tq", self._Xq, w_hat)
        mse = (preds - self._yq) ** 2
        return mse.mean(dim=1)


def quick_sanity_check() -> bool:
    """Return True if ridge solution achieves noise-level MSE within tolerance."""

    taskset = LinearRegressionTaskset(
        d=5,
        n_context=20,
        n_query=20,
        sigma_w=1.0,
        sigma_eps=0.05,
        n_tasks=8,
        seed=0,
    )
    mse = taskset.ridge_mse().mean()
    return torch.isclose(mse, torch.tensor(taskset.sigma_eps**2), rtol=0.5, atol=0.5)


if __name__ == "__main__":
    assert quick_sanity_check(), "Ridge solution sanity check failed."
