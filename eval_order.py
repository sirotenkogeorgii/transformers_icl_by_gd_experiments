"""Run ordering experiments and aggregate metrics into a CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from analysis.alignment import compute_alignment
from analysis.attention_stats import compute_attention_stats
from analysis.ood_eval import evaluate_ood
from runners import common
from runners import run_concat_inputs_targets as runner_lsa1
from runners import run_softmax_copy as runner_twolayer

DEFAULT_ALPHAS = (1.0, 1.5, 2.0, 3.0)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_summary(run_dir: Path) -> Dict:
    summary_path = run_dir / "metrics_summary.json"
    with summary_path.open() as f:
        return json.load(f)


def _read_ood_alpha3(run_dir: Path) -> float:
    ood_path = run_dir / "analysis" / "ood.csv"
    with ood_path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if abs(float(row["alpha"]) - 3.0) < 1e-8:
            return float(row["test_mse"])
    return float("nan")


def _run_model(model: str, run_args: common.RunArgs) -> Dict:
    args_namespace = argparse.Namespace()
    if model == "lsa1":
        runner_lsa1.configure(args_namespace, run_args)
    elif model == "twolayer":
        runner_twolayer.configure(args_namespace, run_args)
    else:
        raise ValueError(f"Unsupported model: {model}")
    return common.train_loop(run_args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ordering evaluation sweep")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model identifiers")
    parser.add_argument(
        "--ordering",
        type=str,
        required=True,
        help="Comma-separated ordering modes",
    )
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    parser.add_argument("--eta", type=float, default=0.1, help="GD baseline step size")
    parser.add_argument("--out", type=Path, default=Path("results.csv"), help="Output CSV path")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/eval_order"), help="Directory to place runs")
    parser.add_argument(
        "--train_ordering",
        type=str,
        default="auto",
        choices=common.ORDERING_CHOICES + ("auto",),
        help="Override ordering used while training (auto => use --ordering)",
    )
    parser.add_argument(
        "--eval_ordering",
        type=str,
        default="auto",
        choices=common.ORDERING_CHOICES + ("auto",),
        help="Override ordering used for evaluation batches (auto => use --ordering)",
    )
    parser.add_argument(
        "--noise_mode",
        type=str,
        default="clean",
        choices=common.NOISE_MODE_CHOICES,
        help="Noise mode applied by default to training/eval batches",
    )
    parser.add_argument(
        "--noise_p",
        type=str,
        default="0.0",
        help="Noise corruption probability (comma-separated for multiple values)",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.0,
        help="Noise standard deviation for label corruption",
    )
    parser.add_argument(
        "--noise_placement",
        type=str,
        default="mixed",
        choices=common.PLACEMENT_CHOICES,
        help="Relative placement of noisy vs clean context pairs",
    )
    parser.add_argument(
        "--train_noise_mode",
        type=str,
        default="auto",
        choices=common.NOISE_MODE_CHOICES + ("auto",),
        help="Override noise mode during training batches",
    )
    parser.add_argument(
        "--train_noise_p",
        type=float,
        default=None,
        help="Override noise probability during training batches",
    )
    parser.add_argument(
        "--train_noise_sigma",
        type=float,
        default=None,
        help="Override noise sigma during training batches",
    )
    parser.add_argument(
        "--train_noise_placement",
        type=str,
        default="auto",
        choices=common.PLACEMENT_CHOICES + ("auto",),
        help="Override noise placement during training batches",
    )
    parser.add_argument(
        "--eval_noise_mode",
        type=str,
        default="auto",
        choices=common.NOISE_MODE_CHOICES + ("auto",),
        help="Override noise mode during evaluation batches",
    )
    parser.add_argument(
        "--eval_noise_p",
        type=float,
        default=None,
        help="Override noise probability during evaluation batches",
    )
    parser.add_argument(
        "--eval_noise_sigma",
        type=float,
        default=None,
        help="Override noise sigma during evaluation batches",
    )
    parser.add_argument(
        "--eval_noise_placement",
        type=str,
        default="auto",
        choices=common.PLACEMENT_CHOICES + ("auto",),
        help="Override noise placement during evaluation batches",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    orderings = [o.strip() for o in args.ordering.split(",") if o.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows: List[Dict] = []
    _ensure_dir(args.runs_dir)

    def _norm_mode(value: str | None) -> str | None:
        if value in (None, "auto"):
            return None
        return value

    def _norm_place(value: str | None) -> str | None:
        if value in (None, "auto"):
            return None
        return value

    def _parse_float_list(values: str) -> List[float]:
        if values is None:
            return []
        parts = [v.strip() for v in values.split(",") if v.strip()]
        if not parts:
            return []
        return [float(v) for v in parts]

    noise_p_list = _parse_float_list(args.noise_p)
    if not noise_p_list:
        noise_p_list = [0.0]

    train_noise_mode = _norm_mode(args.train_noise_mode)
    train_noise_place = _norm_place(args.train_noise_placement)
    eval_noise_mode = _norm_mode(args.eval_noise_mode)
    eval_noise_place = _norm_place(args.eval_noise_placement)

    multiple_noise_ps = len(noise_p_list) > 1

    for model in models:
        for ordering in orderings:
            for noise_p_value in noise_p_list:
                for seed in seeds:
                    run_suffix = f"_p{noise_p_value}" if multiple_noise_ps else ""
                    run_name = f"{model}_{ordering}{run_suffix}_seed{seed}"
                    run_dir = (args.runs_dir / run_name).resolve()
                    run_args = common.RunArgs(
                        seed=seed,
                        steps=args.steps or int(common.config.training_steps),
                        eval_every=500,
                        run_dir=run_dir,
                        save_attention=True,
                        ordering=ordering,
                        noise_mode=args.noise_mode,
                        noise_p=noise_p_value,
                        noise_sigma=args.noise_sigma,
                        noise_placement=args.noise_placement,
                        train_ordering=args.train_ordering,
                        eval_ordering=args.eval_ordering,
                        train_noise_mode=train_noise_mode,
                        train_noise_p=args.train_noise_p,
                        train_noise_sigma=args.train_noise_sigma,
                        train_noise_placement=train_noise_place,
                        eval_noise_mode=eval_noise_mode,
                        eval_noise_p=args.eval_noise_p,
                        eval_noise_sigma=args.eval_noise_sigma,
                        eval_noise_placement=eval_noise_place,
                    )

                    print(
                        f"[eval_order] Running {model} ordering={ordering} p={noise_p_value} seed={seed}"
                    )
                    _run_model(model, run_args)

                    attn_stats = compute_attention_stats(run_dir)
                    alignment_stats = compute_alignment(run_dir, eta=args.eta)
                    evaluate_ood(run_dir, DEFAULT_ALPHAS)
                    summary = _load_summary(run_dir)
                    ood_alpha3 = _read_ood_alpha3(run_dir)

                    # Derive effective evaluation noise parameters for logging.
                    _, _, _, _ = common.resolve_noise(
                        run_args.noise_mode,
                        run_args.noise_p,
                        run_args.noise_sigma,
                        run_args.noise_placement,
                        run_args.train_noise_mode,
                        run_args.train_noise_p,
                        run_args.train_noise_sigma,
                        run_args.train_noise_placement,
                    )
                    (
                        eval_noise_mode_eff,
                        eval_noise_p_eff,
                        eval_noise_sigma_eff,
                        eval_noise_place_eff,
                    ) = common.resolve_noise(
                        run_args.noise_mode,
                        run_args.noise_p,
                        run_args.noise_sigma,
                        run_args.noise_placement,
                        run_args.eval_noise_mode,
                        run_args.eval_noise_p,
                        run_args.eval_noise_sigma,
                        run_args.eval_noise_placement,
                    )

                    rows.append(
                        {
                            "exp": str(run_dir.relative_to(Path.cwd())),
                            "model": model,
                            "ordering": ordering,
                            "noise_mode": eval_noise_mode_eff,
                            "p": eval_noise_p_eff,
                            "sigma": eval_noise_sigma_eff,
                            "placement": eval_noise_place_eff,
                            "seed": seed,
                            "test_mse": summary.get("test_loss"),
                            "align_cos": alignment_stats["align_cos"],
                            "align_l2": alignment_stats["l2_pred_diff"],
                            "auroc": attn_stats["auroc"],
                            "ood_alpha3": ood_alpha3,
                        }
                    )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp",
                "model",
                "ordering",
                "noise_mode",
                "p",
                "sigma",
                "placement",
                "seed",
                "test_mse",
                "align_cos",
                "align_l2",
                "auroc",
                "ood_alpha3",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[eval_order] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
