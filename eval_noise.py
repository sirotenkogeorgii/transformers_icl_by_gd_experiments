"""Noise robustness evaluation producing aggregated metrics."""

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


def _run_model(model: str, run_args: common.RunArgs) -> None:
    args_namespace = argparse.Namespace()
    if model == "lsa1":
        runner_lsa1.configure(args_namespace, run_args)
    elif model == "twolayer":
        runner_twolayer.configure(args_namespace, run_args)
    else:
        raise ValueError(f"Unsupported model: {model}")
    common.train_loop(run_args)


def _parse_list(values: str, cast=float) -> List:
    return [cast(v) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Noise robustness sweep")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated models")
    parser.add_argument("--noise_mode", type=str, required=True, help="Noise modes (comma-separated)")
    parser.add_argument("--p", type=str, default="0.0", help="Comma-separated corruption rates")
    parser.add_argument("--sigma", type=str, default="0.0", help="Comma-separated sigmas")
    parser.add_argument("--placement", type=str, default="mixed", help="Comma-separated placements")
    parser.add_argument(
        "--ordering", type=str, default="random", help="Ordering applied during training"
    )
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--steps", type=int, default=None, help="Training steps override")
    parser.add_argument("--eta", type=float, default=0.1, help="GD baseline step size")
    parser.add_argument("--out", type=Path, default=Path("results.csv"), help="Output CSV path")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/eval_noise"), help="Directory to place runs")
    parser.add_argument(
        "--train_ordering",
        type=str,
        default="auto",
        choices=common.ORDERING_CHOICES + ("auto",),
        help="Override ordering used for training batches (auto => use --ordering)",
    )
    parser.add_argument(
        "--eval_ordering",
        type=str,
        default="auto",
        choices=common.ORDERING_CHOICES + ("auto",),
        help="Override ordering used for evaluation batches",
    )
    parser.add_argument(
        "--train_noise_mode",
        type=str,
        default="auto",
        choices=common.NOISE_MODE_CHOICES + ("auto",),
        help="Override training noise mode",
    )
    parser.add_argument(
        "--train_noise_p",
        type=float,
        default=None,
        help="Override training noise probability",
    )
    parser.add_argument(
        "--train_noise_placement",
        type=str,
        default="auto",
        choices=common.PLACEMENT_CHOICES + ("auto",),
        help="Override training noise placement",
    )
    parser.add_argument(
        "--train_noise_sigma",
        type=float,
        default=None,
        help="Override training noise sigma",
    )
    parser.add_argument(
        "--eval_noise_mode",
        type=str,
        default="auto",
        choices=common.NOISE_MODE_CHOICES + ("auto",),
        help="Override evaluation noise mode",
    )
    parser.add_argument(
        "--eval_noise_p",
        type=float,
        default=None,
        help="Override evaluation noise probability",
    )
    parser.add_argument(
        "--eval_noise_placement",
        type=str,
        default="auto",
        choices=common.PLACEMENT_CHOICES + ("auto",),
        help="Override evaluation noise placement",
    )
    parser.add_argument(
        "--eval_noise_sigma",
        type=float,
        default=None,
        help="Override evaluation noise sigma",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    noise_modes = [m.strip() for m in args.noise_mode.split(",") if m.strip()]
    ps = _parse_list(args.p, float)
    sigmas = _parse_list(args.sigma, float)
    placements = [p.strip() for p in args.placement.split(",") if p.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not ps:
        ps = [0.0]
    if not sigmas:
        sigmas = [0.0]
    if not placements:
        placements = ["mixed"]

    def _norm_mode(value: str):
        return None if value in (None, "auto") else value

    def _norm_place(value: str):
        return None if value in (None, "auto") else value

    rows: List[Dict] = []
    _ensure_dir(args.runs_dir)

    for model in models:
        for noise_mode in noise_modes:
            for p_val in ps:
                for sigma_val in sigmas:
                    for placement in placements:
                        for seed in seeds:
                            run_name = f"{model}_{noise_mode}_p{p_val}_s{sigma_val}_{placement}_seed{seed}"
                            run_dir = (args.runs_dir / run_name).resolve()
                            run_args = common.RunArgs(
                                seed=seed,
                                steps=args.steps or int(common.config.training_steps),
                                eval_every=500,
                                run_dir=run_dir,
                                save_attention=True,
                                ordering=args.ordering,
                                noise_mode=noise_mode,
                                noise_p=p_val,
                                noise_sigma=sigma_val,
                                noise_placement=placement,
                                train_ordering=args.train_ordering,
                                eval_ordering=args.eval_ordering,
                                train_noise_mode=_norm_mode(args.train_noise_mode),
                                train_noise_p=args.train_noise_p,
                                train_noise_sigma=args.train_noise_sigma,
                                train_noise_placement=_norm_place(args.train_noise_placement),
                                eval_noise_mode=_norm_mode(args.eval_noise_mode),
                                eval_noise_p=args.eval_noise_p,
                                eval_noise_sigma=args.eval_noise_sigma,
                                eval_noise_placement=_norm_place(args.eval_noise_placement),
                            )

                            print(
                                f"[eval_noise] model={model} noise={noise_mode} p={p_val} sigma={sigma_val} placement={placement} seed={seed}"
                            )
                            _run_model(model, run_args)

                            attn_stats = compute_attention_stats(run_dir)
                            alignment_stats = compute_alignment(run_dir, eta=args.eta)
                            evaluate_ood(run_dir, DEFAULT_ALPHAS)
                            summary = _load_summary(run_dir)
                            ood_alpha3 = _read_ood_alpha3(run_dir)

                            rows.append(
                                {
                                    "exp": str(run_dir.relative_to(Path.cwd())),
                                    "model": model,
                                    "ordering": args.ordering,
                                    "noise_mode": noise_mode,
                                    "p": p_val,
                                    "sigma": sigma_val,
                                    "placement": placement,
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

    print(f"[eval_noise] Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
