# Transformers Learn In-Context by Gradient Descent — Order & Noise Studies

> **Repro badge:** Additive modules only; core `src/` files remain intact except for flag-guarded hooks that expose metrics and attention.

## Table of Contents
- [Transformers Learn In-Context by Gradient Descent — Order \& Noise Studies](#transformers-learn-in-context-by-gradient-descent--order--noise-studies)
  - [Table of Contents](#table-of-contents)
  - [Project Title + One-liner](#project-title--one-liner)
  - [Motivation \& Prior Work](#motivation--prior-work)
  - [What this project does](#what-this-project-does)
  - [Repository Tree](#repository-tree)
  - [Directory \& Module Structure](#directory--module-structure)
  - [Installation \& Environment](#installation--environment)
  - [Reproducibility Defaults (from the paper)](#reproducibility-defaults-from-the-paper)
  - [Quickstart (Day-1 smoke)](#quickstart-day-1-smoke)
  - [CLI \& Flags](#cli--flags)
    - [`runners/run_softmax_copy.py`](#runnersrun_softmax_copypy)
    - [`runners/run_concat_inputs_targets.py`](#runnersrun_concat_inputs_targetspy)
  - [Experiment Pipelines](#experiment-pipelines)
    - [Order sensitivity](#order-sensitivity)
    - [Noise / outlier robustness](#noise--outlier-robustness)
    - [Attention capture \& statistics](#attention-capture--statistics)
    - [GD baseline \& alignment](#gd-baseline--alignment)
    - [OOD scaling](#ood-scaling)
    - [Evaluation drivers](#evaluation-drivers)
    - [Figures](#figures)
  - [Makefile Targets](#makefile-targets)
  - [Expected Artifacts](#expected-artifacts)
  - [Troubleshooting](#troubleshooting)
  - [FAQ](#faq)
  - [Citing \& References](#citing--references)
  - [License / Acknowledgments](#license--acknowledgments)

## Project Title + One-liner
Reproduction and extension of the *Transformers Learn In-Context by Gradient Descent* experiments with programmable order/noise manipulations, attention tracing, GD baselines, OOD stress tests, and plotting utilities.

## Motivation & Prior Work
This repository builds on the findings of von Oswald *et al.* ("Transformers Learn In-Context by Gradient Descent")—see the bundled paper at [`./docs/paper.pdf`](./paper.pdf). The goal is to reproduce the reported behaviours (copying, GD-like dynamics) and extend them with controlled curricula, noise injection, and diagnostic tooling for attention and alignment.

## What this project does
- Reproduces the original copy vs. concatenation setups using the authors’ Haiku/JAX stack without refactoring core code.
- Adds flexible APIs for context ordering and noise corruption to study both train-time curricula and inference-time sensitivity.
- Captures attention weights, extracts GD baselines, measures alignment/OUROC, and runs OOD scaling sweeps.
- Provides evaluation drivers and plotting templates to batch experiments into CSV summaries and publication-ready figures.

## Repository Tree
```
./
├── analysis/              # Post-run analytics (alignment, attention stats, ood)
├── baselines/             # Closed-form GD baseline utilities
├── demos_ref/             # Original notebook-derived demo scripts (reference only)
├── plots/                 # Plot factories (quick attention + figure templates)
├── report/figures/        # Generated figures (order/noise/alignment/OOD)
├── runners/               # CLI wrappers for training the two canonical models
├── runs/                  # Saved experiments, metrics, attention, params, etc.
├── scripts/               # Convenience scripts (e.g., smoke.sh)
├── src/                   # Authors’ core implementation (untouched logic)
├── utils_attention.py     # Attention capture helpers (additive)
├── utils_log.py           # JSONL logger utility (additive)
├── eval_order.py          # Order sensitivity sweep driver
├── eval_noise.py          # Noise/outlier robustness driver
├── plots/fig_templates.py # Report figure generator
└── requirements.txt       # Exact dependency spec
```

## Directory & Module Structure
- `utils_attention.py`, `utils_log.py`: Lightweight helpers for attention capture and structured JSONL logging.
- `runners/`: Module-mode CLIs (`run_softmax_copy.py`, `run_concat_inputs_targets.py`, shared `common.py`) exposing curriculum/noise flags and emitting run artifacts.
- `analysis/`: Standalone scripts for attention statistics (`attention_stats.py`), GD alignment (`alignment.py`), and OOD scaling (`ood_eval.py`).
- `baselines/`: `gd_step.py` implements the closed-form one-step GD baseline used by alignment.
- `plots/`: `attention_quick.py` (heatmap from saved tensors) and `fig_templates.py` (aggregate figures from CSV summaries).
- `demos_ref/`: Frozen copies of the original Colab exports; kept for reference only.
- `report/`: Houses generated figures under `report/figures/`.
- `runs/`: Canonical location for metrics, attention dumps, params, and CSV/JSON analyses per experiment.
- `scripts/`: Shell helpers (e.g., `scripts/smoke.sh`) to orchestrate small batteries of runs.
- `src/attn.py`, `src/data.py`, `src/transformer.py`, `src/train.py`, `src/config.py`: Original research code; modifications are flag-guarded and limited to exposing hooks.

## Installation & Environment
- **Python**: 3.10+ recommended.
- **Dependencies**: `pip install -r requirements.txt` (installs JAX, Haiku, Optax, matplotlib, etc.).
- **macOS note**: Apple Silicon/CPU builds can rely on the pure-CPU JAX wheels included in the requirements. For GPU acceleration, install platform-specific wheels per JAX docs.
- Ensure commands are run from the repository root with `PYTHONPATH=.` (explicit in all examples below).

## Reproducibility Defaults (from the paper)
- Optimizer: Adam, `lr=1e-3` for depth `<3`, `5e-4` otherwise; `betas=(0.9,0.999)`, no weight decay.
- Batch size: 2048 for Transformer runs; GD pretraining uses matching defaults.
- Gradient clipping: global-norm `10` (GD and TF).
- Initialisation: standard deviation `0.002 / num_layers` (matches authors’ Haiku settings).
- Positional encodings: concatenated sinusoidal PE of size `20`; clamping inputs to `[-10, 10]` when depth > 2.
- Seeds: deterministic via `config.seed`, `PYTHONHASHSEED`, NumPy/JAX RNGs.
- Teacher setup: `W₀ = 0`, linear context features, K/Q/V square projections; no additional regularisation.

## Quickstart (Day-1 smoke)
```bash
make demo_copy             # reproduces the softmax copy behaviour
make demo_concat           # concatenation baseline
bash scripts/smoke.sh      # runs Day-1 acceptance checks (ordering + noise toggles)
```
Expected artifacts:
- `runs/<exp>/metrics.jsonl` — step-by-step logging (train/eval loss).
- `runs/<exp>/attn/` — attention tensors (when `--save_attention` is used).
- `runs/<exp>/figs/attention_layer0_head0.png` — quick heatmaps from `attention_quick.py` (smoke script).

## CLI & Flags
All commands assume `PYTHONPATH=.`, module mode (`python -m ...`), and execution from repo root.

### `runners/run_softmax_copy.py`
- `--seed <int>`: RNG seed passed into `config.seed`; governs data + init.
- `--out <path>`: Output directory under `runs/`; contains metrics, params, attention.
- `--save_attention`: Save query-token attention weights at end of training.
- `--ordering {random,smallnorm2largenorm,easy2hard,hard2easy}`: Default ordering for both training and evaluation unless overridden.
- `--noise_mode {clean,label_noise,random_pairs}` (plus `--noise_p`, `--noise_sigma`, `--noise_placement {mixed,clean_first,noisy_first}`): Apply noise transforms to training/eval batches.
- `--train_ordering / --eval_ordering`: Override training vs. evaluation ordering separately (`auto` falls back to `--ordering`).
- `--train_noise_* / --eval_noise_*`: Override noise mode, probability, sigma, placement independently for curriculum vs. inference-time sensitivity.
- Outputs: `metrics.jsonl`, `metrics_summary.json`, `analysis/` (alignment, ood, attention stats), `params.pkl`, `ordering_idx.npy`, `noise_mask.npy`, optional `attn/` directory.

### `runners/run_concat_inputs_targets.py`
Same flag surface as above, targeting the concatenation architecture. Produces identical artifact layout; attention capture is identical when `--save_attention` is passed.

## Experiment Pipelines

### Order sensitivity
```bash
# Baseline curriculum: train and eval on easy→hard
PYTHONPATH=. python -m runners.run_concat_inputs_targets \
  --ordering easy2hard --out runs/curriculum_easy2hard --save_attention

# Inference-time sensitivity: train random, eval easy→hard
PYTHONPATH=. python -m runners.run_concat_inputs_targets \
  --ordering random --train_ordering auto --eval_ordering easy2hard \
  --out runs/inference_order
```
Generated artifacts include `analysis/analysis_metadata.json` describing both train/eval settings, plus `ordering_idx.npy` for the eval batch.

### Noise / outlier robustness
```bash
# Label noise curriculum (train & eval noisy first)
PYTHONPATH=. python -m runners.run_concat_inputs_targets \
  --noise_mode label_noise --noise_p 0.25 --noise_sigma 0.5 \
  --noise_placement noisy_first --out runs/noisy_curriculum

# Inference-only noise (train clean, eval noisy last)
PYTHONPATH=. python -m runners.run_concat_inputs_targets \
  --noise_mode clean \
  --eval_noise_mode label_noise --eval_noise_p 0.25 --eval_noise_sigma 0.5 \
  --eval_noise_placement clean_first --out runs/inference_noise

# Random pairs ablation (train clean, eval mixed random pairs)
PYTHONPATH=. python -m runners.run_concat_inputs_targets \
  --noise_mode clean --eval_noise_mode random_pairs --eval_noise_p 0.2 \
  --eval_noise_placement mixed --out runs/quick_random_pairs
```
Attention masks for noisy entries are saved as `noise_mask.npy` under `runs/<exp>/` and `runs/<exp>/analysis/`.

### Attention capture & statistics
1. Train with `--save_attention` to populate `runs/<exp>/attn/` (layer/head `.npy` plus raw `.npz`).
2. Run attention stats:
   ```bash
   PYTHONPATH=. python -m analysis.attention_stats --run runs/<exp>
   ```
   Produces `runs/<exp>/analysis/attention_stats.json` and `attention_stats.png` summarising clean vs. noisy attention and AUROC.

### GD baseline & alignment
```bash
PYTHONPATH=. python -m analysis.alignment --run runs/<exp> --eta 0.1
```
Outputs `runs/<exp>/analysis/alignment.json` with `l2_pred_diff` and `align_cos` (cosine similarity of sensitivities).

### OOD scaling
```bash
PYTHONPATH=. python -m analysis.ood_eval --run runs/<exp> --alphas 1 1.5 2 3
```
Writes `runs/<exp>/analysis/ood.csv` containing test MSE vs. scale factor.

### Evaluation drivers
```bash
# Order sweep (curriculum vs. sensitivity)
PYTHONPATH=. python -m eval_order \
  --models lsa1,twolayer --ordering random,smallnorm2largenorm \
  --train_ordering auto --eval_ordering auto \
  --seeds 0,1 --out results_order.csv --runs-dir runs/eval_order

# Noise sweep
PYTHONPATH=. python -m eval_noise \
  --models lsa1,twolayer --noise_mode label_noise,random_pairs \
  --p 0.0,0.25 --sigma 0.5 --placement clean_first,noisy_first \
  --train_noise_mode auto --eval_noise_mode auto \
  --ordering random --seeds 0 --out results_noise.csv --runs-dir runs/eval_noise
```
Each run directory automatically captures attention, alignment, OOD, ordering/noise metadata, and serialised parameters. Summaries land in the specified `results_*.csv`.

### Figures
```bash
PYTHONPATH=. python -m plots.fig_templates --from results_order.csv --out report/figures/order
PYTHONPATH=. python -m plots.fig_templates --from results_noise.csv --out report/figures/noise
```
Generates `order_sensitivity.png`, `noise_robustness.png`, `attention_separation.png`, `alignment_metrics.png`, `ood_curve.png` with white backgrounds suitable for reports.

## Makefile Targets
| Target        | Description                                           | Key outputs |
|---------------|-------------------------------------------------------|-------------|
| `demo_copy`   | Runs softmax-copy reproduction (`run_softmax_copy`)    | `runs/demo_copy/*` |
| `demo_concat` | Runs concatenation reproduction (`run_concat_inputs_targets`) | `runs/demo_concat/*` |
| `smoke_train` | Executes `scripts/smoke.sh` (Day-1 checks)             | `runs/smoke_*` dirs |
| `figs`        | Renders attention heatmaps for `runs/demo_concat`      | `runs/demo_concat/figs/` |
| `eval_order`  | Convenience wrapper around `eval_order.py`             | `runs/eval_order/*`, `results_order.csv` |
| `eval_noise`  | Convenience wrapper around `eval_noise.py`             | `runs/eval_noise/*`, `results_noise.csv` |

## Expected Artifacts
- `runs/<exp>/metrics.jsonl` & `metrics_summary.json` — scalar logs.
- `runs/<exp>/params.pkl` — Haiku parameters snapshot.
- `runs/<exp>/ordering_idx.npy`, `runs/<exp>/noise_mask.npy` — eval-batch metadata (mirrored under `analysis/`).
- `runs/<exp>/analysis/attention_stats.json` (and `.png`).
- `runs/<exp>/analysis/alignment.json` — GD alignment metrics.
- `runs/<exp>/analysis/ood.csv` — scaling sweep results.
- `runs/<exp>/attn/` — per-layer/head attention arrays when enabled.
- `results_order.csv`, `results_noise.csv` — aggregate experiment tables.
- `report/figures/*` — generated figures from plotting templates.

## Troubleshooting
- **`ModuleNotFoundError: runners...`** — include `PYTHONPATH=.` and run modules via `python -m runners.run_concat_inputs_targets` (the runners package has an `__init__.py`).
- **JAX wheel issues on macOS** — stick to the CPU wheels supplied in `requirements.txt`; see the JAX docs for GPU-enabled binaries.
- **Make target not found** — ensure you execute `make` from repo root; verify tabs (not spaces) if editing `Makefile`.
- **Shape/import errors** — commands must run from repo root so Haiku/Optax modules resolve correctly; double-check module-mode invocations.

## FAQ
- **Can I train on random order but evaluate on a curriculum?** Yes — use `--train_ordering auto --eval_ordering <mode>` on any runner/eval driver.
- **Where are attention weights stored?** Under `runs/<exp>/attn/` (per-layer `.npy` plus raw `.npz`), with summary stats in `runs/<exp>/analysis/attention_stats.json`.
- **How do I shrink runtime for quick tests?** Lower `--steps` (or edit `config.training_steps`) and reduce `--p`/`--sigma` grids when invoking `eval_*` drivers.
- **How do I inspect noisy vs. clean tokens?** Check `runs/<exp>/noise_mask.npy` (bool mask) and `analysis/analysis_metadata.json` for placement metadata.

## Citing & References
- Johannes von Oswald *et al.*, *Transformers Learn In-Context by Gradient Descent*. See [`./docs/paper.pdf`](./paper.pdf) for the bundled copy and cite accordingly when publishing derived work.

## License / Acknowledgments
The original implementation originates from the Google Research team (see paper above). This repository layers additive tooling atop their code; consult the upstream project for licensing details and acknowledge both sources when reusing results.
