# Reproducing experiment from the paper “Transformers Learn In-Context by Gradient Descent” by Johannes von Oswald (Linear Regression)

**Goal:** Reproduce the key result that, on synthetic linear regression tasks, the loss curve of a trained transformer measured across layers closely matches the loss curve of true gradient descent (GD) across steps. Produce a plot overlaying **GD Steps** (green line) vs **Transformer Layers** (purple + markers) with near overlap.

---

## 0) Scope & Target Figure

* **Task family:** Linear regression with Gaussian features and noise.
* **Per-task data:**

  * Sample $w_* \sim \mathcal N(0, \sigma_w^2 I_d)$
  * Context: $n_c$ pairs $(x_i, y_i)$ where $x_i \sim \mathcal N(0, I_d)$, $y_i = x_i^\top w_* + \varepsilon_i$, $\varepsilon_i \sim \mathcal N(0, \sigma_\varepsilon^2)$
  * Queries: $n_q$ pairs $(x_j, y_j)$ drawn the same way; only $x_j$ (and mask) are shown to the model during training; loss computed on query targets.
* **Baselines & Outputs:**

  1. **True GD** on squared loss using only context: compute test MSE on queries after $t=1..T$ steps.
  2. **Transformer** of depth $L$: compute MSE from **layerwise readouts** at layers $\ell=1..L$ on the same queries.
  3. **Plot** both curves on one axis.

---

## 1) Repo Audit & Minimal Additions

**Codex — Task:** Read the repo and summarize model, attention, config, and training utilities. Propose a minimal set of new files under `experiments/ic_gd/` and how they integrate with existing components.

**Deliverable:** A tree diff similar to:

```
experiments/ic_gd/
  data.py
  gd_baseline.py
  io.py
  heads.py
  train_transformer.py
  eval_layerwise.py
  plot_compare.py
  configs/linreg.yaml
```

**Acceptance:** Clear mapping to existing Transformer class & config system; no duplication of core modules.

---

## 2) Linear Regression Task Sampler

**File:** `experiments/ic_gd/data.py`

**Implement:** `LinearRegressionTaskset` (PyTorch Dataset)

* Args: `d, n_context, n_query, sigma_w, sigma_eps, n_tasks, seed`.
* `__getitem__` returns dict of tensors: `Xc (n_c×d)`, `yc (n_c)`, `Xq (n_q×d)`, `yq (n_q)`, `w_star (d)`.
* Deterministic given `seed`.

**Quick tests:**

* Shapes correct and repeatable.
* Closed-form ridge solution on a sampled task achieves MSE ≈ `sigma_eps**2` (sanity check).

---

## 3) Exact GD Baseline

**File:** `experiments/ic_gd/gd_baseline.py`

**Functions:**

* `gd_predict(Xc, yc, Xq, lr, steps, init="zero"|"ridge", ridge=0.0)` → list of length `steps`; each element is predictions on `Xq` after step `t`.
* `gd_curve(taskset, lr, steps, lr_search=False)` → mean and sem of MSE over tasks at each step. Optionally sweep or line-search LR per task.

**Update rule:**
$w_{t+1} = w_t - \eta \frac{1}{n_c} X_c^\top (X_c w_t - y_c)$

**CLI:** Save JSON `{"mse": [...], "sem": [...], "steps": T}` to `save_dir/gd_metrics.json`.

**Acceptance:** For stable `lr`, the loss is monotone decreasing.

---

## 4) Transformer IO: Packing Sequences

**File:** `experiments/ic_gd/io.py`

**Implement:**

* `pack_sequence(Xc, yc, Xq)` → tensor `[seq_len, d+2]` where each token holds `[x (d), y_or_0 (1), sep (1)]`.
* Create attention masks so **queries may attend to all context** but not future queries (causal over queries).
* Produce position/type embeddings if required by the base model.

**Loss:** Compute only on query positions.

**Acceptance:** Tiny overfit (single task) reaches near-zero train loss.

---

## 5) Layerwise Readouts (Shared Head)

**File:** `experiments/ic_gd/heads.py`

**Implement:**

* `LinearReadout(d_model)` → `ŷ = proj(h_state)` for query tokens.
* Wrap transformer to **return hidden states for each layer** (use forward hooks or modify forward to cache intermediates).
* `LayerwisePredictor(transformer, readout)` → given a batch, returns per-layer MSE on queries for `ℓ=1..L`.

**Acceptance:** `eval_layerwise.py` prints a decreasing vector of losses across layers on a trained model.

---

## 6) Training Script

**File:** `experiments/ic_gd/train_transformer.py`

**Behavior:**

* Stream tasks from `LinearRegressionTaskset` (meta-learning by sampling new tasks each step).
* Model hyperparams: `d_model, n_heads, ffw_mult, L` (reuse repo’s Transformer).
* Optimizer: AdamW; `lr`, `weight_decay`, `warmup`, optional cosine decay, `grad_clip`.
* Log: train MSE on queries each step; periodic eval on a held-out taskset.
* Save best checkpoint and config to `save_dir`.

**Default hyperparams:**

* Data: `d=20, n_context=20, n_query=20, sigma_w=1.0, sigma_eps=0.05`.
* Model: `d_model=128, n_heads=4, ffw_mult=4, L in {2,4,8,12,24}`.
* Train: `batch_size=64, steps=50k`, `lr=1e-4`, `wd=0.01`, `grad_clip=1.0`.

**Acceptance:** Training/eval losses stable; checkpoints saved.

---

## 7) Evaluation: Layer-Depth Curve

**File:** `experiments/ic_gd/eval_layerwise.py`

**Behavior:**

* Load a checkpoint.
* For `nsamples` random tasks, compute per-layer MSE on queries.
* Aggregate mean and SEM for `ℓ=1..L`.
* Save `layerwise_metrics.json` with `{layers: L, mse: [...], sem: [...]}`.

**Acceptance:** Loss decreases with depth; arrays consistent.

---

## 8) GD Curve on Same Distribution

**File:** `experiments/ic_gd/gd_baseline.py` (CLI)

**Behavior:**

* Use same `d, n_context, n_query, sigma_eps` as training.
* Either (a) fixed `lr` chosen by spectral bound estimate (power iteration on `XᵀX/n_c`), or (b) sweep over a grid and take lower envelope.
* Save to `gd_metrics.json`.

**Acceptance:** Reasonable, smooth, decreasing MSE vs steps.

---

## 9) Plotting Overlay

**File:** `experiments/ic_gd/plot_compare.py`

**Behavior:**

* Load `gd_metrics.json` and `layerwise_metrics.json`.
* Plot green solid line for GD; purple plus markers for Transformer.
* Labels: **x:** `GD Steps / Transformer Layers`, **y:** `Loss`.
* Legend entries exactly: `Gradient descent` and `Trained Transformer`.
* Save `compare_gd_vs_transformer.png` and `.pdf` in `save_dir`.

**Acceptance:** Curves overlap qualitatively like the target figure.

---

## 10) Configs & Sweeps

**File:** `experiments/ic_gd/configs/linreg.yaml`

**Include presets:**

* `depth_sweep: [2,4,8,12,24]`
* `noise_sweep: [0.0, 0.02, 0.05, 0.1]`
* `context_sweep: [10,20,40]`

**Utility:** `run_depth_sweep.sh` to train for each `L`, then evaluate and plot.

---

## 11) Reproducibility & Logging

* Implement `utils/repro.py` to set seeds for PyTorch/NumPy, control cuDNN determinism, and log:

  * git commit hash, full config, seeds, device info.
* Store as `meta.json` in each `save_dir`.

---

## 12) Tests (CI Optional)

**File:** `tests/test_ic_gd.py`

* Dataset determinism & shapes.
* GD baseline loss decreases with steps (given safe `lr`).
* Transformer can overfit a single task.
* On a trained checkpoint, layerwise losses are non-increasing up to noise (allow small tolerance).

---

## 13) Runbook

**Train (example, L=12):**

```bash
python experiments/ic_gd/train_transformer.py \
  d=20 n_context=20 n_query=20 sigma_w=1.0 sigma_eps=0.05 \
  d_model=128 n_heads=4 ffw_mult=4 L=12 \
  batch_size=64 lr=1e-4 wd=0.01 steps=50000 warmup=2000 \
  save_dir=runs/ic_gd/d20_L12 seed=1
```

**Evaluate layerwise:**

```bash
python experiments/ic_gd/eval_layerwise.py \
  ckpt=runs/ic_gd/d20_L12/best.pt nsamples=5000 \
  save_dir=runs/ic_gd/d20_L12
```

**GD baseline:**

```bash
python experiments/ic_gd/gd_baseline.py \
  d=20 n_context=20 n_query=20 nsamples=5000 T=50 \
  lr_search=true save_dir=runs/ic_gd/baseline
```

**Plot overlay:**

```bash
python experiments/ic_gd/plot_compare.py \
  gd=runs/ic_gd/baseline/gd_metrics.json \
  tr=runs/ic_gd/d20_L12/layerwise_metrics.json \
  out=runs/ic_gd/d20_L12/compare_gd_vs_transformer
```

---

## 14) Acceptance Checklist

* [ ] Dataset sampler returns correct shapes; seed-stable.
* [ ] GD MSE decreases for safe LR; JSON saved.
* [ ] Transformer trains stably; checkpoint saved.
* [ ] Layerwise MSE decreases with depth; JSON saved.
* [ ] Final plot shows near overlap between GD steps and Transformer layers.

---

## 15) Common Pitfalls & Notes

* **Distribution match:** Use identical data distribution for GD and Transformer eval.
* **Masking:** Allow queries to attend to all context; prevent leakage across queries (causal among queries).
* **Shared head:** Use a **single** linear readout for all layers for comparability; do not refit per layer.
* **LR selection for GD:** Too large diverges; use spectral bound or LR sweep.
* **Eval size:** Use thousands of tasks to reduce variance in curves.
* **Ridge reference:** (Optional) plot closed-form ridge solution MSE as a horizontal line.

---

## 16) Ready‑to‑Paste Codex Prompts

### A) Create dataset

> Write `experiments/ic_gd/data.py` implementing `LinearRegressionTaskset` with args `(d, n_context, n_query, sigma_w, sigma_eps, n_tasks, seed)`. `__getitem__` returns tensors `Xc, yc, Xq, yq, w_star`. Make it deterministic with `seed`. Add a quick test that the ridge closed-form achieves MSE ≈ `sigma_eps**2` on a sampled task.

### B) GD baseline

> Implement `experiments/ic_gd/gd_baseline.py` with `gd_predict` and `gd_curve` per the update rule `w_{t+1} = w_t - eta*(Xc.T @ (Xc @ w_t - yc))/n_context`. Add a CLI to compute and save `{mse, sem, steps}` to `gd_metrics.json`. Include a unit test that loss decreases for a safe `lr`.

### C) Transformer IO & masking

> Implement `experiments/ic_gd/io.py` with `pack_sequence(Xc, yc, Xq)` forming tokens `[x (d), y_or_0 (1), sep (1)]`. Provide attention masks so queries can attend to all context but not future queries. Ensure loss is computed only on query tokens.

### D) Layerwise readouts

> Implement `experiments/ic_gd/heads.py` with a shared `LinearReadout(d_model)` and a wrapper that returns hidden states after each layer and computes per-layer query MSE. Do not train separate heads per layer.

### E) Training script

> Create `experiments/ic_gd/train_transformer.py` that streams fresh tasks, optimizes with AdamW, logs, evaluates, and saves best checkpoints. Add config args listed in the plan.

### F) Eval & plot

> Implement `experiments/ic_gd/eval_layerwise.py` to save `{layers, mse, sem}` and `experiments/ic_gd/plot_compare.py` to reproduce the target figure with styles/labels from the plan.

---

**End of Plan**