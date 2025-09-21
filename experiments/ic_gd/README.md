# In-Context GD Experiment Integration Plan

This repository already provides flexible Transformer components implemented in JAX/Haiku (`src/transformer.py`), attention utilities (`src/attn.py`), configuration defaults (`src/config.py`), and the existing training harness (`src/train.py`). The new in-context gradient descent (IC-GD) replication stack will live under `experiments/ic_gd/` and reuse the core model where possible while offering a PyTorch-based pipeline specialised for the linear regression benchmark.

## Existing building blocks
- **Model** (`src/transformer.py`): defines a configurable Transformer with optional attention-only mode, positional encodings, and DEQ-style recurrence. We will instantiate this module for the IC-GD experiment by mirroring key hyperparameters (depth, heads, embedding size) inside experiment-specific configs.
- **Attention utilities** (`src/attn.py`): provides multi-head attention, MLP blocks, layer norm, and token embedding helpers. The IC-GD training script will import these abstractions when constructing the Transformer backbone.
- **Configuration** (`src/config.py`): exposes a `ConfigDict` describing model/data defaults. We will build experiment-specific YAML configs that feed a lightweight dataclass mirroring the same fields to keep interoperability with analysis notebooks.
- **Training utilities** (`src/train.py`): contains the Haiku/Optax loop that current experiments use. The IC-GD training entry point will take inspiration from this structure but operate directly in PyTorch, while keeping logging and checkpoint conventions compatible.

## New experiment files
The IC-GD workflow introduces the following files:

```
experiments/ic_gd/
  data.py              # LinearRegressionTaskset dataset & task sampling utilities
  gd_baseline.py       # Closed-form gradient descent simulator + CLI
  io.py                # Sequence packing & masking helpers for transformer inputs
  heads.py             # Shared linear readout & layerwise evaluation helpers
  train_transformer.py # Training script for transformer on IC-GD tasks
  eval_layerwise.py    # Script to evaluate layerwise losses on checkpoints
  plot_compare.py      # Plotting utility to compare GD baseline vs transformer
  utils.py             # Shared helpers (seeding, logging)
  configs/
    linreg.yaml        # Default configuration for the replication experiment
```

These components collectively implement the plan in `plans/experiment_replication.md` while minimising duplication with core modules. Subsequent steps will populate each file with the required functionality.
