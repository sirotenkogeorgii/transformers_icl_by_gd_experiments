#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

mkdir -p runs

run() {
  echo "[smoke] $*"
  eval "$*"
}

run "$PYTHON -m runners.run_softmax_copy --seed 0 --out runs/smoke_copy"
run "$PYTHON -m runners.run_softmax_copy --seed 0 --save_attention --out runs/smoke_copy_attn"
run "$PYTHON -m runners.run_concat_inputs_targets --seed 0 --ordering smallnorm2largenorm --out runs/smoke_concat_order"
run "$PYTHON -m runners.run_concat_inputs_targets --seed 0 --noise_mode label_noise --noise_p 0.25 --noise_sigma 0.5 --out runs/smoke_concat_noise"

if command -v tree >/dev/null 2>&1; then
  echo "[smoke] Produced artifacts:"
  tree -L 2 runs | sed 's/^/[smoke] /'
fi
