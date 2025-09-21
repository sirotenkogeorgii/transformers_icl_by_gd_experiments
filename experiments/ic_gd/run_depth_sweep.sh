#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-"experiments/ic_gd/configs/linreg.yaml"}
DEPTHS=(2 4 8 12 24)
for L in "${DEPTHS[@]}"; do
  python experiments/ic_gd/train_transformer.py --config "$CONFIG" save_dir=runs/ic_gd/d20_L${L} L=${L}
done
