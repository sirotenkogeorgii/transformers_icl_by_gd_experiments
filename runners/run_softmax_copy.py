"""CLI runner for the softmax-copy curriculum experiment.

This script wraps the original notebook logic into a reproducible command-line
entry point that relies exclusively on the authors' modules under ``src/``.
"""

from __future__ import annotations

import json
from typing import Any

from runners import common
from src.config import config


def configure(args: Any, run_args: common.RunArgs) -> None:
    """Apply experiment-specific configuration overrides."""

    config.classic_token_const = True
    config.local_usage = True
    config.distract_size = 0
    config.training_steps = run_args.steps
    config.training_steps_gd = run_args.steps
    config.use_softmax = False
    config.first_layer_sm = True
    config.use_non_lin_mix = False
    config.deq = False
    config.att_only_trans = True
    config.layer_norm = False
    config.out_proj = False
    config.in_proj = False
    config.adam = True
    config.pre_train_gd = True
    config.gd_lr = 1e-3
    config.dataset_size = 10
    config.input_size = 10
    config.num_layers = 2
    config.num_heads = 1
    config.init_scale = 0.02 / config.num_layers
    config.wd = 0.0
    config.dropout_rate = 0.0
    config.y_update = False
    config.input_range = 1.0
    config.gd_deq = True
    config.key_size = 20
    config.analyse = False
    config.ana_copy = True
    config.num_seeds = 1

    config.save_attention = run_args.save_attention
    config.ordering = run_args.ordering
    config.noise_mode = run_args.noise_mode
    config.noise_p = run_args.noise_p
    config.noise_sigma = run_args.noise_sigma
    config.noise_placement = run_args.noise_placement

    # common.configure_training_defaults(depth=config.num_layers, pos_enc_size=10)
    common.configure_training_defaults(depth=config.num_layers, pos_enc_size=6)


def main() -> None:
    parser = common.build_common_parser("Softmax-copy runner")
    args = parser.parse_args()
    run_args = common.normalize_run_args(args, default_run_name="demo_copy")
    configure(args, run_args)
    summary = common.train_loop(run_args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
