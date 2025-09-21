import torch
from pathlib import Path

from torch.utils.data import DataLoader

from experiments.ic_gd.data import LinearRegressionTaskset
from experiments.ic_gd.gd_baseline import gd_curve
from experiments.ic_gd.heads import LayerwisePredictor, LinearReadout
from experiments.ic_gd.eval_layerwise import evaluate_layerwise
from experiments.ic_gd.train_transformer import ICGDTransformer, _collate, train


def test_linear_regression_taskset_deterministic():
    taskset_a = LinearRegressionTaskset(
        d=4,
        n_context=6,
        n_query=5,
        sigma_w=1.0,
        sigma_eps=0.1,
        n_tasks=3,
        seed=123,
    )
    taskset_b = LinearRegressionTaskset(
        d=4,
        n_context=6,
        n_query=5,
        sigma_w=1.0,
        sigma_eps=0.1,
        n_tasks=3,
        seed=123,
    )

    assert len(taskset_a) == 3
    sample = taskset_a[0]
    assert sample["Xc"].shape == (6, 4)
    assert sample["yc"].shape == (6,)
    assert sample["Xq"].shape == (5, 4)
    assert sample["yq"].shape == (5,)

    for idx in range(3):
        a = taskset_a[idx]
        b = taskset_b[idx]
        for key in ["Xc", "yc", "Xq", "yq", "w_star"]:
            assert torch.allclose(a[key], b[key])


def test_gd_curve_monotonic():
    taskset = LinearRegressionTaskset(
        d=5,
        n_context=12,
        n_query=6,
        sigma_w=1.0,
        sigma_eps=0.1,
        n_tasks=32,
        seed=0,
    )
    metrics = gd_curve(taskset, lr=0.1, steps=6, lr_search=True)
    mse = metrics["mse"]
    assert all(mse[i] <= mse[i - 1] + 1e-6 for i in range(1, len(mse)))


def test_transformer_overfits_and_layerwise(tmp_path):
    config = {
        "seed": 0,
        "save_dir": str(tmp_path / "run"),
        "dataset": {
            "d": 4,
            "n_context": 6,
            "n_query": 6,
            "sigma_w": 1.0,
            "sigma_eps": 0.0,
            "n_tasks": 128,
            "seed": 0,
        },
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "ff_mult": 2,
            "depth": 3,
            "dropout": 0.0,
        },
        "training": {
            "batch_size": 16,
            "steps": 60,
            "lr": 5e-4,
            "weight_decay": 0.0,
            "warmup": 0,
            "eval_interval": 30,
            "grad_clip": 0.0,
        },
        "eval": {"nsamples": 64},
    }
    train(config)

    ckpt = Path(config["save_dir"]) / "best.pt"
    state = torch.load(ckpt, map_location="cpu")
    assert state["val_loss"] < 1e-2

    dataset = LinearRegressionTaskset(**config["dataset"], n_tasks=32)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=_collate,
    )

    model = ICGDTransformer(
        input_dim=config["dataset"]["d"] + 2,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        depth=config["model"]["depth"],
        ff_mult=config["model"].get("ff_mult", 2),
        dropout=0.0,
        seq_len=config["dataset"]["n_context"] + config["dataset"]["n_query"],
    )
    readout = LinearReadout(config["model"]["d_model"])
    model.load_state_dict(state["model"])
    readout.load_state_dict(state["readout"])

    predictor = LayerwisePredictor(model, readout)
    metrics = evaluate_layerwise(predictor, loader, torch.device("cpu"))
    mse = metrics["mse"]
    for i in range(1, len(mse)):
        assert mse[i] <= mse[i - 1] + 1e-2
