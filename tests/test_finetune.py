"""Fine-tuning protocol — in particular, that the test fold never drives model selection."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from chroma_dcnn.training import finetune_chroma as fc
from chroma_dcnn.training.pretrain_chroma import pretrain_chroma

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_class_weights_balanced():
    w = fc._class_weights(np.array([0, 0, 0, 1]), num_classes=3)
    assert w.tolist() == pytest.approx([4 / 6, 4 / 2, 0.0])


def test_class_weights_absent_class_is_zero():
    w = fc._class_weights(np.array([2, 2]), num_classes=3)
    assert w[0] == 0 and w[1] == 0 and w[2] == pytest.approx(1.0)


def test_inner_val_split_disjoint_and_stratified():
    y = np.repeat([0, 1, 2], 10)
    fit, val = fc._inner_val_split(y, 0.2, seed=0)
    assert set(fit).isdisjoint(val)
    assert sorted(np.concatenate([fit, val])) == list(range(len(y)))
    assert len(val) == 6
    assert np.bincount(y[val]).tolist() == [2, 2, 2]


def test_inner_val_split_is_deterministic():
    y = np.repeat([0, 1], 10)
    a = fc._inner_val_split(y, 0.3, seed=7)
    b = fc._inner_val_split(y, 0.3, seed=7)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_inner_val_split_grouped_keeps_groups_whole():
    y = np.repeat([0, 1], 12)
    groups = np.array([f"g{i // 3}" for i in range(24)])
    fit, val = fc._inner_val_split(y, 0.25, seed=0, groups=groups)
    assert set(groups[fit]).isdisjoint(groups[val])
    assert len(val) > 0


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1, 1.5])
def test_inner_val_split_rejects_bad_fraction(frac):
    with pytest.raises(ValueError, match="val_fraction"):
        fc._inner_val_split(np.repeat([0, 1], 5), frac, seed=0)


# ---------------------------------------------------------------------------
# _train_fold
# ---------------------------------------------------------------------------

def _model(cfg):
    m = cfg["model"]
    return fc.ChromatogramCNN(fc.ChromaCNNConfig(
        mz_max=m["mz_max"], cnn_channels=m["cnn_channels"], kernel_size=m["kernel_size"],
        num_classes=cfg["task"]["num_classes"], dropout=m["dropout"],
    ))


def test_train_fold_selects_on_val_and_scores_on_test(monkeypatch, chroma_dataset, finetune_cfg):
    """Regression test for the leak: early stopping must only ever see the val set."""
    paths, y, _ = chroma_dataset
    fit, val, test = slice(0, 14), slice(14, 18), slice(18, 24)
    order = np.random.default_rng(0).permutation(len(y))
    paths = [paths[i] for i in order]
    y = y[order]

    eval_sets, metric_sets = [], []
    real_eval, real_metrics = fc._eval_loss, fc._compute_metrics

    def spy_eval(model, loader, criterion, device):
        eval_sets.append(loader.dataset)
        return real_eval(model, loader, criterion, device)

    def spy_metrics(model, loader, device):
        metric_sets.append(loader.dataset)
        return real_metrics(model, loader, device)

    monkeypatch.setattr(fc, "_eval_loss", spy_eval)
    monkeypatch.setattr(fc, "_compute_metrics", spy_metrics)

    metrics = fc._train_fold(
        _model(finetune_cfg),
        paths[fit], paths[val], paths[test],
        y[fit], y[val], y[test],
        finetune_cfg, torch.device("cpu"),
    )

    assert eval_sets and all(len(ds) == 4 for ds in eval_sets)  # val only
    assert len(metric_sets) == 1 and len(metric_sets[0]) == 6  # test, once
    assert eval_sets[0] is not metric_sets[0]
    assert set(metrics) == {"balanced_accuracy", "macro_f1"}
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_train_fold_restores_best_checkpoint(monkeypatch, chroma_dataset, finetune_cfg):
    """Val loss is best at epoch 1 then worsens → final weights equal epoch-1 weights."""
    paths, y, _ = chroma_dataset
    finetune_cfg["finetuning"].update(epochs=5, early_stopping_patience=10)
    losses = iter([1.0, 2.0, 3.0, 4.0, 5.0])
    snapshots = []

    def fake_eval(model, loader, criterion, device):
        snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
        return next(losses)

    captured = {}

    def fake_metrics(model, loader, device):
        captured["state"] = model.state_dict()
        return {"balanced_accuracy": 0.0, "macro_f1": 0.0}

    monkeypatch.setattr(fc, "_eval_loss", fake_eval)
    monkeypatch.setattr(fc, "_compute_metrics", fake_metrics)
    fc._train_fold(_model(finetune_cfg), paths[:12], paths[12:18], paths[18:],
                   y[:12], y[12:18], y[18:], finetune_cfg, torch.device("cpu"))

    assert len(snapshots) == 5
    for k, v in captured["state"].items():
        assert torch.equal(v, snapshots[0][k]), k


def test_train_fold_early_stops(monkeypatch, chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    finetune_cfg["finetuning"].update(epochs=50, early_stopping_patience=2)
    calls = []

    def flat_eval(model, loader, criterion, device):
        calls.append(1)
        return 1.0

    monkeypatch.setattr(fc, "_eval_loss", flat_eval)
    fc._train_fold(_model(finetune_cfg), paths[:12], paths[12:18], paths[18:],
                   y[:12], y[12:18], y[18:], finetune_cfg, torch.device("cpu"), lr=1e-3)
    assert len(calls) == 3  # first epoch sets best, then 2 stale epochs


def test_train_fold_learns_separable_data(chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    finetune_cfg["finetuning"].update(epochs=30, early_stopping_patience=30)
    order = np.random.default_rng(1).permutation(len(y))
    paths, y = [paths[i] for i in order], y[order]
    m = fc._train_fold(_model(finetune_cfg), paths[:15], paths[15:19], paths[19:],
                       y[:15], y[15:19], y[19:], finetune_cfg, torch.device("cpu"))
    assert m["balanced_accuracy"] > 0.5


def test_eval_loss_empty_loader_is_zero():
    assert fc._eval_loss(None, [], None, "cpu") == 0.0


# ---------------------------------------------------------------------------
# ChromaFinetuner
# ---------------------------------------------------------------------------

def _record_folds(monkeypatch):
    calls = []

    def fake_train_fold(model, fit_p, val_p, test_p, fit_y, val_y, test_y, cfg, device, lr=None):
        calls.append({"fit": set(fit_p), "val": set(val_p), "test": set(test_p), "lr": lr,
                      "val_y": val_y})
        return {"balanced_accuracy": 1.0, "macro_f1": 1.0}

    monkeypatch.setattr(fc, "_train_fold", fake_train_fold)
    return calls


def test_evaluate_condition_partitions_every_fold(monkeypatch, chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    finetune_cfg["task"]["cv_seeds"] = [0, 1]
    calls = _record_folds(monkeypatch)

    res = fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu").evaluate_condition(
        "from_scratch")

    assert len(calls) == 6 and len(res["balanced_accuracy"]) == 6
    for c in calls:
        assert c["fit"].isdisjoint(c["val"])
        assert c["fit"].isdisjoint(c["test"])
        assert c["val"].isdisjoint(c["test"])
        assert c["fit"] | c["val"] | c["test"] == set(paths)
        assert len(set(c["val_y"])) == 3  # stratified: every class in val
    for seed_calls in (calls[:3], calls[3:]):
        tests = [p for c in seed_calls for p in c["test"]]
        assert sorted(tests) == sorted(paths)  # outer folds cover each sample once


def test_evaluate_condition_grouped_never_splits_groups(monkeypatch, chroma_dataset,
                                                        finetune_cfg):
    paths, y, groups = chroma_dataset
    finetune_cfg["task"]["cv_strategy"] = "grouped"
    calls = _record_folds(monkeypatch)
    fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu",
                       groups=list(groups)).evaluate_condition("from_scratch")
    g = dict(zip(paths, groups))
    for c in calls:
        fit_g, val_g, test_g = ({g[p] for p in c[k]} for k in ("fit", "val", "test"))
        assert fit_g.isdisjoint(val_g) and fit_g.isdisjoint(test_g) and val_g.isdisjoint(test_g)


def test_evaluate_condition_learning_rates(monkeypatch, chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    finetune_cfg["finetuning"].update(lr=5e-5, lr_scratch=1e-3)
    calls = _record_folds(monkeypatch)
    ft = fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu")
    ft.evaluate_condition("from_scratch", seeds=[0])
    ft.evaluate_condition("chroma_pretrain", seeds=[0])
    assert {c["lr"] for c in calls[:3]} == {1e-3}
    assert {c["lr"] for c in calls[3:]} == {5e-5}


def test_evaluate_condition_end_to_end(chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    res = fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu").evaluate_condition(
        "from_scratch")
    assert len(res["balanced_accuracy"]) == 3 == len(res["macro_f1"])


def test_run_all_conditions_without_checkpoint(monkeypatch, chroma_dataset, finetune_cfg):
    paths, y, _ = chroma_dataset
    _record_folds(monkeypatch)
    res = fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu").run_all_conditions()
    assert list(res) == ["from_scratch"]


def test_run_all_conditions_with_checkpoint(chroma_dataset, finetune_cfg, pretrain_cfg, spectra):
    ckpt = pretrain_chroma(pretrain_cfg, spectra, torch.device("cpu"))
    finetune_cfg["pretrained_checkpoints"]["chroma_pretrain"] = ckpt
    finetune_cfg["finetuning"]["epochs"] = 1
    paths, y, _ = chroma_dataset
    res = fc.ChromaFinetuner(finetune_cfg, paths, y, device="cpu").run_all_conditions()
    assert list(res) == ["from_scratch", "chroma_pretrain"]


def test_build_model_loads_checkpoint(pretrain_cfg, finetune_cfg, spectra):
    ckpt = pretrain_chroma(pretrain_cfg, spectra, torch.device("cpu"))
    ft = fc.ChromaFinetuner(finetune_cfg, [], np.array([]), device="cpu")
    pre = fc._build_model(ft.model_config, "chroma_pretrain", ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert torch.equal(pre.spec_proj.weight, saved["spec_proj_state"]["weight"])
    scratch = fc._build_model(ft.model_config, "chroma_pretrain", None)
    assert not torch.equal(scratch.spec_proj.weight, saved["spec_proj_state"]["weight"])


@pytest.mark.parametrize(("cuda", "mps", "expected"), [
    (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_device_autoselect(monkeypatch, finetune_cfg, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    ft = fc.ChromaFinetuner(finetune_cfg, [], np.array([]))
    assert ft.device.type == expected


def test_explicit_device(finetune_cfg):
    assert fc.ChromaFinetuner(finetune_cfg, [], np.array([]), device="cpu").device.type == "cpu"
