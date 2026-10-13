"""Baseline classifiers, their CV wrapper, and statistical comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chroma_dcnn.evaluation import baselines as bl
from chroma_dcnn.evaluation.stats import (
    bonferroni_correct,
    compare_conditions,
    print_results_table,
    rank_biserial_r,
    results_table,
)


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 12)
    X = rng.normal(size=(36, 20)) * 0.1
    X[np.arange(36), y * 3] += 3.0
    return X, y


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_plsda_classifier_fits_separable(separable):
    X, y = separable
    clf = bl.PLSDAClassifier(n_components=10).fit(X, y)
    assert clf.pls.n_components == 10  # not capped at n_classes - 1
    assert clf.score(X, y) == 1.0


def test_plsda_clamps_components_to_data(separable):
    X, y = separable
    assert bl.PLSDAClassifier(n_components=50).fit(X, y).pls.n_components == X.shape[1]


def test_plsda_preserves_label_values(separable):
    X, y = separable
    labels = np.array(["a", "b", "c"])[y]
    assert set(bl.PLSDAClassifier(2).fit(X, labels).predict(X)) == {"a", "b", "c"}


def test_run_plsda_selects_components(separable):
    X, y = separable
    clf = bl.run_plsda(X, y)
    assert 1 <= clf.n_components <= 15
    assert clf.score(X, y) == 1.0
    assert bl.run_plsda(X, y, n_components_range=[1]).n_components == 1


@pytest.mark.parametrize("fn", [bl.run_random_forest, bl.run_svm, bl.run_mlp])
def test_classifiers_fit_separable(separable, fn):
    X, y = separable
    assert (fn(X, y).predict(X) == y).mean() > 0.9


@pytest.mark.parametrize("model", ["plsda", "rf", "svm", "mlp"])
def test_baseline_cv_kfold(separable, model):
    X, y = separable
    res = bl.baseline_cv(X, y, model, seeds=[0, 1], cv_folds=3)
    assert len(res["balanced_accuracy"]) == 6 == len(res["macro_f1"])
    assert np.mean(res["balanced_accuracy"]) > 0.6  # chance = 1/3


@pytest.mark.filterwarnings("ignore:A single label was found")  # LOSO folds hold one sample
def test_baseline_cv_strategies(separable):
    X, y = separable
    groups = np.repeat(np.arange(12), 3)  # 3 samples per group; groups span classes evenly
    assert len(bl.baseline_cv(X, y, "rf", seeds=[0], cv_strategy="grouped", cv_folds=3,
                              groups=groups)["balanced_accuracy"]) == 3
    assert len(bl.baseline_cv(X, y, "rf", seeds=[0], cv_strategy="loso",
                              groups=groups)["balanced_accuracy"]) == 12
    assert len(bl.baseline_cv(X[:9], y[:9], "rf", seeds=[0],
                              cv_strategy="loso")["balanced_accuracy"]) == 9
    # grouped without groups falls back to stratified k-fold
    assert len(bl.baseline_cv(X, y, "rf", seeds=[0], cv_strategy="grouped",
                              cv_folds=4)["balanced_accuracy"]) == 4


def test_baseline_cv_defaults_to_ten_seeds(separable, monkeypatch):
    X, y = separable
    stub = type("C", (), {"predict": lambda self, Z: np.zeros(len(Z))})
    monkeypatch.setattr(bl, "run_random_forest", lambda X, y, **kw: stub())
    assert len(bl.baseline_cv(X, y, "rf", cv_folds=3)["balanced_accuracy"]) == 30


def test_baseline_cv_pca_is_fit_on_train_only(separable, monkeypatch):
    """PCA must never see test rows: fit inputs have exactly the train-fold size."""
    from sklearn import decomposition

    X, y = separable
    fitted_sizes = []
    real_fit = decomposition.PCA.fit

    def spy(self, X, y=None):
        fitted_sizes.append(len(X))
        return real_fit(self, X, y)

    monkeypatch.setattr(decomposition.PCA, "fit", spy)
    res = bl.baseline_cv(X, y, "rf", seeds=[0], cv_folds=3, n_pca_components=5)
    assert fitted_sizes == [24, 24, 24]
    assert np.mean(res["balanced_accuracy"]) > 0.5  # chance = 1/3


@pytest.mark.parametrize("model", ["plsda", "svm"])
def test_baseline_cv_pca_is_refit_within_inner_cv(separable, monkeypatch, model):
    """Hyperparameter search must refit PCA on each inner training split."""
    from joblib import parallel_config
    from sklearn import decomposition

    X, y = separable
    fitted_sizes = []
    real_fit = decomposition.PCA.fit

    def spy(self, X, y=None):
        fitted_sizes.append(len(X))
        return real_fit(self, X, y)

    monkeypatch.setattr(decomposition.PCA, "fit", spy)
    with parallel_config(backend="threading"):  # keep GridSearchCV fits in-process
        bl.baseline_cv(X, y, model, seeds=[0], cv_folds=3, n_pca_components=5)
    assert max(fitted_sizes) == 24           # never sees the outer test fold
    assert any(n < 24 for n in fitted_sizes)  # refit on inner training splits


def test_baseline_cv_unknown_model(separable):
    X, y = separable
    with pytest.raises(ValueError, match="Unknown model_name"):
        bl.baseline_cv(X, y, "xgb", seeds=[0], cv_folds=3)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_rank_biserial_sign_convention():
    from scipy.stats import mannwhitneyu

    a, b = [1, 2, 3], [4, 5, 6]
    u, _ = mannwhitneyu(a, b)
    assert rank_biserial_r(u, 3, 3) == 1.0  # b > a → +1
    u, _ = mannwhitneyu(b, a)
    assert rank_biserial_r(u, 3, 3) == -1.0


def test_compare_conditions():
    df = compare_conditions({"a": [0.1, 0.2, 0.3, 0.2, 0.1], "b": [0.9, 0.8, 0.95, 0.85, 0.9],
                             "c": [0.15, 0.25, 0.2, 0.3, 0.1]})
    assert len(df) == 3
    ab = df[(df.condition_a == "a") & (df.condition_b == "b")].iloc[0]
    assert ab.effect_r == 1.0
    assert ab.p_corrected == pytest.approx(min(ab.p_raw * 3, 1.0))
    assert bool(ab.significant)
    assert df.p_corrected.is_monotonic_increasing


def test_compare_conditions_single_condition_is_empty():
    df = compare_conditions({"only": [0.5, 0.6]})
    assert df.empty and "effect_r" in df.columns


def test_bonferroni_correct():
    assert bonferroni_correct([0.01, 0.2, 0.6]) == pytest.approx([0.03, 0.6, 1.0])


def test_results_table_and_print(capsys):
    res = {"x": {"t": [0.5, 0.7]}, "y": {"t": [0.9, 0.9, 0.9]}}
    df = results_table(res)
    assert isinstance(df, pd.DataFrame)
    assert df.loc["x", "t_mean"] == pytest.approx(0.6) and df.loc["y", "t_n"] == 3
    print_results_table(res)
    out = capsys.readouterr().out
    assert "0.600 ± 0.100 (n=2)" in out and "0.900 ± 0.000 (n=3)" in out
