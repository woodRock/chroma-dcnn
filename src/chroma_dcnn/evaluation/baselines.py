"""
Published-baseline reproduction and sanity-check classifiers.

Baselines evaluated:
  PLS-DA  — matches Christmann et al. 2022 (olive oil) and Sangkanu et al. 2025 (hemp)
  RF      — Random Forest (100 trees, default sklearn params)
  SVM     — SVM with RBF kernel (C, gamma tuned via inner CV)

All baselines use the same CV protocol as the transformer conditions so that
comparisons are fair.  10 seeds × k-fold → flat list of fold scores.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneGroupOut,
    LeaveOneOut,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# PLS-DA helper
# ---------------------------------------------------------------------------

class PLSDAClassifier:
    """
    PLS discriminant analysis: project X into LVs then classify by
    nearest centroid to the PLS score means per class.

    Wraps PLSRegression with one-hot Y encoding.
    """

    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components, max_iter=500)
        self.le = LabelEncoder()
        self.class_centroids_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDAClassifier":
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        n_classes = len(self.classes_)
        Y_dummy = np.eye(n_classes)[y_enc]
        # sklearn caps n_components at n_samples - 1; clamp here so small folds don't error.
        # Components are not capped at n_classes - 1: PLS can extract further X-space
        # components after Y is spanned, and the paper's results used up to 15.
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pls = PLSRegression(n_components=n_comp, max_iter=500)
        self.pls.fit(X, Y_dummy)
        T = self.pls.transform(X)
        if not np.isfinite(T).all():
            # Degenerate fit (e.g. Y residual exhausted before n_comp components)
            raise ValueError(f"PLS-DA with {n_comp} components produced non-finite scores")
        self.class_centroids_ = np.array(
            [T[y_enc == c].mean(axis=0) for c in range(n_classes)]
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T = self.pls.transform(X)
        dists = np.linalg.norm(T[:, None] - self.class_centroids_[None], axis=-1)
        return self.le.inverse_transform(dists.argmin(axis=1))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return balanced_accuracy_score(y, self.predict(X))


class _ClampedPCA(BaseEstimator, TransformerMixin):
    """PCA whose component count is clamped to what each fit's data allows."""

    def __init__(self, n_components: int = 50, random_state: int | None = None) -> None:
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        n = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.pca_ = PCA(n_components=n, svd_solver="randomized",
                        random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.pca_.transform(X)


def make_pca_preprocessor(n_components: int, random_state: int | None = None) -> Pipeline:
    """StandardScaler + PCA, refit wherever it is used (outer or inner CV fold)."""
    return Pipeline([
        ("pca_scaler", StandardScaler()),
        ("pca", _ClampedPCA(n_components, random_state=random_state)),
    ])


class _PreprocessedClassifier:
    """A fitted preprocessor followed by a fitted classifier."""

    def __init__(self, preprocess, clf) -> None:
        self.preprocess, self.clf = preprocess, clf

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(self.preprocess.transform(X))


def _with_preprocess(preprocess, steps: list) -> Pipeline:
    return Pipeline(([("pre", clone(preprocess))] if preprocess is not None else []) + steps)


def run_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components_range: list[int] | None = None,
    cv_folds: int = 5,
    preprocess=None,
):
    """
    Fit PLS-DA with inner CV to select n_components.
    Returns fitted model on full X, y.

    ``preprocess`` (an unfitted sklearn transformer) is refit on each inner
    training split, so inner validation rows never influence it.
    """
    def _fit_pre(X_fit):
        if preprocess is None:
            return None, X_fit
        pre = clone(preprocess)
        return pre, pre.fit_transform(X_fit)

    if n_components_range is None:
        max_comp = min(X.shape[1], X.shape[0] - 1, 15)
        n_components_range = list(range(1, max_comp + 1))

    best_score, best_n = -1.0, 1
    skf = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y))), shuffle=True, random_state=0)
    for n in n_components_range:
        scores = []
        clf = PLSDAClassifier(n_components=n)
        try:
            for tr, te in skf.split(X, y):
                pre, X_fit = _fit_pre(X[tr])
                X_val = X[te] if pre is None else pre.transform(X[te])
                clf.fit(X_fit, y[tr])
                scores.append(clf.score(X_val, y[te]))
        except ValueError:
            continue  # this many components is degenerate on some inner fold
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score, best_n = mean_score, n

    pre, X_fit = _fit_pre(X)
    final_clf = PLSDAClassifier(n_components=best_n)
    final_clf.fit(X_fit, y)
    return final_clf if pre is None else _PreprocessedClassifier(pre, final_clf)


def run_random_forest(X: np.ndarray, y: np.ndarray, preprocess=None) -> Pipeline:
    clf = _with_preprocess(preprocess, [
        ("rf", RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)),
    ])
    clf.fit(X, y)
    return clf


def run_mlp(X: np.ndarray, y: np.ndarray, preprocess=None) -> Pipeline:
    """2-layer MLP (1000→256→64→C) with early stopping; input standardised."""
    pipe = _with_preprocess(preprocess, [
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=500,
            early_stopping=True,
            random_state=0,
        )),
    ])
    pipe.fit(X, y)
    return pipe


def run_svm(X: np.ndarray, y: np.ndarray, cv_folds: int = 3, preprocess=None) -> Pipeline:
    """
    SVM with RBF kernel; C and gamma tuned via inner grid-search CV.
    ``preprocess`` is part of the pipeline, so it is refit within each inner fold.
    """
    pipe = _with_preprocess(preprocess, [
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=False)),
    ])
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto"],
    }
    n_splits = min(cv_folds, len(np.unique(y)))
    gs = GridSearchCV(pipe, param_grid, cv=n_splits, scoring="balanced_accuracy", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# CV evaluation wrapper
# ---------------------------------------------------------------------------

def baseline_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    seeds: list[int] | None = None,
    cv_strategy: str = "kfold",
    cv_folds: int = 5,
    groups: np.ndarray | None = None,
    n_pca_components: int | None = None,
) -> dict[str, list[float]]:
    """
    Evaluate a baseline model using the same CV protocol as transformer conditions.

    model_name       : 'plsda' | 'rf' | 'svm' | 'mlp'
    cv_strategy      : 'kfold' | 'loso'
    n_pca_components : if set, apply StandardScaler + PCA before the classifier.
                       It is fit on the outer training split only, and refit
                       within every inner CV split used for hyperparameter
                       selection (PLS-DA, SVM), so validation rows never shape it.

    Returns {'balanced_accuracy': [...], 'macro_f1': [...]}  (n_seeds × n_folds entries)
    """
    if seeds is None:
        seeds = list(range(10))

    all_ba, all_f1 = [], []

    for seed in seeds:
        np.random.seed(seed)

        if cv_strategy == "loso" and groups is not None:
            splits = list(LeaveOneGroupOut().split(X, y, groups=groups))
        elif cv_strategy == "loso":
            splits = list(LeaveOneOut().split(X))
        elif cv_strategy == "grouped" and groups is not None:
            sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            splits = list(sgkf.split(X, y, groups=groups))
        else:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            splits = list(skf.split(X, y))

        for tr_idx, te_idx in splits:
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            pre = (make_pca_preprocessor(n_pca_components, random_state=seed)
                   if n_pca_components is not None else None)

            if model_name == "plsda":
                clf = run_plsda(X_tr, y_tr, preprocess=pre)
                preds = clf.predict(X_te)
            elif model_name == "rf":
                clf = run_random_forest(X_tr, y_tr, preprocess=pre)
                preds = clf.predict(X_te)
            elif model_name == "svm":
                clf = run_svm(X_tr, y_tr, preprocess=pre)
                preds = clf.predict(X_te)
            elif model_name == "mlp":
                clf = run_mlp(X_tr, y_tr, preprocess=pre)
                preds = clf.predict(X_te)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            all_ba.append(balanced_accuracy_score(y_te, preds))
            all_f1.append(f1_score(y_te, preds, average="macro", zero_division=0))

    return {"balanced_accuracy": all_ba, "macro_f1": all_f1}
