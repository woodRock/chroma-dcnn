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
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, LeaveOneOut, StratifiedGroupKFold, StratifiedKFold
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
        # PLS Y-rank = n_classes - 1; fitting more components makes Y residual constant → NaN
        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1], len(self.classes_) - 1)
        self.pls = PLSRegression(n_components=n_comp, max_iter=500)
        self.pls.fit(X, Y_dummy)
        T = self.pls.transform(X)
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


def run_plsda(
    X: np.ndarray,
    y: np.ndarray,
    n_components_range: list[int] | None = None,
    cv_folds: int = 5,
) -> PLSDAClassifier:
    """
    Fit PLS-DA with inner CV to select n_components.
    Returns fitted model on full X, y.
    """
    if n_components_range is None:
        n_classes = len(np.unique(y))
        max_comp = min(X.shape[1], X.shape[0] - 1, 15, n_classes - 1)
        n_components_range = list(range(1, max_comp + 1))

    best_score, best_n = -1.0, 1
    skf = StratifiedKFold(n_splits=min(cv_folds, len(np.unique(y))), shuffle=True, random_state=0)
    for n in n_components_range:
        scores = []
        clf = PLSDAClassifier(n_components=n)
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            scores.append(clf.score(X[te], y[te]))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score, best_n = mean_score, n

    final_clf = PLSDAClassifier(n_components=best_n)
    final_clf.fit(X, y)
    return final_clf


def run_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    clf.fit(X, y)
    return clf


def run_mlp(X: np.ndarray, y: np.ndarray) -> Pipeline:
    """2-layer MLP (1000→256→64→C) with early stopping; input standardised."""
    pipe = Pipeline([
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


def run_svm(X: np.ndarray, y: np.ndarray, cv_folds: int = 3) -> Pipeline:
    """SVM with RBF kernel; C and gamma tuned via inner grid-search CV."""
    pipe = Pipeline([
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
    n_pca_components : if set, apply StandardScaler + PCA within each fold before
                       fitting the classifier.  PCA is fit on the training split
                       only (no information leakage into the test split).

    Returns {'balanced_accuracy': [...], 'macro_f1': [...]}  (n_seeds × n_folds entries)
    """
    from sklearn.decomposition import PCA

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

            if n_pca_components is not None:
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)
                n_comp = min(n_pca_components, X_tr.shape[0] - 1, X_tr.shape[1])
                pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)

            if model_name == "plsda":
                clf = run_plsda(X_tr, y_tr)
                preds = clf.predict(X_te)
            elif model_name == "rf":
                clf = run_random_forest(X_tr, y_tr)
                preds = clf.predict(X_te)
            elif model_name == "svm":
                clf = run_svm(X_tr, y_tr)
                preds = clf.predict(X_te)
            elif model_name == "mlp":
                clf = run_mlp(X_tr, y_tr)
                preds = clf.predict(X_te)
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            all_ba.append(balanced_accuracy_score(y_te, preds))
            all_f1.append(f1_score(y_te, preds, average="macro", zero_division=0))

    return {"balanced_accuracy": all_ba, "macro_f1": all_f1}
