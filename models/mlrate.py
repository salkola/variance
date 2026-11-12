"""
MLRATE: cross-fit a prediction, then Lin regression adjustment with an interaction.

The first stage predicts ``y_post`` from pre-treatment columns only (``y_pre`` and ``z``).
The second stage reads the average treatment effect from the coefficient on ``w``
after the prediction has been centered (Lin, 2013).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from config import MLRATE_GBDT_PARAMS, MLRATE_N_FOLDS


def estimate_ate_mlrate(
    y: np.ndarray,
    w: np.ndarray,
    y_pre: np.ndarray,
    z: np.ndarray,
    *,
    n_folds: int = MLRATE_N_FOLDS,
    random_state: int = 0,
) -> float:
    """
    Cross-fit ``E[y_post | y_pre, z]``, then OLS of ``y_post`` on ``w``, the centered
    prediction, and their product. Returns the coefficient on ``w``.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    treat = np.asarray(w, dtype=np.float64).ravel()
    pre = np.asarray(y_pre, dtype=np.float64).ravel()
    features = np.asarray(z, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    n = len(y)
    if len(treat) != n or len(pre) != n or features.shape[0] != n:
        raise ValueError("y, w, y_pre, and z must have the same number of rows.")
    if n_folds < 2:
        raise ValueError("mlrate needs at least two cross-fitting folds.")

    x = np.column_stack([pre, features])
    yhat = np.empty(n, dtype=np.float64)
    folds = np.array_split(np.random.default_rng(random_state).permutation(n), n_folds)
    for k, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        model = GradientBoostingRegressor(**MLRATE_GBDT_PARAMS, random_state=random_state + k)
        model.fit(x[train], y[train])
        yhat[test] = model.predict(x[test])

    centered = yhat - float(yhat.mean())
    design = np.column_stack([np.ones(n), treat, centered, treat * centered])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])
