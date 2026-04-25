"""
Double ML (partially linear regression) ATE for RCT-style data.

Uses ``doubleml`` with nuisance learners; valid under unconfoundedness / RCT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV

import doubleml as dml

from config import (
    DML_GBDT_L_PARAMS,
    DML_N_FOLDS,
    DML_RF_M_PARAMS,
    DML_SCORE,
)

@dataclass
class DMLResult:
    ate_hat: float
    se: float


def _dml_first_coef_se(model: object) -> tuple[float, float]:
    """``doubleml`` versions differ: ``coef`` / ``se`` may be 1d or 2d."""
    c = np.asarray(getattr(model, "coef")).reshape(-1)
    s = np.asarray(getattr(model, "se")).reshape(-1)
    return float(c[0]), float(s[0])


def estimate_ate_dml_plr(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    n_folds: int = DML_N_FOLDS,
    score: str = DML_SCORE,
    random_state: int | None = 0,
) -> DMLResult:
    """
    Partially linear model: Y = theta * W + g(Z) + eps.

    ml_l models E[Y|Z], ml_m models E[W|Z] (under RCT, m is constant but cross-fitting still applies).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Flexible outcome model; simple propensity / conditional mean of W given Z
    ml_l = GradientBoostingRegressor(
        **DML_GBDT_L_PARAMS,
        random_state=random_state,
    )
    ml_m = RandomForestRegressor(
        random_state=random_state,
        **DML_RF_M_PARAMS,
    )

    data = dml.DoubleMLData.from_arrays(z, y, w)
    model = dml.DoubleMLPLR(data, clone(ml_l), clone(ml_m), n_folds=n_folds, score=score)
    if random_state is not None:
        np.random.seed(random_state)
    model.fit()

    ate, se = _dml_first_coef_se(model)
    return DMLResult(ate_hat=ate, se=se)


def estimate_ate_dml_lasso(
    y: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    *,
    n_folds: int = DML_N_FOLDS,
    random_state: int | None = 0,
) -> DMLResult:
    """Same PLR, but both nuisances use LassoCV (high-dimensional linear benchmark)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    w = np.asarray(w, dtype=np.float64).ravel()
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    ml_l = LassoCV(cv=5, random_state=random_state, max_iter=20_000)
    ml_m = LassoCV(cv=5, random_state=random_state, max_iter=20_000)

    data = dml.DoubleMLData.from_arrays(z, y, w)
    model = dml.DoubleMLPLR(data, clone(ml_l), clone(ml_m), n_folds=n_folds)
    if random_state is not None:
        np.random.seed(random_state)
    model.fit()
    ate, se = _dml_first_coef_se(model)
    return DMLResult(ate_hat=ate, se=se)
