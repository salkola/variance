"""
CUPED-style variance reduction for an RCT with a scalar or vector pre-period outcome.

References (conceptual):
- Deng, Xu, Kohavi, Walker (2013) "Improving the Sensitivity of Online Controlled Experiments..."
  (CUPED)
- Generalized multivariate extension: regression of post on pre with centered pre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import CUPED_DEFAULT_PRE_MEAN

CupedMode = Literal[
    "naive_diff",
    "cuped_known_mean",
    "cuped_one_theta",
    "cuped_two_theta",
    "cuped_full_sample",
    "cuped_multivariate",
    "ancova_ols",
]


@dataclass
class CupedResult:
    ate_hat: float
    se_delta: float
    theta: np.ndarray | float | None


def _se_two_sample_mean_diff(y1: np.ndarray, y0: np.ndarray) -> float:
    n1, n0 = len(y1), len(y0)
    v1 = float(np.var(y1, ddof=1)) if n1 > 1 else 0.0
    v0 = float(np.var(y0, ddof=1)) if n0 > 1 else 0.0
    return float(np.sqrt(v1 / n1 + v0 / n0))


def estimate_ate_cuped(
    y_post: np.ndarray,
    w: np.ndarray,
    y_pre: np.ndarray,
    mode: CupedMode,
    y_pre_pop_mean: float | np.ndarray | None = None,
) -> CupedResult:
    """
    Estimate ATE on ``y_post`` using different CUPED / ANCOVA constructions.

    Parameters
    ----------
    y_pre
        Pre-period outcome(s): shape (n,) or (n, k) for multivariate CUPED.
    y_pre_pop_mean
        For ``cuped_known_mean``, population mean(s) for centering ``y_pre``.
        If None and scalar ``y_pre``, uses ``CUPED_DEFAULT_PRE_MEAN`` from config.
    """
    y = np.asarray(y_post, dtype=np.float64).ravel()
    treat = np.asarray(w, dtype=np.float64).ravel()
    pre = np.asarray(y_pre, dtype=np.float64)

    if pre.ndim == 1:
        pre = pre.reshape(-1, 1)

    n = len(y)
    if len(treat) != n or pre.shape[0] != n:
        raise ValueError("y_post, w, and y_pre must have compatible lengths.")

    w1 = treat == 1
    w0 = treat == 0

    if mode == "naive_diff":
        ate = float(y[w1].mean() - y[w0].mean())
        se = _se_two_sample_mean_diff(y[w1], y[w0])
        return CupedResult(ate_hat=ate, se_delta=se, theta=None)

    if mode == "ancova_ols":
        pre_c = pre - pre.mean(axis=0, keepdims=True)
        design = np.column_stack([np.ones(n), treat, pre_c])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        ate = float(beta[1])
        resid = y - design @ beta
        dof = max(n - design.shape[1], 1)
        inv_xt_x = np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(inv_xt_x[1, 1] * (resid @ resid) / dof))
        return CupedResult(ate_hat=ate, se_delta=se, theta=beta[2:].copy())

    if mode == "cuped_known_mean":
        if pre.shape[1] != 1:
            raise ValueError("cuped_known_mean expects a scalar pre-period outcome.")
        mu = CUPED_DEFAULT_PRE_MEAN if y_pre_pop_mean is None else float(y_pre_pop_mean)
        x0 = pre[:, 0]
        theta = float(np.cov(y, x0, ddof=1)[0, 1] / np.var(x0, ddof=1))
        y_adj = y - theta * (x0 - mu)
        ate = float(y_adj[w1].mean() - y_adj[w0].mean())
        se = _se_two_sample_mean_diff(y_adj[w1], y_adj[w0])
        return CupedResult(ate_hat=ate, se_delta=se, theta=theta)

    if mode == "cuped_one_theta":
        if pre.shape[1] != 1:
            raise ValueError("cuped_one_theta expects a scalar pre-period outcome.")
        x0 = pre[:, 0]
        theta = float(np.cov(y, x0, ddof=1)[0, 1] / np.var(x0, ddof=1))
        mu = float(x0.mean())
        y_adj = y - theta * (x0 - mu)
        ate = float(y_adj[w1].mean() - y_adj[w0].mean())
        se = _se_two_sample_mean_diff(y_adj[w1], y_adj[w0])
        return CupedResult(ate_hat=ate, se_delta=se, theta=theta)

    if mode == "cuped_two_theta":
        # Arm-specific pre→post slopes θ_T, θ_C; common centering μ_pre.
        #
        # Important: if you center within each arm (μ_T = mean_pre_T, μ_C = mean_pre_C),
        # then the adjusted within-arm means equal the original within-arm means exactly,
        # so the ATE point estimate becomes identical to naive for every dataset (no variance reduction).
        if pre.shape[1] != 1:
            raise ValueError("cuped_two_theta expects a scalar pre-period outcome.")
        x0 = pre[:, 0]
        n1, n0 = int(w1.sum()), int(w0.sum())
        if n1 < 2 or n0 < 2:
            raise ValueError("cuped_two_theta needs at least two units per arm.")
        v1 = float(np.var(x0[w1], ddof=1))
        v0 = float(np.var(x0[w0], ddof=1))
        if v1 <= 0.0 or v0 <= 0.0:
            raise ValueError("cuped_two_theta needs positive within-arm variance of y_pre.")
        theta_t = float(np.cov(y[w1], x0[w1], ddof=1)[0, 1] / v1)
        theta_c = float(np.cov(y[w0], x0[w0], ddof=1)[0, 1] / v0)
        mu_pre = float(x0.mean())
        y_adj = np.empty_like(y)
        y_adj[w1] = y[w1] - theta_t * (x0[w1] - mu_pre)
        y_adj[w0] = y[w0] - theta_c * (x0[w0] - mu_pre)
        ate = float(y_adj[w1].mean() - y_adj[w0].mean())
        se = _se_two_sample_mean_diff(y_adj[w1], y_adj[w0])
        return CupedResult(ate_hat=ate, se_delta=se, theta=np.array([theta_t, theta_c]))

    if mode == "cuped_full_sample":
        return estimate_ate_cuped(y_post, w, y_pre, "cuped_one_theta", y_pre_pop_mean)

    if mode == "cuped_multivariate":
        k = pre.shape[1]
        if y_pre_pop_mean is None:
            mu = np.full(k, CUPED_DEFAULT_PRE_MEAN / max(k, 1))
        else:
            mu = np.asarray(y_pre_pop_mean, dtype=np.float64).ravel()
            if mu.shape[0] != k:
                raise ValueError("y_pre_pop_mean must match number of pre columns.")
        pre_c = pre - mu.reshape(1, -1)
        theta, *_ = np.linalg.lstsq(pre_c, y, rcond=None)
        y_adj = y - pre_c @ theta
        ate = float(y_adj[w1].mean() - y_adj[w0].mean())
        se = _se_two_sample_mean_diff(y_adj[w1], y_adj[w0])
        return CupedResult(ate_hat=ate, se_delta=se, theta=theta.copy())

    raise ValueError(f"Unknown mode: {mode}")
