"""
Synthetic ad conversion panel with a sharp intervention at t=0.

Designed for variance-reduction comparisons (CUPED, Double ML):
- Pre- and post-period outcomes share the same systematic component g(Z); independent
  noise is drawn for each period. Post adds a heterogeneous treatment effect tau(Z) * w
  whose sample average equals the configured ATE.
- ``y_pre`` is shifted to mean ``pre_mean_target`` (default from ``DGP_PRE_MEAN_TARGET``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import (
    DGP_HTE_SCALE,
    DGP_N_FEATURES,
    DGP_PRE_MEAN_TARGET,
    DGP_TRUE_ATE,
    DGP_Y_POST_NOISE_SCALE,
    DGP_Y_PRE_NOISE_SCALE,
)


@dataclass
class AdsRCTDGP:
    """Data generating process for geo/user-like units with one pre and one post outcome."""

    n_units: int = 5000
    n_features: int = DGP_N_FEATURES
    seed: int | None = 42
    true_ate: float = DGP_TRUE_ATE
    """Average treatment effect on the post-period outcome scale."""

    hte_scale: float = DGP_HTE_SCALE
    """Scale of the mean-zero deviation of the unit-level treatment effect around ``true_ate``."""

    pre_mean_target: float = DGP_PRE_MEAN_TARGET
    """Target sample mean for ``y_pre`` after shifting."""

    y_pre_noise_scale: float = DGP_Y_PRE_NOISE_SCALE
    y_post_noise_scale: float = DGP_Y_POST_NOISE_SCALE

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def sample(self) -> dict[str, np.ndarray]:
        rng = self._rng()
        n, p = self.n_units, self.n_features

        # Latent features (campaign context, audience mix, seasonality, etc.)
        z = rng.standard_normal((n, p))
        chol = np.linalg.cholesky(self._feature_corr(p))
        z = z @ chol.T

        w = rng.binomial(1, 0.5, size=n).astype(np.float64)

        # Shared signal; independent pre/post noise (same functional form g(Z)).
        g_z = self._nonlinear_signal(z)
        eps_pre = self.y_pre_noise_scale * rng.standard_normal(n)
        eps_post = self.y_post_noise_scale * rng.standard_normal(n)

        y_pre_raw = g_z + eps_pre
        y_pre = y_pre_raw - float(y_pre_raw.mean()) + self.pre_mean_target
        # Sample average of tau(z) is exactly true_ate. y_pre does not contain it.
        tau = self.true_ate + self.hte_scale * self._hte_deviation(z)
        y_post = tau * w + g_z + eps_post

        return {
            "z": z.astype(np.float64),
            "w": w,
            "y_pre": y_pre.astype(np.float64),
            "y_post": y_post.astype(np.float64),
        }

    @staticmethod
    def _feature_corr(p: int) -> np.ndarray:
        idx = np.arange(p)
        corr = 0.65 ** np.abs(idx[:, None] - idx[None, :])
        return corr

    @staticmethod
    def _nonlinear_signal(z: np.ndarray) -> np.ndarray:
        """GBDT-like structure: interactions + hinges + smooth terms."""
        a = z[:, 0]
        b = z[:, 1]
        c = z[:, 2]
        hinge1 = np.maximum(a - 0.4, 0.0)
        hinge2 = np.maximum(-0.25 - b, 0.0)
        interaction = 0.35 * np.tanh(1.2 * a * c)
        smooth = 0.22 * np.sin(2.1 * c) + 0.18 * (z[:, 3] ** 2)
        tail = 0.05 * np.tanh(z[:, 4:].sum(axis=1))
        return interaction + 0.55 * hinge1 + 0.42 * hinge2 + smooth + tail

    @staticmethod
    def _hte_deviation(z: np.ndarray) -> np.ndarray:
        """Nonlinear treatment-effect deviation, centered to mean 0 on this sample."""
        h = np.tanh(z[:, 0] * z[:, 1]) + 0.5 * np.maximum(z[:, 2], 0.0)
        return h - float(h.mean())


def make_multivariate_y_pre(
    z: np.ndarray,
    rng: np.random.Generator,
    *,
    pre_mean_target: float = DGP_PRE_MEAN_TARGET,
    noise_scale: float = DGP_Y_PRE_NOISE_SCALE,
) -> np.ndarray:
    """
    Several pre-period outcomes, each ``g(Z) + noise`` with independent noise, then column-centered.

    Returns shape (n, k); column means approximately ``pre_mean_target / k``.
    """
    n, p = z.shape
    k = 3
    g_z = AdsRCTDGP._nonlinear_signal(z)
    cols = []
    for _ in range(k):
        raw = g_z + noise_scale * rng.standard_normal(n)
        cols.append(raw - raw.mean() + pre_mean_target / k)
    return np.stack(cols, axis=1)
