"""
Central configuration for the variance-reduction comparisons.

Keep all knobs here so simulations/estimators stay consistent.
"""

from __future__ import annotations

# Data generating process (DGP)

DGP_TRUE_ATE: float = 0.08
DGP_PRE_MEAN_TARGET: float = 1.0

DGP_N_FEATURES: int = 5
# Independent noise for pre vs post outcome;
# same default scale (same “shape”, different draws).
DGP_Y_PRE_NOISE_SCALE: float = 0.35
DGP_Y_POST_NOISE_SCALE: float = 0.35
# Scale of the mean-zero unit-level deviation around DGP_TRUE_ATE.
# The sample average of tau(z) is exactly DGP_TRUE_ATE.
DGP_HTE_SCALE: float = 0.25

# Monte Carlo / experiment run

MC_N_REPS: int = 200
MC_N_UNITS: int = 10000
MC_SEED_OFFSET: int = 0

# Bootstrap percentile intervals (Monte Carlo uncertainty)
MC_CI_LEVEL: float = 0.90  # central interval for percentile endpoints
MC_BOOTSTRAP_B: int = 2000
MC_BOOTSTRAP_SEED: int = 0

# CUPED

# Default population mean used to center multivariate pre columns when a
# per-column reference is not passed in. A common constant cancels in the
# between-arm difference of means (same as scalar CUPED).
CUPED_DEFAULT_PRE_MEAN: float = DGP_PRE_MEAN_TARGET

# For multivariate pre-metric, default per-column mean is total/k.
CUPED_MULTI_K: int = 3

# MLRATE: cross-fit gradient boosting, then Lin regression with an interaction.

MLRATE_N_FOLDS: int = 5
MLRATE_GBDT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
}

# Plotting / reporting

PLOT_DPI: int = 160

METHOD_ORDER: list[str] = [
    "naive",
    "cuped_one_theta",
    "cuped_two_theta",
    "cuped_multi",
    "ancova",
    "mlrate",
]

