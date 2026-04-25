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

# Monte Carlo / experiment run

MC_N_REPS: int = 100
MC_N_UNITS: int = 3000
MC_SEED_OFFSET: int = 0

# Bootstrap percentile intervals (Monte Carlo uncertainty)
MC_CI_LEVEL: float = 0.90  # central interval for percentile endpoints
MC_BOOTSTRAP_B: int = 2000
MC_BOOTSTRAP_SEED: int = 12345

# CUPED

# Default population mean used for centering when a known mean is assumed.
CUPED_DEFAULT_PRE_MEAN: float = DGP_PRE_MEAN_TARGET

# For multivariate pre-metric, default per-column mean is total/k.
CUPED_MULTI_K: int = 3

# Double ML

DML_N_FOLDS: int = 5
DML_SCORE: str = "partialling out"

# Nuisance learners (defaults)
DML_GBDT_L_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
}

DML_RF_M_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "n_jobs": -1,
}

# Plotting / reporting

PLOT_DPI: int = 160

METHOD_ORDER: list[str] = [
    "naive",
    "cuped_known_pre_mean",
    "cuped_one_theta",
    "cuped_two_theta",
    "cuped_multi",
    "ancova_zx",
    "dml_plr_gbdt",
    "dml_plr_lasso",
]

