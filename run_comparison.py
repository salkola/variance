#!/usr/bin/env python3
"""
Monte Carlo comparison: naive diff, CUPED variants, and Double ML on synthetic ad conversions.

Run from repo root:
  python run_comparison.py

Outputs formatted tables (``.txt``), printed summaries, ρ = Corr(Y_post, Y_pre), and figures under ``output/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    DGP_PRE_MEAN_TARGET,
    DGP_TRUE_ATE,
    MC_BOOTSTRAP_B,
    MC_BOOTSTRAP_SEED,
    MC_CI_LEVEL,
    MC_N_REPS,
    MC_N_UNITS,
    MC_SEED_OFFSET,
    METHOD_ORDER,
    PLOT_DPI,
)

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic_ads import AdsRCTDGP, make_multivariate_y_pre  # noqa: E402
from models.cuped import estimate_ate_cuped  # noqa: E402
from models.dml import estimate_ate_dml_lasso, estimate_ate_dml_plr  # noqa: E402

RHO_COL = "rho_y_pre_y_post"


def _write_formatted_table(
    path: Path,
    df: pd.DataFrame,
    *,
    title: str | None = None,
    decimals: int = 6,
    index: bool = True,
) -> None:
    """Write a fixed-width, human-readable table (UTF-8 text)."""
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(decimals)
    body = out.to_string(index=index)
    parts = [title, "", body] if title else [body]
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _pearson_corr_y_pre_y_post(y_post: np.ndarray, y_pre: np.ndarray) -> float:
    """ρ = Corr(Y_post, Y_pre) (sample Pearson correlation)."""
    y_post = np.asarray(y_post, dtype=np.float64).ravel()
    y_pre = np.asarray(y_pre, dtype=np.float64).ravel()
    if y_post.shape != y_pre.shape:
        raise ValueError("y_post and y_pre must have the same length.")
    if len(y_post) < 2:
        return float("nan")
    return float(np.corrcoef(y_post, y_pre)[0, 1])


def _one_rep(seed: int, n_units: int, pre_mean_target: float, true_ate: float) -> dict[str, float]:
    dgp = AdsRCTDGP(
        n_units=n_units,
        seed=seed,
        true_ate=true_ate,
        pre_mean_target=pre_mean_target,
    )
    d = dgp.sample()
    z, y_pre, w, y_post = d["z"], d["y_pre"], d["w"], d["y_post"]
    rng = np.random.default_rng(seed + 999)
    y_pre_multi = make_multivariate_y_pre(z, rng, pre_mean_target=pre_mean_target)

    out: dict[str, float] = {}
    out[RHO_COL] = _pearson_corr_y_pre_y_post(y_post, y_pre)

    def put(name: str, ate: float):
        out[name] = ate

    put("naive", estimate_ate_cuped(y_post, w, y_pre, "naive_diff").ate_hat)
    put(
        "cuped_known_pre_mean",
        estimate_ate_cuped(y_post, w, y_pre, "cuped_known_mean", y_pre_pop_mean=pre_mean_target).ate_hat,
    )
    put("cuped_one_theta", estimate_ate_cuped(y_post, w, y_pre, "cuped_one_theta").ate_hat)
    put("cuped_two_theta", estimate_ate_cuped(y_post, w, y_pre, "cuped_two_theta").ate_hat)
    put("ancova_zx", estimate_ate_cuped(y_post, w, z, "ancova_ols").ate_hat)
    put(
        "cuped_multi",
        estimate_ate_cuped(y_post, w, y_pre_multi, "cuped_multivariate", y_pre_pop_mean=None).ate_hat,
    )

    dml_gb = estimate_ate_dml_plr(y_post, w, z, random_state=seed % (2**31))
    put("dml_plr_gbdt", dml_gb.ate_hat)

    dml_la = estimate_ate_dml_lasso(y_post, w, z, random_state=seed % (2**31))
    put("dml_plr_lasso", dml_la.ate_hat)

    return out


def run_mc(
    *,
    n_reps: int = MC_N_REPS,
    n_units: int = MC_N_UNITS,
    pre_mean_target: float = DGP_PRE_MEAN_TARGET,
    true_ate: float = DGP_TRUE_ATE,
) -> pd.DataFrame:
    rows = []
    for r in range(n_reps):
        rows.append(
            _one_rep(MC_SEED_OFFSET + r, n_units=n_units, pre_mean_target=pre_mean_target, true_ate=true_ate)
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, truth: float, ate_cols: list[str]) -> pd.DataFrame:
    d = df[ate_cols]
    bias = d.mean(axis=0) - truth
    rmse = np.sqrt(((d - truth) ** 2).mean(axis=0))
    var = d.var(axis=0, ddof=1)
    rel_var = var / var["naive"]
    out = pd.DataFrame({"bias": bias, "rmse": rmse, "var": var, "var_vs_naive": rel_var})
    return out


def _percentile_endpoints(level: float) -> tuple[float, float]:
    alpha = (1.0 - level) / 2.0
    return 100.0 * alpha, 100.0 * (1.0 - alpha)


def summarize_bootstrap_intervals(
    draws: pd.DataFrame,
    ate_cols: list[str],
    *,
    n_boot: int = MC_BOOTSTRAP_B,
    seed: int = MC_BOOTSTRAP_SEED,
    ci_level: float = MC_CI_LEVEL,
) -> pd.DataFrame:
    """
    Nonparametric bootstrap over Monte Carlo rows:

    - **tau**: percentile CI for the *mean* of ``\\hat\\tau`` across replications (i.e. for
      ``E[\\hat\\tau]``; under unbiasedness this centers on the true ``\\tau``).
    - **var_vs_naive**: percentile CI for ``Var(\\hat\\tau_m) / Var(\\hat\\tau_{naive})`` computed
      on each bootstrap resample of rows.
    """
    rng = np.random.default_rng(seed)
    d = draws[ate_cols].to_numpy(dtype=np.float64)
    r, k = d.shape
    lo_p, hi_p = _percentile_endpoints(ci_level)
    naive_idx = ate_cols.index("naive")

    tau_mean = d.mean(axis=0)
    boot_tau_mean = np.empty((n_boot, k), dtype=np.float64)
    boot_var_ratio = np.empty((n_boot, k), dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, r, size=r)
        sample = d[idx, :]
        boot_tau_mean[b, :] = sample.mean(axis=0)
        v = sample.var(axis=0, ddof=1)
        v0 = float(v[naive_idx])
        boot_var_ratio[b, :] = np.where(v0 > 1e-20, v / v0, np.nan)

    tau_lo = np.percentile(boot_tau_mean, lo_p, axis=0)
    tau_hi = np.percentile(boot_tau_mean, hi_p, axis=0)
    vr_lo = np.nanpercentile(boot_var_ratio, lo_p, axis=0)
    vr_hi = np.nanpercentile(boot_var_ratio, hi_p, axis=0)

    vars_pop = draws[ate_cols].var(ddof=1)
    rel_var = (vars_pop / vars_pop["naive"]).reindex(ate_cols).to_numpy(dtype=np.float64)

    pct = int(round(100 * ci_level))
    return pd.DataFrame(
        {
            "tau_hat_mean": tau_mean,
            f"tau_ci{pct}_lo": tau_lo,
            f"tau_ci{pct}_hi": tau_hi,
            "var_vs_naive": rel_var,
            f"var_ratio_ci{pct}_lo": vr_lo,
            f"var_ratio_ci{pct}_hi": vr_hi,
        },
        index=list(ate_cols),
    )


def _fmt_mean_ci(mean: float, lo: float, hi: float, *, decimals: int = 4) -> str:
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ± [{fmt.format(lo)}, {fmt.format(hi)}]"


def summarize_rho(df: pd.DataFrame) -> pd.DataFrame:
    rho = df[RHO_COL]
    return pd.DataFrame(
        [
            {
                "rho_mean": float(rho.mean()),
                "rho_std": float(rho.std(ddof=1)) if len(rho) > 1 else 0.0,
                "rho_min": float(rho.min()),
                "rho_max": float(rho.max()),
            }
        ]
    )


def plot_violin(df: pd.DataFrame, truth: float, outdir: Path, method_order: list[str]) -> None:
    """``method_order`` lists categories left-to-right (ascending sampling ``var`` matches ``relative_variance.png``)."""
    long = df[method_order].melt(var_name="method", value_name="ate_hat")
    long["method"] = pd.Categorical(long["method"], categories=method_order, ordered=True)

    plt.figure(figsize=(11, 5))
    sns.violinplot(data=long, x="method", y="ate_hat", inner="box", cut=0, density_norm="width")
    plt.axhline(truth, color="black", linestyle="--", linewidth=0.5, label="true ATE")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Estimated ATE")
    plt.title("Distribution of ATE estimates (Monte Carlo) · Sorted by sampling Var")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outdir / "violin_ate_estimates.png", dpi=PLOT_DPI)
    plt.close()


def plot_variance_bar(
    summary: pd.DataFrame,
    outdir: Path,
    ci_intervals: pd.DataFrame,
    *,
    pct: int,
) -> None:
    s = summary.drop(index="naive", errors="ignore")
    idx = s.sort_values("var").index.tolist()
    y = s.loc[idx, "var_vs_naive"].to_numpy(dtype=float)
    lo_col = f"var_ratio_ci{pct}_lo"
    hi_col = f"var_ratio_ci{pct}_hi"
    ci = ci_intervals.reindex(idx)
    lo = ci[lo_col].to_numpy(dtype=float)
    hi = ci[hi_col].to_numpy(dtype=float)
    yerr = np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])

    x = np.arange(len(idx))
    plt.figure(figsize=(9, 4.5))
    plt.bar(x, y, yerr=yerr, capsize=4, color="steelblue", ecolor="black", alpha=0.85)
    plt.xticks(x, [str(i) for i in idx], rotation=25, ha="right")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, label="naive variance")
    plt.ylabel("Var(hat) / Var(naive)")
    plt.title(f"Variance ratio vs naive ({pct}% bootstrap MC intervals)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "relative_variance.png", dpi=PLOT_DPI)
    plt.close()


def plot_forest_tau(
    ci_intervals: pd.DataFrame,
    truth: float,
    outdir: Path,
    *,
    pct: int,
    method_order: list[str],
) -> None:
    """Monte Carlo mean ATE with bootstrap percentile intervals (horizontal).

    ``method_order`` should list methods bottom-to-top on the y-axis (we use
    descending sampling variance so **lowest** ``var`` / best variance reduction
    appears at the **top**, matching ``relative_variance.png``).
    """
    order = list(method_order)
    means = ci_intervals.loc[order, "tau_hat_mean"].to_numpy(dtype=float)
    lo = ci_intervals.loc[order, f"tau_ci{pct}_lo"].to_numpy(dtype=float)
    hi = ci_intervals.loc[order, f"tau_ci{pct}_hi"].to_numpy(dtype=float)
    y = np.arange(len(order))
    xerr = np.vstack([np.maximum(0.0, means - lo), np.maximum(0.0, hi - means)])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(means, y, xerr=xerr, fmt="o", color="steelblue", ecolor="black", capsize=4, markersize=5)
    ax.axvline(truth, color="steelblue", linestyle="--", linewidth=0.5, label="true ATE")
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlabel("Mean ATE ± MC-bootstrap CI", fontsize=15)
    ax.set_title(f"ATE by method · Sorted by sampling Var · {pct}% CI", fontsize=18)
    ax.legend(loc="upper right", fontsize=14)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.90, bottom=0.14)
    fig.savefig(outdir / "forest_tau.png", dpi=PLOT_DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def main() -> None:
    outdir = ROOT / "output"
    outdir.mkdir(parents=True, exist_ok=True)

    truth = DGP_TRUE_ATE
    print("Running Monte Carlo (this may take a few minutes)...")
    draws = run_mc()
    _write_formatted_table(
        outdir / "mc_draws.txt",
        draws,
        title="Monte Carlo ATE draws (one row per replication; columns = methods + rho_y_pre_y_post)",
        decimals=6,
    )

    rho_summ = summarize_rho(draws)
    _write_formatted_table(
        outdir / "rho_summary.txt",
        rho_summ,
        title="Correlation Y_post vs Y_pre across Monte Carlo draws",
        decimals=5,
        index=False,
    )

    summ = summarize(draws, truth, METHOD_ORDER)
    ci_summ = summarize_bootstrap_intervals(draws, METHOD_ORDER)
    pct = int(round(100 * MC_CI_LEVEL))
    tau_lo_col = f"tau_ci{pct}_lo"
    tau_hi_col = f"tau_ci{pct}_hi"
    vr_lo_col = f"var_ratio_ci{pct}_lo"
    vr_hi_col = f"var_ratio_ci{pct}_hi"

    merged = summ.reindex(METHOD_ORDER).copy()
    merged[f"tau_mean_{pct}%"] = [
        _fmt_mean_ci(m, lo, hi, decimals=4)
        for m, lo, hi in zip(
            ci_summ.reindex(METHOD_ORDER)["tau_hat_mean"],
            ci_summ.reindex(METHOD_ORDER)[tau_lo_col],
            ci_summ.reindex(METHOD_ORDER)[tau_hi_col],
            strict=True,
        )
    ]
    merged[f"var_ratio_{pct}%"] = [
        _fmt_mean_ci(m, lo, hi, decimals=4)
        for m, lo, hi in zip(
            ci_summ.reindex(METHOD_ORDER)["var_vs_naive"],
            ci_summ.reindex(METHOD_ORDER)[vr_lo_col],
            ci_summ.reindex(METHOD_ORDER)[vr_hi_col],
            strict=True,
        )
    ]
    merged = merged.sort_values("var")
    _write_formatted_table(
        outdir / "summary_table.txt",
        merged,
        title=(
            f"Monte Carlo summary + {pct}% bootstrap CIs over MC rows (B={MC_BOOTSTRAP_B}, seed={MC_BOOTSTRAP_SEED}). "
            f"true ATE tau = {DGP_TRUE_ATE}, pre-period mean anchor = {DGP_PRE_MEAN_TARGET}. "
            f"Numeric columns from MC draws; last two columns are mean ± [lo, hi] (4 dp)."
        ),
        decimals=5,
    )
    print(
        f"\n=== Summary (true ATE = {DGP_TRUE_ATE}, pre-period scalar mean anchored at {DGP_PRE_MEAN_TARGET}) ===\n"
    )
    r = rho_summ.iloc[0]
    print(
        f"ρ = Corr(Y_post, Y_pre) over MC draws: mean={r['rho_mean']:.5f}, "
        f"std={r['rho_std']:.5f}, min={r['rho_min']:.5f}, max={r['rho_max']:.5f}\n"
    )
    print("(See output/rho_summary.txt)\n")
    print(merged.to_string())

    violin_order = summ.sort_values("var", ascending=True).index.tolist()
    plot_violin(draws, truth, outdir, violin_order)
    plot_variance_bar(summ, outdir, ci_summ, pct=pct)
    forest_order = summ.sort_values("var", ascending=False).index.tolist()
    plot_forest_tau(ci_summ, truth, outdir, pct=pct, method_order=forest_order)

    print(f"\nWrote: {outdir / 'mc_draws.txt'}")
    print(f"Wrote: {outdir / 'rho_summary.txt'}")
    print(f"Wrote: {outdir / 'summary_table.txt'}")
    print(f"Wrote: {outdir / 'violin_ate_estimates.png'}")
    print(f"Wrote: {outdir / 'relative_variance.png'}")
    print(f"Wrote: {outdir / 'forest_tau.png'}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid", context="talk")
    main()
