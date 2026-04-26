# Variance reduction: CUPED vs Double ML

In randomized experiments, the **average treatment effect (ATE)** on a post-period outcome is often estimated by a simple **difference in means**. That estimator is unbiased, but it can be **noisy** when units differ a lot in baseline “potential” outcomes. A common fix is to use **pre-experiment information** (a pre-period outcome $y_{\text{pre}}$) or **covariates** $z$ to **partial out** predictable heterogeneity and shrink the sampling variance of the ATE—without changing the estimand under randomization.

This repository **simulates** an ad-style **RCT** (Randomized Controlled Trial; treatment $w$ is assigned at random, independent of $y_{\text{pre}}$ and $z$) (see `data/synthetic_ads.py`), then compares several **ATE estimators** that differ in how aggressively they exploit $y_{\text{pre}}$ and $z$. The point is not any single table of numbers: those numbers are just **one Monte Carlo snapshot** for whatever settings live in `config.py` (sample size, noise, number of replications).

Run:

```bash
python3 run_comparison.py
```

Plots and formatted tables are written under `output/`.

---

## Goal and estimand

We observe **post-period outcome** $y_{\text{post}}$, **binary treatment** $w$ ∈ {0, 1} ($w = 1$: treated arm, $w = 0$: control arm; assigned at experiment start t=0), and (optionally) **pre-period outcome** $y_{\text{pre}}$ and features $z$. Under **random assignment**, the ATE on $y_{\text{post}}$ is

$$
\tau = \mathbb{E}[y_{\text{post}} \mid w{=}1] - \mathbb{E}[y_{\text{post}} \mid w{=}0].
$$

Every estimator below is built to target **that same** $\tau$ in this RCT setting; they differ in **efficiency** (sampling variance) and sometimes in **finite-sample bias** if the working model is a poor approximation.

### Notation: outcomes $y$, features $z$, and noise

We write **outcomes with a period subscript** so “y” is not a single ambiguous symbol:

- **$y_{\text{pre}}$**: pre-period (pre-experiment) outcome for each unit—what you would have measured before treatment starts (t < 0).
- **$y_{\text{post}}$**: post-period outcome used for the experiment’s primary analysis and for $\tau$ (t $\ge$ 0).

**$z$**: vector of **unit-level features** (in the simulator: latent “context” after a fixed correlation transform—think campaign/audience/seasonality–like inputs). Some estimators adjust using $z$ directly (`ancova_zx`, Double ML); CUPED-style rows use $y_{\text{pre}}$ (and multivariate pre columns built from the same $z$).

**Noise (simulator only):** the synthetic **data generating process (DGP)** in `data/synthetic_ads.py` builds a shared nonlinear signal $g(z)$, then adds **independent** mean-zero Gaussian shocks to pre and post:

- **$\epsilon_\text{pre}$**, **$\epsilon_\text{post}$**: idiosyncratic noise in $y_{\text{pre}}$ and $y_{\text{post}}$ (scaled by `DGP_Y_PRE_NOISE_SCALE` and `DGP_Y_POST_NOISE_SCALE` in `config.py`). They are **independent of each other** and of $w$.

**$g(z)$**: deterministic part of both periods’ outcomes before noise; $y_{\text{post}}$ also adds **`true_ate * w`**. Details and functional form live in `AdsRCTDGP` in code, not in the table.

---

## Estimators

### Naive difference in means

$$
\hat\tau_{\text{naive}} = \bar y_{\text{post},1} - \bar y_{\text{post},0}.
$$

Uses only $y_{\text{post}}$ and $w$. This is the **reference**: easy to interpret and correctly centered under RCT, but it ignores information in $y_{\text{pre}}$ and $z$ that could reduce noise.

### CUPED (scalar $y_{\text{pre}}$): `cuped_known_pre_mean` and `cuped_one_theta`

**CUPED** (Controlled-experiment Using Pre-Experiment Data) uses a pre-period outcome that is **correlated** with $y_{\text{post}}$ to re-express the analysis in terms of an adjusted post outcome whose **treatment contrast**—the **difference in mean** $y_{\text{post}}$ between arms—has lower variance.

Here “**scalar**” means each unit contributes **one number** $y_{\text{pre},i}$ (a single pre-period metric), rather than a vector of multiple pre metrics.

#### ATE

Define an adjusted outcome:

$$
y^{\text{adj}}_i(\theta,\mu)=y_{\text{post},i}-\theta\bigl(y_{\text{pre},i}-\mu\bigr),
$$

and estimate ATE by a plain difference in means on $y^{\text{adj}}$:

$$
\hat\tau_{\text{cuped}}(\theta,\mu)=\overline{y^{\text{adj}}}_T-\overline{y^{\text{adj}}}_C.
$$

Under random assignment, $\mathbb{E}[\bar y_{\text{pre},T}-\bar y_{\text{pre},C}]=0$, so $\hat\tau_{\text{cuped}}(\theta,\mu)$ targets the same $\tau$ as $\hat\tau_{\text{naive}}$ for any fixed $(\theta,\mu)$. The whole point of CUPED is choosing $\theta$ to **minimize variance**.

#### Slope

Let $X_i=y_{\text{pre},i}-\mu$ and $Y_i=y_{\text{post},i}$. The adjusted outcome is $Y_i-\theta X_i$. Its variance is

$$
\text{Var}(Y-\theta X)=\text{Var}(Y)-2\theta\,\text{Cov}(Y,X)+\theta^2\text{Var}(X).
$$

This quadratic is minimized at

$$
\theta^\star=\frac{\text{Cov}(Y,X)}{\text{Var}(X)}=\frac{\text{Cov}(y_{\text{post}},y_{\text{pre}})}{\text{Var}(y_{\text{pre}})},
$$

which is the pooled OLS slope of $y_{\text{post}}$ on $y_{\text{pre}}$ (centering by $\mu$ does not change the slope). In practice, we plug in the sample analogue $\hat\theta$ computed on the pooled data, form $y^{\text{adj}}$, and then take a standard difference in means.

#### Intuition

For a single realized RCT dataset, it is helpful to rewrite CUPED directly in terms of the **naive point estimate** and the **observed pre-period shift** between arms. Throughout this section, a **contrast** is a **between-arm difference in sample means** for a named outcome (e.g. $\bar y_{\text{post},T}-\bar y_{\text{post},C}$ for post, or $\Delta_{\text{pre}}$ for pre). Let

$$
\Delta_{\text{pre}}=\bar y_{\text{pre},T}-\bar y_{\text{pre},C}.
$$

With a common reference $\mu$ (pooled mean or a known population mean), the CUPED estimate can be written as

$$
\hat\tau_{\text{cuped}}=\hat\tau_{\text{naive}}-\hat\theta\,\Delta_{\text{pre}}.
$$

Interpretation:

- Under randomization, $\mathbb{E}[\Delta_{\text{pre}}]=0$, but in any **finite** sample $\Delta_{\text{pre}}$ is typically **nonzero** because treatment and control groups differ slightly in baseline composition.
- The naive ATE is driven by **between-arm mean differences** in $y_{\text{post}}$. Part of the sampling variability in that between-arm contrast is tied to the same finite-sample imbalance in **between-arm** pre means, $\Delta_{\text{pre}}$. When $y_{\text{pre}}$ is correlated with $y_{\text{post}}$, the between-arm post contrast and $\Delta_{\text{pre}}$ **covary** across repeated experiments. Subtracting $\hat\theta\Delta_{\text{pre}}$ removes part of the sampling noise in the naive contrast. Under random assignment, it does not change the estimand: the average treatment effect (ATE) on $y_{\text{post}}$.
- **Single sample (one run).** If, by chance, the treated group has a higher pre mean ($\Delta_{\text{pre}}>0$) and pre predicts post ($\hat\theta>0$), then part of the naive post-period contrast $\hat\tau_{\text{naive}}$ is **attributable to baseline composition**, not the treatment. The term $-\hat\theta\,\Delta_{\text{pre}}$ subtracts that component so the adjusted estimate is closer to the contrast you would expect if pre means were **perfectly balanced** in that draw. If $\Delta_{\text{pre}}\approx 0$, there is little imbalance to remove and CUPED is close to naive.
- **Repeated sampling.** Over hypothetical replications of the experiment, the variance of $\hat\tau$ is smaller than that of $\hat\tau_{\text{naive}}$ when the adjustment successfully removes the **covarying** part of sampling noise. In a **single** realized trial you obtain one CUPED point estimate; its **standard error** and **confidence interval** are then typically **narrower** than for naive, because the sampling distribution of $\hat\tau$ is tighter (analytically, or from a **unit-level** bootstrap that resamples entities and recomputes $\hat\theta$ each resample).
- If $\Delta_{\text{pre}}=0$ in a particular draw, the adjustment term is zero and CUPED coincides with naive for that draw; the variance gain is a **repeated-sampling** property, not a guarantee in every single finite sample.

The two variants differ only in **$\mu$** (centering of $y_{\text{pre}}$):

- **`cuped_known_pre_mean`**: $\mu$ is set to the configured “known” pre mean (`DGP_PRE_MEAN_TARGET` in `config.py`), aligned with how the simulator anchors $y_{\text{pre}}$.
- **`cuped_one_theta`**: $\mu$ is the **sample mean** of $y_{\text{pre}}$ (pooled **single** $\hat\theta$).

When $\mu$ is well specified and $n$ is large, these often behave almost the same.

### Arm-specific θ (two θ’s): `cuped_two_theta`

Instead of one pooled slope $\hat\theta$, estimate **separate** pre→post slopes within each arm, $\hat\theta_T$ and $\hat\theta_C$, then compare arms after adjusting each unit’s $y_{\text{post}}$ with its arm’s slope and a **common** pre reference.

$$
\mu_{\text{pre}} = \bar y_{\text{pre}}.
$$

Per unit:

$$
y^{\text{adj}}_i = y_{\text{post},i} - \hat\theta_{w_i}\bigl(y_{\text{pre},i} - \mu_{\text{pre}}\bigr)
$$

and $\hat\tau$ is the difference in means of $y^{\text{adj}}$ between arms. When $\theta_T \neq \theta_C$, this differs from pooled scalar CUPED (one slope for both arms). With finite arms, estimating two slopes can be a bit noisier than one pooled $\hat\theta$.

### Multivariate CUPED: `cuped_multi`

Instead of one $y_{\text{pre}}$, we build several pre columns (`make_multivariate_y_pre` in `data/synthetic_ads.py`), then run the **multivariate** analogue: pooled linear adjustment using all pre columns (centered at default per-column means), then difference in means on the adjusted post outcome. Extra columns help when they carry **non-overlapping** predictive information about $y_{\text{post}}$.

### ANCOVA / regression adjustment on features: `ancova_zx`

Fit an OLS regression of $y_{\text{post}}$ on treatment and **centered** covariates:

$$
y_{\text{post}} \approx \beta_0 + \tau w + \beta_z^\top (z - \bar z),
$$

and read off $\hat\tau$ as the coefficient on $w$. Under an RCT, this is a standard **covariate adjustment** path: if the linear model captures outcome heterogeneity along $z$, the ATE estimate can be less noisy than the naive contrast. If the true conditional mean $\mathbb{E}[y_{\text{post}} \mid z]$ is **nonlinear**, a linear ANCOVA can leave residual structure on the table.

### Double ML (partially linear regression): `dml_plr_gbdt` and `dml_plr_lasso`

**Double machine learning** targets the same partially linear structure

$$
y_{\text{post}} = \tau w + g(z) + \varepsilon,
$$

but treats $g(z)$ as **unknown** and learns it with machine learning, using **cross-fitting** so overfitting bias is controlled (`doubleml`).

- **`dml_plr_gbdt`**: flexible **gradient boosting** for the outcome nuisance $g(z)$, with a **random forest** for $\mathbb{E}[w \mid z]$ (under RCT the latter is nearly flat, but the estimator is written generically).
- **`dml_plr_lasso`**: both nuisances use **`LassoCV`**—a linear, sparse high-dimensional workhorse. It shines when $g(z)$ is well approximated linearly; when $g(z)$ is curved or interactive, GBDT-style learners often fit better and can yield sharper ATE estimates.

---

## Monte Carlo summary

`run_comparison.py` draws many **independent** synthetic datasets. For each replication it computes every $\hat\tau$ above and stores them in `output/mc_draws.txt`. `output/summary_table.txt` combines point summaries (bias, RMSE, variance, `var_vs_naive`) with **90% bootstrap percentile intervals** over Monte Carlo rows (`MC_BOOTSTRAP_B`, `MC_BOOTSTRAP_SEED`, `MC_CI_LEVEL = 0.90` in `config.py`) for $\mathbb{E}[\hat\tau]$ and for the variance ratio. Rows are sorted by **`var`** (ascending). Figures: `violin_ate_estimates.png`, `relative_variance.png` (bars include bootstrap error bars), `forest_tau.png` (MC mean ATE with horizontal bootstrap intervals and a line at true $\tau$; **y-axis sorted by sampling variance**—lowest `var` / best variance reduction at the **top**, same ordering idea as the summary table).

Let $\tau$ be the simulator’s true ATE (`DGP_TRUE_ATE`)—**constant across draws**—and let $\hat\tau_r$ be the **estimated** ATE from replication $r$ (one draw of data, one $\hat\tau$ per method). We do **not** use $\tau_r$: the estimand does not vary with $r$ in this setup.

<!-- GitHub math rendering note:
Use $...$ for inline math and $$...$$ for display math. -->


| Column                   | Meaning                                                                                                                                                                                                                                  |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **bias**                 | $\overline{\hat\tau} - \tau$: average deviation from truth across replications. |
| **rmse**                 | $\sqrt{\frac{1}{R}\sum_r (\hat\tau_r - \tau)^2}$: typical error magnitude; combines bias and noise. |
| **var**                  | Sample variance of $\hat\tau_r$ across replications (sampling variance of the point estimate). |
| **var_vs_naive**         | $\mathrm{Var}(\hat\tau) / \mathrm{Var}(\hat\tau_{\text{naive}})$. Values **below 1** mean that estimator’s $\hat\tau$ is **less dispersed** across runs than the naive difference in means—i.e. **variance reduction** in repeated sampling. |
| **`tau_mean_90%`**  | Monte Carlo mean $\hat\tau$ with **90% bootstrap** percentile interval over MC rows (mean ± [lo, hi]). |
| **`var_ratio_90%`** | Point `var_vs_naive` with **90% bootstrap** percentile interval for the ratio on each resample (mean ± [lo, hi]). |


Sorting and magnitudes **change** if you change noise scales, n, the number of MC replications, or the DGP. Bootstrap columns quantify **Monte Carlo uncertainty** in the reported summaries, not uncertainty from a single field experiment.

For the strength of pre/post alignment in a given run batch, see **`output/rho_summary.txt`**: $\rho \approx \mathrm{Corr}(y_{\text{post}}, y_{\text{pre}})$ per draw, then summarized.

---

## Simulation results

This section summarizes the latest Monte Carlo outputs written under `output/` by `python3 run_comparison.py` (see top of this file).

### Setup and pre/post linkage

The synthetic RCT is constructed so that $y_{\text{pre}}$ (and features $z$) are predictive of $y_{\text{post}}$ while treatment $w$ is randomized. Across replications, the **pre/post correlation** is stable and fairly high (see `output/rho_summary.txt`):

- $\rho = \mathrm{Corr}(y_{\text{post}}, y_{\text{pre}})$: mean $\approx 0.684$, std $\approx 0.013$, min $\approx 0.648$, max $\approx 0.720$

That correlation is exactly what scalar CUPED exploits; multivariate CUPED and DML can additionally exploit other predictive directions (multiple pre columns or nonlinear structure in $z$).

### Point summaries

Rows in `output/summary_table.txt` are sorted by **sampling variance** `var` (ascending). Here are the key columns (bias, RMSE, var, and variance ratio vs naive):


| Method               | bias     | rmse        | var         | var_vs_naive |
| -------------------- | -------- | ----------- | ----------- | ------------ |
| **dml_plr_gbdt**     | −0.00191 | **0.01334** | **0.00018** | **0.32776**  |
| **cuped_multi**      | −0.00074 | 0.01370     | 0.00019     | 0.35215      |
| cuped_known_pre_mean | −0.00091 | 0.01665     | 0.00028     | 0.52011      |
| cuped_one_theta      | −0.00091 | 0.01665     | 0.00028     | 0.52011      |
| cuped_two_theta      | −0.00088 | 0.01666     | 0.00028     | 0.52061      |
| ancova_zx            | −0.00128 | 0.02155     | 0.00047     | 0.87074      |
| dml_plr_lasso        | −0.00130 | 0.02174     | 0.00048     | 0.88613      |
| **naive**            | −0.00084 | **0.02307** | 0.00054     | 1.00000      |


Interpretation:

- **Variance reduction**: every adjusted method has `var_vs_naive < 1`. The best precision in this run is **`dml_plr_gbdt`** (about **0.33×** the naive variance), followed by **`cuped_multi`** (**0.35×**). Scalar CUPED variants cluster at **0.52×**, and linear adjustment paths (**`ancova_zx`**, **`dml_plr_lasso`**) are only modestly better than naive (**0.87–0.89×**).
- **Bias vs RMSE**: biases are small relative to RMSE, so error is dominated by **variance** rather than systematic shift.
- **Bootstrap intervals**: the table also reports 90% bootstrap intervals over Monte Carlo rows for the MC mean ATE and for the variance ratio; these quantify **Monte Carlo uncertainty** in the reported summaries.

### Figures

All figures are generated from the same MC draws and are ordered consistently by sampling variance.

- **Relative variance vs naive** (`output/relative_variance.png`): bars show `Var(τ̂)/Var(naive)` with 90% bootstrap intervals.
  - **Ordering**: methods run **left → right** by increasing sampling variance (same idea as the summary table); the dashed line at **1** is naive.
  - **Takeaway**: **`dml_plr_gbdt`** and **`cuped_multi`** show the **largest** variance deflation; linear-on-$z$ methods sit only modestly below **1** (often ~**0.87–0.9×** naive in this run), with comparatively short bars but ratios still far from the best few.

![](output/relative_variance.png)

- **Forest plot of MC mean ATE** (`output/forest_tau.png`): points show MC mean $\hat\tau$ with 90% bootstrap intervals; the dashed vertical line is the true ATE.
  - **Precision**: **narrower** horizontal intervals at the **top** = lower `var` / more stable $\hat\tau$ across MC replications; **`naive`** is widest at the bottom.
  - **Coverage**: in this snapshot, every interval **crosses** the true $\tau$ line—differences are mostly about **width** (efficiency), not obvious systematic miss.

![](output/forest_tau.png)

- **Violin plot of ATE draws** (`output/violin_ate_estimates.png`): full distribution of $\hat\tau$ across replications; dashed horizontal line is the true ATE.
  - **Spread**: violins get **wider** left → right as `var` increases; **`dml_plr_gbdt`** is the tightest cloud, **`naive`** the most dispersed.
  - **Centering**: box medians sit **near** the true ATE for all methods here; the plot emphasizes **sampling noise** more than large bias.

![](output/violin_ate_estimates.png)



### Takeaways

- With $\rho \approx 0.68$, **CUPED** delivers meaningful variance reduction; in this DGP, **multivariate CUPED** improves further over scalar CUPED.
- **DML with a flexible learner** (`dml_plr_gbdt`) is most efficient here, consistent with nonlinear outcome structure.
- Linear adjustments on $z$ (`ancova_zx`, `dml_plr_lasso`) are close to each other and only slightly better than naive in this configuration.

These rankings can change with `config.py` (sample size, noise scales, DGP details, number of MC replications).

---

## Sanity check: scalar CUPED and $1 - \rho^2$

For **scalar** CUPED (one $y_{\text{pre}}$), a useful back-of-the-envelope is that variance reduction tracks the fraction of post variance not linearly explained by pre:

$$
\mathrm{Var}(\hat\tau_{\text{cuped}}) / \mathrm{Var}(\hat\tau_{\text{naive}}) \approx 1 - \rho^2,
$$

where $\rho = \mathrm{Corr}(y_{\text{post}}, y_{\text{pre}})$.

In the current run outputs (from `output/mc_draws.txt` / `output/rho_summary.txt`):

- $\bar\rho \approx 0.684$ so $1 - \bar\rho^2 \approx 0.532$
- empirical $\mathrm{Var}(\hat\tau_{\text{cuped}}) / \mathrm{Var}(\hat\tau_{\text{naive}}) \approx 0.520$ for **`cuped_one_theta`**

In practice the empirical variance ratio and $1-\rho^2$ are **approximately the same** here; **finite-sample** noise (including estimating $\theta$ from data each replication) and mild departures from a purely linear pre–post link can shift the ratio by a **small** amount that is not meaningfully different.