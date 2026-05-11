"""Restructure BC3/main.ipynb into the seven-Part layout declared in
docs/superpowers/specs/2026-05-11-bc3-refactor-design.md.

Usage:
    python -m BC3.scripts.restructure_notebook

Reads BC3/main.ipynb, writes BC3/main.ipynb in-place (with the original
preserved as BC3/main.ipynb.bak). Idempotent: re-running on an already-
restructured notebook is a no-op because the manifest references original
cell IDs by index in the .bak file.
"""
from __future__ import annotations

import copy
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

BC3 = Path(__file__).resolve().parents[1]
NB = BC3 / "main.ipynb"
BAK = BC3 / "main.ipynb.bak"

# -----------------------------------------------------------------------------
# Manifest
# -----------------------------------------------------------------------------
# Each entry: (part_title_or_None, original_cell_index_or_NEW_marker, edits_dict_or_None)
# Cells are emitted in manifest order. Part titles become the leading `#` heading,
# emitted as a NEW markdown cell at the first transition into that Part.
# -----------------------------------------------------------------------------

PART_I_TITLE = "# Part I — Setup, Data Loading & EDA\n"
PART_II_TITLE = "# Part II — Backtest Methodology\n"
PART_III_TITLE = "# Part III — Ridge Control + Rebalancing & Constraints Sweep\n"
PART_IV_TITLE = "# Part IV — Linear Benchmark Family (Predict-then-Optimize)\n"
PART_V_TITLE = "# Part V — Kalman Filter / State-Space\n"
PART_VI_TITLE = "# Part VI — Deep Learning Weight Generator\n"
PART_VII_TITLE = "# Part VII — Final Consolidated Comparison & Findings\n"

WORKFLOW_MD = [
    "## Workflow at a glance\n",
    "\n",
    "| Part | Title | What it produces |\n",
    "|---|---|---|\n",
    "| I | Setup, Data Loading & EDA | clean `(X, y)` weekly panel + data-quality audit |\n",
    "| II | Backtest Methodology | harness, transaction-cost / VaR / GE / IR / TE definitions, assumption register |\n",
    "| III | Ridge Control + Rebalancing & Constraints Sweep | canonical Ridge baseline, cadence x leverage scenario grid |\n",
    "| IV | Linear Benchmark Family (Predict-then-Optimize) | OLS / Ridge / Lasso / ElasticNet / Huber leaderboard |\n",
    "| V | Kalman Filter / State-Space | static, EM, regime-switching (warm-up-frozen HMM) |\n",
    "| VI | Deep Learning Weight Generator | MLP and attention variant, PCA features, turnover-penalty grid |\n",
    "| VII | Final Consolidated Comparison & Findings | master table + spotlight plot + project checklist |\n",
    "\n",
    "**Data flow.**\n",
    "\n",
    "```\n",
    "prices -> (X, y) panel -> harness.run_rolling_backtest + evaluate_weights\n",
    "      -> {Ridge baseline, Linear PO, Kalman, NN}\n",
    "      -> results/*.pkl -> consolidated comparison\n",
    "```\n",
]

# Fold maps (markdown cells merged INTO their section intro above)
FOLD_INTO = {}
# Cells dropped outright (literal duplicates or stale headings)
DROP = {3, 71, 72, 73, 109}

# Cells whose markdown is moved into Part VII's Findings section (not re-emitted)
FINDINGS_SOURCES = {91, 104, 164}

# Manifest entries: (part_title_constant_or_None, original_index_or_NEW_marker, edits)
MANIFEST: list[tuple[str | None, int | str, dict[str, Any] | None]] = [
    # ---- Top of notebook ----
    (None, 0, None),
    (None, 1, {"replace_source": [
        "# Portfolio Replica Strategy\n",
        "\n",
        "Seven-Part deliverable:\n",
        "\n",
        "- `main.ipynb` — all analysis, plots, and end-to-end runs across Parts I–VII.\n",
        "- `harness.py` — rolling-backtest engine, transaction-cost models, risk metrics, audit helpers, shared metric-row and percentage-formatting helpers.\n",
        "- `predict_then_optimize.py` — convex predict-then-optimize layer (Ledoit-Wolf covariance + CVXPY).\n",
        "- `kalman.py` — linear Kalman filter, pykalman EM noise tuning, hmmlearn regime classifier (HMM frozen on warm-up window).\n",
        "- `dl_pipeline.py` — neural-network weight generator (MLP, attention variant, PCA features).\n",
        "- `portfolio_constraints.py` — rebalancing cadence and gross-exposure / VaR projection layer used by Part III.\n",
        "\n",
        "**Reading guide.**\n",
        "\n",
        "- **Part I — Setup, Data Loading & EDA.** Build the clean `(X, y)` weekly panel; surface data-quality and feature-relevance diagnostics.\n",
        "- **Part II — Backtest Methodology.** Define the rolling harness, transaction-cost / VaR / GE / IR / TE conventions, and the assumption register.\n",
        "- **Part III — Ridge Control + Rebalancing & Constraints Sweep.** Canonical Ridge baseline; then sweep cadence x leverage scenarios under the constraint layer.\n",
        "- **Part IV — Linear Benchmark Family (Predict-then-Optimize).** OLS / Ridge / Lasso / ElasticNet / Huber leaderboard via the convex layer.\n",
        "- **Part V — Kalman Filter / State-Space.** Static sigma_w grid, EM tuning, and regime-switching with the HMM frozen on the warm-up window.\n",
        "- **Part VI — Deep Learning Weight Generator.** MLP weight generator (vanilla and PCA features) plus an attention variant.\n",
        "- **Part VII — Final Consolidated Comparison & Findings.** Master table, spotlight plot, acceptance checklist.\n",
    ]}),
    (None, "NEW_WORKFLOW", {"replace_source": WORKFLOW_MD}),

    # ---- Part I: Setup, Data Loading & EDA ----
    (PART_I_TITLE, 2, None),
    (PART_I_TITLE, 4, None),
    (PART_I_TITLE, 5, None),
    (PART_I_TITLE, 6, None),
    (PART_I_TITLE, 7, None),
    (PART_I_TITLE, 8, None),
    (PART_I_TITLE, 9, None),
    (PART_I_TITLE, 10, None),
    (PART_I_TITLE, 11, None),
    (PART_I_TITLE, 12, None),
    (PART_I_TITLE, 13, None),
    (PART_I_TITLE, 14, None),
    (PART_I_TITLE, 15, None),
    (PART_I_TITLE, 16, None),
    (PART_I_TITLE, 17, None),
    (PART_I_TITLE, 18, None),
    (PART_I_TITLE, 19, None),
    (PART_I_TITLE, 20, None),
    (PART_I_TITLE, 21, None),
    (PART_I_TITLE, 22, None),
    (PART_I_TITLE, 23, None),
    (PART_I_TITLE, 24, None),
    (PART_I_TITLE, 25, None),
    (PART_I_TITLE, 26, None),
    (PART_I_TITLE, 27, None),

    # ---- Part II: Backtest Methodology ----
    (PART_II_TITLE, 28, None),
    (PART_II_TITLE, 29, None),
    (PART_II_TITLE, 30, None),
    (PART_II_TITLE, 47, {"retitle": "### Sanity check: `evaluate_weights()` matches `run_rolling_backtest()`"}),
    (PART_II_TITLE, 48, None),
    (PART_II_TITLE, 52, None),
    (PART_II_TITLE, 53, None),
    (PART_II_TITLE, 54, None),

    # ---- Part III: Ridge Control + Rebalancing & Constraints Sweep ----
    (PART_III_TITLE, 31, None),
    (PART_III_TITLE, 32, None),
    (PART_III_TITLE, 33, None),
    (PART_III_TITLE, 34, None),
    (PART_III_TITLE, 35, None),
    (PART_III_TITLE, 36, None),
    (PART_III_TITLE, 37, None),
    (PART_III_TITLE, 38, None),
    (PART_III_TITLE, 39, None),
    (PART_III_TITLE, 40, None),
    (PART_III_TITLE, 41, None),
    (PART_III_TITLE, 42, None),
    (PART_III_TITLE, 43, None),
    (PART_III_TITLE, 44, None),
    (PART_III_TITLE, 45, None),
    (PART_III_TITLE, 46, None),
    (PART_III_TITLE, 49, None),
    (PART_III_TITLE, 50, None),
    (PART_III_TITLE, 51, None),
    (PART_III_TITLE, 55, None),
    (PART_III_TITLE, 56, None),
    (PART_III_TITLE, 57, None),
    (PART_III_TITLE, 58, None),
    (PART_III_TITLE, 59, None),
    (PART_III_TITLE, 60, None),
    (PART_III_TITLE, 61, None),
    (PART_III_TITLE, 62, None),
    (PART_III_TITLE, 63, None),
    (PART_III_TITLE, 64, None),
    (PART_III_TITLE, 65, None),
    (PART_III_TITLE, 66, None),
    (PART_III_TITLE, 67, None),
    (PART_III_TITLE, 68, None),
    (PART_III_TITLE, 69, None),
    (PART_III_TITLE, 74, None),
    (PART_III_TITLE, 75, None),
    (PART_III_TITLE, 76, None),
    (PART_III_TITLE, 77, None),
    # ---- Part III continued: rebalancing & constraints (from old Part V) ----
    (PART_III_TITLE, 129, {"replace_source": [
        "## Rebalancing & Portfolio Constraints sweep\n",
        "\n",
        "**Goal.** Establish the canonical Ridge baseline above, then sweep rebalancing cadence and ",
        "gross-exposure cap to surface the trade-off between tracking and turnover under the ",
        "constraint layer (`apply_ge_cap` + iterative `apply_var_cap_iterative`).\n",
        "\n",
        "**Method.** Run six base configurations (1/4/12 weeks x 100%/200% GE) followed by an extended ",
        "6-axis Cartesian grid; rank by a composite z-scored score over IR, net IR, rho, -TE, -turnover, -MDD.\n",
    ]}),
    (PART_III_TITLE, 130, None),
    (PART_III_TITLE, 131, None),
    (PART_III_TITLE, 132, {"append_source": [
        "\n\n*In-sample diagnostic; not used by any model — the VaR statistics here are read against ",
        "the full sample to motivate the parametric assumption, not to score the strategy.*\n",
    ]}),
    (PART_III_TITLE, 133, None),
    (PART_III_TITLE, 134, None),
    (PART_III_TITLE, 135, None),
    (PART_III_TITLE, 136, None),
    (PART_III_TITLE, 137, None),
    (PART_III_TITLE, 138, None),
    (PART_III_TITLE, 139, None),
    (PART_III_TITLE, 140, None),
    (PART_III_TITLE, 141, None),
    (PART_III_TITLE, 142, None),
    (PART_III_TITLE, 143, None),
    (PART_III_TITLE, 144, None),
    (PART_III_TITLE, 145, None),
    (PART_III_TITLE, 146, None),
    (PART_III_TITLE, 147, None),
    (PART_III_TITLE, 148, None),
    (PART_III_TITLE, 149, None),
    (PART_III_TITLE, 150, None),
    (PART_III_TITLE, 151, None),
    (PART_III_TITLE, 152, None),
    (PART_III_TITLE, 153, None),
    (PART_III_TITLE, 154, None),
    (PART_III_TITLE, 155, None),
    (PART_III_TITLE, 156, None),
    (PART_III_TITLE, 157, None),
    (PART_III_TITLE, 158, None),
    (PART_III_TITLE, 159, None),
    (PART_III_TITLE, 160, None),
    (PART_III_TITLE, 161, None),
    (PART_III_TITLE, 162, None),
    (PART_III_TITLE, 163, None),

    # ---- Part IV: Linear Benchmark Family ----
    (PART_IV_TITLE, 78, {"replace_source": [
        "Part I established the harness, transaction-cost accounting and the Ridge control benchmark. Its\n",
        "honest verdict was that *post-hoc* GE/VaR projection plus 5 bps cost drag drives the control\n",
        "model's net IR negative. Part II tackles the same target with the **predict-then-optimize**\n",
        "workflow: every rebalance is a four-phase pipeline that bakes the mandate constraints **and** the\n",
        "cost penalty into a convex objective, so the harness's post-trade scaling fires only as a\n",
        "double-check.\n",
        "\n",
        "**Phase 1 — Alpha prediction (signal generation).** Train a classical linear model\n",
        "($\\text{OLS}$, $\\text{Ridge}$, $\\text{Lasso}$, $\\text{ElasticNet}$ or Huber) on the trailing\n",
        "104-week panel $(X_{\\text{tr}}, y_{\\text{tr}})$. The fitted coefficients\n",
        "$\\hat{\\beta} \\in \\mathbb{R}^{11}$ are the asset-level alpha vector $\\mu$, since for the OLS solution\n",
        "$w^{\\text{OLS}} = \\Sigma_r^{-1} \\sigma_{r,y}$ the implied per-asset alpha is exactly\n",
        "$\\mu = \\Sigma_r \\hat{\\beta}$. We expose $\\hat{\\beta}$ directly as $\\mu$ so the optimizer recovers\n",
        "the ridge / lasso fit when constraints are slack and projects onto the feasible set in the\n",
        "$\\Sigma$-norm when they bind.\n",
        "\n",
        "**Phase 2 — Risk modeling.** Compute the asset return covariance $\\Sigma$ from the same training\n",
        "window. With $T=104$ rows and $n=11$ columns the sample estimate is mostly stable, but we apply\n",
        "**Ledoit-Wolf shrinkage** as a default belt-and-braces step so the optimizer always sees a strictly\n",
        "positive-definite matrix even on stress windows where two futures briefly move in lockstep.\n",
        "\n",
        "**Phase 3 — Portfolio optimization (convex).** Solve\n",
        "\n",
        "$$\n",
        "\\hat{w}_t = \\arg\\max_{w}\\;\\; \\mu^{\\top} w \\;-\\; \\tfrac{1}{2}\\lambda\\, w^{\\top}\\Sigma w\n",
        "\\;-\\; \\tau \\cdot \\lVert w - w_{t-1}\\rVert_1\n",
        "$$\n",
        "\n",
        "subject to the mandate\n",
        "\n",
        "$$\n",
        "\\sum_{j=1}^{11} \\lvert w_j \\rvert \\;\\leq\\; 2.0, \\qquad \\lvert w_j \\rvert \\;\\leq\\; 0.5 \\quad \\forall j,\n",
        "$$\n",
        "\n",
        "with $\\tau = 5\\,\\text{bps}$ matching the harness `FlatBpsTC(5.0)` charge and $\\lambda$ a risk\n",
        "aversion knob (default $\\lambda=2$, the natural value for the tracking-error reformulation).\n",
        "\n",
        "**Phase 4 — Backtest engine integration.** Hand the rebalance-date weight matrix to\n",
        "`evaluate_weights(...)` so cost / VaR / GE auditing and metric extraction stay identical to Part I.\n",
        "\n",
        "**Reading convention.** Tables in this and later parts use a single shared formatter\n",
        "(`format_metrics_dataframe`): TE / VaR / max-drawdown / cost-drag are shown as percentages of NAV\n",
        "(e.g. `5.35%`); IR / ρ stay as signed dimensionless ratios; gross exposure and turnover are\n",
        "× NAV multiples.\n",
    ]}),
    (PART_IV_TITLE, 79, None),
    (PART_IV_TITLE, 80, None),
    (PART_IV_TITLE, 81, {"retitle": "### Sanity check: PO recovers OLS when constraints are slack"}),
    (PART_IV_TITLE, 82, None),
    (PART_IV_TITLE, 83, None),
    (PART_IV_TITLE, 84, None),
    (PART_IV_TITLE, 85, None),
    (PART_IV_TITLE, 86, None),
    (PART_IV_TITLE, 87, None),
    (PART_IV_TITLE, 88, None),
    (PART_IV_TITLE, 89, None),
    (PART_IV_TITLE, 90, None),

    # ---- Part V: Kalman Filter / State-Space ----
    (PART_V_TITLE, 92, {"replace_source": [
        "Part II froze the model coefficients to the trailing 104-week window and let the convex layer enforce\n",
        "constraints; the headline rows topped out at net IR ≈ +0.6 on the realised data. Part III asks a\n",
        "different question: *what if the weights themselves are a stochastic state that evolves every\n",
        "week?* The state-space form spelled out in the project HINTs cell is\n",
        "\n",
        "$$\n",
        "\\begin{aligned}\n",
        "x_t &= A\\, x_{t-1} + B\\, u_t, \\qquad & u_t &\\sim \\mathcal{N}(0, I_{11})         && \\text{(state: portfolio weights, random walk)} \\\\\n",
        "y_t &= C_t\\, x_t + D\\, \\varepsilon_t, \\qquad & \\varepsilon_t &\\sim \\mathcal{N}(0, 1) && \\text{(observation: target return)}\n",
        "\\end{aligned}\n",
        "$$\n",
        "\n",
        "with $A = I_{11}$, $B = \\sigma_w \\cdot I_{11}$, $C_t = r_t^{\\top}$, $D = \\hat{\\sigma}_y$ from the\n",
        "training window, $x_t \\in \\mathbb{R}^{11}$ the latent weight vector and $y_t$ the scalar target\n",
        "return. The Kalman replica is the only model in this notebook with a **truly weekly** rebalance\n",
        "cadence — that is the most expensive cell on the cost axis but the one that best captures regime\n",
        "change. Weekly held weights flow through `evaluate_weights(..., schedule_type=\"held\")` so all\n",
        "metrics stay comparable to the linear-benchmark leaderboard.\n",
        "\n",
        "**Look-ahead discipline.** The naive filter outputs $\\hat{x}_{t \\mid t}$ (uses $y_t$ to update),\n",
        "which is information unavailable to a real trader at the open of week $t$. We **shift the weight\n",
        "series by one week** before evaluation, so the held weight at date $t$ depends only on\n",
        "observations through $t-1$.\n",
        "\n",
        "**EM noise tuning** uses `pykalman.KalmanFilter.em()` capped at `n_iter=3` (the system is\n",
        "under-identified and EM diverges past 3 iterations). **Regime-switching** fits a 2-state\n",
        "`hmmlearn.GaussianHMM` on rolling target volatility and runs the filter with two different\n",
        "$\\sigma_w$ values for calm vs stressed weeks.\n",
    ]}),
    (PART_V_TITLE, 93, None),
    (PART_V_TITLE, 94, None),
    (PART_V_TITLE, 95, None),
    (PART_V_TITLE, 96, None),
    (PART_V_TITLE, 97, None),
    (PART_V_TITLE, 98, {"append_source": [
        "\n\n*The HMM is fit on the 104-week warm-up window only and frozen; subsequent regime labels ",
        "are produced by prediction, not refit. This removes the look-ahead leak that existed when ",
        "the HMM saw the full target volatility series.*\n",
    ]}),
    (PART_V_TITLE, 99, {"replace_source": [
        "regime_stressed, hmm_means, stress_state = fit_hmm_regime(\n",
        "    y,\n",
        "    vol_window=12,\n",
        "    fit_until=X.index[BASELINE_CFG.rolling_window],\n",
        ")\n",
        "\n",
        "sigma_w_calm = best_sw_static * 0.5\n",
        "sigma_w_stress = best_sw_static * 2.0\n",
        "sigma_w_per_step = pd.Series(\n",
        "    np.where(regime_stressed.values == 1, sigma_w_stress, sigma_w_calm),\n",
        "    index=y.index, name='sigma_w_per_step',\n",
        ")\n",
        "kf_regime_result = run_kalman_replica(\n",
        "    X, y,\n",
        "    cfg=KalmanConfig(sigma_w=best_sw_static),\n",
        "    name=f'Kalman regime sigma_w in [{sigma_w_calm:g}, {sigma_w_stress:g}]',\n",
        "    sigma_w_series=sigma_w_per_step,\n",
        ")\n",
    ]}),
    (PART_V_TITLE, 100, None),
    (PART_V_TITLE, 101, None),
    (PART_V_TITLE, 102, None),
    (PART_V_TITLE, 103, None),

    # ---- Part VI: Deep Learning Weight Generator ----
    (PART_VI_TITLE, 105, {"replace_source": [
        "This part trains an end-to-end neural network whose **output is portfolio weights**, not returns:\n",
        "\n",
        "$$\n",
        "w_t = f_\\theta(\\phi_t), \\qquad r^{\\text{replica}}_t = w_t^{\\top} r_t.\n",
        "$$\n",
        "\n",
        "The same `(X, y)` and transaction-cost assumptions from Part I are reused, so every model in this notebook lands in the same leaderboard.\n",
    ]}),
    (PART_VI_TITLE, 106, None),
    (PART_VI_TITLE, 107, None),
    (PART_VI_TITLE, 108, None),
    (PART_VI_TITLE, 110, None),
    (PART_VI_TITLE, 111, None),
    (PART_VI_TITLE, 112, None),
    (PART_VI_TITLE, 113, None),
    (PART_VI_TITLE, 114, None),
    (PART_VI_TITLE, 115, None),
    (PART_VI_TITLE, 116, None),
    (PART_VI_TITLE, 117, None),
    (PART_VI_TITLE, 118, None),
    (PART_VI_TITLE, 119, None),
    (PART_VI_TITLE, 120, None),
    (PART_VI_TITLE, 121, None),
    (PART_VI_TITLE, 122, None),
    (PART_VI_TITLE, 123, None),
    (PART_VI_TITLE, 124, None),
    (PART_VI_TITLE, 125, None),

    # ---- Part VII: Final Consolidated Comparison & Findings ----
    (PART_VII_TITLE, 126, None),
    (PART_VII_TITLE, 127, None),
    (PART_VII_TITLE, 128, None),
    (PART_VII_TITLE, "NEW_FINDINGS", {"replace_source": [
        "## Findings\n",
        "\n",
        "**Part III (Ridge + Constraints).** Cadence sets the cost / fit trade-off. Weekly rebalancing ",
        "pays the highest turnover and hence the heaviest 5 bps drag; quarterly rebalancing loses ",
        "tracking precision but survives crises with the lowest max drawdown. The cadence x leverage ",
        "grid identifies the composite-score winner at the project's net IR criterion.\n",
        "\n",
        "**Part IV (Linear PO).** Across the {OLS, Ridge, Lasso, ElasticNet, Huber} x {all-11, top-5} ",
        "leaderboard, the longest rolling window (3y) and modest Ridge regularisation dominate on net IR. ",
        "Huber is competitive only in the stress windows; Lasso's sparsity gains are offset by higher TE.\n",
        "\n",
        "**Part V (Kalman).** The static sigma_w = 1e-3 grid point is consistent with the EM-tuned ",
        "(sigma_w, sigma_y) under 3 EM iterations. The regime-switching variant (warm-up-frozen HMM) ",
        "narrows TE in the 2008 crisis window but at the cost of higher turnover.\n",
        "\n",
        "**Part VI (DL).** The MLP weight generator with turnover penalty lambda = 1e-3 lands inside the ",
        "linear-PO leaderboard but does not surpass the best PO + ElasticNet on net IR. The attention ",
        "variant is parameter-hungry and underperforms on this 700-week sample.\n",
        "\n",
        "**Overall.** PO + ElasticNet (top-5, monthly rebalance, GE = 150%) wins the consolidated ",
        "net-IR criterion. Survival of the 2008 and 2020 crisis windows is the binding constraint for ",
        "any deployable configuration; the cadence x leverage grid in Part III makes that explicit.\n",
    ]}),
    (PART_VII_TITLE, 70, {"retitle": "## Acceptance checklist against the project brief"}),
]


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def _new_cell(cell_type: str, source: list[str]) -> dict[str, Any]:
    base: dict[str, Any] = {
        "cell_type": cell_type,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _apply_edits(cell: dict[str, Any], edits: dict[str, Any] | None) -> dict[str, Any]:
    if not edits:
        return cell
    cell = copy.deepcopy(cell)
    if "replace_source" in edits:
        cell["source"] = edits["replace_source"]
    if "prepend_source" in edits:
        cell["source"] = list(edits["prepend_source"]) + cell["source"]
    if "append_source" in edits:
        cell["source"] = cell["source"] + list(edits["append_source"])
    if "retitle" in edits:
        new_title = edits["retitle"].rstrip("\n") + "\n"
        src = cell["source"]
        if src and src[0].lstrip().startswith("#"):
            src[0] = new_title
        else:
            cell["source"] = [new_title] + src
    return cell


def _fold_markdown(cells: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    folded: dict[int, dict[str, Any]] = {}
    for host_idx, addendum_idxs in FOLD_INTO.items():
        host = copy.deepcopy(cells[host_idx])
        merged = list(host.get("source", []))
        for j in addendum_idxs:
            if j >= len(cells):
                continue
            add = cells[j]
            if add.get("cell_type") != "markdown":
                continue
            merged += ["\n\n"]
            merged += add.get("source", [])
        host["source"] = merged
        folded[host_idx] = host
    return folded


def main() -> None:
    if not BAK.exists():
        shutil.copyfile(NB, BAK)
        print(f"Backed up {NB} -> {BAK}")

    nb = json.loads(BAK.read_text())
    cells_in = nb["cells"]
    folded = _fold_markdown(cells_in)

    consumed_by_fold = set()
    for addendums in FOLD_INTO.values():
        consumed_by_fold.update(addendums)

    cells_out: list[dict[str, Any]] = []
    current_part: str | None = None

    for part_title, ref, edits in MANIFEST:
        if part_title and part_title != current_part:
            cells_out.append(_new_cell("markdown", [part_title]))
            current_part = part_title

        if isinstance(ref, int):
            if ref in DROP:
                raise ValueError(f"Manifest references dropped cell {ref}")
            if ref in consumed_by_fold:
                raise ValueError(f"Manifest references consumed-by-fold cell {ref}")
            cell = folded.get(ref, cells_in[ref])
            cell = _apply_edits(cell, edits)
            cells_out.append(cell)
        elif ref == "NEW_WORKFLOW":
            cells_out.append(_new_cell("markdown", edits["replace_source"]))
        elif ref == "NEW_FINDINGS":
            cells_out.append(_new_cell("markdown", edits["replace_source"]))
        else:
            raise ValueError(f"Unknown manifest reference: {ref!r}")

    referenced = {ref for _, ref, _ in MANIFEST if isinstance(ref, int)}
    expected_emitted = set(range(len(cells_in))) - DROP - consumed_by_fold - FINDINGS_SOURCES
    missing = expected_emitted - referenced
    if missing:
        raise SystemExit(
            f"Manifest missing original cells: {sorted(missing)}. "
            "Either add them to the manifest, the DROP set, or the FOLD_INTO map."
        )

    nb_out = copy.deepcopy(nb)
    nb_out["cells"] = cells_out
    NB.write_text(json.dumps(nb_out, indent=1))
    print(f"Wrote {NB} with {len(cells_out)} cells (was {len(cells_in)}).")


if __name__ == "__main__":
    main()
