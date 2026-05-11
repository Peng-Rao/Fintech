# BC3 Notebook & Module Refactor — Design

**Date:** 2026-05-11
**Author:** Peng Rao (with Claude assistance)
**Status:** Approved — ready for implementation plan
**Branch:** `refactor/bc3-cleanup`

---

## 1. Goals

Refactor the BC3 Portfolio Replica deliverable so a reader can follow it top-to-bottom without backtracking and a grader can find evidence for every claim. Concretely:

- **Clear heading hierarchy** — one `#` per Part, consistent `##`/`###` nesting, no orphan sub-headings.
- **Coherent content** — fragmented "Graph explanation" markdown cells folded into their section intros; literal duplicates removed.
- **Minimised code reuse** — `portfolio_constraints.py` stops re-implementing harness primitives; one canonical place for VaR / GE / metrics helpers.
- **No data leakage** in any model-consumed feature or parameter.
- **Workflow follows financial methodology** — data → harness/methodology → controlled baseline → model families → synthesis, rather than the historical M1 → M5 narrative.
- **Workflow rendered as a single markdown cell** at the top of the notebook.

Out of scope: re-running the notebook end-to-end, regenerating `results/*.pkl`, updating `README.md` / `TODO.md` / `presentation.tex` to reflect new Part numbering, renaming `main.ipynb`.

---

## 2. Constraints & decisions made during brainstorming

| Decision | Choice |
|---|---|
| Scope | Full refactor: `main.ipynb` + all five `.py` modules. |
| Part ordering | Reorder to financial-methodology arc (see §3). |
| Output location | Feature branch `refactor/bc3-cleanup`. User merges. |
| Aggressiveness | Moderate — merge fragmented markdown, drop literal duplicates, keep all unique analysis content. |
| Data-leakage policy | Fix leakage in model-consumed sites; label in-sample diagnostics. |
| Module consolidation | Option A — keep all five module files (mirrors M1–M5 team allocation), eliminate duplicates in `portfolio_constraints.py`, fix HMM leakage. |

---

## 3. New Part structure

| Part | Title | Content origin |
|---|---|---|
| **I** | Setup, Data Loading & EDA | Old cells 0–27. |
| **II** | Backtest Methodology | Old cells 28–30, 52–54, 70–73. |
| **III** | Ridge Control + Rebalancing & Constraints Sweep | Old cells 31–77 + old Part V cells 129–164. |
| **IV** | Linear Benchmark Family (Predict-then-Optimize) | Old Part II cells 78–91. |
| **V** | Kalman Filter / State-Space | Old Part III cells 92–104. |
| **VI** | Deep Learning Weight Generator | Old Part IV cells 105–128. |
| **VII** | Final Consolidated Comparison & Findings | Old cells 126–128 + scattered "Findings." blocks (91, 104, 164). |

**New top-of-notebook cell** — single markdown cell `## Workflow at a glance` after the Colab badge, containing the table above and a four-line data-flow narrative:

```
prices
  → (X, y) panel
  → harness.run_rolling_backtest + evaluate_weights
  → {Ridge baseline, Linear PO, Kalman, NN}
  → results/*.pkl
  → consolidated comparison
```

---

## 4. Notebook-level cell hygiene

**Heading rules**
- One `#` per Part.
- `##` for the named sections inside a Part (Setup, Data loading, EDA & data quality, etc.).
- `###` only when its parent `##` has 2+ true sub-blocks. Single-block `###` headings collapse upward.
- Every code cell that produces a plot or table sits under a `##` or `###`; no naked code cells.

**Markdown consolidation rules**
- One-line "**Graph explanation.**" cells (old indices 17, 19, 27, 36, 42, 46, 58, 60, 67, 73, 76) merge **into the markdown cell immediately above** the plot.
- Each named section adopts a consistent block structure: **Goal → Method → How to read it**, three short paragraphs max.
- "**Findings.**" blocks at the end of old Parts II, III, V (cells 91, 104, 164) move to Part VII; each model family contributes one paragraph.

**Literal-duplicate deletions**
- Second `assumption_register` display (old cell 72) — drop.
- Inline `variable_info` redefinition (old cell 109) — drop; rely on Part I canonical definition.
- Old cell 71 "Limitations and assumption register" — duplicates cell 52; drop, one sentence absorbed into Part II.
- Old cell 70 "Checklist against the project brief" — keep, but **move to end of Part VII** (whole-notebook checklist, not Part-I-only).

**Retitle & relocate**
- Old cell 47 "Cross-check the second entry point" → `### Sanity check: evaluate_weights() matches run_rolling_backtest()`, moved into Part II methodology.
- Old cells 81–82 "Smoke test — recover the OLS fit" → `### Sanity check: PO recovers OLS when constraints are slack`, moved to top of Part IV.

---

## 5. Module-level changes (Option A)

### `harness.py` — one addition
- **New public helper:** `apply_var_cap_iterative(weights, X_hist, var_confidence, var_horizon, max_var, step, min_scaling)` — the iterative historical-VaR shrinker currently inside `portfolio_constraints.apply_var_cap`. Returns `(scaled_weights, scaling, final_var, history_df)`.
- No other surface changes; `harness.py` remains the import target.

### `portfolio_constraints.py` — heavy surgery
- **Delete** (replaced by re-exports of harness helpers): `calculate_var_gaussian`, `calculate_historical_var`, `apply_gross_exposure_cap`, `compute_turnover`, `compute_metrics`.
- **Wrap** `apply_var_cap` around `harness.apply_var_cap_iterative` to preserve the existing four-tuple signature.
- **Rewrite** `run_backtest` to call `harness.run_rolling_backtest` with an ElasticNet `model_factory`; apply GE / VaR projection between fit and `evaluate_weights`. Output dict keeps the existing keys (`weights_history`, `replica_returns`, `target_returns`, `metrics`, `var_series`, `ge_series`, `scaling_series`) so notebook cells 136–164 do not change.
- **Keep unchanged** (genuine track-specific logic): `evaluate_crisis_window`, `build_scenario_grid`, `run_scenario_search`, `select_best_scenarios`.
- Expected file size: 447 → ~180 lines.

### `kalman.py` — leakage fix
- `fit_hmm_regime(y, vol_window=12, ..., fit_until: pd.Timestamp | None = None)`. When `fit_until` is provided, HMM is fit on `y.loc[:fit_until]` only; `.predict` then runs on the full sample. `None` preserves current behaviour (back-compat).
- The notebook **must** pass `fit_until=X.index[BASELINE_CFG.rolling_window]` (i.e., the warm-up boundary, matching the Ridge initialisation).
- Add one markdown sentence to Part V explaining: *"The HMM is fit on the 104-week warm-up window only and frozen; subsequent regime labels are produced by prediction, not refit."*

### `predict_then_optimize.py` — docstring only
- Already plugs cleanly into the harness; no logic edits.
- Update module docstring to reference "Part IV" (was "linear-benchmark track").

### `dl_pipeline.py` — docstring only
- Audit-confirmed leakage-safe (`build_features_pca` refits PCA per step; `_ridge_warmstart` uses trailing window; `build_features` ends with `.shift(1)`).
- Update module docstring to reference "Part VI" (was "M5 / NN weight generator").

---

## 6. Data-leakage policy

**Single rule:** anything fit on data wider than its consumer's training window must be either (a) refit per-window, (b) frozen on a clearly-named warm-up slice, or (c) labelled "in-sample diagnostic, not used by any model."

| Site | Status | Action |
|---|---|---|
| `kalman.fit_hmm_regime` consumed by Part V regime switching | Leaks today | Add `fit_until` (option b). |
| `dl_pipeline.build_features_pca` | Safe (rolling refit) | No change. |
| `dl_pipeline._ridge_warmstart` | Safe (trailing-window + `.shift(1)`) | No change. |
| `dl_pipeline.build_features` regime block | Safe (`.shift(1)` at end) | No change. |
| `harness.run_rolling_backtest` | Safe | No change. |
| `predict_then_optimize.run_predict_then_optimize_backtest` | Safe | No change. |
| `portfolio_constraints.run_backtest` (post-rewrite) | Inherits harness's slicing | No change. |

**Diagnostics — label, do not fix.** Each gets a one-line caption ending **"In-sample diagnostic; not used by any model."**

- `data_quality_report`, `market_stress_outlier_audit`, `compute_vif`.
- Full-sample correlation heatmaps, full-sample annualised return statistics.
- `var_gaussian_target`, target return-distribution overlay, Jarque-Bera normality test (old cells 132–134).
- Full-sample histograms / QQ-plots.

**Outlier-audit decision rule retained verbatim:** *"The pipeline removes only invalid inputs such as missing required returns or non-positive required prices. It does not winsorize or drop flagged outlier weeks; these are reported for transparency and feed the regime audit downstream."*

**End-of-refactor verification:** grep the codebase for `.fit(` calls outside the whitelisted diagnostic helpers; any new hit is a bug to fix before merge.

---

## 7. Verification scaffolding

Two short standalone scripts under `BC3/scripts/`, deleted in the final commit:

- `smoke_constraints.py` — runs `BASELINE_CONFIGS[0]` before and after the `portfolio_constraints.py` rewrite, asserts `metrics["IR"]`, `metrics["TE"]`, `metrics["turnover"]` match to 1e-6.
- `smoke_kalman.py` — runs `KalmanConfig(sigma_w=1e-3)` with `fit_until=None` (legacy mode) and asserts `replica_returns` and `metrics["IR" | "TE" | "turnover"]` match the pre-refactor `main` reference to 1e-6.

These are verification scaffolding, not production tests, and are removed in commit 6 below.

---

## 8. Branch & commit plan

Branch: `refactor/bc3-cleanup`, branched from `main`. User drives the merge.

| # | Commit | Files touched | Purpose |
|---|---|---|---|
| 1 | `chore(BC3): add refactor scaffolding` | `docs/superpowers/specs/2026-05-11-bc3-refactor-design.md`, `BC3/scripts/smoke_*.py` | Spec + smoke scripts. |
| 2 | `refactor(BC3): consolidate portfolio_constraints onto harness` | `portfolio_constraints.py`, `harness.py` | Shrink `portfolio_constraints.py`; add `harness.apply_var_cap_iterative`. Smoke script confirms parity. |
| 3 | `fix(BC3): freeze HMM regime on warm-up window` | `kalman.py` | Add `fit_until` parameter. Smoke script confirms legacy mode unchanged. |
| 4 | `docs(BC3): align module docstrings with new Part numbering` | `predict_then_optimize.py`, `dl_pipeline.py`, `kalman.py`, `portfolio_constraints.py` | Docstring + section-banner cleanup only. |
| 5 | `refactor(BC3): restructure main.ipynb into 7 parts` | `BC3/main.ipynb` | Reorganisation, workflow markdown cell, heading/markdown consolidation, diagnostic-labelling pass. |
| 6 | `chore(BC3): drop smoke scaffolding` | `BC3/scripts/` | Delete after commits 2 and 3 are individually verified. |

Each commit is independently revertable. The notebook rewrite is the last logic-bearing commit, so reverting it does not undo the `.py` fixes.

---

## 9. Acceptance criteria

A refactor is complete when **all** of the following hold:

1. `git diff main` shows the seven-part notebook reorganisation plus the four `.py` changes above and nothing else.
2. `smoke_constraints.py` and `smoke_kalman.py` exit 0 in the pre-merge run.
3. The notebook contains exactly one `#` heading per Part (seven total), exactly one `assumption_register()` display, and a `## Workflow at a glance` markdown cell after the badge.
4. Grep for unguarded full-sample `.fit(` calls outside the whitelisted diagnostic helpers returns zero hits.
5. `kalman.fit_hmm_regime` accepts `fit_until=` and Part V passes it.
6. `portfolio_constraints.py` is under 200 lines.
7. `BC3/scripts/` is removed by the final commit on the branch.
