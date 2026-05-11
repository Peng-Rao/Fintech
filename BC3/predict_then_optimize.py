"""Predict-then-Optimize layer (notebook Part IV: Linear Benchmark Family).

Four phases:
    1. Alpha:    classical linear model on (X_tr, y_tr) -> coef_ as mu
    2. Risk:     Ledoit-Wolf shrunken sample covariance
    3. Optimize: convex max mu'w - 0.5*lambda*w'Sigma w - tau*||w-w_prev||_1
                 s.t. ||w||_1 <= ge_cap, |w_j| <= w_cap   (CVXPY)
    4. Audit:    rebalance-date weights -> evaluate_weights(...) for cost / VaR / GE accounting

Every alpha factory must accept (X_tr, y_tr) at .fit() and expose .coef_, the
sklearn convention. OLS, Ridge, Lasso, ElasticNet and HuberRegressor all conform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from harness import FlatBpsTC, HarnessConfig, ReplicaResult, evaluate_weights
from sklearn.covariance import LedoitWolf

__all__ = [
    "POConfig",
    "alpha_from_linear_fit",
    "shrunk_covariance",
    "solve_po",
    "top_k_active",
    "run_predict_then_optimize_backtest",
]


@dataclass
class POConfig:
    """Predict-then-Optimize configuration. Defaults match the project mandate.

    risk_aversion=2.0 is the natural value for the tracking-error reformulation
    (Var(w'r-y) = w'Sigma w - 2 w'sigma_{ry} + ...), so the convex objective recovers
    the OLS/Ridge fit when constraints are slack.
    """

    risk_aversion: float = 2.0  # lambda in 0.5*lambda*w'Sigma w
    tc_bps: float = 5.0  # tau in tau * ||w - w_prev||_1 (one-way bps)
    ge_cap: float = 2.0  # gross-exposure cap: sum |w_j| <= ge_cap
    w_cap: float = 0.5  # per-asset cap: |w_j| <= w_cap
    use_lw_shrinkage: bool = True  # Ledoit-Wolf shrinkage on Sigma
    top_k: Optional[int] = None  # optional |corr(r_j, y)| pre-selection
    long_only: bool = False  # long-only: w >= 0 (no shorts)
    solver: str = "CLARABEL"  # CVXPY solver; SCS fallback on failure


def alpha_from_linear_fit(
    model_factory: Callable[[], Any], X_tr: np.ndarray, y_tr: np.ndarray
) -> np.ndarray:
    """Phase 1: train a classical linear model on (X_tr, y_tr) and return its
    coefficients as the asset-level alpha mu. Coefficients live in raw return units
    (no standardisation) so the optimiser can compare mu and sqrt(diag(Sigma)) directly.
    """
    mdl = model_factory()
    mdl.fit(X_tr, y_tr)
    coef = np.asarray(getattr(mdl, "coef_"), dtype=float).ravel()
    if coef.shape[0] != X_tr.shape[1]:
        raise ValueError(
            f"alpha factory returned coef of shape {coef.shape}, expected ({X_tr.shape[1]},)"
        )
    return coef


def shrunk_covariance(X_tr: np.ndarray, use_lw: bool) -> np.ndarray:
    """Phase 2: Ledoit-Wolf shrunken sample covariance (or sample covariance if disabled).

    LW automatically shrinks toward a constant-variance diagonal target, well-suited to
    futures returns where the universe is small but cross-correlations are noisy.
    """
    if use_lw:
        lw = LedoitWolf()
        lw.fit(X_tr)
        Sigma = np.asarray(lw.covariance_, dtype=float)
    else:
        Sigma = np.cov(X_tr, rowvar=False, ddof=1)
    Sigma = (Sigma + Sigma.T) / 2.0
    eig_min = float(np.linalg.eigvalsh(Sigma).min())
    if eig_min < 1e-10:
        Sigma = Sigma + (1e-10 - eig_min) * np.eye(Sigma.shape[0])
    return Sigma


def solve_po(
    mu: np.ndarray,
    Sigma: np.ndarray,
    w_prev: np.ndarray,
    *,
    risk_aversion: float,
    tc_bps: float,
    ge_cap: float,
    w_cap: float,
    solver: str = "CLARABEL",
    long_only: bool = False,
) -> np.ndarray:
    """Phase 3: convex max mu'w - 0.5*lambda*w'Sigma w - tau*||w-w_prev||_1
    s.t. ||w||_1 <= ge_cap, |w_j| <= w_cap. With long_only=True, additionally enforces w >= 0
    (long-only variant). Falls back to SCS, then to w_prev on failure.
    """
    n = mu.size
    w = cp.Variable(n)
    tau = tc_bps / 1e4
    objective = cp.Maximize(
        mu @ w
        - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(Sigma))
        - tau * cp.norm1(w - w_prev)
    )
    constraints = [cp.norm1(w) <= ge_cap, cp.abs(w) <= w_cap]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    for s in (solver, "SCS"):
        try:
            prob.solve(solver=s, verbose=False)
            if w.value is not None and np.all(np.isfinite(w.value)):
                return np.asarray(w.value, dtype=float).ravel()
        except (cp.error.SolverError, ValueError):
            continue
    return w_prev.copy()


def top_k_active(X_tr: np.ndarray, y_tr: np.ndarray, k: int) -> np.ndarray:
    """Top-K asset pre-selection by |corr(r_j, y)| inside the training window."""
    if k >= X_tr.shape[1]:
        return np.arange(X_tr.shape[1])
    cors = np.zeros(X_tr.shape[1])
    for j in range(X_tr.shape[1]):
        s = X_tr[:, j].std(ddof=1)
        if s > 0:
            cors[j] = np.corrcoef(X_tr[:, j], y_tr)[0, 1]
    order = np.argsort(-np.abs(cors))
    return np.sort(order[:k])


def run_predict_then_optimize_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    alpha_factory: Callable[[], Any],
    *,
    config: HarnessConfig,
    po_config: POConfig,
    name: str,
) -> ReplicaResult:
    """Predict-then-Optimize rolling backtest plugged into the harness.

    Walks the same (rolling_window, rebalance_every) grid as run_rolling_backtest, but at each
    rebalance: (1) trains the alpha factory to get mu, (2) builds shrunken Sigma, (3) solves the
    convex problem with mandate constraints baked in, (4) hands the rebalance-date weight matrix
    to evaluate_weights for cost / risk auditing.
    """
    cfg = config
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    n_obs, n_feat = Xv.shape
    if cfg.rolling_window >= n_obs:
        raise ValueError(
            f"rolling_window={cfg.rolling_window} must be smaller than len(X)={n_obs}"
        )

    eval_dates = X.index[cfg.rolling_window :]
    rb_offsets = list(range(0, len(eval_dates), cfg.rebalance_every))
    rb_dates = pd.DatetimeIndex([eval_dates[k] for k in rb_offsets])

    rows: list[np.ndarray] = []
    w_prev = np.zeros(n_feat)
    for k in rb_offsets:
        t = cfg.rolling_window + k
        tr_lo = max(cfg.fit_window_start, t - cfg.rolling_window)
        X_tr = Xv[tr_lo:t]
        y_tr = yv[tr_lo:t]

        active = (
            top_k_active(X_tr, y_tr, po_config.top_k)
            if po_config.top_k is not None
            else np.arange(n_feat)
        )
        Xa_tr = X_tr[:, active]

        mu_a = alpha_from_linear_fit(alpha_factory, Xa_tr, y_tr)
        Sigma_a = shrunk_covariance(Xa_tr, po_config.use_lw_shrinkage)

        # Transform the linear-model coefficients (which are optimal unconstrained weights)
        # into the implied 'mu' vector for the standard mean-variance objective:
        # FOC: mu - lambda * Sigma * w = 0  => mu = lambda * Sigma * w_opt
        implied_mu = po_config.risk_aversion * (Sigma_a @ mu_a)

        w_a = solve_po(
            implied_mu,
            Sigma_a,
            w_prev[active],
            risk_aversion=po_config.risk_aversion,
            tc_bps=po_config.tc_bps,
            ge_cap=po_config.ge_cap,
            w_cap=po_config.w_cap,
            solver=po_config.solver,
            long_only=po_config.long_only,
        )
        w_full = np.zeros(n_feat)
        w_full[active] = w_a
        rows.append(w_full)
        w_prev = w_full

    weights_history = pd.DataFrame(np.stack(rows), index=rb_dates, columns=X.columns)

    return evaluate_weights(
        X,
        y,
        weights_history,
        schedule_type="rebalance",
        tc_model=FlatBpsTC(po_config.tc_bps),
        config=cfg,
        name=name,
    )
