"""Kalman / state-space layer.

Linear KF with random-walk weights as the latent state and the scalar target return
as the observation. Includes:
    - hand-rolled forward filter with optional VaR guardrail (`kalman_run_full`)
    - Ridge-fit initialisation `_kalman_init`
    - notebook-friendly wrapper `run_kalman_replica` that returns a ReplicaResult
    - pykalman EM helper (`fit_em_noise`)
    - hmmlearn 2-state HMM regime classifier (`fit_hmm_regime`)
    - metrics-row helper (`kf_metrics_row`) for leaderboard tables
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from harness import (
    FlatBpsTC,
    HarnessConfig,
    ReplicaResult,
    evaluate_weights,
    gaussian_var,
    metrics_row_from_replica,
)

# Back-compat alias: kf_metrics_row was the Kalman-specific name; the canonical helper now
# lives in harness.metrics_row_from_replica and is shared by every track.
kf_metrics_row = metrics_row_from_replica

__all__ = [
    "KalmanConfig",
    "kalman_run_full",
    "run_kalman_replica",
    "fit_em_noise",
    "fit_hmm_regime",
    "kf_metrics_row",
]


@dataclass
class KalmanConfig:
    """Hyper-parameters for the linear Kalman filter on portfolio weights."""
    sigma_w: float = 1e-3                    # process noise std (B = sigma_w * I)
    sigma_y: Optional[float] = None          # obs noise std; None -> sample std on init window
    init_window: int = 52                    # weeks Ridge uses to seed x_0
    init_ridge_alpha: float = 1.0
    P0_scale: float = 0.01                   # P_0 = P0_scale * I_11
    var_cap: float = 0.08
    var_history_w: int = 52
    var_horizon_w: int = 4
    var_conf: float = 0.01
    apply_var_guardrail: bool = True


def _kalman_init(
    X: pd.DataFrame, y: pd.Series, cfg: KalmanConfig
) -> Tuple[np.ndarray, float]:
    """Ridge-fit x_0 on the first init_window weeks; default sigma_y from the same window."""
    init = max(2, cfg.init_window)
    ridge = Ridge(alpha=cfg.init_ridge_alpha, fit_intercept=False)
    ridge.fit(X.iloc[:init].values, y.iloc[:init].values)
    sigma_y = cfg.sigma_y if cfg.sigma_y is not None else float(y.iloc[:init].std(ddof=1))
    return ridge.coef_.copy(), sigma_y


def kalman_run_full(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: KalmanConfig,
    *,
    sigma_w_series: Optional[pd.Series] = None,
    sigma_y_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Linear Kalman filter (random-walk weights) with optional per-step VaR guardrail.

    sigma_w_series / sigma_y_series allow per-step noise overrides for regime/adaptive variants.
    Returns the post-update posterior means x_{t|t} as a weekly DataFrame.
    """
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    T, n_assets = Xv.shape
    A = np.eye(n_assets)
    x_t, sigma_y_default = _kalman_init(X, y, cfg)
    P_t = np.eye(n_assets) * cfg.P0_scale

    weights = np.empty((T, n_assets))
    for t in range(T):
        sw = float(sigma_w_series.iloc[t]) if sigma_w_series is not None else cfg.sigma_w
        sy = float(sigma_y_series.iloc[t]) if sigma_y_series is not None else sigma_y_default
        Q_t = (sw ** 2) * np.eye(n_assets)
        R_t = sy ** 2

        Ct = Xv[t].reshape(1, -1)
        x_pred = A @ x_t
        P_pred = A @ P_t @ A.T + Q_t
        innov = float(yv[t] - (Ct @ x_pred).item())
        S = float((Ct @ P_pred @ Ct.T).item() + R_t)
        K = (P_pred @ Ct.T).ravel() / S
        x_t = x_pred + K * innov
        P_t = P_pred - np.outer(K, (Ct @ P_pred).ravel())

        if cfg.apply_var_guardrail and t >= cfg.init_window:
            lo = max(0, t - cfg.var_history_w)
            v = gaussian_var(Xv[lo:t + 1] @ x_t, conf=cfg.var_conf, horizon=cfg.var_horizon_w)
            if np.isfinite(v) and v > cfg.var_cap and v > 0:
                x_t = x_t * (cfg.var_cap / v)
        weights[t] = x_t

    return pd.DataFrame(weights, index=X.index, columns=X.columns)


def run_kalman_replica(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cfg: KalmanConfig,
    name: str,
    eval_window: int = 104,
    tc_bps: float = 5.0,
    harness_cfg: Optional[HarnessConfig] = None,
    sigma_w_series: Optional[pd.Series] = None,
    sigma_y_series: Optional[pd.Series] = None,
) -> ReplicaResult:
    """Filter -> shift-by-1 (no look-ahead) -> evaluate_weights(schedule_type='held')."""
    weights_df = kalman_run_full(
        X, y, cfg, sigma_w_series=sigma_w_series, sigma_y_series=sigma_y_series,
    )
    weights_eval = weights_df.shift(1).dropna()
    weights_eval = weights_eval.loc[weights_eval.index >= X.index[eval_window]]
    if harness_cfg is None:
        from harness import DEFAULT_VAR_CAP, DEFAULT_GE_CAP   # late import to avoid cycle
        harness_cfg = HarnessConfig()
    return evaluate_weights(
        X, y, weights_eval,
        schedule_type="held",
        tc_model=FlatBpsTC(tc_bps),
        config=harness_cfg,
        name=name,
    )


def fit_em_noise(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: KalmanConfig,
    *,
    n_iter: int = 3,
    sigma_w_seed: float = 1e-3,
) -> Tuple[float, float, float, float]:
    """Pykalman EM on (transition_covariance, observation_covariance).

    Caps n_iter to 3 because the (1 obs, 11 latent state) system is under-identified;
    pykalman's coordinate-ascent EM diverges past 3 iterations on this problem.

    Returns (sigma_w, sigma_y, loglik, elapsed_s).
    """
    import time as _time
    from pykalman import KalmanFilter

    n_assets = X.shape[1]
    x0, sigma_y_init = _kalman_init(X, y, cfg)
    obs_matrices = X.values.reshape(len(X), 1, n_assets)

    kf = KalmanFilter(
        transition_matrices=np.eye(n_assets),
        observation_matrices=obs_matrices,
        initial_state_mean=x0,
        initial_state_covariance=np.eye(n_assets) * cfg.P0_scale,
        transition_covariance=np.eye(n_assets) * (sigma_w_seed ** 2),
        observation_covariance=np.array([[sigma_y_init ** 2]]),
    )
    t0 = _time.perf_counter()
    kf = kf.em(y.values, n_iter=n_iter, em_vars=["transition_covariance", "observation_covariance"])
    elapsed = _time.perf_counter() - t0

    sigma_w = float(np.sqrt(np.mean(np.diag(kf.transition_covariance))))
    sigma_y = float(np.sqrt(kf.observation_covariance[0, 0]))
    loglik = float(kf.loglikelihood(y.values))
    return sigma_w, sigma_y, loglik, elapsed


def fit_hmm_regime(
    y: pd.Series,
    *,
    vol_window: int = 12,
    n_iter: int = 100,
    random_state: int = 42,
) -> Tuple[pd.Series, np.ndarray, int]:
    """Two-state Gaussian HMM on rolling realised target volatility.

    Returns (regime_indicator [0=calm, 1=stressed], state_means, stress_state_index).
    """
    from hmmlearn.hmm import GaussianHMM

    vol_obs = y.rolling(vol_window, min_periods=4).std().bfill().values.reshape(-1, 1)
    hmm_model = GaussianHMM(
        n_components=2, covariance_type="full",
        n_iter=n_iter, random_state=random_state,
    )
    hmm_model.fit(vol_obs)
    states = hmm_model.predict(vol_obs)
    stress_state = int(np.argmax(hmm_model.means_.ravel()))
    regime = pd.Series(
        (states == stress_state).astype(int), index=y.index, name="regime",
    )
    return regime, hmm_model.means_.ravel(), stress_state


# kf_metrics_row alias is defined at the top of the module (re-export of harness helper).
