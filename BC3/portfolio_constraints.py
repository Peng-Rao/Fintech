"""Rebalancing & portfolio constraints (extended track).

Standalone Elastic-Net rolling backtest with:
    - per-rebalance gross-exposure cap projection (`apply_gross_exposure_cap`)
    - per-rebalance Gaussian / historical VaR cap projection (`apply_var_cap`)
    - crisis-window evaluation (`evaluate_crisis_window`)
    - 2D scenario grid (`build_scenario_grid` + `run_scenario_search`)
    - composite-score scenario selector (`select_best_scenarios`)

Adapted from the team-member's `M3_Rebalancing_PortfolioConstraints_extended.ipynb`.
Logic is preserved as-is so results match the standalone notebook; main.ipynb just supplies
the `(X, y)` panel produced by Part I.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    "calculate_var_gaussian",
    "calculate_historical_var",
    "test_normality",
    "apply_gross_exposure_cap",
    "apply_var_cap",
    "compute_turnover",
    "compute_metrics",
    "fit_elastic_net",
    "run_backtest",
    "evaluate_crisis_window",
    "build_scenario_grid",
    "run_scenario_search",
    "select_best_scenarios",
]


# ---------------------------------------------------------------------------
# VaR helpers
# ---------------------------------------------------------------------------

def calculate_var_gaussian(returns, confidence: float = 0.01, horizon: int = 4) -> float:
    """Parametric Gaussian VaR as a positive loss fraction of NAV."""
    sigma = np.std(returns)
    z_score = stats.norm.ppf(confidence)
    return float(-z_score * sigma * np.sqrt(horizon))


def calculate_historical_var(returns, confidence: float = 0.01, horizon: int = 4) -> float:
    """Historical-quantile VaR over a multi-week horizon."""
    returns = np.asarray(returns, dtype=float)
    series = pd.Series(returns[~np.isnan(returns)])
    horizon_returns = (1 + series).rolling(window=horizon).apply(np.prod, raw=True) - 1
    horizon_returns = horizon_returns.dropna()
    if len(horizon_returns) == 0:
        return float("nan")
    return float(-np.quantile(horizon_returns, confidence))


def test_normality(returns, alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Jarque-Bera test for return normality."""
    stat_jb, p_jb = stats.jarque_bera(returns)
    return {
        "Jarque-Bera": {"stat": float(stat_jb), "p_value": float(p_jb), "normal": bool(p_jb > alpha)},
    }


# ---------------------------------------------------------------------------
# Constraint projections
# ---------------------------------------------------------------------------

def apply_gross_exposure_cap(weights, max_gross_exposure: float) -> Tuple[np.ndarray, float, float]:
    """Scale weights so sum |w_j| <= max_gross_exposure."""
    weights = np.asarray(weights, dtype=float)
    gross_exposure = float(np.sum(np.abs(weights)))
    scaling_factor = 1.0
    if gross_exposure > max_gross_exposure:
        scaling_factor = max_gross_exposure / gross_exposure
        weights = weights * scaling_factor
        gross_exposure = float(np.sum(np.abs(weights)))
    return weights, scaling_factor, gross_exposure


def apply_var_cap(
    weights,
    X_values,
    var_confidence: float = 0.01,
    var_horizon: int = 4,
    max_var: float = 0.08,
    step: float = 0.01,
    min_scaling: float = 0.0,
) -> Tuple[np.ndarray, float, float, pd.DataFrame]:
    """Iteratively shrink weights until historical VaR <= max_var.

    Returns (scaled_weights, scaling, final_var, scan_history_df).
    """
    scaling = 1.0
    history: list[dict] = []
    while scaling >= min_scaling:
        weights_scaled = weights * scaling
        portfolio_returns = X_values @ weights_scaled
        var_value = calculate_historical_var(
            portfolio_returns, confidence=var_confidence, horizon=var_horizon,
        )
        history.append({"scaling": scaling, "VaR": var_value})
        if not np.isnan(var_value) and var_value <= max_var:
            return weights_scaled, scaling, var_value, pd.DataFrame(history)
        scaling -= step
    history_df = pd.DataFrame(history)
    return weights * min_scaling, min_scaling, history_df["VaR"].iloc[-1], history_df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_turnover(weights_history) -> float:
    """Mean L1 weight change between consecutive rebalances."""
    if isinstance(weights_history, pd.DataFrame):
        vals = weights_history.values
    else:
        vals = weights_history
    diffs = [np.sum(np.abs(vals[t] - vals[t - 1])) for t in range(1, len(vals))]
    return float(np.mean(diffs)) if diffs else 0.0


def compute_metrics(
    asset_returns: pd.Series,
    target_returns: pd.Series,
    weights_history,
    *,
    cost_bps: float = 5,
    var_values: Optional[Iterable[float]] = None,
) -> pd.DataFrame:
    """Wide set of replication metrics, returned as a (Metric × Value) DataFrame."""
    ann = 52
    diff = asset_returns - target_returns
    rep_ann = float(asset_returns.mean() * ann)
    tgt_ann = float(target_returns.mean() * ann)
    rep_vol = float(asset_returns.std() * np.sqrt(ann))
    tgt_vol = float(target_returns.std() * np.sqrt(ann))
    TE = float(diff.std() * np.sqrt(ann))
    IR = (rep_ann - tgt_ann) / TE if TE > 0 else float("nan")
    rho = float(asset_returns.corr(target_returns))
    sharpe = rep_ann / rep_vol if rep_vol > 0 else float("nan")

    wh_vals = weights_history.values if isinstance(weights_history, pd.DataFrame) else weights_history
    GE_mean = float(np.mean([np.sum(np.abs(w)) for w in wh_vals]))
    avg_var = float(np.nanmean(list(var_values))) if var_values is not None else float("nan")

    turnover = compute_turnover(weights_history)
    cost_per_period = cost_bps / 10_000 * turnover
    net_returns = asset_returns - cost_per_period
    net_ann = float(net_returns.mean() * ann)
    net_TE = float((net_returns - target_returns).std() * np.sqrt(ann))
    net_IR = (net_ann - tgt_ann) / net_TE if net_TE > 0 else float("nan")

    cum = (1 + asset_returns).cumprod()
    mdd = float((1 - cum / cum.cummax()).max())

    metrics = {
        "IR": IR, "TE": TE, "p": rho, "GE": GE_mean, "VaR": avg_var,
        "turnover": turnover, "net_IR": net_IR, "net_TE": net_TE,
        "rep_ann": rep_ann, "tgt_ann": tgt_ann, "rep_vol": rep_vol, "tgt_vol": tgt_vol,
        "sharpe": sharpe, "mdd": mdd, "net_ann": net_ann,
    }
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
    metrics_df.index.name = "Metric"
    return metrics_df


# ---------------------------------------------------------------------------
# Elastic-Net rolling backtest
# ---------------------------------------------------------------------------

def fit_elastic_net(X_train, y_train, alpha: float = 0.001, l1_ratio: float = 0.5) -> np.ndarray:
    """MinMax-normalised Elastic-Net fit; coefficients rescaled to raw return units."""
    scaler_X = MinMaxScaler()
    X_norm = scaler_X.fit_transform(X_train)
    scaler_y = MinMaxScaler()
    y_norm = scaler_y.fit_transform(np.asarray(y_train).reshape(-1, 1)).flatten()

    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=False, max_iter=10000, tol=1e-4,
    )
    model.fit(X_norm, y_norm)
    return model.coef_ / scaler_X.scale_


def run_backtest(
    X_values,
    y_values,
    asset_names: List[str],
    dates,
    *,
    rolling_window: int = 104,
    rebal_freq: int = 4,
    max_gross_exposure: float = 1.0,
    max_var: float = 0.08,
    step: float = 0.005,
    alpha: float = 0.001,
    l1_ratio: float = 0.5,
    cost_bps: float = 5,
    var_confidence: float = 0.01,
    var_horizon: int = 4,
) -> Dict[str, Any]:
    """Rolling Elastic-Net backtest with GE and VaR projection at every rebalance."""
    X_values = np.asarray(X_values)
    y_values = np.asarray(y_values).reshape(-1)
    n_feat = X_values.shape[1]

    weights_list: list[np.ndarray] = []
    replica_list: list[float] = []
    target_list: list[float] = []
    date_list: list = []
    var_list: list[float] = []
    scaling_list: list[float] = []
    ge_list: list[float] = []

    current_weights = np.zeros(n_feat)
    last_rebal_idx = -999
    scaling = 1.0
    var = float("nan")

    for i in range(len(X_values) - rolling_window - 1):
        end_idx = i + rolling_window
        if (i - last_rebal_idx) >= rebal_freq:
            X_train = X_values[i:end_idx]
            y_train = y_values[i:end_idx]
            raw_weights = fit_elastic_net(X_train, y_train, alpha=alpha, l1_ratio=l1_ratio)

            weights_ge, ge_scaling, _final_ge = apply_gross_exposure_cap(
                raw_weights, max_gross_exposure=max_gross_exposure,
            )
            X_hist = X_values[max(0, end_idx - 52):end_idx]
            weights_after_var, var_scaling, final_var, _var_history = apply_var_cap(
                weights=weights_ge,
                X_values=X_hist,
                var_confidence=var_confidence,
                var_horizon=var_horizon,
                max_var=max_var,
                step=step,
            )
            current_weights = weights_after_var
            scaling = ge_scaling * var_scaling
            var = final_var
            last_rebal_idx = i
        else:
            scaling = 1.0
            hist_rets = replica_list[max(0, len(replica_list) - 52):] if replica_list else []
            if len(hist_rets) >= max(12, var_horizon):
                var = calculate_historical_var(
                    hist_rets, confidence=var_confidence, horizon=var_horizon,
                )
            else:
                var = float("nan")

        replica_ret = float(np.dot(X_values[end_idx], current_weights))
        replica_list.append(replica_ret)
        target_list.append(float(y_values[end_idx]))
        date_list.append(dates[end_idx])
        weights_list.append(current_weights.copy())
        var_list.append(var)
        scaling_list.append(scaling)
        ge_list.append(float(np.sum(np.abs(current_weights))))

    replica_returns = pd.Series(replica_list, index=date_list, name="replica")
    target_series = pd.Series(target_list, index=date_list, name="target")
    weights_history = pd.DataFrame(weights_list, index=date_list, columns=asset_names)

    metrics_df = compute_metrics(
        replica_returns, target_series, weights_history,
        cost_bps=cost_bps, var_values=var_list,
    )
    metadata_df = pd.DataFrame.from_dict(
        {
            "rolling_window": rolling_window,
            "rebal_freq": rebal_freq,
            "max_gross_exposure": max_gross_exposure,
            "max_var": max_var,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
        },
        orient="index", columns=["Value"],
    )

    return {
        "weights_history": weights_history,
        "replica_returns": replica_returns,
        "target_returns": target_series,
        "metrics": metrics_df,
        "metadata": metadata_df,
        "var_series": pd.Series(var_list, index=date_list, name="VaR"),
        "ge_series": pd.Series(ge_list, index=date_list, name="Gross Exposure"),
        "scaling_series": pd.Series(scaling_list, index=date_list, name="Scaling"),
    }


# ---------------------------------------------------------------------------
# Crisis-window evaluation
# ---------------------------------------------------------------------------

def evaluate_crisis_window(replica_series: pd.Series, target_series: pd.Series, start, end) -> Dict[str, float]:
    """Cumulative return, drawdown, correlation, TE and realised VaR over a crisis slice."""
    rep = replica_series.loc[start:end]
    tgt = target_series.loc[start:end]
    if len(rep) == 0:
        return {k: float("nan") for k in
                ["cum_rep", "cum_tgt", "mdd", "corr", "te_crisis", "var_realized"]}
    cum_rep = float((1 + rep).prod() - 1)
    cum_tgt = float((1 + tgt).prod() - 1)
    cum_path = (1 + rep).cumprod()
    mdd = float((1 - cum_path / cum_path.cummax()).max())
    corr = float(rep.corr(tgt))
    te_c = float((rep - tgt).std() * np.sqrt(52))
    var_r = calculate_historical_var(rep.values, confidence=0.01, horizon=4)
    return {
        "cum_rep": cum_rep, "cum_tgt": cum_tgt,
        "mdd": mdd, "corr": corr, "te_crisis": te_c, "var_realized": var_r,
    }


# ---------------------------------------------------------------------------
# Scenario search
# ---------------------------------------------------------------------------

def build_scenario_grid(
    rebal_freqs: Iterable[int] = (1, 2, 4, 8, 12),
    gross_exposure_caps: Iterable[float] = (1.0, 1.5, 2.0),
    alphas: Iterable[float] = (0.0001, 0.001, 0.01),
    l1_ratios: Iterable[float] = (0.2, 0.5, 0.8),
    rolling_windows: Iterable[int] = (52, 104),
    max_vars: Iterable[float] = (0.06, 0.08, 0.12),
) -> List[Dict[str, Any]]:
    """Cartesian product of the six hyper-parameters; returns config dicts for `run_backtest`."""
    freq_tags = {1: "Wkly", 2: "BiWkly", 4: "Mthly", 8: "BiMthly", 12: "Qtrly"}
    configs: list[dict] = []
    for idx, (rf, ge, a, l1, rw, mv) in enumerate(
        itertools.product(rebal_freqs, gross_exposure_caps, alphas, l1_ratios, rolling_windows, max_vars),
        start=1,
    ):
        configs.append({
            "label": f"S{idx:03d}",
            "desc": (
                f"{freq_tags.get(rf, f'{rf}w')} / {int(ge*100)}%GE / a{a} / L1={l1} "
                f"/ RW{rw}w / VaR{int(mv*100)}%"
            ),
            "rebal_freq": rf,
            "max_gross_exposure": ge,
            "alpha": a,
            "l1_ratio": l1,
            "rolling_window": rw,
            "max_var": mv,
        })
    return configs


def run_scenario_search(
    configs,
    X_values,
    y_values,
    asset_names,
    dates,
    *,
    cost_bps: float = 5,
    var_confidence: float = 0.01,
    var_horizon: int = 4,
    step: float = 0.005,
    verbose: bool = True,
    verbose_every: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Run `run_backtest` for every grid config; return labelled results."""
    scenario_results: Dict[str, Dict[str, Any]] = {}
    n = len(configs)
    for i, config in enumerate(configs, start=1):
        label = config["label"]
        try:
            result = run_backtest(
                X_values=X_values, y_values=y_values,
                asset_names=asset_names, dates=dates,
                rolling_window=config["rolling_window"],
                rebal_freq=config["rebal_freq"],
                max_gross_exposure=config["max_gross_exposure"],
                max_var=config["max_var"],
                step=step,
                alpha=config["alpha"],
                l1_ratio=config["l1_ratio"],
                cost_bps=cost_bps,
                var_confidence=var_confidence,
                var_horizon=var_horizon,
            )
            scenario_results[label] = {"config": config, **result}
            if verbose and (i % verbose_every == 0 or i == n):
                m = result["metrics"]
                IR = m.loc["IR", "Value"]
                TE = m.loc["TE", "Value"]
                rho = m.loc["p", "Value"]
                print(f"  [{i:>4}/{n}]  {label}  IR={IR:+.3f}  TE={TE:.2%}  ρ={rho:.3f}  — {config['desc']}")
        except Exception as e:
            if verbose:
                print(f"  [{i:>4}/{n}]  {label}  ✗ ERROR: {e}")
    print(f"\nCompleted {len(scenario_results)}/{n} scenarios successfully.")
    return scenario_results


def select_best_scenarios(scenario_results) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Per-criterion winners + composite-score-ranked DataFrame of every scenario."""
    criteria = {
        "Best Gross IR":    max(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["IR", "Value"]),
        "Best Net IR":      max(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["net_IR", "Value"]),
        "Best Correlation": max(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["p", "Value"]),
        "Lowest TE":        min(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["TE", "Value"]),
        "Lowest Turnover":  min(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["turnover", "Value"]),
        "Lowest Drawdown":  min(scenario_results, key=lambda k: scenario_results[k]["metrics"].loc["mdd", "Value"]),
    }
    rows = []
    for label, res in scenario_results.items():
        m = res["metrics"]
        rows.append({
            "label": label,
            "desc": res["config"]["desc"],
            "IR": m.loc["IR", "Value"],
            "net_IR": m.loc["net_IR", "Value"],
            "p": m.loc["p", "Value"],
            "TE": m.loc["TE", "Value"],
            "turnover": m.loc["turnover", "Value"],
            "mdd": m.loc["mdd", "Value"],
            "rep_ann": m.loc["rep_ann", "Value"],
            "rep_vol": m.loc["rep_vol", "Value"],
            "sharpe": m.loc["sharpe", "Value"],
            "GE": m.loc["GE", "Value"],
            "VaR": m.loc["VaR", "Value"],
        })
    df = pd.DataFrame(rows).set_index("label")
    for col in ("IR", "net_IR", "p"):
        df[f"z_{col}"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-12)
    for col in ("TE", "turnover", "mdd"):
        df[f"z_{col}"] = -(df[col] - df[col].mean()) / (df[col].std() + 1e-12)
    z_cols = ["z_IR", "z_net_IR", "z_p", "z_TE", "z_turnover", "z_mdd"]
    df["composite_score"] = df[z_cols].mean(axis=1)
    df = df.sort_values("composite_score", ascending=False)
    return criteria, df
