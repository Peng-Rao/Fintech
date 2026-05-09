"""harness.py — portfolio-replica backtesting library.

Extracted from main.ipynb to keep the notebook short. Public names re-exported
from this module via `from harness import *` are listed in __all__.

Project-interface guarantees
----------------------------
1. Inputs: X = 11 futures weekly returns, y = target Monster Index weekly return.
2. Output: weights_history = DataFrame[rebalance_dates × 11].
3. Output: replica_returns = Series[T_eval], plus metrics dict containing
   {IR, TE, rho, GE, VaR, turnover, net_IR, net_TE}.
4. Persistence: export_result_artifacts(...) writes the canonical pickle and
   human-readable CSV/JSON side files for later consolidation.
"""

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    cast,
)

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Lasso

ANNUAL_FACTOR = 52

FUTURES_COLS = [
    "RX1",
    "TY1",
    "GC1",
    "CO1",
    "ES1",
    "VG1",
    "NQ1",
    "LLL1",
    "TP1",
    "DU1",
    "TU2",
]

TARGET_WEIGHTS = {"HFRXGL": 0.50, "MXWD": 0.25, "LEGATRUU": 0.25}

DEFAULT_TC_BPS = (
    5.0  # conservative baseline; project brief mentions 2-4 bps as a realistic range
)

DEFAULT_VAR_CONF = 0.01  # 1% left-tail => 99% VaR

DEFAULT_VAR_HORIZON_W = 4  # about one month

DEFAULT_VAR_CAP = 0.20  # one-month 99% VaR <= 20%

DEFAULT_GE_CAP = 2.0  # gross exposure <= 200%

DEFAULT_RNG_SEED = 42

StandardizationMode = Literal["none", "scale_only", "zscore"]

ScheduleType = Literal["rebalance", "held"]


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for one backtest run.

    Defaults are intentionally conservative and finance-aware:
    - rolling_window=104: two years of weekly history per fit.
    - rebalance_every=4: approximately monthly rebalancing.
    - standardization_mode="scale_only": improves numerical conditioning without
      demeaning returns, avoiding an implicit hidden intercept/cash component.
    - var_cap=0.20 and ge_cap=2.0: assignment/UCITS-style risk limits.

    Execution filters are optional but useful because the assignment asks for
    careful treatment of rebalancing, transaction costs and tricks for reducing
    unnecessary trades.
    """

    rolling_window: int = 104
    rebalance_every: int = 4
    standardization_mode: StandardizationMode = "scale_only"
    var_cap: Optional[float] = DEFAULT_VAR_CAP
    ge_cap: Optional[float] = DEFAULT_GE_CAP
    var_conf: float = DEFAULT_VAR_CONF
    var_horizon_w: int = DEFAULT_VAR_HORIZON_W
    var_history_w: int = 52
    fit_window_start: int = 0
    seed: int = DEFAULT_RNG_SEED
    name: str = "model"

    # Execution realism controls
    min_trade_abs: float = (
        0.0  # zero per-asset trades smaller than this absolute weight
    )
    rebalance_threshold_l1: float = (
        0.0  # skip whole rebalance if total |Δw| is below this
    )

    # Rollover placeholder for futures. Because the dataset contains generic/continuous
    # futures series, exact contract calendars are unavailable. This option charges a
    # transparent conservative roll friction proportional to gross notional exposure.
    rollover_cost_bps: float = 0.0
    rollover_every_weeks: int = (
        13  # quarterly roll proxy; used only when rollover_cost_bps > 0
    )
    rollover_start_offset: int = 0

    # Optional robust training preprocessing. It clips only the training window used
    # for model fitting, never the realised returns used for PnL evaluation.
    fit_clip_quantiles: Optional[Tuple[float, float]] = None

    def __post_init__(self) -> None:
        allowed = {"none", "scale_only", "zscore"}
        if self.standardization_mode not in allowed:
            raise ValueError(
                f"standardization_mode must be one of {allowed}; got {self.standardization_mode!r}"
            )
        if self.rolling_window < 26:
            raise ValueError("rolling_window must be at least 26 weekly observations.")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1.")
        if self.var_cap is not None and self.var_cap <= 0:
            raise ValueError("var_cap must be positive or None.")
        if self.ge_cap is not None and self.ge_cap <= 0:
            raise ValueError("ge_cap must be positive or None.")
        if not 0 < self.var_conf < 0.5:
            raise ValueError("var_conf should be a left-tail probability, e.g. 0.01.")
        if self.var_horizon_w < 1 or self.var_history_w < 2:
            raise ValueError("VaR horizon/history values are too short.")
        if self.min_trade_abs < 0 or self.rebalance_threshold_l1 < 0:
            raise ValueError("Execution thresholds must be non-negative.")
        if self.rollover_cost_bps < 0:
            raise ValueError("rollover_cost_bps must be non-negative.")
        if self.rollover_every_weeks < 1:
            raise ValueError("rollover_every_weeks must be >= 1.")
        if self.fit_clip_quantiles is not None:
            lo, hi = self.fit_clip_quantiles
            if not (0 <= lo < hi <= 1):
                raise ValueError(
                    "fit_clip_quantiles must be (lo, hi) with 0 <= lo < hi <= 1."
                )


def _as_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.astype(float)


def load_bloomberg_weekly(
    path: Union[str, Path], sheet: str = "Copia_statica"
) -> pd.DataFrame:
    """Load the Bloomberg weekly price panel robustly.

    The source Excel file has a header row whose first cell is "Ticker".
    The old version silently accepted the wrong row when "Ticker" was missing;
    this final version fails loudly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    mask = raw.iloc[:, 0].astype(str).str.strip().eq("Ticker")
    if not mask.any():
        raise ValueError("Could not find the 'Ticker' header row in the Excel sheet.")

    header_idx = int(mask[mask].index[0])
    headers = raw.iloc[header_idx].tolist()
    headers[0] = "Date"

    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = headers
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = _as_float_frame(df)
    return clean_price_panel(df)


def clean_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean the price panel without altering valid market observations.

    Cleaning performed:
    - sort by date;
    - remove duplicate dates, keeping the last observation;
    - convert all values to floats;
    - drop columns/rows that are completely empty;
    - reject non-positive prices for required series because pct_change on them
      would be financially meaningless.

    This function deliberately does NOT forward-fill prices. Forward-filling before
    returns would manufacture zero returns and understate volatility/TE.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index, errors="coerce")
    prices = prices[~prices.index.isna()].sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = _as_float_frame(prices)
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")

    required = set(FUTURES_COLS) | set(TARGET_WEIGHTS)
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Missing required columns in price panel: {missing}")

    bad_non_positive = [c for c in required if (prices[c].dropna() <= 0).any()]
    if bad_non_positive:
        raise ValueError(
            f"Non-positive prices found in required columns: {bad_non_positive}"
        )
    return prices


def build_replication_panel(
    prices: pd.DataFrame,
    futures_cols: List[str] = FUTURES_COLS,
    target_weights: Dict[str, float] = TARGET_WEIGHTS,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X and y for the project interface.

    X = weekly arithmetic returns of the 11 futures.
    y = weekly arithmetic return of the synthetic Monster Index:
        50% HFRXGL + 25% MXWD + 25% LEGATRUU by default.

    We use pct_change(fill_method=None) so missing prices are not silently
    forward-filled by pandas.
    """
    prices = clean_price_panel(prices)
    if abs(sum(target_weights.values()) - 1.0) > 1e-9:
        raise ValueError(
            f"target_weights must sum to 1.0; got {sum(target_weights.values())}"
        )

    required = set(futures_cols) | set(target_weights)
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    returns = prices[list(required)].pct_change(fill_method=None)
    X = returns[futures_cols].copy()
    y = sum(returns[col] * weight for col, weight in target_weights.items()).rename(
        "Monster_Index"
    )
    panel = pd.concat([X, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    X_clean = panel[futures_cols]
    y_clean = panel["Monster_Index"]
    return X_clean, y_clean


def data_quality_report(
    prices: pd.DataFrame, X: pd.DataFrame, y: pd.Series
) -> Dict[str, pd.DataFrame]:
    """Return compact EDA/data-quality tables used by the final notebook."""
    required = FUTURES_COLS + list(TARGET_WEIGHTS)
    price_required = prices[required]
    missing = pd.DataFrame(
        {
            "missing_count": price_required.isna().sum(),
            "missing_pct": price_required.isna().mean(),
            "non_positive_count": (price_required <= 0).sum(),
        }
    )
    ret_stats = X.join(y).agg(["mean", "std", "min", "max", "skew", "kurt"]).T
    ret_stats["ann_vol"] = X.join(y).std() * np.sqrt(ANNUAL_FACTOR)
    corr_to_target = X.corrwith(y).rename("corr_to_target").to_frame()
    return {
        "missing_prices": missing,
        "return_stats": ret_stats,
        "corr_to_target": corr_to_target.sort_values("corr_to_target", ascending=False),
    }


def market_stress_outlier_audit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    robust_z_threshold: float = 5.0,
    top_n: int = 25,
    stress_windows: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """Flag extreme weekly returns without deleting valid market observations.

    The audit uses a robust z-score based on the median absolute deviation (MAD).
    Observations are flagged when |robust_z| exceeds ``robust_z_threshold``.
    This is a diagnostic layer, not an automatic deletion rule: finance data often
    contains genuine crisis observations, and removing them would make the
    backtest unrealistically smooth.

    Returns
    -------
    Dict with:
    - asset_summary: counts and severity of flagged observations by series;
    - top_observations: largest absolute robust-z observations with regime labels;
    - robust_z: date × series robust-z panel for plots/review.
    """
    if robust_z_threshold <= 0:
        raise ValueError("robust_z_threshold must be positive.")

    panel = (
        X.join(y.rename("Monster_Index"))
        .replace([np.inf, -np.inf], np.nan)
        .dropna(how="all")
    )
    median = panel.median()
    mad = (panel - median).abs().median().replace(0, np.nan)
    robust_z = 0.6745 * (panel - median) / mad
    extreme_mask = robust_z.abs() >= robust_z_threshold

    def _first_date(series: pd.Series, func: str) -> str:
        clean = series.dropna()
        if clean.empty:
            return ""
        idx = clean.idxmin() if func == "min" else clean.idxmax()
        return pd.Timestamp(str(idx)).date().isoformat()

    summary = pd.DataFrame(index=panel.columns)
    summary["observations"] = panel.count()
    summary["extreme_count"] = extreme_mask.sum().astype(int)
    summary["extreme_pct"] = summary["extreme_count"] / summary["observations"].replace(
        0, np.nan
    )
    summary["max_abs_robust_z"] = robust_z.abs().max()
    summary["worst_weekly_return"] = panel.min()
    summary["worst_week_date"] = [_first_date(panel[c], "min") for c in panel.columns]
    summary["best_weekly_return"] = panel.max()
    summary["best_week_date"] = [_first_date(panel[c], "max") for c in panel.columns]
    summary["annualised_vol"] = panel.std() * np.sqrt(ANNUAL_FACTOR)
    summary["skew"] = panel.skew()
    summary["kurtosis"] = panel.kurt()
    summary = summary.sort_values(
        ["extreme_count", "max_abs_robust_z"], ascending=False
    )

    long = panel.stack().rename("weekly_return").to_frame()
    long.index.names = ["date", "series"]
    rz_long = robust_z.stack().rename("robust_z")
    rz_long.index.names = ["date", "series"]
    flagged = long.join(rz_long).dropna()
    flagged = flagged[flagged["robust_z"].abs() >= robust_z_threshold].copy()
    flagged["abs_robust_z"] = flagged["robust_z"].abs()

    def _regime_label(date_like: Any) -> str:
        if not stress_windows:
            return "ordinary/non-labelled week"
        d = pd.Timestamp(date_like)
        for label, (start, end) in stress_windows.items():
            if pd.Timestamp(start) <= d <= pd.Timestamp(end):
                return label
        return "ordinary/non-labelled week"

    if not flagged.empty:
        flagged = flagged.reset_index()
        flagged["date"] = pd.to_datetime(flagged["date"]).dt.date.astype(str)
        flagged["regime_label"] = flagged["date"].map(_regime_label)
        flagged = flagged.sort_values("abs_robust_z", ascending=False).head(top_n)
    else:
        flagged = pd.DataFrame(
            columns=[
                "date",
                "series",
                "weekly_return",
                "robust_z",
                "abs_robust_z",
                "regime_label",
            ]
        )

    return {
        "asset_summary": summary,
        "top_observations": flagged,
        "robust_z": robust_z,
    }


def hash_inputs(X: pd.DataFrame, y: pd.Series) -> str:
    """Deterministic short hash of X/y values, columns and date span."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(X.values).tobytes())
    h.update(np.ascontiguousarray(y.values).tobytes())
    h.update("|".join(map(str, X.columns)).encode())
    h.update(str(X.index[0]).encode())
    h.update(str(X.index[-1]).encode())
    return h.hexdigest()


class TCModel(Protocol):
    """Transaction-cost model returning cost as fraction of NAV."""

    def __call__(
        self, dw: np.ndarray, w_old: np.ndarray, w_new: np.ndarray
    ) -> float: ...

    @property
    def label(self) -> str: ...


@dataclass(frozen=True)
class FlatBpsTC:
    """Flat bps cost on one-way turnover: cost = bps × Σ|Δw|."""

    bps: float = DEFAULT_TC_BPS

    def __call__(self, dw: np.ndarray, w_old: np.ndarray, w_new: np.ndarray) -> float:
        return float(np.sum(np.abs(dw)) * self.bps / 1e4)

    @property
    def label(self) -> str:
        return f"Flat {self.bps:.1f} bps"


@dataclass(frozen=True)
class HalfSpreadPlusImpactTC:
    """Half-spread plus square-root market-impact proxy.

    This is not a calibrated execution model. It is a transparent stress model
    to test whether a strategy survives more realistic, nonlinear cost pressure.
    """

    half_spread_bps: float = 2.0
    impact_coef_bps: float = 8.0

    def __call__(self, dw: np.ndarray, w_old: np.ndarray, w_new: np.ndarray) -> float:
        a = np.abs(dw)
        spread = self.half_spread_bps * a.sum() / 1e4
        impact = self.impact_coef_bps * np.power(a, 1.5).sum() / 1e4
        return float(spread + impact)

    @property
    def label(self) -> str:
        return f"Half-spread {self.half_spread_bps:.1f}bps + impact {self.impact_coef_bps:.1f}"


def gaussian_var(
    returns: Union[np.ndarray, pd.Series],
    conf: float = DEFAULT_VAR_CONF,
    horizon: int = DEFAULT_VAR_HORIZON_W,
) -> float:
    """Parametric Gaussian VaR as a positive loss fraction of NAV."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(-norm.ppf(conf) * np.std(r, ddof=1) * np.sqrt(horizon))


def historical_var(
    returns: Union[np.ndarray, pd.Series],
    conf: float = DEFAULT_VAR_CONF,
    horizon: int = DEFAULT_VAR_HORIZON_W,
) -> float:
    """Historical quantile VaR as a positive loss fraction of NAV."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(-np.quantile(r, conf) * np.sqrt(horizon))


def expected_shortfall(
    returns: Union[np.ndarray, pd.Series],
    conf: float = DEFAULT_VAR_CONF,
    horizon: int = DEFAULT_VAR_HORIZON_W,
) -> float:
    """Expected shortfall/CVaR as a positive loss fraction of NAV."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    q = np.quantile(r, conf)
    tail = r[r <= q]
    if tail.size == 0:
        return float("nan")
    return float(-tail.mean() * np.sqrt(horizon))


def max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return float("nan")
    wealth = (1 + returns).cumprod()
    return float(1 - (wealth / wealth.cummax()).min())


def tracking_error(active_returns: pd.Series) -> float:
    return (
        float(active_returns.std(ddof=1) * np.sqrt(ANNUAL_FACTOR))
        if len(active_returns) > 1
        else float("nan")
    )


def information_ratio(active_returns: pd.Series) -> float:
    te = tracking_error(active_returns)
    if not np.isfinite(te) or te <= 0:
        return 0.0
    return float(active_returns.mean() * ANNUAL_FACTOR / te)


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or a.std(ddof=1) == 0 or b.std(ddof=1) == 0:
        return float("nan")
    return float(a.corr(b))


def _safe_beta(replica: pd.Series, target: pd.Series) -> float:
    var = target.var(ddof=1)
    if len(replica) < 2 or var <= 0:
        return float("nan")
    return float(replica.cov(target) / var)


def metrics_from_returns(
    replica_gross: pd.Series,
    replica_net: pd.Series,
    target: pd.Series,
    *,
    gross_exposure: Optional[pd.Series] = None,
    var_series: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    tc_total_bps: Optional[float] = None,
    n_rebalances: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the project-interface metrics plus finance-audit extras."""
    idx = replica_net.index.intersection(target.index).intersection(replica_gross.index)
    rg, rn, y = replica_gross.loc[idx], replica_net.loc[idx], target.loc[idx]
    if len(idx) == 0:
        base = {
            k: float("nan")
            for k in ["IR", "TE", "rho", "GE", "VaR", "turnover", "net_IR", "net_TE"]
        }
        return {**base, "n_obs": 0}

    active_gross = rg - y
    active_net = rn - y
    years = len(rn) / ANNUAL_FACTOR
    vol = rn.std(ddof=1) * np.sqrt(ANNUAL_FACTOR) if len(rn) > 1 else float("nan")

    avg_turnover_weekly = (
        float(turnover.mean())
        if turnover is not None and len(turnover)
        else float("nan")
    )
    total_turnover = (
        float(turnover.sum())
        if turnover is not None and len(turnover)
        else float("nan")
    )
    nonzero_turnover = (
        turnover[turnover > 1e-12]
        if turnover is not None and len(turnover)
        else pd.Series(dtype=float)
    )

    return {
        # Core interface keys
        "IR": information_ratio(active_gross),
        "TE": tracking_error(active_gross),
        "rho": _safe_corr(rn, y),
        "GE": float(gross_exposure.mean())
        if gross_exposure is not None and len(gross_exposure)
        else float("nan"),
        "VaR": float(var_series.mean())
        if var_series is not None and len(var_series)
        else float("nan"),
        "turnover": avg_turnover_weekly,
        "net_IR": information_ratio(active_net),
        "net_TE": tracking_error(active_net),
        # Finance-reporting extras
        "ann_ret_gross": float(rg.mean() * ANNUAL_FACTOR),
        "ann_ret_net": float(rn.mean() * ANNUAL_FACTOR),
        "ann_ret_target": float(y.mean() * ANNUAL_FACTOR),
        "ann_active_return_net": float(active_net.mean() * ANNUAL_FACTOR),
        "ann_vol_net": float(vol),
        "sharpe_net": float((rn.mean() * ANNUAL_FACTOR) / vol)
        if np.isfinite(vol) and vol > 0
        else 0.0,
        "max_drawdown_net": max_drawdown(rn),
        "hit_ratio_direction": float((np.sign(rn) == np.sign(y)).mean()),
        "beta_to_target": _safe_beta(rn, y),
        "tc_total_bps": float(tc_total_bps)
        if tc_total_bps is not None
        else float("nan"),
        "tc_bps_per_year": float(tc_total_bps / years)
        if tc_total_bps is not None and years > 0
        else float("nan"),
        "total_turnover": total_turnover,
        "annual_turnover": float(total_turnover / years)
        if years > 0 and np.isfinite(total_turnover)
        else float("nan"),
        "avg_turnover_per_rebalance": float(nonzero_turnover.mean())
        if len(nonzero_turnover)
        else 0.0,
        "max_turnover_rebalance": float(nonzero_turnover.max())
        if len(nonzero_turnover)
        else 0.0,
        "n_rebalances": int(n_rebalances)
        if n_rebalances is not None
        else int(len(nonzero_turnover)),
        "n_obs": int(len(idx)),
    }


@dataclass
class ReplicaResult:
    """Canonical result object saved by the harness and consumed by later comparison steps.

    weights_history is interface-compliant: rows are rebalance dates only.
    held_weights_history is the weekly forward-filled schedule used for PnL.
    """

    name: str
    input_hash: str
    replica_gross: pd.Series
    replica_net: pd.Series
    target: pd.Series
    weights_history: pd.DataFrame
    held_weights_history: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex
    gross_exposure: pd.Series
    var_series: pd.Series
    var_at_rebalance: pd.Series
    scaling: pd.Series
    turnover: pd.Series
    execution_tc_per_period: pd.Series
    rollover_tc_per_period: pd.Series
    tc_per_period: pd.Series
    tc_cumulative: pd.Series
    trade_blotter: pd.DataFrame
    config: Dict[str, Any]
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def replica_returns(self) -> pd.Series:
        """Project-interface alias. We expose gross replica returns explicitly."""
        return self.replica_gross

    @property
    def weights(self) -> pd.DataFrame:
        """Backward-compatible alias for older notebooks."""
        return self.weights_history

    @property
    def metrics(self) -> Dict[str, float]:
        m = metrics_from_returns(
            self.replica_gross,
            self.replica_net,
            self.target,
            gross_exposure=self.gross_exposure,
            var_series=self.var_series,
            turnover=self.turnover,
            tc_total_bps=float(self.tc_cumulative.iloc[-1] * 1e4)
            if len(self.tc_cumulative)
            else None,
            n_rebalances=len(self.rebalance_dates),
        )
        cfg_ge = self.config.get("ge_cap")
        cfg_var = self.config.get("var_cap")
        m.update(
            {
                "max_GE": float(self.gross_exposure.max())
                if len(self.gross_exposure)
                else float("nan"),
                "max_VaR": float(self.var_series.max())
                if len(self.var_series)
                else float("nan"),
                "max_abs_weight": float(self.held_weights_history.abs().max().max())
                if len(self.held_weights_history)
                else float("nan"),
                "GE_breach_count_all_weeks": int(
                    (self.gross_exposure > cfg_ge + 1e-10).sum()
                )
                if cfg_ge is not None
                else 0,
                "VaR_breach_count_all_weeks": int(
                    (self.var_series > cfg_var + 1e-10).sum()
                )
                if cfg_var is not None
                else 0,
                "VaR_breach_count_rebalance": int(
                    (self.var_at_rebalance > cfg_var + 1e-10).sum()
                )
                if cfg_var is not None
                else 0,
            }
        )
        return m

    def to_metrics_row(self) -> pd.Series:
        return pd.Series({"model": self.name, **self.metrics})

    def restrict(
        self,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        new_name: Optional[str] = None,
    ) -> "ReplicaResult":
        """Return a cropped result for crisis-window evaluation."""
        start = cast(pd.Timestamp, pd.Timestamp(start))
        end = cast(pd.Timestamp, pd.Timestamp(end))
        idx = self.replica_net.index[
            (self.replica_net.index >= start) & (self.replica_net.index <= end)
        ]
        rb = self.rebalance_dates[
            (self.rebalance_dates >= start) & (self.rebalance_dates <= end)
        ]
        tb = self.trade_blotter
        if len(tb) and "date" in tb.columns:
            tb = tb[(tb["date"] >= start) & (tb["date"] <= end)].copy()
        return ReplicaResult(
            name=new_name or f"{self.name}_{start.date()}_{end.date()}",
            input_hash=self.input_hash,
            replica_gross=self.replica_gross.loc[idx],
            replica_net=self.replica_net.loc[idx],
            target=self.target.loc[idx],
            weights_history=self.weights_history.loc[
                self.weights_history.index.intersection(rb)
            ],
            held_weights_history=self.held_weights_history.loc[idx],
            rebalance_dates=rb,
            gross_exposure=self.gross_exposure.loc[idx],
            var_series=self.var_series.loc[idx],
            var_at_rebalance=self.var_at_rebalance.loc[
                self.var_at_rebalance.index.intersection(rb)
            ],
            scaling=self.scaling.loc[self.scaling.index.intersection(rb)],
            turnover=self.turnover.loc[idx],
            execution_tc_per_period=self.execution_tc_per_period.loc[idx],
            rollover_tc_per_period=self.rollover_tc_per_period.loc[idx],
            tc_per_period=self.tc_per_period.loc[idx],
            tc_cumulative=self.tc_per_period.loc[idx].cumsum(),
            trade_blotter=tb,
            config=dict(self.config),
            extra={**self.extra, "restricted_to": (str(start.date()), str(end.date()))},
        )


def _validate_X_y(X: pd.DataFrame, y: pd.Series) -> None:
    if not X.index.equals(y.index):
        raise ValueError("X and y indices must be identical.")
    if list(X.columns) != FUTURES_COLS:
        # This is strict because the project interface uses 11 futures in this order.
        raise ValueError(
            f"X columns must be FUTURES_COLS in the agreed order: {FUTURES_COLS}"
        )
    if X.isna().any().any() or y.isna().any():
        raise ValueError("X and y must be NaN-free before entering the harness.")
    if not np.isfinite(X.values).all() or not np.isfinite(y.values).all():
        raise ValueError("X and y must contain only finite values.")
    if len(X) < 30:
        raise ValueError("Too few observations for a rolling backtest.")


def _clip_training_window(
    X_tr: np.ndarray, y_tr: np.ndarray, quantiles: Optional[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    if quantiles is None:
        return X_tr, y_tr
    lo, hi = quantiles
    X_lo, X_hi = np.quantile(X_tr, [lo, hi], axis=0)
    y_lo, y_hi = np.quantile(y_tr, [lo, hi])
    return np.clip(X_tr, X_lo, X_hi), np.clip(y_tr, y_lo, y_hi)


def _standardize_for_fit(
    X_tr: np.ndarray, y_tr: np.ndarray, mode: StandardizationMode
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], float]:
    """Return transformed X/y, coefficient back-transformer and intercept proxy.

    intercept_proxy is the raw-scale intercept implied by z-scoring. It is zero
    for scale_only and none. It is returned for audit purposes because the
    portfolio PnL formula has no free intercept unless a cash column is added.
    """
    if mode == "none":
        return X_tr, y_tr, lambda coef: coef.copy(), 0.0

    x_scale = X_tr.std(axis=0, ddof=0)
    x_scale = np.where(x_scale > 0, x_scale, 1.0)
    y_scale = y_tr.std(ddof=0)
    y_scale = y_scale if y_scale > 0 else 1.0

    if mode == "scale_only":
        X_fit = X_tr / x_scale
        y_fit = y_tr / y_scale
        return X_fit, y_fit, lambda coef: coef * (y_scale / x_scale), 0.0

    # zscore mode: allowed, but not default because it implies a raw intercept.
    x_mean = X_tr.mean(axis=0)
    y_mean = y_tr.mean()
    X_fit = (X_tr - x_mean) / x_scale
    y_fit = (y_tr - y_mean) / y_scale

    def back_transform(coef: np.ndarray) -> np.ndarray:
        return coef * (y_scale / x_scale)

    # We cannot know the coefficient yet; final intercept_proxy is computed after fit.
    return X_fit, y_fit, back_transform, float("nan")


def _extract_coefficients(model: Any, n_features: int) -> np.ndarray:
    coef = getattr(model, "coef_", None)
    if coef is None:
        raise AttributeError(f"Model {type(model).__name__} has no .coef_.")
    coef = np.asarray(coef, dtype=float).ravel()
    if coef.shape[0] != n_features:
        raise ValueError(f"Coefficient shape {coef.shape} != n_features={n_features}.")
    intercept = getattr(model, "intercept_", 0.0)
    intercept_arr = np.asarray(intercept).ravel()
    intercept_float = float(intercept_arr[0]) if intercept_arr.size else 0.0
    return coef


def _apply_ge_cap(
    weights: np.ndarray, ge_cap: Optional[float]
) -> Tuple[np.ndarray, float]:
    if ge_cap is None:
        return weights, 1.0
    ge = float(np.sum(np.abs(weights)))
    if ge > ge_cap and ge > 0:
        scale = ge_cap / ge
        return weights * scale, scale
    return weights, 1.0


def _apply_var_cap(
    weights: np.ndarray,
    X_hist: np.ndarray,
    var_cap: Optional[float],
    conf: float,
    horizon: int,
) -> Tuple[np.ndarray, float, float]:
    projected = X_hist @ weights
    projected_var = gaussian_var(projected, conf, horizon)
    if (
        var_cap is not None
        and np.isfinite(projected_var)
        and projected_var > var_cap
        and projected_var > 0
    ):
        scale = var_cap / projected_var
        return weights * scale, var_cap, scale
    return weights, projected_var, 1.0


def _is_roll_week(k: int, cfg: HarnessConfig) -> bool:
    return cfg.rollover_cost_bps > 0 and (
        (k - cfg.rollover_start_offset) % cfg.rollover_every_weeks == 0
    )


def _rollover_cost(weights: np.ndarray, cfg: HarnessConfig) -> float:
    if cfg.rollover_cost_bps <= 0:
        return 0.0
    return float(np.sum(np.abs(weights)) * cfg.rollover_cost_bps / 1e4)


def _allocate_trade_cost(dw: np.ndarray, total_cost: float) -> np.ndarray:
    a = np.abs(dw)
    total = a.sum()
    if total <= 0 or total_cost == 0:
        return np.zeros_like(dw, dtype=float)
    return total_cost * a / total


def _make_trade_rows(
    date: pd.Timestamp,
    assets: Iterable[str],
    w_old: np.ndarray,
    w_new: np.ndarray,
    execution_cost: float,
) -> List[Dict[str, Any]]:
    dw = w_new - w_old
    allocated = _allocate_trade_cost(dw, execution_cost)
    rows = []
    for asset, old, new, trade, cost in zip(assets, w_old, w_new, dw, allocated):
        rows.append(
            {
                "date": pd.Timestamp(date),
                "asset": asset,
                "old_weight": float(old),
                "new_weight": float(new),
                "trade": float(trade),
                "abs_trade": float(abs(trade)),
                "execution_tc": float(cost),
            }
        )
    return rows


def run_rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    *,
    config: Optional[HarnessConfig] = None,
    tc_model: Optional[TCModel] = None,
    progress: bool = False,
) -> ReplicaResult:
    """Walk-forward fit, rebalance, apply risk caps and account for costs.

    No look-ahead rule: at evaluation week t, the model can only be fitted on
    observations strictly before t. Realised return at t uses weights decided at
    the latest rebalance at or before t, with costs charged only at rebalance/roll
    events.
    """
    cfg = config or HarnessConfig()
    _validate_X_y(X, y)
    if cfg.rolling_window >= len(X):
        raise ValueError(
            f"rolling_window={cfg.rolling_window} must be smaller than len(X)={len(X)}."
        )

    # deterministic seed for any stochastic estimator that accepts random_state in its constructor
    np.random.default_rng(cfg.seed)

    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    n_obs, n_feat = Xv.shape
    out_dates = X.index[cfg.rolling_window :]
    n_eval = len(out_dates)

    rg = np.empty(n_eval)
    rn = np.empty(n_eval)
    held_w = np.zeros((n_eval, n_feat))
    ge = np.empty(n_eval)
    var_series = np.empty(n_eval)
    turnover = np.zeros(n_eval)
    exec_tc = np.zeros(n_eval)
    roll_tc = np.zeros(n_eval)
    total_tc = np.zeros(n_eval)

    rb_dates: List[pd.Timestamp] = []
    rb_weights: List[np.ndarray] = []
    rb_var: List[float] = []
    rb_scaling: List[float] = []
    trade_rows: List[Dict[str, Any]] = []
    implied_intercepts: List[float] = []

    w_current = np.zeros(n_feat)
    t_start = time.perf_counter()

    for k in range(n_eval):
        t = cfg.rolling_window + k
        date = X.index[t]
        is_rebalance = k % cfg.rebalance_every == 0

        if is_rebalance:
            tr_lo = max(cfg.fit_window_start, t - cfg.rolling_window)
            X_tr_raw = Xv[tr_lo:t]
            y_tr_raw = yv[tr_lo:t]
            X_tr_raw, y_tr_raw = _clip_training_window(
                X_tr_raw, y_tr_raw, cfg.fit_clip_quantiles
            )

            X_fit, y_fit, back_transform, _ = _standardize_for_fit(
                X_tr_raw, y_tr_raw, cfg.standardization_mode
            )
            model = model_factory()
            model.fit(X_fit, y_fit)
            coef_fit = _extract_coefficients(model, n_feat)
            w_new = back_transform(coef_fit)

            # Audit implied raw intercept if the user enables z-score mode.
            if cfg.standardization_mode == "zscore":
                x_mean = X_tr_raw.mean(axis=0)
                y_mean = y_tr_raw.mean()
                implied_intercepts.append(float(y_mean - x_mean @ w_new))

            # Cap exposures first, then cap projected VaR using recent history.
            w_new, ge_scale = _apply_ge_cap(w_new, cfg.ge_cap)
            var_lo = max(0, t - cfg.var_history_w)
            w_new, var_projected, var_scale = _apply_var_cap(
                w_new, Xv[var_lo:t], cfg.var_cap, cfg.var_conf, cfg.var_horizon_w
            )
            # GE may change after VaR scaling, but cannot increase; no need to re-apply.

            # Execution-realism filters: skip very small rebalances/trades.
            dw_proposed = w_new - w_current
            l1 = float(np.sum(np.abs(dw_proposed)))
            if cfg.rebalance_threshold_l1 > 0 and l1 < cfg.rebalance_threshold_l1:
                w_new = w_current.copy()
                dw_proposed = np.zeros_like(dw_proposed)
                var_projected = gaussian_var(
                    Xv[var_lo:t] @ w_new, cfg.var_conf, cfg.var_horizon_w
                )
            elif cfg.min_trade_abs > 0:
                small = np.abs(dw_proposed) < cfg.min_trade_abs
                if small.any():
                    dw_proposed = np.where(small, 0.0, dw_proposed)
                    w_new = w_current + dw_proposed
                    var_projected = gaussian_var(
                        Xv[var_lo:t] @ w_new, cfg.var_conf, cfg.var_horizon_w
                    )

            # The no-trade filters above are execution-realism adjustments, but
            # they must never break the mandate. Re-apply hard risk caps after
            # filtering so GE/VaR limits remain authoritative.
            w_new, ge_scale_after_filter = _apply_ge_cap(w_new, cfg.ge_cap)
            w_new, var_projected_after_filter, var_scale_after_filter = _apply_var_cap(
                w_new, Xv[var_lo:t], cfg.var_cap, cfg.var_conf, cfg.var_horizon_w
            )
            ge_scale *= ge_scale_after_filter
            var_scale *= var_scale_after_filter
            var_projected = var_projected_after_filter

            dw = w_new - w_current
            period_turnover = float(np.sum(np.abs(dw)))
            period_exec_tc = tc_model(dw, w_current, w_new) if tc_model else 0.0

            rb_dates.append(cast(pd.Timestamp, pd.Timestamp(date)))
            rb_weights.append(w_new.copy())
            rb_var.append(float(var_projected))
            rb_scaling.append(float(ge_scale * var_scale))
            trade_rows.extend(
                _make_trade_rows(
                    cast(pd.Timestamp, pd.Timestamp(date)), X.columns, w_current, w_new, period_exec_tc
                )
            )

            w_current = w_new
        else:
            period_turnover = 0.0
            period_exec_tc = 0.0

        # Rollover friction, charged transparently when enabled.
        period_roll_tc = (
            _rollover_cost(w_current, cfg) if _is_roll_week(k, cfg) else 0.0
        )

        # Ex-post reporting VaR with the current held weights.
        lo = max(0, t - cfg.var_history_w)
        var_now = gaussian_var(Xv[lo:t] @ w_current, cfg.var_conf, cfg.var_horizon_w)

        r_gross = float(Xv[t] @ w_current)
        r_net = r_gross - period_exec_tc - period_roll_tc

        rg[k] = r_gross
        rn[k] = r_net
        held_w[k] = w_current
        ge[k] = float(np.sum(np.abs(w_current)))
        var_series[k] = var_now
        turnover[k] = period_turnover
        exec_tc[k] = period_exec_tc
        roll_tc[k] = period_roll_tc
        total_tc[k] = period_exec_tc + period_roll_tc

    elapsed = time.perf_counter() - t_start
    assert np.isfinite(rn).all(), "Non-finite returns produced."
    assert len(rb_dates) == len(rb_weights)

    result = ReplicaResult(
        name=cfg.name,
        input_hash=hash_inputs(X, y),
        replica_gross=pd.Series(rg, index=out_dates, name="replica_gross"),
        replica_net=pd.Series(rn, index=out_dates, name="replica_net"),
        target=y.loc[out_dates].rename("target"),
        weights_history=pd.DataFrame(
            rb_weights, index=pd.DatetimeIndex(rb_dates), columns=X.columns
        ),
        held_weights_history=pd.DataFrame(held_w, index=out_dates, columns=X.columns),
        rebalance_dates=pd.DatetimeIndex(rb_dates),
        gross_exposure=pd.Series(ge, index=out_dates, name="gross_exposure"),
        var_series=pd.Series(var_series, index=out_dates, name="VaR_1m_99"),
        var_at_rebalance=pd.Series(
            rb_var, index=pd.DatetimeIndex(rb_dates), name="VaR_at_rebalance"
        ),
        scaling=pd.Series(
            rb_scaling, index=pd.DatetimeIndex(rb_dates), name="risk_scaling"
        ),
        turnover=pd.Series(turnover, index=out_dates, name="turnover"),
        execution_tc_per_period=pd.Series(
            exec_tc, index=out_dates, name="execution_tc"
        ),
        rollover_tc_per_period=pd.Series(roll_tc, index=out_dates, name="rollover_tc"),
        tc_per_period=pd.Series(total_tc, index=out_dates, name="total_tc"),
        tc_cumulative=pd.Series(
            total_tc, index=out_dates, name="tc_cumulative"
        ).cumsum(),
        trade_blotter=pd.DataFrame(trade_rows),
        config=asdict(cfg),
        extra={
            "tc_label": tc_model.label if tc_model else "frictionless",
            "elapsed_s": elapsed,
            "standardization_note": (
                "scale_only avoids hidden intercept"
                if cfg.standardization_mode == "scale_only"
                else "zscore implies an intercept not used in futures PnL"
                if cfg.standardization_mode == "zscore"
                else "no standardization"
            ),
            "implied_intercepts_if_zscore": implied_intercepts,
        },
    )
    return result


def evaluate_weights(
    X: pd.DataFrame,
    y: pd.Series,
    weights_history: pd.DataFrame,
    *,
    schedule_type: ScheduleType = "rebalance",
    tc_model: Optional[TCModel] = None,
    config: Optional[HarnessConfig] = None,
    name: str = "external",
) -> ReplicaResult:
    """Evaluate weights produced by another modelling step using the same machinery.

    schedule_type="rebalance": weights_history contains rows only at rebalance dates.
    schedule_type="held": weights_history contains weekly held weights.

    This prevents the common error of evaluating sparse rebalance weights only on
    rebalance dates. In rebalance mode, the harness forward-fills holdings to all
    weekly dates and charges costs only on the supplied rebalance dates.
    """
    cfg = config or HarnessConfig(name=name)
    _validate_X_y(X, y)
    if not weights_history.columns.equals(X.columns):
        raise ValueError("weights_history columns must match X.columns exactly.")
    if weights_history.isna().any().any():
        raise ValueError(
            "weights_history contains NaNs; clean or forward-fill before evaluation."
        )

    weights_history = weights_history.sort_index()
    if schedule_type == "rebalance":
        rb_dates = pd.DatetimeIndex(weights_history.index.intersection(X.index))
        if len(rb_dates) == 0:
            raise ValueError(
                "No rebalance-date overlap between weights_history and X.index."
            )
        full_idx = X.index[X.index >= rb_dates.min()]
        held = weights_history.reindex(full_idx).ffill().dropna()
        eval_idx = held.index
        rebal_set = set(rb_dates)
    elif schedule_type == "held":
        eval_idx = pd.DatetimeIndex(weights_history.index.intersection(X.index))
        if len(eval_idx) == 0:
            raise ValueError("No overlap between held weights and X.index.")
        held = weights_history.loc[eval_idx]
        changed = held.diff().abs().sum(axis=1).fillna(np.inf) > 1e-12
        rb_dates = pd.DatetimeIndex(held.index[changed])
        rebal_set = set(rb_dates)
    else:
        raise ValueError("schedule_type must be 'rebalance' or 'held'.")

    n_eval, n_feat = held.shape
    X_eval = X.loc[eval_idx]
    rg = np.empty(n_eval)
    rn = np.empty(n_eval)
    ge = np.empty(n_eval)
    var_series = np.empty(n_eval)
    turnover = np.zeros(n_eval)
    exec_tc = np.zeros(n_eval)
    roll_tc = np.zeros(n_eval)
    total_tc = np.zeros(n_eval)
    rb_weights = []
    rb_var = []
    rb_scaling = []
    trade_rows: List[Dict[str, Any]] = []

    Xv_full = X.to_numpy(dtype=float)
    date_to_pos = {d: i for i, d in enumerate(X.index)}
    w_prev = np.zeros(n_feat)

    for k, d in enumerate(eval_idx):
        w_now = held.loc[d].to_numpy(dtype=float)
        if d in rebal_set:
            dw = w_now - w_prev
            turnover[k] = float(np.sum(np.abs(dw)))
            exec_tc[k] = tc_model(dw, w_prev, w_now) if tc_model else 0.0
            trade_rows.extend(
                _make_trade_rows(cast(pd.Timestamp, pd.Timestamp(d)), X.columns, w_prev, w_now, exec_tc[k])
            )
            rb_weights.append(w_now.copy())
            w_prev = w_now

        if _is_roll_week(k, cfg):
            roll_tc[k] = _rollover_cost(w_now, cfg)

        t = date_to_pos[d]
        lo = max(0, t - cfg.var_history_w)
        var_now = gaussian_var(Xv_full[lo:t] @ w_now, cfg.var_conf, cfg.var_horizon_w)
        var_series[k] = var_now
        if d in rebal_set:
            rb_var.append(var_now)
            rb_scaling.append(1.0)

        r_gross = float(X_eval.loc[d].to_numpy(dtype=float) @ w_now)
        rg[k] = r_gross
        total_tc[k] = exec_tc[k] + roll_tc[k]
        rn[k] = r_gross - total_tc[k]
        ge[k] = float(np.sum(np.abs(w_now)))

    return ReplicaResult(
        name=name,
        input_hash=hash_inputs(X, y),
        replica_gross=pd.Series(rg, index=eval_idx, name="replica_gross"),
        replica_net=pd.Series(rn, index=eval_idx, name="replica_net"),
        target=y.loc[eval_idx].rename("target"),
        weights_history=weights_history.loc[rb_dates]
        if schedule_type == "rebalance"
        else held.loc[rb_dates],
        held_weights_history=held,
        rebalance_dates=rb_dates,
        gross_exposure=pd.Series(ge, index=eval_idx, name="gross_exposure"),
        var_series=pd.Series(var_series, index=eval_idx, name="VaR_1m_99"),
        var_at_rebalance=pd.Series(rb_var, index=rb_dates, name="VaR_at_rebalance"),
        scaling=pd.Series(rb_scaling, index=rb_dates, name="risk_scaling"),
        turnover=pd.Series(turnover, index=eval_idx, name="turnover"),
        execution_tc_per_period=pd.Series(exec_tc, index=eval_idx, name="execution_tc"),
        rollover_tc_per_period=pd.Series(roll_tc, index=eval_idx, name="rollover_tc"),
        tc_per_period=pd.Series(total_tc, index=eval_idx, name="total_tc"),
        tc_cumulative=pd.Series(
            total_tc, index=eval_idx, name="tc_cumulative"
        ).cumsum(),
        trade_blotter=pd.DataFrame(trade_rows),
        config={**asdict(cfg), "evaluation_schedule_type": schedule_type},
        extra={
            "tc_label": tc_model.label if tc_model else "frictionless",
            "entry_point": "evaluate_weights",
        },
    )


def make_static_ols_weights(
    X: pd.DataFrame,
    y: pd.Series,
    rebalance_dates: Union[pd.DatetimeIndex, Iterable[pd.Timestamp]],
    *,
    ge_cap: Optional[float] = DEFAULT_GE_CAP,
) -> pd.DataFrame:
    """Diagnostic full-sample OLS weights for a static benchmark.

    This function is intentionally labelled diagnostic because it uses the full
    sample. It is useful as an upper-bound sanity check, not as a tradable
    no-look-ahead strategy.
    """
    _validate_X_y(X, y)
    coef = np.linalg.lstsq(
        X.to_numpy(dtype=float), y.to_numpy(dtype=float), rcond=None
    )[0]
    coef, _ = _apply_ge_cap(coef, ge_cap)
    return pd.DataFrame(
        [coef] * len(pd.DatetimeIndex(rebalance_dates)),
        index=pd.DatetimeIndex(rebalance_dates),
        columns=X.columns,
    )


def make_beta_scaled_single_future_weights(
    X: pd.DataFrame,
    y: pd.Series,
    future: str,
    rebalance_dates: Union[pd.DatetimeIndex, Iterable[pd.Timestamp]],
    *,
    ge_cap: Optional[float] = DEFAULT_GE_CAP,
) -> pd.DataFrame:
    """Static one-future benchmark scaled by covariance beta to the target."""
    _validate_X_y(X, y)
    if future not in X.columns:
        raise ValueError(f"future must be one of {list(X.columns)}; got {future!r}")
    beta = float(y.cov(X[future]) / X[future].var(ddof=1))
    w = np.zeros(X.shape[1], dtype=float)
    w[list(X.columns).index(future)] = beta
    w, _ = _apply_ge_cap(w, ge_cap)
    return pd.DataFrame(
        [w] * len(pd.DatetimeIndex(rebalance_dates)),
        index=pd.DatetimeIndex(rebalance_dates),
        columns=X.columns,
    )


def build_optimizer_weights(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: Optional[HarnessConfig] = None,
    method: Literal["constrained_te", "cost_aware_l1_delta"] = "constrained_te",
    l2_penalty: float = 1e-6,
    turnover_l1_penalty: float = 1e-5,
) -> pd.DataFrame:
    """Build rebalance-date weights for two benchmark optimizers.

    method="constrained_te" solves a rolling ridge/TE proxy and then enforces the
    same GE/VaR scaling rules used by the harness. method="cost_aware_l1_delta"
    solves a rolling delta update with an L1 turnover penalty, which directly
    discourages unnecessary changes in weights before the transaction-cost audit.

    The L1 turnover update is implemented through an augmented Lasso problem:
    minimise tracking error + l2_penalty*||w||² + turnover_l1_penalty*||w-w_old||₁.
    GE and VaR are then enforced as hard post-solve mandate checks.
    """
    _validate_X_y(X, y)
    if l2_penalty < 0 or turnover_l1_penalty < 0:
        raise ValueError("penalties must be non-negative")
    cfg = config or HarnessConfig()
    n_feat = X.shape[1]
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    w_current = np.zeros(n_feat, dtype=float)
    dates: List[pd.Timestamp] = []
    weights: List[np.ndarray] = []

    for t in range(cfg.rolling_window, len(X)):
        k = t - cfg.rolling_window
        if k % cfg.rebalance_every != 0:
            continue
        X_tr = Xv[t - cfg.rolling_window : t]
        y_tr = yv[t - cfg.rolling_window : t]

        if method == "constrained_te":
            A = X_tr.T @ X_tr + float(l2_penalty) * np.eye(n_feat)
            b = X_tr.T @ y_tr
            try:
                w_new = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w_new = np.linalg.pinv(A) @ b
        elif method == "cost_aware_l1_delta":
            y_adj = y_tr - X_tr @ w_current
            scale = np.sqrt(len(y_tr) * float(l2_penalty))
            X_aug = np.vstack([X_tr, scale * np.eye(n_feat)])
            y_aug = np.concatenate([y_adj, -scale * w_current])
            model = Lasso(
                alpha=float(turnover_l1_penalty), fit_intercept=False, max_iter=20000
            )
            model.fit(X_aug, y_aug)
            w_new = w_current + model.coef_
        else:
            raise ValueError("method must be 'constrained_te' or 'cost_aware_l1_delta'")

        w_new, _ = _apply_ge_cap(w_new, cfg.ge_cap)
        w_new, _, _ = _apply_var_cap(
            w_new,
            Xv[max(0, t - cfg.var_history_w) : t],
            cfg.var_cap,
            cfg.var_conf,
            cfg.var_horizon_w,
        )
        dates.append(cast(pd.Timestamp, pd.Timestamp(X.index[t])))
        weights.append(w_new.copy())
        w_current = w_new.copy()

    return pd.DataFrame(weights, index=pd.DatetimeIndex(dates), columns=X.columns)


def split_diagnostics(
    results: Dict[str, ReplicaResult],
    windows: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """Chronological validation/test diagnostics for candidate controls."""
    rows: List[Dict[str, Any]] = []
    for label, result in results.items():
        for window, (start, end) in windows.items():
            sub = result.restrict(start, end, new_name=f"{result.name}__{window}")
            m = sub.metrics
            rows.append(
                {
                    "model": label,
                    "window": window,
                    "start": start,
                    "end": end,
                    "n_obs": m.get("n_obs"),
                    "rho": m.get("rho"),
                    "beta_to_target": m.get("beta_to_target"),
                    "net_TE": m.get("net_TE"),
                    "net_IR": m.get("net_IR"),
                    "ann_ret_net": m.get("ann_ret_net"),
                    "max_drawdown_net": m.get("max_drawdown_net"),
                    "annual_turnover": m.get("annual_turnover"),
                    "tc_total_bps": m.get("tc_total_bps"),
                    "GE": m.get("GE"),
                    "max_GE": m.get("max_GE"),
                    "max_VaR": m.get("max_VaR"),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "window"])


def validate_project_interface(
    result: ReplicaResult, X: pd.DataFrame, y: pd.Series, *, raise_on_error: bool = True
) -> bool:
    """Validate that a result is safe for later comparison."""
    errors: List[str] = []
    required_metrics = {"IR", "TE", "rho", "GE", "VaR", "turnover", "net_IR", "net_TE"}

    if result.input_hash != hash_inputs(X, y):
        errors.append("input_hash does not match current X/y")
    if not result.replica_returns.index.equals(result.target.index):
        errors.append("replica_returns and target indices are not aligned")
    if not result.replica_net.index.equals(result.target.index):
        errors.append("replica_net and target indices are not aligned")
    if not result.held_weights_history.index.equals(result.replica_returns.index):
        errors.append("held_weights_history must have one row per evaluated week")
    if not result.weights_history.columns.equals(X.columns):
        errors.append("weights_history columns do not match X.columns")
    if not result.held_weights_history.columns.equals(X.columns):
        errors.append("held_weights_history columns do not match X.columns")
    if not result.weights_history.index.equals(result.rebalance_dates):
        errors.append("weights_history rows must be exactly the rebalance_dates")
    if len(result.weights_history) > len(result.held_weights_history):
        errors.append("rebalance-date weights cannot exceed weekly held weights")
    if not required_metrics.issubset(result.metrics):
        errors.append(
            f"missing required metrics: {sorted(required_metrics - set(result.metrics))}"
        )
    arrays_to_check = {
        "replica_returns": result.replica_returns.values,
        "replica_net": result.replica_net.values,
        "weights_history": result.weights_history.values,
        "held_weights_history": result.held_weights_history.values,
    }
    for name, arr in arrays_to_check.items():
        if not np.isfinite(arr).all():
            errors.append(f"{name} contains non-finite values")

    if errors and raise_on_error:
        raise AssertionError(
            "Project interface validation failed:\n- " + "\n- ".join(errors)
        )
    return not errors


def risk_audit_table(result: ReplicaResult) -> pd.DataFrame:
    cfg = result.config
    rows = {
        "Average GE": result.gross_exposure.mean(),
        "Maximum GE": result.gross_exposure.max(),
        "GE cap": cfg.get("ge_cap", np.nan),
        "GE breaches - all weeks": result.metrics.get(
            "GE_breach_count_all_weeks", np.nan
        ),
        "Average VaR": result.var_series.mean(),
        "Maximum VaR": result.var_series.max(),
        "VaR cap": cfg.get("var_cap", np.nan),
        "VaR breaches - all weeks": result.metrics.get(
            "VaR_breach_count_all_weeks", np.nan
        ),
        "VaR breaches - rebalance dates": result.metrics.get(
            "VaR_breach_count_rebalance", np.nan
        ),
        "Maximum absolute single future weight": result.metrics.get(
            "max_abs_weight", np.nan
        ),
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=["value"])


def asset_cost_attribution(result: ReplicaResult) -> pd.DataFrame:
    """Attribute turnover, execution cost and average exposure by futures contract."""
    assets = result.held_weights_history.columns
    avg_abs_weight = result.held_weights_history.abs().mean()
    max_abs_weight = result.held_weights_history.abs().max()

    if result.trade_blotter.empty:
        total_turnover = pd.Series(0.0, index=assets)
        exec_cost = pd.Series(0.0, index=assets)
    else:
        total_turnover = (
            result.trade_blotter.groupby("asset")["abs_trade"]
            .sum()
            .reindex(assets)
            .fillna(0.0)
        )
        exec_cost = (
            result.trade_blotter.groupby("asset")["execution_tc"]
            .sum()
            .reindex(assets)
            .fillna(0.0)
        )

    df = pd.DataFrame(
        {
            "total_turnover": total_turnover,
            "execution_cost_bps": exec_cost * 1e4,
            "avg_abs_weight": avg_abs_weight,
            "max_abs_weight": max_abs_weight,
        }
    )
    total_cost = df["execution_cost_bps"].sum()
    df["cost_share"] = df["execution_cost_bps"] / total_cost if total_cost > 0 else 0.0
    return df.sort_values("execution_cost_bps", ascending=False)


def cumulative_returns_table(result: ReplicaResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_cumulative": (1 + result.target).cumprod(),
            "replica_gross_cumulative": (1 + result.replica_gross).cumprod(),
            "replica_net_cumulative": (1 + result.replica_net).cumprod(),
            "tracking_gap_net": (1 + result.replica_net).cumprod()
            - (1 + result.target).cumprod(),
        }
    )


def rolling_diagnostics(result: ReplicaResult, window: int = 52) -> pd.DataFrame:
    active = result.replica_net - result.target
    return pd.DataFrame(
        {
            "rolling_TE_net": active.rolling(window).std() * np.sqrt(ANNUAL_FACTOR),
            "rolling_corr_net": result.replica_net.rolling(window).corr(result.target),
            "rolling_beta_net": result.replica_net.rolling(window).cov(result.target)
            / result.target.rolling(window).var(),
        }
    )


def exposure_dashboard(result: ReplicaResult) -> pd.DataFrame:
    """Weekly exposure diagnostics for futures notional weights.

    Positive weights are long notional exposure; negative weights are short notional
    exposure. Gross exposure is the sum of absolute weights and net exposure is the
    signed sum of weights.
    """
    held = result.held_weights_history
    return pd.DataFrame(
        {
            "long_exposure": held.clip(lower=0).sum(axis=1),
            "short_exposure": held.clip(upper=0).sum(axis=1),
            "net_exposure": held.sum(axis=1),
            "gross_exposure": held.abs().sum(axis=1),
            "max_abs_weight": held.abs().max(axis=1),
        },
        index=held.index,
    )


def rolling_risk_dashboard(result: ReplicaResult, window: int = 52) -> pd.DataFrame:
    """Rolling portfolio-risk diagnostics used for the notebook dashboard."""
    r = result.replica_net
    return pd.DataFrame(
        {
            "rolling_gaussian_VaR_1m_99": r.rolling(window).apply(
                lambda x: gaussian_var(x, DEFAULT_VAR_CONF, DEFAULT_VAR_HORIZON_W),
                raw=False,
            ),
            "rolling_historical_VaR_1m_99": r.rolling(window).apply(
                lambda x: historical_var(x, DEFAULT_VAR_CONF, DEFAULT_VAR_HORIZON_W),
                raw=False,
            ),
            "rolling_expected_shortfall_1m_99": r.rolling(window).apply(
                lambda x: expected_shortfall(
                    x, DEFAULT_VAR_CONF, DEFAULT_VAR_HORIZON_W
                ),
                raw=False,
            ),
            "rolling_worst_weekly_loss": r.rolling(window).min().mul(-1),
            "rolling_vol_ann": r.rolling(window).std().mul(np.sqrt(ANNUAL_FACTOR)),
        },
        index=r.index,
    )


def stress_window_diagnostics(
    result: ReplicaResult, windows: Dict[str, Tuple[str, str]], *, min_obs: int = 8
) -> pd.DataFrame:
    """Evaluate one result inside named historical windows.

    Windows with fewer than min_obs evaluated weeks are kept but marked, because
    that is more transparent than silently dropping stress periods.
    """
    rows: List[Dict[str, Any]] = []
    for label, (start, end) in windows.items():
        sub = result.restrict(start, end, new_name=f"{result.name}__{label}")
        active = sub.replica_net - sub.target
        rows.append(
            {
                "window": label,
                "start": str(pd.Timestamp(start).date()),
                "end": str(pd.Timestamp(end).date()),
                "n_obs": int(len(sub.replica_net)),
                "enough_data": bool(len(sub.replica_net) >= min_obs),
                "rho": _safe_corr(sub.replica_net, sub.target),
                "net_TE": tracking_error(active),
                "net_IR": information_ratio(active),
                "max_abs_weekly_tracking_gap": float(active.abs().max())
                if len(active)
                else float("nan"),
                "net_max_drawdown": max_drawdown(sub.replica_net),
                "target_max_drawdown": max_drawdown(sub.target),
                "average_GE": float(sub.gross_exposure.mean())
                if len(sub.gross_exposure)
                else float("nan"),
                "max_GE": float(sub.gross_exposure.max())
                if len(sub.gross_exposure)
                else float("nan"),
                "average_VaR": float(sub.var_series.mean())
                if len(sub.var_series)
                else float("nan"),
                "max_VaR": float(sub.var_series.max())
                if len(sub.var_series)
                else float("nan"),
                "turnover": float(sub.turnover.sum())
                if len(sub.turnover)
                else float("nan"),
                "tc_bps": float(sub.tc_per_period.sum() * 1e4)
                if len(sub.tc_per_period)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("window")


def tracking_error_failure_table(
    result: ReplicaResult, *, top_n: int = 10
) -> pd.DataFrame:
    """Worst absolute weekly tracking-gap observations with portfolio context."""
    active = result.replica_net - result.target
    rows: List[Dict[str, Any]] = []
    for d in active.abs().sort_values(ascending=False).head(top_n).index:
        w = result.held_weights_history.loc[d]
        top = w.abs().sort_values(ascending=False).head(3)
        signed = [f"{asset}={w[asset]:+.3f}" for asset in top.index]
        rows.append(
            {
                "date": pd.Timestamp(d).date().isoformat(),
                "target_return": float(result.target.loc[d]),
                "replica_net_return": float(result.replica_net.loc[d]),
                "active_return": float(active.loc[d]),
                "abs_active_return": float(abs(active.loc[d])),
                "gross_exposure": float(result.gross_exposure.loc[d]),
                "VaR_1m_99": float(result.var_series.loc[d]),
                "turnover": float(result.turnover.loc[d]),
                "transaction_cost": float(result.tc_per_period.loc[d]),
                "largest_exposures": ", ".join(signed),
            }
        )
    return pd.DataFrame(rows)


def benchmark_result_table(results: Dict[str, ReplicaResult]) -> pd.DataFrame:
    """Compact comparison table for pipeline-validation benchmark models."""
    rows = []
    for label, result in results.items():
        m = result.metrics
        rows.append(
            {
                "benchmark": label,
                "rho": m.get("rho"),
                "TE": m.get("TE"),
                "net_TE": m.get("net_TE"),
                "IR": m.get("IR"),
                "net_IR": m.get("net_IR"),
                "ann_ret_net": m.get("ann_ret_net"),
                "beta_to_target": m.get("beta_to_target"),
                "max_drawdown_net": m.get("max_drawdown_net"),
                "GE": m.get("GE"),
                "max_GE": m.get("max_GE"),
                "VaR": m.get("VaR"),
                "max_VaR": m.get("max_VaR"),
                "annual_turnover": m.get("annual_turnover"),
                "tc_total_bps": m.get("tc_total_bps"),
                "n_rebalances": m.get("n_rebalances"),
                "n_obs": m.get("n_obs"),
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("benchmark")
        .sort_values(["net_TE", "tc_total_bps"])
    )


def assumption_register() -> pd.DataFrame:
    """Visible model/data/cost assumptions for audit and presentation."""
    rows = [
        {
            "area": "Data",
            "assumption": "Weekly returns are computed from the supplied Bloomberg-style price panel.",
            "implementation": "pct_change on cleaned positive prices; rows with missing required returns are removed.",
            "risk_if_wrong": "Bad data would directly affect weights, TE, VaR and transaction-cost estimates.",
            "control": "Strict required-column checks, non-positive price rejection and input hash exported with results.",
        },
        {
            "area": "Data",
            "assumption": "Prices are not forward-filled before return calculation.",
            "implementation": "clean_price_panel deliberately avoids forward fill.",
            "risk_if_wrong": "Forward fill can create artificial zero returns and understate volatility and VaR.",
            "control": "Documented in code and notebook; NaN rows are removed only after return construction.",
        },
        {
            "area": "Data",
            "assumption": "Extreme returns are flagged for review, not automatically deleted.",
            "implementation": "market_stress_outlier_audit uses robust z-scores and labels crisis/stress windows.",
            "risk_if_wrong": "Deleting genuine market stress would overstate replication quality and understate risk.",
            "control": "Only invalid prices/missing required returns are removed; valid extreme market observations remain in the backtest.",
        },
        {
            "area": "Model",
            "assumption": "Ridge is a transparent control model, not the final portfolio claim.",
            "implementation": "Rolling two-year fit, monthly rebalance, fit_intercept=False.",
            "risk_if_wrong": "Interpreting it as the final best model would overstate the modelling conclusion.",
            "control": "Benchmark and validation tables separate pipeline validation from final strategy selection.",
        },
        {
            "area": "Model",
            "assumption": "Cost-aware optimisation is included as a benchmark control, not as an over-claimed final trading system.",
            "implementation": "build_optimizer_weights supports constrained TE and L1 turnover-penalised weight updates.",
            "risk_if_wrong": "Ignoring turnover during weight construction can make a model look good before costs but weak after costs.",
            "control": "Leaderboard compares ordinary rolling estimators with constrained and cost-aware optimizer controls.",
        },
        {
            "area": "Backtest",
            "assumption": "No look-ahead is allowed.",
            "implementation": "At week t, the model is fitted only on observations before t.",
            "risk_if_wrong": "Look-ahead would overstate performance and reduce apparent tracking error.",
            "control": "run_rolling_backtest uses explicit rolling train windows and rebalance dates.",
        },
        {
            "area": "Exposure",
            "assumption": "Futures weights are notional exposures, not cash allocations.",
            "implementation": "Gross exposure equals sum(abs(weights)); GE cap is enforced before PnL.",
            "risk_if_wrong": "Risk could be understated if weights were treated like long-only cash shares.",
            "control": "GE time series, breach counts, long/short/net exposure dashboard.",
        },
        {
            "area": "Risk",
            "assumption": "One-month 99% Gaussian VaR is the hard portfolio-risk control.",
            "implementation": "VaR is projected on the recent training history and weights are scaled if needed.",
            "risk_if_wrong": "Gaussian VaR can miss tail risk and regime shifts.",
            "control": "Historical VaR, Expected Shortfall and stress-window diagnostics are also reported.",
        },
        {
            "area": "Transaction costs",
            "assumption": "5 bps per unit one-way turnover is the baseline cost stress.",
            "implementation": "FlatBpsTC charges cost = bps × sum(abs(delta weights)).",
            "risk_if_wrong": "Too-low costs would inflate net IR; too-high costs would penalise high-turnover models.",
            "control": "Cost sensitivity from 0 to 20 bps plus half-spread/impact model.",
        },
        {
            "area": "Transaction costs",
            "assumption": "ADV-based costs are illustrative unless real futures ADV data is supplied.",
            "implementation": "TieredADVTC is available as a stress-test framework, not used as calibrated baseline.",
            "risk_if_wrong": "A calibrated liquidity conclusion cannot be claimed from placeholder ADV values.",
            "control": "Notebook labels the ADV model as illustrative only.",
        },
        {
            "area": "Handoff",
            "assumption": "All candidate strategies can be evaluated through the same contract.",
            "implementation": "Either run_rolling_backtest(model_factory) or evaluate_weights(weights_history).",
            "risk_if_wrong": "Strategies would not be comparable if dates, costs, risk limits or metrics differ.",
            "control": "validate_project_interface, exported CSV/JSON/pickle artifacts and contract tests.",
        },
    ]
    return pd.DataFrame(rows)


def cost_sensitivity_sweep(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: Callable[[], Any],
    tc_models: Dict[str, Optional[TCModel]],
    *,
    config: Optional[HarnessConfig] = None,
) -> pd.DataFrame:
    """Run the same model under several TC scenarios."""
    rows = []
    base_cfg = config or HarnessConfig()
    for label, tc in tc_models.items():
        cfg = replace(base_cfg, name=f"{base_cfg.name}__{label}")
        res = run_rolling_backtest(X, y, model_factory, config=cfg, tc_model=tc)
        rows.append({"scenario": label, **res.metrics})
    df = pd.DataFrame(rows).set_index("scenario")
    ordered = [
        "IR",
        "TE",
        "net_IR",
        "net_TE",
        "rho",
        "GE",
        "VaR",
        "turnover",
        "annual_turnover",
        "tc_total_bps",
        "tc_bps_per_year",
        "ann_ret_net",
        "max_drawdown_net",
        "n_rebalances",
        "n_obs",
    ]
    return df[[c for c in ordered if c in df.columns]]


def stationary_block_bootstrap_ci(
    replica: pd.Series,
    target: pd.Series,
    *,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = DEFAULT_RNG_SEED,
) -> Dict[str, float]:
    """Stationary block bootstrap CI for net TE and net IR."""
    rng = np.random.default_rng(seed)
    diffs = (replica - target).dropna().to_numpy(dtype=float)
    n = len(diffs)
    if n < 5:
        return {
            k: float("nan")
            for k in ["te_mean", "te_lo", "te_hi", "ir_mean", "ir_lo", "ir_hi"]
        }
    block = max(2, int(np.sqrt(n)))
    p_geo = 1.0 / block
    te_samples = np.empty(n_boot)
    ir_samples = np.empty(n_boot)
    for b in range(n_boot):
        idx_buf: List[int] = []
        while len(idx_buf) < n:
            start = int(rng.integers(0, n))
            length = int(rng.geometric(p_geo))
            idx_buf.extend(((start + np.arange(length)) % n).tolist())
        d = diffs[np.asarray(idx_buf[:n], dtype=int)]
        te = d.std(ddof=1) * np.sqrt(ANNUAL_FACTOR)
        te_samples[b] = te
        ir_samples[b] = (d.mean() * ANNUAL_FACTOR / te) if te > 0 else 0.0
    alpha = (1 - ci_level) / 2
    return {
        "te_mean": float(te_samples.mean()),
        "te_lo": float(np.quantile(te_samples, alpha)),
        "te_hi": float(np.quantile(te_samples, 1 - alpha)),
        "ir_mean": float(ir_samples.mean()),
        "ir_lo": float(np.quantile(ir_samples, alpha)),
        "ir_hi": float(np.quantile(ir_samples, 1 - alpha)),
    }


def save_result(result: ReplicaResult, path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "input_hash": result.input_hash,
        "name": result.name,
        "result": result,
    }
    with path.open("wb") as fh:
        pickle.dump(envelope, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_result(
    path: Union[str, Path], *, expected_input_hash: Optional[str] = None
) -> ReplicaResult:
    path = Path(path)
    with path.open("rb") as fh:
        envelope = pickle.load(fh)
    if (
        expected_input_hash is not None
        and envelope.get("input_hash") != expected_input_hash
    ):
        raise ValueError("input hash mismatch; result was built on different X/y")
    return envelope["result"]


def export_result_artifacts(
    result: ReplicaResult, results_dir: Union[str, Path], stem: str = "harness"
) -> Dict[str, Path]:
    """Export the canonical result plus human-readable audit files."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    paths["pickle"] = save_result(result, results_dir / f"{stem}.pkl")
    paths["weights_rebalance"] = results_dir / f"{stem}_weights_rebalance.csv"
    paths["holdings_weekly"] = results_dir / f"{stem}_holdings_weekly.csv"
    paths["returns"] = results_dir / f"{stem}_returns.csv"
    paths["metrics"] = results_dir / f"{stem}_metrics.json"
    paths["trade_blotter"] = results_dir / f"{stem}_trade_blotter.csv"
    paths["asset_cost_attribution"] = results_dir / f"{stem}_asset_cost_attribution.csv"
    paths["risk_audit"] = results_dir / f"{stem}_risk_audit.csv"
    paths["rolling_diagnostics"] = results_dir / f"{stem}_rolling_diagnostics.csv"
    paths["cumulative_returns"] = results_dir / f"{stem}_cumulative_returns.csv"

    result.weights_history.to_csv(paths["weights_rebalance"])
    result.held_weights_history.to_csv(paths["holdings_weekly"])
    pd.DataFrame(
        {
            "replica_returns": result.replica_returns,
            "replica_gross": result.replica_gross,
            "replica_net": result.replica_net,
            "target": result.target,
            "turnover": result.turnover,
            "execution_tc": result.execution_tc_per_period,
            "rollover_tc": result.rollover_tc_per_period,
            "total_tc": result.tc_per_period,
            "tc_cumulative": result.tc_cumulative,
            "gross_exposure": result.gross_exposure,
            "VaR_1m_99": result.var_series,
        }
    ).to_csv(paths["returns"])
    result.trade_blotter.to_csv(paths["trade_blotter"], index=False)
    asset_cost_attribution(result).to_csv(paths["asset_cost_attribution"])
    risk_audit_table(result).to_csv(paths["risk_audit"])
    rolling_diagnostics(result).to_csv(paths["rolling_diagnostics"])
    cumulative_returns_table(result).to_csv(paths["cumulative_returns"])

    payload = {
        "input_hash": result.input_hash,
        "name": result.name,
        "metrics": result.metrics,
        "config": result.config,
        "extra": result.extra,
        "interface_note": "weights_history rows are rebalance dates; held_weights_history rows are weekly evaluation dates.",
    }
    with paths["metrics"].open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return paths


__all__ = [
    "ANNUAL_FACTOR",
    "DEFAULT_GE_CAP",
    "DEFAULT_RNG_SEED",
    "DEFAULT_TC_BPS",
    "DEFAULT_VAR_CAP",
    "DEFAULT_VAR_CONF",
    "DEFAULT_VAR_HORIZON_W",
    "FUTURES_COLS",
    "FlatBpsTC",
    "HalfSpreadPlusImpactTC",
    "HarnessConfig",
    "ReplicaResult",
    "ScheduleType",
    "StandardizationMode",
    "TARGET_WEIGHTS",
    "TCModel",
    "asset_cost_attribution",
    "assumption_register",
    "benchmark_result_table",
    "build_optimizer_weights",
    "build_replication_panel",
    "clean_price_panel",
    "cost_sensitivity_sweep",
    "cumulative_returns_table",
    "data_quality_report",
    "evaluate_weights",
    "expected_shortfall",
    "export_result_artifacts",
    "exposure_dashboard",
    "gaussian_var",
    "hash_inputs",
    "historical_var",
    "information_ratio",
    "load_bloomberg_weekly",
    "load_result",
    "make_beta_scaled_single_future_weights",
    "make_static_ols_weights",
    "market_stress_outlier_audit",
    "max_drawdown",
    "metrics_from_returns",
    "risk_audit_table",
    "rolling_diagnostics",
    "rolling_risk_dashboard",
    "run_rolling_backtest",
    "save_result",
    "split_diagnostics",
    "stationary_block_bootstrap_ci",
    "stress_window_diagnostics",
    "tracking_error",
    "tracking_error_failure_table",
    "validate_project_interface",
]
