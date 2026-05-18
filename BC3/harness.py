"""harness.py — portfolio-replica backtesting library.

Extracted from main.ipynb to keep the notebook short. The notebook brings every
top-level public name into scope via `from harness import *`.

Project-interface guarantees
----------------------------
1. Inputs: X = 11 futures weekly returns, y = target Monster Index weekly return.
2. Output: weights_history = DataFrame[rebalance_dates × 11].
3. Output: replica_returns = Series[T_eval], plus metrics dict containing
   {IR, TE, rho, GE, VaR, turnover, net_IR, net_TE}.
"""

import hashlib
import time
from dataclasses import asdict, dataclass, field
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



def historical_var_compound(
    returns: Union[np.ndarray, pd.Series],
    conf: float = DEFAULT_VAR_CONF,
    horizon: int = DEFAULT_VAR_HORIZON_W,
) -> float:
    """Historical VaR from the empirical distribution of compounded ``horizon``-step returns.

    Unlike ``historical_var`` (which takes the 1-step quantile and scales by ``sqrt(horizon)``),
    this helper builds the rolling compound-return series and quantiles that directly. Use this
    when the multi-period return distribution is fat-tailed and the parametric square-root-of-time
    scaling would understate the tail risk. Returns NaN if fewer than ``horizon`` finite
    observations are available.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < horizon:
        return float("nan")
    series = pd.Series(r)
    horizon_returns = (1 + series).rolling(window=horizon).apply(np.prod, raw=True) - 1
    horizon_returns = horizon_returns.dropna()
    if len(horizon_returns) == 0:
        return float("nan")
    return float(-np.quantile(horizon_returns, conf))



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


def apply_ge_cap(
    weights: np.ndarray, ge_cap: Optional[float]
) -> Tuple[np.ndarray, float]:
    """Public alias for the gross-exposure-cap projection.

    Same contract as ``_apply_ge_cap`` (returns ``(scaled_weights, scaling)``); exposed so
    downstream modules don't need to reach into private names.
    """
    return _apply_ge_cap(weights, ge_cap)


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


def apply_var_cap_iterative(
    weights: np.ndarray,
    X_hist: np.ndarray,
    *,
    var_confidence: float = DEFAULT_VAR_CONF,
    var_horizon: int = DEFAULT_VAR_HORIZON_W,
    max_var: float = DEFAULT_VAR_CAP,
    step: float = 0.01,
    min_scaling: float = 0.0,
) -> Tuple[np.ndarray, float, float, pd.DataFrame]:
    """Iteratively shrink ``weights`` until historical VaR over ``X_hist`` is at most ``max_var``.

    Mirrors the projection layer the rebalancing & portfolio-constraints track uses. The
    returned tuple matches the four-tuple ``portfolio_constraints.apply_var_cap`` exposes today
    so the notebook keeps working without signature changes.

    Returns
    -------
    scaled_weights, scaling, final_var, history_df
    """
    weights = np.asarray(weights, dtype=float)
    history: list[dict] = []
    scaling = 1.0
    while scaling >= min_scaling:
        weights_scaled = weights * scaling
        portfolio_returns = X_hist @ weights_scaled
        var_value = historical_var_compound(
            portfolio_returns, conf=var_confidence, horizon=var_horizon
        )
        history.append({"scaling": scaling, "VaR": var_value})
        if not np.isnan(var_value) and var_value <= max_var:
            return weights_scaled, scaling, var_value, pd.DataFrame(history)
        scaling -= step
    history_df = pd.DataFrame(history)
    final_var = (
        float(history_df["VaR"].iloc[-1]) if not history_df.empty else float("nan")
    )
    return weights * min_scaling, min_scaling, final_var, history_df


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
                    cast(pd.Timestamp, pd.Timestamp(date)),
                    X.columns,
                    w_current,
                    w_new,
                    period_exec_tc,
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
                _make_trade_rows(
                    cast(pd.Timestamp, pd.Timestamp(d)),
                    X.columns,
                    w_prev,
                    w_now,
                    exec_tc[k],
                )
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



















PERCENT_METRIC_COLS: Tuple[str, ...] = (
    "TE",
    "net_TE",
    "VaR",
    "max_VaR",
    "max_drawdown",
    "max_drawdown_net",
    "cost_drag",
    "ann_ret",
    "ann_ret_net",
)
RATIO_METRIC_COLS: Tuple[str, ...] = (
    "IR",
    "net_IR",
    "rho",
    "beta_to_target",
)
MULTIPLE_METRIC_COLS: Tuple[str, ...] = (
    "GE",
    "GE_mean",
    "max_GE",
    "annual_turnover",
    "weekly_turnover",
    "turnover",
)
BPS_METRIC_COLS: Tuple[str, ...] = ("tc_total_bps",)


def metrics_row_from_replica(
    label: str,
    res: "ReplicaResult",
    **extra: Any,
) -> Dict[str, Any]:
    """Compact metrics dict for leaderboard tables; shared across all model tracks.

    Standard columns: model | IR | TE | rho | GE_mean | max_GE | VaR | max_VaR |
    annual_turnover | tc_total_bps | net_IR | net_TE. Extra keyword args are appended
    as-is so callers can tag rows with hyper-parameters (sigma_w, alpha, cadence, ...).
    """
    m = res.metrics
    return {
        "model": label,
        "IR": m["IR"],
        "TE": m["TE"],
        "rho": m["rho"],
        "GE_mean": m["GE"],
        "max_GE": m["max_GE"],
        "VaR": m["VaR"],
        "max_VaR": m["max_VaR"],
        "annual_turnover": m["annual_turnover"],
        "tc_total_bps": m["tc_total_bps"],
        "net_IR": m["net_IR"],
        "net_TE": m["net_TE"],
        **extra,
    }


def format_metrics_dataframe(
    df: pd.DataFrame,
    *,
    percent_cols: Iterable[str] = PERCENT_METRIC_COLS,
    ratio_cols: Iterable[str] = RATIO_METRIC_COLS,
    multiple_cols: Iterable[str] = MULTIPLE_METRIC_COLS,
    bps_cols: Iterable[str] = BPS_METRIC_COLS,
):
    """Apply a consistent display formatting to a metrics DataFrame and return a Styler.

    - Columns in `percent_cols` are rendered as `12.34%` (units of NAV).
    - Columns in `ratio_cols` are rendered as `+0.523` (signed dimensionless).
    - Columns in `multiple_cols` are rendered as `2.00` (× NAV multiples).
    - Columns in `bps_cols` are rendered as `337.4 bps` (basis points).
    - Anything else uses pandas defaults.

    The underlying DataFrame is left untouched (numeric, sortable). This helper is
    the single source of truth for metric presentation across all track tables.
    """
    fmt: Dict[str, Callable[[Any], str]] = {}
    pct_set = set(percent_cols)
    ratio_set = set(ratio_cols)
    multiple_set = set(multiple_cols)
    bps_set = set(bps_cols)
    for col in df.columns:
        if col in pct_set:
            fmt[col] = "{:.2%}".format
        elif col in ratio_set:
            fmt[col] = "{:+.3f}".format
        elif col in multiple_set:
            fmt[col] = "{:.2f}".format
        elif col in bps_set:
            fmt[col] = "{:.1f} bps".format
    return df.style.format(fmt)
