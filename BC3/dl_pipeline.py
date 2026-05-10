"""Deep-learning pipeline for the NN weight generator.

Splits naturally into five layers:
    1. Feature engineering — build_features (vanilla) and build_features_pca (PCA variant).
    2. Models                — WeightMLP and WeightTransformer.
    3. Loss & windowing      — make_supervised_windows, te_mse_loss, turnover_penalty,
                               annualized_te_from_weights, _drift_weights, project_var_cap,
                               make_attention_windows.
    4. Trainer               — TrainConfig + train_weight_mlp (chronological split, early stop).
    5. Rolling backtest      — compute_metrics + run_nn_rolling_backtest +
                               run_attn_rolling_backtest (training / extended-feature entry points).

Functions infer the active torch device from `next(model.parameters()).device`, so the module
has no global `device` dependency — drop the model on whatever device you like and pass it in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

__all__ = [
    # Features
    "FeatureConfig",
    "build_features",
    "build_features_pca",
    # Models
    "WeightMLP",
    "WeightTransformer",
    # Windowing & loss
    "make_supervised_windows",
    "make_attention_windows",
    "te_mse_loss",
    "turnover_penalty",
    "annualized_te_from_weights",
    "project_var_cap",
    # Training
    "TrainConfig",
    "train_weight_mlp",
    # Backtest & metrics
    "compute_metrics",
    "run_nn_rolling_backtest",
    "run_attn_rolling_backtest",
]


# =============================================================================
# 1. Feature engineering
# =============================================================================


@dataclass
class FeatureConfig:
    """Feature design knobs."""
    return_lookbacks: tuple = (4, 12, 52)
    vol_lookbacks: tuple = (12, 52)
    use_regime: bool = True
    use_warmstart: bool = True
    warmstart_window: int = 52
    warmstart_alpha: float = 1e-3


def _compounded_return(s: pd.Series, k: int) -> pd.Series:
    return (1.0 + s).rolling(k).apply(np.prod, raw=True) - 1.0


def _ridge_warmstart(
    X: pd.DataFrame, y: pd.Series, window: int, alpha: float
) -> pd.DataFrame:
    """Trailing-window Ridge fit with MinMax normalization, mirroring the EN baseline."""
    cols = [f"w_en_{c}" for c in X.columns]
    out = pd.DataFrame(index=X.index, columns=cols, dtype=float)
    Xv, yv = X.values, y.values
    for i in range(window, len(X)):
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(Xv[i - window:i])
        mdl = Ridge(alpha=alpha, fit_intercept=False)
        mdl.fit(X_tr, yv[i - window:i])
        out.iloc[i] = mdl.coef_ / scaler.scale_
    return out


def build_features(
    X: pd.DataFrame,
    y: pd.Series,
    data: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """Build phi_t for every rebalance date.

    phi_t uses information up to t-1 only (lagged by one step) so the network can
    never peek at the contemporaneous return it is asked to weight.
    """
    feats: dict[str, pd.Series] = {}

    for k in cfg.return_lookbacks:
        for c in X.columns:
            feats[f"ret_{k}w_{c}"] = _compounded_return(X[c], k)

    for k in cfg.vol_lookbacks:
        vol = X.rolling(k).std() * np.sqrt(52)
        for c in X.columns:
            feats[f"vol_{k}w_{c}"] = vol[c]

    if cfg.use_regime:
        mxwo_ret = data["MXWO Index"].pct_change().reindex(X.index)
        bond_ret = data["LEGATRUU Index"].pct_change().reindex(X.index)
        feats["regime_msci_12w"] = _compounded_return(mxwo_ret, 12)
        feats["regime_bond_12w"] = _compounded_return(bond_ret, 12)
        feats["regime_target_vol_12w"] = y.rolling(12).std() * np.sqrt(52)

    if cfg.use_warmstart:
        warm = _ridge_warmstart(X, y, cfg.warmstart_window, cfg.warmstart_alpha)
        for c in warm.columns:
            feats[c] = warm[c]

    return pd.DataFrame(feats).shift(1).dropna()


def build_features_pca(
    X: pd.DataFrame,
    y: pd.Series,
    data: pd.DataFrame,
    cfg: FeatureConfig = FeatureConfig(),
    *,
    n_components: int = 5,
    pca_window: int = 52,
) -> pd.DataFrame:
    """PCA variant of build_features: the trailing-return block is replaced by the most recent
    PCA-score vector of the last `pca_window` weeks of X.
    """
    feats: dict = {}

    pca_cols = [f"pca_{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(index=X.index, columns=pca_cols, dtype=float)
    Xv = X.values
    for i in range(pca_window, len(X)):
        Xw = Xv[i - pca_window:i]
        Xw_c = Xw - Xw.mean(axis=0, keepdims=True)
        scores = PCA(n_components=n_components).fit_transform(Xw_c)
        pca_df.iloc[i] = scores[-1]
    for c in pca_cols:
        feats[c] = pca_df[c]

    for k in cfg.vol_lookbacks:
        vol = X.rolling(k).std() * np.sqrt(52)
        for c in X.columns:
            feats[f"vol_{k}w_{c}"] = vol[c]

    if cfg.use_regime:
        mxwo_ret = data["MXWO Index"].pct_change().reindex(X.index)
        bond_ret = data["LEGATRUU Index"].pct_change().reindex(X.index)
        feats["regime_msci_12w"] = _compounded_return(mxwo_ret, 12)
        feats["regime_bond_12w"] = _compounded_return(bond_ret, 12)
        feats["regime_target_vol_12w"] = y.rolling(12).std() * np.sqrt(52)

    if cfg.use_warmstart:
        warm = _ridge_warmstart(X, y, cfg.warmstart_window, cfg.warmstart_alpha)
        for c in warm.columns:
            feats[c] = warm[c]

    return pd.DataFrame(feats).shift(1).dropna()


# =============================================================================
# 2. Models
# =============================================================================


class WeightMLP(nn.Module):
    """End-to-end weight generator: phi_t -> 64 -> 32 -> w_t in R^11.

    No softmax: long/short positions are allowed.
    """

    def __init__(
        self,
        in_dim: int,
        n_assets: int = 11,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
        gross_cap: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        final = nn.Linear(prev, n_assets)
        # Bias init so that at init w_t.sum() ≈ 1. Without this, w_t.sum() ≈ 0 makes the
        # buy-and-hold drift renormaliser blow up (the 10^19 epoch-0 loss spike).
        nn.init.constant_(final.bias, 1.0 / n_assets)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.gross_cap = gross_cap

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        w = self.net(phi)
        if self.gross_cap is not None:
            ge = w.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.minimum(torch.ones_like(ge), self.gross_cap / ge)
            w = w * scale
        return w


class WeightTransformer(nn.Module):
    """Tiny Transformer over [T_w, n_features] -> w_t in R^n_assets."""

    def __init__(
        self,
        n_features: int,
        n_assets: int = 11,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        gross_cap: float | None = None,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_assets)
        self.gross_cap = gross_cap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(self.proj(x))
        w = self.head(z[:, -1, :])
        if self.gross_cap is not None:
            ge = w.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = torch.minimum(torch.ones_like(ge), self.gross_cap / ge)
            w = w * scale
        return w


# =============================================================================
# 3. Windowing & loss
# =============================================================================


def make_supervised_windows(
    phi: pd.DataFrame, X: pd.DataFrame, y: pd.Series, H: int = 12,
):
    """Pair each phi_t with the contemporaneous H-week window of (r_s, y_s)."""
    H = int(H)
    n_samples = len(phi) - H + 1
    if n_samples <= 0:
        raise ValueError(f"len(phi)={len(phi)} too short for H={H}")
    X_aligned = X.loc[phi.index].values
    y_aligned = y.loc[phi.index].values
    phi_arr = phi.values[:n_samples].astype(np.float32)
    X_win = np.stack([X_aligned[i:i + H] for i in range(n_samples)]).astype(np.float32)
    y_win = np.stack([y_aligned[i:i + H] for i in range(n_samples)]).astype(np.float32)
    sample_dates = phi.index[:n_samples]
    return phi_arr, X_win, y_win, sample_dates


def make_attention_windows(X: pd.DataFrame, y: pd.Series, T_w: int = 52, H: int = 12):
    """Build [phi_seq_t, X_win_t, y_win_t] tuples for the transformer variant.

    phi_seq_t = X.iloc[t-T_w:t] (lagged trailing window) so the model only sees information
    available before time t.
    """
    n = len(X)
    starts = list(range(T_w, n - H + 1))
    if not starts:
        raise ValueError("not enough data for attention windows")
    Xv = X.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    phi_seq = np.stack([Xv[t - T_w:t] for t in starts])
    X_win = np.stack([Xv[t:t + H] for t in starts])
    y_win = np.stack([yv[t:t + H] for t in starts])
    sample_dates = X.index[starts[0]:starts[-1] + 1]
    return phi_seq, X_win, y_win, sample_dates


def _drift_weights(weights: torch.Tensor, X_win: torch.Tensor) -> torch.Tensor:
    """Compound w_t through the realised window into per-step weights w_s (buy-and-hold drift)."""
    B, H, N = X_win.shape
    gross = torch.cumprod(1.0 + X_win, dim=1)
    gross_prev = torch.cat(
        [torch.ones(B, 1, N, device=X_win.device, dtype=X_win.dtype), gross[:, :-1]],
        dim=1,
    )
    w_drift = weights.unsqueeze(1) * gross_prev
    # clamp_min at 1e-3 (not 1e-12) so a hedged init (sum ≈ 0) cannot blow the loss up to 1e19.
    w_drift = w_drift / w_drift.sum(dim=-1, keepdim=True).clamp_min(1e-3)
    return w_drift


def te_mse_loss(
    weights: torch.Tensor,
    X_win: torch.Tensor,
    y_win: torch.Tensor,
    *,
    gamma_budget: float = 0.0,
    drift: bool = True,
) -> torch.Tensor:
    """Forward-window MSE of (w_s^T r_s - y_s), with optional weight drift and budget penalty."""
    if drift:
        w_eff = _drift_weights(weights, X_win)
        replica = (w_eff * X_win).sum(dim=-1)
    else:
        replica = (weights.unsqueeze(1) * X_win).sum(dim=-1)
    excess = replica - y_win
    loss = (excess ** 2).mean()
    if gamma_budget > 0.0:
        loss = loss + gamma_budget * (weights.sum(dim=-1) - 1.0).pow(2).mean()
    return loss


def turnover_penalty(
    weights: torch.Tensor,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
) -> torch.Tensor:
    """L1 (linear t-cost) + L2 (market impact) on Δw between adjacent rows."""
    if weights.size(0) < 2 or (lambda_l1 == 0.0 and lambda_l2 == 0.0):
        return torch.zeros((), device=weights.device, dtype=weights.dtype)
    dw = weights[1:] - weights[:-1]
    pen = torch.zeros((), device=weights.device, dtype=weights.dtype)
    if lambda_l1 > 0.0:
        pen = pen + lambda_l1 * dw.abs().sum(dim=-1).mean()
    if lambda_l2 > 0.0:
        pen = pen + lambda_l2 * dw.pow(2).sum(dim=-1).mean()
    return pen


def annualized_te_from_weights(
    weights: torch.Tensor,
    X_win: torch.Tensor,
    y_win: torch.Tensor,
    *,
    drift: bool = True,
) -> float:
    """TE = sqrt(52 * Var(replica - y)) on a held-out fold (weekly data)."""
    with torch.no_grad():
        if drift:
            w_eff = _drift_weights(weights, X_win)
            replica = (w_eff * X_win).sum(dim=-1)
        else:
            replica = (weights.unsqueeze(1) * X_win).sum(dim=-1)
        var = (replica - y_win).var(unbiased=False).item()
    return float(np.sqrt(52.0 * var))


def project_var_cap(
    weights: np.ndarray,
    recent_returns: np.ndarray,
    confidence: float = 0.01,
    horizon: int = 4,
    var_cap: float = 0.08,
) -> np.ndarray:
    """Post-hoc Gaussian VaR projection (matches the Elastic Net baseline)."""
    sigma = np.std(recent_returns @ weights)
    z = norm.ppf(confidence)
    var = -z * sigma * np.sqrt(horizon)
    if var > var_cap:
        return weights * (var_cap / var)
    return weights


# =============================================================================
# 4. Trainer
# =============================================================================


@dataclass
class TrainConfig:
    """training knobs: temporal split, optimiser, early stopping, loss."""
    H: int = 12
    train_frac: float = 0.6
    val_frac: float = 0.2
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    gamma_budget: float = 0.0
    drift: bool = True
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 15
    seed: int = 42
    refit_on_trainval: bool = True


def _temporal_slices(n: int, train_frac: float, val_frac: float):
    n_tr = int(train_frac * n)
    n_va = int(val_frac * n)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def train_weight_mlp(
    model: nn.Module,
    phi_arr: np.ndarray,
    X_win: np.ndarray,
    y_win: np.ndarray,
    cfg: TrainConfig = TrainConfig(),
) -> dict:
    """Train phi_t -> w_t with the MSE+turnover loss; early-stop on val TE.

    Splits supervised samples chronologically (no shuffle across the boundary), standardises
    features with the training-fold mean/std, and refits on train+val for best_epoch+1 epochs.
    Reports annualised TE on all three folds so train≈val≪test (regime shift) is distinguishable
    from train≪val (classical overfit). Device is taken from the model's first parameter.
    """
    device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    init_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    sl_tr, sl_va, sl_te = _temporal_slices(len(phi_arr), cfg.train_frac, cfg.val_frac)

    mu = phi_arr[sl_tr].mean(axis=0, keepdims=True)
    sd = phi_arr[sl_tr].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    phi_norm = ((phi_arr - mu) / sd).astype(np.float32)

    phi_all = torch.tensor(phi_norm, device=device)
    X_all = torch.tensor(X_win, device=device)
    y_all = torch.tensor(y_win, device=device)

    phi_tr, phi_va, phi_te = phi_all[sl_tr], phi_all[sl_va], phi_all[sl_te]
    X_tr, X_va, X_te = X_all[sl_tr], X_all[sl_va], X_all[sl_te]
    y_tr, y_va, y_te = y_all[sl_tr], y_all[sl_va], y_all[sl_te]

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    best_val_te = float("inf")
    best_state: dict | None = None
    patience_left = cfg.patience
    history = {"epoch": [], "train_loss": [], "val_te": []}

    n_train = phi_tr.size(0)
    for epoch in range(cfg.max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, cfg.batch_size):
            stop = min(start + cfg.batch_size, n_train)
            phi_b = phi_tr[start:stop]
            X_b = X_tr[start:stop]
            y_b = y_tr[start:stop]

            w_b = model(phi_b)
            loss = te_mse_loss(w_b, X_b, y_b, gamma_budget=cfg.gamma_budget, drift=cfg.drift)
            loss = loss + turnover_penalty(w_b, cfg.lambda_l1, cfg.lambda_l2)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            w_val = model(phi_va)
        val_te = annualized_te_from_weights(w_val, X_va, y_va, drift=cfg.drift)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_te"].append(val_te)

        if val_te < best_val_te - 1e-6:
            best_val_te = val_te
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        w_train = model(phi_tr)
    train_te = annualized_te_from_weights(w_train, X_tr, y_tr, drift=cfg.drift)

    refit_epochs = 0
    if cfg.refit_on_trainval and best_state is not None:
        best_epoch_idx = int(np.argmin(history["val_te"]))
        refit_epochs = max(best_epoch_idx + 1, 1)
        model.load_state_dict(init_state)
        optim_refit = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        phi_full = torch.cat([phi_tr, phi_va], dim=0)
        X_full = torch.cat([X_tr, X_va], dim=0)
        y_full = torch.cat([y_tr, y_va], dim=0)
        n_full = phi_full.size(0)
        for _ in range(refit_epochs):
            model.train()
            for start in range(0, n_full, cfg.batch_size):
                stop = min(start + cfg.batch_size, n_full)
                phi_b = phi_full[start:stop]
                X_b = X_full[start:stop]
                y_b = y_full[start:stop]
                w_b = model(phi_b)
                loss = te_mse_loss(w_b, X_b, y_b, gamma_budget=cfg.gamma_budget, drift=cfg.drift)
                loss = loss + turnover_penalty(w_b, cfg.lambda_l1, cfg.lambda_l2)
                optim_refit.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim_refit.step()

    model.eval()
    with torch.no_grad():
        w_test = model(phi_te)
    test_te = annualized_te_from_weights(w_test, X_te, y_te, drift=cfg.drift)

    return {
        "history": history,
        "train_te": train_te,
        "best_val_te": best_val_te,
        "test_te": test_te,
        "refit_epochs": refit_epochs,
        "splits": {"train": sl_tr, "val": sl_va, "test": sl_te},
        "feature_mu": mu,
        "feature_sd": sd,
    }


# =============================================================================
# 5. Rolling backtest & metrics
# =============================================================================


def compute_metrics(
    weights: pd.DataFrame,
    X_oos: pd.DataFrame,
    target: pd.Series,
    replica: pd.Series,
    *,
    cost_bps: float = 5.0,
    var_horizon: int = 4,
    var_confidence: float = 0.01,
) -> dict:
    """Shared metric dict {IR, TE, rho, GE, VaR, turnover, net_IR, net_TE, drag}.

    Net-of-cost subtracts tau * sum_j |w_{t,j} - w_{t-1,j}| at every rebalance, with the first
    row charged against w_0 = 0 (entering from cash).
    """
    excess = (replica - target).values
    te = float(np.sqrt(52.0 * np.var(excess, ddof=0)))
    ir = float(52.0 * np.mean(excess) / te) if te > 0 else float("nan")
    rho = float(np.corrcoef(replica.values, target.values)[0, 1])
    ge = float(weights.abs().sum(axis=1).mean())

    w_prev = np.vstack([np.zeros(weights.shape[1]), weights.values[:-1]])
    dw = np.abs(weights.values - w_prev).sum(axis=1)
    turnover = float(dw.mean())

    tau = cost_bps / 1e4
    cost = tau * dw
    net_replica = replica.values - cost
    net_excess = net_replica - target.values
    net_te = float(np.sqrt(52.0 * np.var(net_excess, ddof=0)))
    net_ir = float(52.0 * np.mean(net_excess) / net_te) if net_te > 0 else float("nan")
    drag = float(52.0 * tau * turnover)

    z = float(norm.ppf(var_confidence))
    sigma_replica = float(np.std(replica.values, ddof=0))
    var_overall = float(-z * sigma_replica * np.sqrt(var_horizon))

    return {
        "IR": ir, "TE": te, "rho": rho, "GE": ge,
        "turnover": turnover, "VaR": var_overall,
        "net_IR": net_ir, "net_TE": net_te, "cost_drag": drag,
    }


def _train_window_mlp(
    phi_tr: np.ndarray,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    *,
    in_dim: int,
    n_assets: int,
    gross_cap: float,
    lambda_l1: float,
    lambda_l2: float,
    gamma_budget: float,
    drift: bool,
    max_epochs: int,
    patience: int,
    val_frac: float,
    init_state: dict | None,
    seed: int,
    device: torch.device,
):
    """Fit a WeightMLP on a single training window. Returns (model, mu, sd)."""
    n = phi_tr.shape[0]
    n_va = max(int(val_frac * n), 8)
    n_in = n - n_va

    mu = phi_tr[:n_in].mean(axis=0, keepdims=True)
    sd = phi_tr[:n_in].std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    phi_norm = ((phi_tr - mu) / sd).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WeightMLP(in_dim=in_dim, n_assets=n_assets, gross_cap=gross_cap).to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    phi_tr_t = torch.tensor(phi_norm[:n_in], device=device)
    X_tr_t = torch.tensor(X_tr[:n_in], device=device)
    y_tr_t = torch.tensor(y_tr[:n_in], device=device)
    phi_va_t = torch.tensor(phi_norm[n_in:], device=device)
    X_va_t = torch.tensor(X_tr[n_in:], device=device)
    y_va_t = torch.tensor(y_tr[n_in:], device=device)

    best_val = float("inf")
    best_state: dict | None = None
    patience_left = patience
    for _ in range(max_epochs):
        model.train()
        w_b = model(phi_tr_t)
        loss = te_mse_loss(w_b, X_tr_t, y_tr_t, gamma_budget=gamma_budget, drift=drift)
        loss = loss + turnover_penalty(w_b, lambda_l1, lambda_l2)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        model.eval()
        with torch.no_grad():
            w_va = model(phi_va_t)
        val_te = annualized_te_from_weights(w_va, X_va_t, y_va_t, drift=drift)
        if val_te < best_val - 1e-6:
            best_val = val_te
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, mu.flatten(), sd.flatten()


def run_nn_rolling_backtest(
    phi: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    H: int = 12,
    train_window: int = 156,
    rebalance_every: int = 4,
    max_epochs: int = 80,
    patience: int = 12,
    lambda_l1: float = 1e-3,
    lambda_l2: float = 1e-3,
    gamma_budget: float = 1e-2,
    drift: bool = True,
    gross_cap: float = 2.0,
    var_cap: float = 0.08,
    var_horizon: int = 4,
    var_confidence: float = 0.01,
    val_frac_within_train: float = 0.2,
    cost_bps: float = 5.0,
    n_ensemble: int = 3,
    seed: int = 42,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """Rolling out-of-sample NN backtest matching the shared contract.

    Retrains the WeightMLP every `rebalance_every` weeks on the trailing `train_window`
    supervised samples and applies the freshly-fit weights for the next `rebalance_every` weeks
    (held constant). Each prediction is projected onto the Gaussian VaR feasible set.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    phi_arr_full, X_win_full, y_win_full, sample_dates = make_supervised_windows(phi, X, y, H=H)
    n = len(phi_arr_full)
    in_dim = phi_arr_full.shape[1]
    n_assets = X.shape[1]
    if n < train_window + 1:
        raise ValueError(f"need n >= train_window+1, got n={n}, train_window={train_window}")

    weights_by_date: dict = {}
    last_states: list[dict | None] = [None] * n_ensemble
    rebalance_indices = list(range(train_window, n, rebalance_every))
    if verbose:
        print(
            f"rebalance_every={rebalance_every}: {len(rebalance_indices)} retrains "
            f"x {n_ensemble} ensemble member(s) "
            f"covering {n - train_window} OOS weeks "
            f"({sample_dates[train_window].date()} -> {sample_dates[-1].date()})"
        )

    for k, t in enumerate(rebalance_indices):
        phi_tr = phi_arr_full[t - train_window:t]
        X_tr = X_win_full[t - train_window:t]
        y_tr = y_win_full[t - train_window:t]

        ensemble_models: list = []
        mu_use: np.ndarray | None = None
        sd_use: np.ndarray | None = None
        for j in range(n_ensemble):
            model_j, mu_j, sd_j = _train_window_mlp(
                phi_tr, X_tr, y_tr,
                in_dim=in_dim, n_assets=n_assets, gross_cap=gross_cap,
                lambda_l1=lambda_l1, lambda_l2=lambda_l2, gamma_budget=gamma_budget,
                drift=drift, max_epochs=max_epochs, patience=patience,
                val_frac=val_frac_within_train, init_state=last_states[j],
                seed=seed + 1000 * j + k, device=device,
            )
            ensemble_models.append(model_j)
            mu_use, sd_use = mu_j, sd_j
            last_states[j] = {k_: v_.detach().clone() for k_, v_ in model_j.state_dict().items()}

        next_t = rebalance_indices[k + 1] if k + 1 < len(rebalance_indices) else n
        for s in range(t, min(next_t, n)):
            phi_s = (phi_arr_full[s] - mu_use) / sd_use
            phi_s_t = torch.tensor(phi_s.astype(np.float32), device=device).unsqueeze(0)
            with torch.no_grad():
                preds = [m(phi_s_t).cpu().numpy().flatten() for m in ensemble_models]
            w_s = np.mean(preds, axis=0)
            date_s = sample_dates[s]
            loc = X.index.get_loc(date_s)
            recent = X.iloc[max(0, loc - 52):loc].values
            if len(recent) >= 12:
                w_s = project_var_cap(
                    w_s, recent,
                    confidence=var_confidence, horizon=var_horizon, var_cap=var_cap,
                )
            weights_by_date[date_s] = w_s

    weights_df = pd.DataFrame(weights_by_date).T.sort_index()
    weights_df.columns = X.columns
    X_oos = X.loc[weights_df.index]
    target_oos = y.loc[weights_df.index]
    replica = pd.Series(
        (weights_df.values * X_oos.values).sum(axis=1),
        index=weights_df.index, name="replica",
    )
    metrics = compute_metrics(
        weights_df, X_oos, target_oos, replica,
        cost_bps=cost_bps, var_horizon=var_horizon, var_confidence=var_confidence,
    )
    return {
        "weights_history": weights_df,
        "replica_returns": replica,
        "target_returns": target_oos,
        "metrics": metrics,
    }


def run_attn_rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    T_w: int = 52,
    H: int = 12,
    train_window: int = 156,
    rebalance_every: int = 4,
    max_epochs: int = 60,
    patience: int = 10,
    lambda_l1: float = 1e-3,
    lambda_l2: float = 1e-3,
    gamma_budget: float = 1e-2,
    drift: bool = True,
    gross_cap: float = 2.0,
    var_cap: float = 0.08,
    var_horizon: int = 4,
    var_confidence: float = 0.01,
    val_frac_within_train: float = 0.2,
    cost_bps: float = 5.0,
    seed: int = 42,
    verbose: bool = True,
    device: Optional[torch.device] = None,
) -> dict:
    """Rolling OOS WeightTransformer backtest. Same contract as run_nn_rolling_backtest."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    phi_seq, X_win, y_win, sample_dates = make_attention_windows(X, y, T_w=T_w, H=H)
    n = len(phi_seq)
    n_features = phi_seq.shape[2]
    n_assets = X.shape[1]
    if n < train_window + 1:
        raise ValueError("not enough samples for attention rolling backtest")

    weights_by_date: dict = {}
    last_state: dict | None = None
    rebalance_indices = list(range(train_window, n, rebalance_every))
    if verbose:
        print(
            f"attn rebalance_every={rebalance_every}: {len(rebalance_indices)} retrains, "
            f"OOS {sample_dates[train_window].date()} -> {sample_dates[-1].date()}"
        )

    for k, t in enumerate(rebalance_indices):
        phi_tr = phi_seq[t - train_window:t]
        X_tr_w = X_win[t - train_window:t]
        y_tr_w = y_win[t - train_window:t]
        n_va = max(int(val_frac_within_train * train_window), 8)
        n_in = train_window - n_va

        mu = phi_tr[:n_in].reshape(-1, n_features).mean(axis=0)
        sd = phi_tr[:n_in].reshape(-1, n_features).std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        phi_norm = ((phi_tr - mu) / sd).astype(np.float32)

        torch.manual_seed(seed + k)
        np.random.seed(seed + k)
        model = WeightTransformer(
            n_features=n_features, n_assets=n_assets, gross_cap=gross_cap,
        ).to(device)
        if last_state is not None:
            model.load_state_dict(last_state)
        optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        phi_tr_t = torch.tensor(phi_norm[:n_in], device=device)
        X_tr_t = torch.tensor(X_tr_w[:n_in], device=device)
        y_tr_t = torch.tensor(y_tr_w[:n_in], device=device)
        phi_va_t = torch.tensor(phi_norm[n_in:], device=device)
        X_va_t = torch.tensor(X_tr_w[n_in:], device=device)
        y_va_t = torch.tensor(y_tr_w[n_in:], device=device)

        best_val = float("inf")
        best_state: dict | None = None
        patience_left = patience
        for _ in range(max_epochs):
            model.train()
            w_b = model(phi_tr_t)
            loss = te_mse_loss(w_b, X_tr_t, y_tr_t, gamma_budget=gamma_budget, drift=drift)
            loss = loss + turnover_penalty(w_b, lambda_l1, lambda_l2)
            optim.zero_grad()
            loss.backward()
            optim.step()
            model.eval()
            with torch.no_grad():
                w_va = model(phi_va_t)
            val_te = annualized_te_from_weights(w_va, X_va_t, y_va_t, drift=drift)
            if val_te < best_val - 1e-6:
                best_val = val_te
                best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        last_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}

        next_t = rebalance_indices[k + 1] if k + 1 < len(rebalance_indices) else n
        for s in range(t, min(next_t, n)):
            phi_s = ((phi_seq[s] - mu) / sd).astype(np.float32)
            with torch.no_grad():
                w_s = (
                    model(torch.tensor(phi_s, device=device).unsqueeze(0))
                    .cpu().numpy().flatten()
                )
            date_s = sample_dates[s]
            loc = X.index.get_loc(date_s)
            recent = X.iloc[max(0, loc - 52):loc].values
            if len(recent) >= 12:
                w_s = project_var_cap(
                    w_s, recent,
                    confidence=var_confidence, horizon=var_horizon, var_cap=var_cap,
                )
            weights_by_date[date_s] = w_s

    weights_df = pd.DataFrame(weights_by_date).T.sort_index()
    weights_df.columns = X.columns
    X_oos = X.loc[weights_df.index]
    target_oos = y.loc[weights_df.index]
    replica = pd.Series(
        (weights_df.values * X_oos.values).sum(axis=1),
        index=weights_df.index, name="replica",
    )
    metrics = compute_metrics(
        weights_df, X_oos, target_oos, replica,
        cost_bps=cost_bps, var_horizon=var_horizon, var_confidence=var_confidence,
    )
    return {
        "weights_history": weights_df,
        "replica_returns": replica,
        "target_returns": target_oos,
        "metrics": metrics,
    }
