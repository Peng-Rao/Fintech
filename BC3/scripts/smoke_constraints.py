"""Reference-vs-current parity check for portfolio_constraints.run_backtest.

Usage
-----
# First run on pre-refactor `main` (or pre-Task-2):
python -m BC3.scripts.smoke_constraints --save

# After Task 2 rewrites portfolio_constraints.py:
python -m BC3.scripts.smoke_constraints --check

Exits 0 on success, 1 on metric mismatch.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import build_replication_panel, clean_price_panel, load_bloomberg_weekly
from portfolio_constraints import run_backtest

REF_PATH = Path(__file__).parent / "reference" / "constraints_baseline.pkl"
TOLERANCE = 1e-6


def _run() -> dict:
    prices_raw = load_bloomberg_weekly(ROOT / "Dataset3_PortfolioReplicaStrategy.xlsx")
    prices = clean_price_panel(prices_raw)
    X, y = build_replication_panel(prices)
    result = run_backtest(
        X_values=X.values,
        y_values=y.values,
        asset_names=X.columns.tolist(),
        dates=X.index.to_numpy(),
        rolling_window=104,
        rebal_freq=1,
        max_gross_exposure=1.0,
        max_var=0.08,
        step=0.01,
        alpha=0.001,
        l1_ratio=0.5,
        cost_bps=5,
    )
    m = result["metrics"]
    return {
        "IR": float(m.loc["IR", "Value"]),
        "TE": float(m.loc["TE", "Value"]),
        "turnover": float(m.loc["turnover", "Value"]),
        "rho": float(m.loc["p", "Value"]),
        "replica_returns": result["replica_returns"].values,
        "target_returns": result["target_returns"].values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Record current run as the reference")
    parser.add_argument("--check", action="store_true", help="Compare current run against the saved reference")
    args = parser.parse_args()
    if args.save == args.check:
        parser.error("Pass exactly one of --save / --check")

    current = _run()
    if args.save:
        REF_PATH.parent.mkdir(parents=True, exist_ok=True)
        with REF_PATH.open("wb") as f:
            pickle.dump(current, f)
        print(f"Saved reference: IR={current['IR']:+.6f} TE={current['TE']:.6f} turnover={current['turnover']:.6f}")
        return 0

    with REF_PATH.open("rb") as f:
        ref = pickle.load(f)

    failures: list[str] = []
    for key in ("IR", "TE", "turnover", "rho"):
        if abs(current[key] - ref[key]) > TOLERANCE:
            failures.append(f"{key}: ref={ref[key]:+.8f} current={current[key]:+.8f} delta={current[key] - ref[key]:+.2e}")
    if not np.allclose(current["replica_returns"], ref["replica_returns"], atol=TOLERANCE):
        max_err = float(np.max(np.abs(current["replica_returns"] - ref["replica_returns"])))
        failures.append(f"replica_returns: max abs deviation = {max_err:.2e}")
    if not np.allclose(current["target_returns"], ref["target_returns"], atol=TOLERANCE):
        max_err = float(np.max(np.abs(current["target_returns"] - ref["target_returns"])))
        failures.append(f"target_returns: max abs deviation = {max_err:.2e}")

    if failures:
        print("PARITY FAIL")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"PARITY OK (IR={current['IR']:+.6f}, TE={current['TE']:.6f}, turnover={current['turnover']:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
