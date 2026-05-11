"""Reference-vs-current parity check for kalman.run_kalman_replica.

Pins the IR/TE/turnover/rho and the full net replica return series for the
canonical KalmanConfig(sigma_w=1e-3) configuration. The HMM-leakage fix
landing in Task 3 of the refactor adds a fit_until parameter to
kalman.fit_hmm_regime (not to run_kalman_replica), so this run-replica
path must remain byte-identical across Task 3.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness import build_replication_panel, clean_price_panel, load_bloomberg_weekly
from kalman import KalmanConfig, run_kalman_replica

REF_PATH = Path(__file__).parent / "reference" / "kalman_baseline.pkl"
TOLERANCE = 1e-6


def _run() -> dict:
    prices_raw = load_bloomberg_weekly(ROOT / "Dataset3_PortfolioReplicaStrategy.xlsx")
    prices = clean_price_panel(prices_raw)
    X, y = build_replication_panel(prices)
    res = run_kalman_replica(
        X, y,
        cfg=KalmanConfig(sigma_w=1e-3),
        name="smoke_kalman",
        eval_window=104,
        tc_bps=5.0,
    )
    return {
        "IR": float(res.metrics["IR"]),
        "TE": float(res.metrics["TE"]),
        "turnover": float(res.metrics["turnover"]),
        "rho": float(res.metrics["rho"]),
        "replica_net": res.replica_net.values,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--check", action="store_true")
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
    if not np.allclose(current["replica_net"], ref["replica_net"], atol=TOLERANCE):
        max_err = float(np.max(np.abs(current["replica_net"] - ref["replica_net"])))
        failures.append(f"replica_net: max abs deviation = {max_err:.2e}")

    if failures:
        print("PARITY FAIL")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"PARITY OK (IR={current['IR']:+.6f}, TE={current['TE']:.6f}, turnover={current['turnover']:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
