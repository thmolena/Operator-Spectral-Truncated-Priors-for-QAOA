from __future__ import annotations

import numpy as np


def normalized_squared_error(theta_ref, mu, var) -> np.ndarray:
    theta_ref = np.asarray(theta_ref, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.maximum(np.asarray(var, dtype=float), 1e-12)
    return ((theta_ref - mu) ** 2) / var


def empirical_coverage(theta_refs, mus, vars_, radius: float) -> float:
    vals = []
    for t, m, v in zip(theta_refs, mus, vars_):
        score = float(normalized_squared_error(t, m, v).sum())
        vals.append(score <= radius)
    return float(np.mean(vals)) if vals else float("nan")


def reliability_bins(predicted, observed, bins: int = 10):
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (predicted >= lo) & (predicted < hi if hi < 1 else predicted <= hi)
        if mask.any():
            rows.append({"bin_low": lo, "bin_high": hi, "predicted": float(predicted[mask].mean()), "observed": float(observed[mask].mean()), "count": int(mask.sum())})
    return rows


def expected_calibration_error(predicted, observed, bins: int = 10) -> float:
    rows = reliability_bins(predicted, observed, bins)
    n = sum(r["count"] for r in rows)
    return float(sum(r["count"] * abs(r["predicted"] - r["observed"]) for r in rows) / max(n, 1))


def uncertainty_error_correlation(var, errors) -> float:
    u = np.asarray(var, dtype=float).reshape(len(errors), -1).mean(axis=1)
    e = np.asarray(errors, dtype=float)
    if u.size < 2 or np.std(u) == 0 or np.std(e) == 0:
        return 0.0
    return float(np.corrcoef(u, e)[0, 1])
