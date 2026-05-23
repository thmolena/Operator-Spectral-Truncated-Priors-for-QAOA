from __future__ import annotations

import numpy as np


def approximation_ratio(value: float, reference: float) -> float:
    return 0.0 if reference <= 0 else float(value) / float(reference)


def best_so_far_curve(values) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(values, dtype=float))


def area_under_query_curve(values) -> float:
    curve = best_so_far_curve(values)
    return float(np.trapezoid(curve, dx=1.0))


def query_to_target(values, target: float):
    curve = best_so_far_curve(values)
    hits = np.flatnonzero(curve >= target)
    return None if hits.size == 0 else int(hits[0] + 1)


def paired_difference(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have equal shape")
    return a - b


def paired_win_rate(a, b) -> float:
    return float(np.mean(paired_difference(a, b) > 0.0))


def standard_error(values) -> float:
    x = np.asarray(values, dtype=float)
    if x.size <= 1:
        return 0.0
    return float(x.std(ddof=1) / np.sqrt(x.size))


def bootstrap_ci(values, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, size=x.size, replace=True).mean() for _ in range(n_boot)]
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def failure_rate(values, target: float) -> float:
    return float(np.mean(np.asarray(values, dtype=float) < target))
