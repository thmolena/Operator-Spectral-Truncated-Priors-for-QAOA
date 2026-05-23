from __future__ import annotations

import numpy as np
from .qaoa_angles import assert_theta, project_angles


def fuse_diagonal_priors(priors: list[np.ndarray], variances: list[np.ndarray] | None, p: int, layout: str = "blocked", variance_floor: float = 1e-8, variance_ceiling: float = 1.0):
    if not priors:
        raise ValueError("at least one prior is required")
    d = 2 * p
    xs = [assert_theta(x, p) for x in priors]
    if variances is None:
        variances = [np.full(d, 0.25, dtype=float) for _ in xs]
    vs = [np.clip(np.asarray(v, dtype=float).reshape(-1), variance_floor, variance_ceiling) for v in variances]
    if any(v.size != d for v in vs):
        raise ValueError("each variance vector must have length 2*p")
    precisions = [1.0 / v for v in vs]
    prec_sum = np.sum(precisions, axis=0)
    mu = np.sum([prec * x for prec, x in zip(precisions, xs)], axis=0) / prec_sum
    var = np.clip(1.0 / prec_sum, variance_floor, variance_ceiling)
    return project_angles(mu, p, layout), var


def ablate_variance(var: np.ndarray, mode: str, seed: int = 0) -> np.ndarray:
    var = np.asarray(var, dtype=float).copy()
    if mode == "none" or mode == "isotropic":
        return np.full_like(var, float(np.mean(var)))
    if mode == "shuffled":
        rng = np.random.default_rng(seed)
        rng.shuffle(var)
        return var
    if mode == "full":
        return var
    raise ValueError("unknown variance ablation mode")
