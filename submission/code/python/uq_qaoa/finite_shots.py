from __future__ import annotations

import numpy as np
from .statevector import qaoa_statevector
from .maxcut import all_cut_values


def sample_objective(n: int, edges, p: int, theta, shots: int, rng: np.random.Generator, layout: str = "blocked", cost_values=None, return_counts: bool = False):
    if shots <= 0:
        raise ValueError("shots must be positive")
    if cost_values is None:
        cost_values = all_cut_values(n, edges)
    state = qaoa_statevector(n, edges, p, theta, layout=layout, cost_values=cost_values)
    probs = np.abs(state) ** 2
    probs = probs / probs.sum()
    samples = rng.choice(probs.size, size=shots, p=probs)
    vals = np.asarray(cost_values, dtype=float)[samples]
    mean = float(vals.mean())
    se = float(vals.std(ddof=1) / np.sqrt(shots)) if shots > 1 else 0.0
    out = {"mean": mean, "standard_error": se, "shots": shots}
    if return_counts:
        unique, counts = np.unique(samples, return_counts=True)
        out["counts"] = {int(k): int(v) for k, v in zip(unique, counts)}
    return out
