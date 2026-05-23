from __future__ import annotations

import numpy as np


def cut_value(bitstring: int | list[int] | np.ndarray, edges, weights=None) -> float:
    if weights is None:
        weights = [1.0] * len(edges)
    if isinstance(bitstring, (int, np.integer)):
        return float(sum(w for (u, v), w in zip(edges, weights) if ((int(bitstring) >> u) & 1) != ((int(bitstring) >> v) & 1)))
    bits = np.asarray(bitstring, dtype=np.int8)
    return float(sum(w for (u, v), w in zip(edges, weights) if bits[u] != bits[v]))


def all_cut_values(n: int, edges, weights=None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    vals = np.zeros(1 << n, dtype=np.float64)
    for z in range(1 << n):
        vals[z] = cut_value(z, edges, weights)
    return vals


def exact_maxcut_bruteforce(n: int, edges, weights=None) -> tuple[float, str]:
    if n > 26:
        raise ValueError("exact brute force is intentionally limited to n<=26")
    return float(all_cut_values(n, edges, weights).max()), "exact"


def classical_heuristic_reference(n: int, edges, seed: int = 0, restarts: int = 256) -> tuple[float, str]:
    rng = np.random.default_rng(seed)
    best = 0.0
    for _ in range(restarts):
        bits = rng.integers(0, 2, size=n, dtype=np.int8)
        improved = True
        while improved:
            improved = False
            current = cut_value(bits, edges)
            for i in range(n):
                bits[i] ^= 1
                val = cut_value(bits, edges)
                if val > current:
                    current = val
                    improved = True
                else:
                    bits[i] ^= 1
        best = max(best, cut_value(bits, edges))
    return float(best), "heuristic"


def normalize_objective(value: float, reference: float) -> float:
    if reference <= 0:
        return 0.0
    return float(value) / float(reference)
