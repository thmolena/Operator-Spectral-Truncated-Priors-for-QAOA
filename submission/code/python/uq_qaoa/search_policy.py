from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any
import numpy as np
from .qaoa_angles import assert_theta, project_angles


@dataclass
class SearchResult:
    theta_hat: np.ndarray
    y_hat: float
    trace: list[dict]


def uq_qaoa_search(objective, p: int, Q: int, anchors: list[tuple[str, np.ndarray]], var_post, layout: str = "blocked", eta0: float = 1.0, shrink_factor: float = 0.5, eta_min: float = 1e-3, rng_seed: int = 0, trace_context: dict | None = None) -> SearchResult:
    if Q <= 0:
        raise ValueError("Q must be positive")
    d = 2 * p
    var = np.asarray(var_post, dtype=float).reshape(-1)
    if var.size != d:
        raise ValueError("var_post length must be 2*p")
    trace_context = trace_context or {}
    trace: list[dict] = []
    best_theta = None
    best_y = -np.inf
    remaining = Q
    qidx = 0

    def eval_record(theta, source: str, coord: Any = "", sign: Any = "", step_size: Any = "", eta: Any = ""):
        nonlocal best_theta, best_y, remaining, qidx
        if remaining <= 0:
            return False
        t0 = time.perf_counter()
        theta = project_angles(assert_theta(theta, p), p, layout)
        y = float(objective(theta))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        qidx += 1
        remaining -= 1
        if y > best_y or best_theta is None:
            best_y = y
            best_theta = theta.copy()
        row = dict(trace_context)
        row.update({
            "query_index": qidx,
            "theta_json": __import__("json").dumps([float(x) for x in theta]),
            "source": source,
            "coordinate_index": coord,
            "step_sign": sign,
            "step_size": step_size,
            "eta": eta,
            "objective_exact": y,
            "incumbent_objective": best_y,
            "incumbent_theta_json": __import__("json").dumps([float(x) for x in best_theta]),
            "elapsed_ms": elapsed_ms,
        })
        trace.append(row)
        return True

    seen: set[bytes] = set()
    for name, theta in anchors:
        theta = project_angles(assert_theta(theta, p), p, layout)
        key = np.round(theta, 14).tobytes()
        if key in seen:
            continue
        seen.add(key)
        eval_record(theta, f"anchor:{name}")
        if remaining <= 0:
            break

    if best_theta is None:
        raise RuntimeError("no anchor could be evaluated")

    eta = eta0
    order = list(np.argsort(-var))
    while remaining > 0 and eta >= eta_min:
        improved = False
        for j in order:
            for sign in (+1, -1):
                if remaining <= 0:
                    break
                step = float(eta * np.sqrt(max(var[j], 0.0)))
                probe = best_theta.copy()
                probe[j] += sign * step
                old_best = best_y
                eval_record(probe, "coordinate_refine", coord=int(j), sign=int(sign), step_size=step, eta=float(eta))
                if best_y > old_best:
                    improved = True
                    break
            if improved or remaining <= 0:
                break
        if not improved:
            eta *= shrink_factor
    return SearchResult(best_theta.copy(), float(best_y), trace)
