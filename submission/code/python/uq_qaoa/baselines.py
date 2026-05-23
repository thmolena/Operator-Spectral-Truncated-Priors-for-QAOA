from __future__ import annotations

import numpy as np
from .qaoa_angles import random_theta, project_angles, assert_theta
from .priors import theta_tqa, theta_global
from .search_policy import uq_qaoa_search, SearchResult


def random_search(objective, p: int, Q: int, seed: int = 0, layout: str = "blocked") -> SearchResult:
    rng = np.random.default_rng(seed)
    best_theta = None
    best_y = -np.inf
    trace = []
    for q in range(1, Q + 1):
        theta = random_theta(p, rng, layout)
        y = float(objective(theta))
        if y > best_y or best_theta is None:
            best_y = y
            best_theta = theta.copy()
        trace.append({"query_index": q, "source": "random", "objective_exact": y, "incumbent_objective": best_y})
    if best_theta is None:
        raise RuntimeError("random_search evaluated no candidates")
    return SearchResult(best_theta, best_y, trace)


def single_anchor(objective, p: int, Q: int, theta, name: str, layout: str = "blocked") -> SearchResult:
    theta = project_angles(assert_theta(theta, p), p, layout)
    y = float(objective(theta))
    trace = [{"query_index": 1, "source": name, "objective_exact": y, "incumbent_objective": y}]
    return SearchResult(theta, y, trace)


def tqa(objective, p: int, Q: int, layout: str = "blocked") -> SearchResult:
    return single_anchor(objective, p, Q, theta_tqa(p, layout), "tqa", layout)


def global_prior(objective, p: int, Q: int, layout: str = "blocked") -> SearchResult:
    return single_anchor(objective, p, Q, theta_global(p, layout), "global_prior", layout)


def tqa_refine(objective, p: int, Q: int, layout: str = "blocked", seed: int = 0) -> SearchResult:
    d = 2 * p
    anchors = [("tqa", theta_tqa(p, layout))]
    return uq_qaoa_search(objective, p, Q, anchors, np.full(d, 0.25), layout=layout, rng_seed=seed)


def posterior_mean_only(objective, p: int, Q: int, mu_post, layout: str = "blocked") -> SearchResult:
    return single_anchor(objective, p, Q, mu_post, "posterior_mean", layout)


def posterior_anchors_only(objective, p: int, Q: int, anchors, layout: str = "blocked") -> SearchResult:
    best_theta = None
    best_y = -np.inf
    trace = []
    for q, (name, theta) in enumerate(anchors[:Q], start=1):
        theta = project_angles(assert_theta(theta, p), p, layout)
        y = float(objective(theta))
        if y > best_y or best_theta is None:
            best_y = y
            best_theta = theta.copy()
        trace.append({"query_index": q, "source": f"anchor:{name}", "objective_exact": y, "incumbent_objective": best_y})
    if best_theta is None:
        raise RuntimeError("posterior_anchors_only evaluated no anchors")
    return SearchResult(best_theta, best_y, trace)


def spsa(objective, p: int, Q: int, seed: int = 0, layout: str = "blocked") -> SearchResult:
    rng = np.random.default_rng(seed)
    theta = random_theta(p, rng, layout)
    best_theta = theta.copy()
    best_y = float(objective(theta))
    trace = [{"query_index": 1, "source": "spsa_init", "objective_exact": best_y, "incumbent_objective": best_y}]
    q = 1
    a = 0.2
    c = 0.1
    while q + 1 <= Q:
        delta = rng.choice([-1.0, 1.0], size=2 * p)
        yp = float(objective(project_angles(theta + c * delta, p, layout)))
        ym = float(objective(project_angles(theta - c * delta, p, layout)))
        q += 2
        ghat = (yp - ym) / (2 * c) * delta
        theta = project_angles(theta + a * ghat, p, layout)
        for y, src in [(yp, "spsa_plus"), (ym, "spsa_minus")]:
            if y > best_y:
                best_y = y
                best_theta = theta.copy()
            trace.append({"query_index": len(trace) + 1, "source": src, "objective_exact": y, "incumbent_objective": best_y})
    return SearchResult(best_theta, best_y, trace[:Q])
