from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .baselines import random_search, tqa_refine
from .graphs import GraphInstance, generate_graph, graph_features
from .maxcut import all_cut_values, exact_maxcut_bruteforce, normalize_objective
from .priors import theta_global, theta_tqa
from .qaoa_angles import assert_theta, project_angles, random_theta
from .search_policy import SearchResult
from .statevector import qaoa_expectation


DEFAULT_FAMILIES = ("er", "random_regular", "watts_strogatz", "barabasi_albert")
DEFAULT_SIZES = (8, 10)
DEFAULT_SEED = 260424803


@dataclass(frozen=True)
class TrainingRecord:
    graph: GraphInstance
    theta: np.ndarray
    ratio: float
    signature: np.ndarray
    operator: np.ndarray
    commutator_norm: float


@dataclass(frozen=True)
class TrainingCore:
    """Rank/commutator-independent offline training targets.

    The expensive part of building a library is the budgeted offline optimisation
    of reference angles on the training graphs; this depends only on the QAOA
    objective, not on the spectral-truncation rank or the commutator weight.
    Caching it once lets the truncation sweep recompute only the cheap operator
    signatures for each rank, which is what makes the sweep figure tractable.
    """

    p: int
    optimizer_budget: int
    seed: int
    graphs: tuple[GraphInstance, ...]
    thetas: tuple[np.ndarray, ...]
    ratios: tuple[float, ...]


@dataclass(frozen=True)
class OperatorLibrary:
    p: int
    rank: int
    commutator_weight: float
    records: tuple[TrainingRecord, ...]
    signature_mean: np.ndarray
    signature_scale: np.ndarray


_CORE_CACHE: dict[tuple, TrainingCore] = {}


@dataclass(frozen=True)
class OperatorPrior:
    mean: np.ndarray
    covariance: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    weights: np.ndarray
    neighbor_indices: np.ndarray
    commutator_norm: float
    effective_rank: float


def stable_seed(seed: int, *parts: object) -> int:
    """Small deterministic mixer that avoids Python hash randomization."""
    value = int(seed) & 0xFFFFFFFF
    for part in parts:
        for ch in str(part):
            value = (1664525 * (value ^ ord(ch)) + 1013904223) & 0xFFFFFFFF
    return int(value % (2**31 - 1))


def effective_dimension(matrix: np.ndarray) -> float:
    """Participation-ratio effective dimension d_eff = (tr M)^2 / tr(M^2).

    For a positive-semidefinite covariance this is the number of directions that
    carry comparable variance. A noncommutative spectral-truncation prior
    concentrates the search variance into a few collective angle modes, lowering
    d_eff; this is the quantity that enters the query-complexity bound in the
    manuscript (fewer effective directions => fewer objective queries to cover
    the near-optimal set).
    """
    m = np.asarray(matrix, dtype=float)
    m = 0.5 * (m + m.T)
    evals = np.clip(np.linalg.eigvalsh(m), 0.0, None)
    s1 = float(evals.sum())
    s2 = float(np.sum(evals ** 2))
    if s2 <= 1e-18:
        return 0.0
    return (s1 ** 2) / s2


def _toeplitz_from(values: np.ndarray, d: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size < d:
        values = np.pad(values, (0, d - values.size), mode="edge")
    idx = np.abs(np.subtract.outer(np.arange(d), np.arange(d)))
    return values[idx]


def _graph_arrays(graph: GraphInstance) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adj = np.zeros((graph.n, graph.n), dtype=float)
    for u, v in graph.edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    deg = adj.sum(axis=1)
    lap = np.diag(deg) - adj
    return adj, deg, lap


def spectral_operator_matrix(
    graph: GraphInstance,
    p: int,
    rank: int | None = None,
    commutator_weight: float = 4.0,
) -> tuple[np.ndarray, float, float]:
    """Build the truncated noncommutative graph operator used by OST-QAOA.

    The construction maps graph spectra into a 2p-by-2p operator on QAOA angle
    coordinates. Two graph-derived generators are multiplied in both orders; the
    resulting commutator is used as a positive interaction term. Truncating the
    spectrum of this angle operator is the rank-control knob used in the paper.
    """
    if p <= 0:
        raise ValueError("p must be positive")
    d = 2 * p
    rank = d if rank is None else int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")

    _adj, deg, lap = _graph_arrays(graph)
    eigvals = np.linalg.eigvalsh(lap)
    denom = float(max(eigvals.max(), 1.0))
    scaled = eigvals / denom

    moments = np.array([np.mean(scaled ** (k + 1)) for k in range(d)], dtype=float)
    centered_deg = deg - deg.mean()
    deg_scale = float(max(np.linalg.norm(centered_deg), 1.0))
    degree_moments = np.array(
        [np.mean((centered_deg / deg_scale) ** (k + 1)) for k in range(d)],
        dtype=float,
    )
    features = graph_features(graph)
    feature_scale = np.array([1.0, 10.0, 10.0, 1.0, 10.0, 20.0, 20.0, 60.0])
    feature_vec = np.resize(np.tanh(features / feature_scale), d)

    a = _toeplitz_from(0.70 * moments + 0.30 * np.abs(feature_vec), d)
    b = _toeplitz_from(0.55 * degree_moments + 0.45 * feature_vec[::-1], d)
    b = np.roll(b, shift=1, axis=0)
    b = 0.5 * (b + b.T)

    commutator = a @ b - b @ a
    commutator_norm = float(np.linalg.norm(commutator, ord="fro"))
    interaction = commutator.T @ commutator
    raw = 0.55 * a + 0.45 * b + float(commutator_weight) * interaction
    raw = 0.5 * (raw + raw.T)

    evals, evecs = np.linalg.eigh(raw)
    order = np.argsort(evals)[::-1]
    keep = order[: min(rank, d)]
    kept = np.clip(evals[keep], 0.0, None)
    op = (evecs[:, keep] * kept) @ evecs[:, keep].T
    op = 0.5 * (op + op.T)
    effective_rank = float((kept.sum() ** 2) / max(np.sum(kept**2), 1e-12))
    return op, commutator_norm, effective_rank


def operator_signature(graph: GraphInstance, p: int, rank: int, commutator_weight: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    op, comm_norm, eff_rank = spectral_operator_matrix(graph, p, rank, commutator_weight)
    tri = op[np.triu_indices_from(op)]
    feat = graph_features(graph)
    raw = np.concatenate([feat, tri, [comm_norm, eff_rank]])
    return raw.astype(float), op, comm_norm, eff_rank


def make_objective(graph: GraphInstance, p: int) -> tuple[Callable[[np.ndarray], float], float]:
    cost_values = all_cut_values(graph.n, graph.edges)
    optimum, _ = exact_maxcut_bruteforce(graph.n, graph.edges)

    def objective(theta: np.ndarray) -> float:
        value = qaoa_expectation(graph.n, graph.edges, p, theta, cost_values=cost_values)
        return normalize_objective(value, optimum)

    return objective, float(optimum)


def _evaluate(theta: np.ndarray, objective: Callable[[np.ndarray], float], p: int) -> tuple[np.ndarray, float]:
    theta = project_angles(assert_theta(theta, p), p)
    return theta, float(objective(theta))


def directional_search(
    objective: Callable[[np.ndarray], float],
    p: int,
    budget: int,
    anchors: list[tuple[str, np.ndarray]],
    covariance: np.ndarray,
    *,
    directions: np.ndarray | None = None,
    scales: tuple[float, ...] = (1.0, 0.55, 0.28),
    trace_context: dict | None = None,
) -> SearchResult:
    """Search anchors, then probe covariance-ranked angle directions."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    d = 2 * p
    cov = np.asarray(covariance, dtype=float).reshape(d, d)
    cov = 0.5 * (cov + cov.T)
    if directions is None:
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvals = np.clip(eigvals[order], 1e-8, None)
        directions = eigvecs[:, order]
    else:
        directions = np.asarray(directions, dtype=float).reshape(d, -1)
        eigvals = np.clip(np.diag(directions.T @ cov @ directions), 1e-8, None)

    trace_context = trace_context or {}
    trace: list[dict] = []
    seen: set[bytes] = set()
    best_theta: np.ndarray | None = None
    best_y = -np.inf
    q = 0

    def record(theta: np.ndarray, source: str, step: float = 0.0, direction_index: int = -1) -> None:
        nonlocal best_theta, best_y, q
        if q >= budget:
            return
        theta, y = _evaluate(theta, objective, p)
        key = np.round(theta, 13).tobytes()
        if key in seen:
            return
        seen.add(key)
        q += 1
        if y > best_y or best_theta is None:
            best_y = y
            best_theta = theta.copy()
        row = dict(trace_context)
        row.update(
            {
                "query_index": q,
                "source": source,
                "direction_index": int(direction_index),
                "step_size": float(step),
                "objective_exact": float(y),
                "incumbent_objective": float(best_y),
            }
        )
        trace.append(row)

    for name, theta in anchors:
        record(theta, f"anchor:{name}")
        if q >= budget:
            break

    if best_theta is None:
        raise RuntimeError("no anchor could be evaluated")

    for scale in scales:
        for j in range(directions.shape[1]):
            if q >= budget:
                break
            direction = directions[:, j]
            norm = float(np.linalg.norm(direction))
            if norm <= 0.0:
                continue
            direction = direction / norm
            step = float(scale * np.sqrt(max(eigvals[min(j, eigvals.size - 1)], 1e-8)))
            for sign in (1.0, -1.0):
                if q >= budget:
                    break
                record(best_theta + sign * step * direction, "operator_refine", step=sign * step, direction_index=j)
    return SearchResult(best_theta.copy(), float(best_y), trace)


def optimize_reference_angles(graph: GraphInstance, p: int, budget: int, seed: int) -> tuple[np.ndarray, float]:
    """Budgeted offline optimizer used only to populate the training library."""
    objective, _ = make_objective(graph, p)
    rng = np.random.default_rng(seed)
    anchors: list[tuple[str, np.ndarray]] = [
        ("tqa", theta_tqa(p)),
        ("global", theta_global(p)),
    ]
    for idx in range(max(0, min(10, budget - len(anchors)))):
        anchors.append((f"random{idx}", random_theta(p, rng)))
    covariance = np.eye(2 * p) * 0.55
    result = directional_search(objective, p, budget, anchors, covariance, scales=(0.9, 0.45, 0.22))
    return result.theta_hat, result.y_hat


def build_training_core(
    p: int = 3,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    train_per_family: int = 6,
    optimizer_budget: int = 42,
    seed: int = DEFAULT_SEED,
) -> TrainingCore:
    """Optimise the rank-independent reference targets once and cache them."""
    key = (p, tuple(families), tuple(int(s) for s in sizes), int(train_per_family), int(optimizer_budget), int(seed))
    cached = _CORE_CACHE.get(key)
    if cached is not None:
        return cached
    graphs: list[GraphInstance] = []
    thetas: list[np.ndarray] = []
    ratios: list[float] = []
    for family in families:
        for idx in range(train_per_family):
            n = int(sizes[idx % len(sizes)])
            graph_seed = stable_seed(seed, "train", family, idx, n)
            graph = generate_graph(family, n, graph_seed)
            theta, ratio = optimize_reference_angles(
                graph,
                p,
                optimizer_budget,
                stable_seed(seed, "opt", family, idx, n),
            )
            graphs.append(graph)
            thetas.append(np.asarray(theta, dtype=float))
            ratios.append(float(ratio))
    core = TrainingCore(
        p=p,
        optimizer_budget=int(optimizer_budget),
        seed=int(seed),
        graphs=tuple(graphs),
        thetas=tuple(thetas),
        ratios=tuple(ratios),
    )
    _CORE_CACHE[key] = core
    return core


def build_operator_library(
    p: int = 3,
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    train_per_family: int = 6,
    rank: int = 4,
    commutator_weight: float = 4.0,
    optimizer_budget: int = 42,
    seed: int = DEFAULT_SEED,
) -> OperatorLibrary:
    core = build_training_core(
        p,
        families=families,
        sizes=sizes,
        train_per_family=train_per_family,
        optimizer_budget=optimizer_budget,
        seed=seed,
    )
    records: list[TrainingRecord] = []
    for graph, theta, ratio in zip(core.graphs, core.thetas, core.ratios):
        sig, op, comm_norm, _eff_rank = operator_signature(graph, p, rank, commutator_weight)
        records.append(
            TrainingRecord(
                graph=graph,
                theta=np.asarray(theta, dtype=float),
                ratio=float(ratio),
                signature=sig,
                operator=op,
                commutator_norm=float(comm_norm),
            )
        )
    signatures = np.vstack([r.signature for r in records])
    mean = signatures.mean(axis=0)
    scale = signatures.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return OperatorLibrary(
        p=p,
        rank=rank,
        commutator_weight=float(commutator_weight),
        records=tuple(records),
        signature_mean=mean,
        signature_scale=scale,
    )


def operator_prior(
    graph: GraphInstance,
    library: OperatorLibrary,
    *,
    k: int = 6,
    tqa_blend: float = 0.25,
    residual_floor: float = 0.025,
    operator_weight: float = 0.8,
) -> OperatorPrior:
    p = library.p
    d = 2 * p
    sig, op, comm_norm, eff_rank = operator_signature(graph, p, library.rank, library.commutator_weight)
    z = (sig - library.signature_mean) / library.signature_scale
    train = np.vstack([(r.signature - library.signature_mean) / library.signature_scale for r in library.records])
    distances = np.linalg.norm(train - z[None, :], axis=1)
    k = max(1, min(int(k), len(library.records)))
    idx = np.argsort(distances)[:k]
    bandwidth = max(float(np.median(distances[idx])), 1e-6)
    weights = np.exp(-0.5 * (distances[idx] / bandwidth) ** 2)
    weights = weights / weights.sum()
    theta_stack = np.vstack([library.records[i].theta for i in idx])
    learned_mean = weights @ theta_stack
    mean = (1.0 - tqa_blend) * learned_mean + tqa_blend * theta_tqa(p)
    mean = project_angles(mean, p)

    # The search frame is the eigenbasis of this covariance. The truncated
    # noncommutative operator is the dominant term so that the spectral
    # truncation parameter and the commutator actually shape the directions the
    # query policy probes; the neighbour scatter supplies a data-driven floor.
    residuals = theta_stack - mean[None, :]
    neighbor_cov = sum(float(w) * np.outer(r, r) for w, r in zip(weights, residuals))
    neighbor_scale = neighbor_cov / max(float(np.trace(neighbor_cov)), 1e-8)
    op_scale = op / max(float(np.trace(op)), 1e-8)
    covariance = (
        float(operator_weight) * op_scale
        + (1.0 - 0.5 * float(operator_weight)) * neighbor_scale
        + np.eye(d) * residual_floor
    )
    covariance = 0.5 * (covariance + covariance.T)
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 1e-8, None)
    evecs = evecs[:, order]
    return OperatorPrior(
        mean=mean,
        covariance=covariance,
        eigvals=evals,
        eigvecs=evecs,
        weights=weights,
        neighbor_indices=idx,
        commutator_norm=float(comm_norm),
        effective_rank=float(eff_rank),
    )


def ost_qaoa_search(
    graph: GraphInstance,
    library: OperatorLibrary,
    *,
    budget: int = 24,
    k: int = 6,
    tqa_blend: float = 0.25,
    operator_weight: float = 0.8,
    trace_context: dict | None = None,
) -> SearchResult:
    objective, _ = make_objective(graph, library.p)
    prior = operator_prior(graph, library, k=k, tqa_blend=tqa_blend, operator_weight=operator_weight)
    nearest = library.records[int(prior.neighbor_indices[0])].theta
    anchors = [
        ("tqa", theta_tqa(library.p)),
        ("operator_mean", prior.mean),
        ("nearest_operator_neighbor", nearest),
        ("global", theta_global(library.p)),
    ]
    return directional_search(
        objective,
        library.p,
        budget,
        anchors,
        prior.covariance,
        directions=prior.eigvecs,
        trace_context=trace_context,
    )


def diagonal_operator_search(
    graph: GraphInstance,
    library: OperatorLibrary,
    *,
    budget: int = 24,
    k: int = 6,
    tqa_blend: float = 0.25,
    operator_weight: float = 0.8,
    trace_context: dict | None = None,
) -> SearchResult:
    objective, _ = make_objective(graph, library.p)
    prior = operator_prior(graph, library, k=k, tqa_blend=tqa_blend, operator_weight=operator_weight)
    nearest = library.records[int(prior.neighbor_indices[0])].theta
    anchors = [
        ("tqa", theta_tqa(library.p)),
        ("operator_mean", prior.mean),
        ("nearest_operator_neighbor", nearest),
        ("global", theta_global(library.p)),
    ]
    return directional_search(
        objective,
        library.p,
        budget,
        anchors,
        np.diag(np.diag(prior.covariance)),
        trace_context=trace_context,
    )


def knn_refine_search(graph: GraphInstance, library: OperatorLibrary, *, budget: int = 24, k: int = 6, trace_context: dict | None = None) -> SearchResult:
    objective, _ = make_objective(graph, library.p)
    prior = operator_prior(graph, library, k=k, tqa_blend=0.0, residual_floor=0.05)
    anchors = [("tqa", theta_tqa(library.p)), ("knn_mean", prior.mean), ("global", theta_global(library.p))]
    return directional_search(
        objective,
        library.p,
        budget,
        anchors,
        np.diag(np.diag(prior.covariance)),
        trace_context=trace_context,
    )


def evaluate_methods_on_graph(
    graph: GraphInstance,
    library: OperatorLibrary,
    *,
    budget: int,
    seed: int,
) -> dict[str, SearchResult]:
    objective, _ = make_objective(graph, library.p)
    context = {"graph_id": graph.graph_id, "family": graph.family, "n": graph.n}
    return {
        "Random": random_search(objective, library.p, budget, seed=seed),
        "TQA": tqa_refine(objective, library.p, 1),
        "TQA+coordinate": tqa_refine(objective, library.p, budget),
        "kNN+coordinate": knn_refine_search(graph, library, budget=budget, trace_context=context),
        "OST diagonal": diagonal_operator_search(graph, library, budget=budget, trace_context=context),
        "OST-QAOA": ost_qaoa_search(graph, library, budget=budget, trace_context=context),
    }


def build_test_graphs(
    *,
    families: tuple[str, ...] = DEFAULT_FAMILIES,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    test_per_family: int = 4,
    seed: int = DEFAULT_SEED,
) -> list[GraphInstance]:
    graphs: list[GraphInstance] = []
    for family in families:
        for idx in range(test_per_family):
            n = int(sizes[(idx + 1) % len(sizes)])
            graphs.append(generate_graph(family, n, stable_seed(seed, "test", family, idx, n)))
    return graphs
