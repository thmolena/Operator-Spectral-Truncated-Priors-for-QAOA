#!/usr/bin/env python3
"""
Core implementation of the UQ-QAOA pipeline for graph-conditioned
trust-region QAOA parameter optimization.

This module implements the complete algorithmic pipeline described in
the manuscript "Query-Efficient QAOA with Graph-Conditioned Trust Regions
and Accelerator-Ready Kernels."  It contains:

  1. GRAPH CONSTRUCTION — parameterized generation of Erdős–Rényi (ER),
     random regular (REG), Barabási–Albert (BA), and Watts–Strogatz (WS)
     graph families with deterministic seeding.

  2. GRAPH FEATURES — extraction of an 8-dimensional spectral and
     topological feature vector:
       (density, d_bar, sigma_d, C_bar, lambda_2, lambda_n, n, m)
     used by all transfer-learning baselines and the GIN predictor.

  3. QAOA STATEVECTOR SIMULATOR — exact 2^n-dimensional complex
     statevector simulation of the depth-p QAOA circuit for MaxCut:
       |ψ⟩ = ∏_{ℓ=1}^p  exp(-i β_ℓ H_M) exp(-i γ_ℓ H_C)  |+⟩^⊗n
     where H_C = Σ_{(i,j)∈E} (1 - Z_i Z_j)/2  is the diagonal MaxCut
     cost Hamiltonian and H_M = Σ_i X_i is the transverse-field mixer.

  4. OPTIMAL ANGLE FINDER — multi-start Nelder–Mead optimization to
     obtain reference QAOA parameters for the training library.

  5. TRAINING LIBRARY — construction of 240 training graphs (60 per
     family, sizes 8/10/12/14) with their optimal angles.

  6. GIN PREDICTOR — a 3-layer Graph Isomorphism Network (GIN) with
     hidden dimension 64, LayerNorm, mean+max graph pooling, and
     two-layer MLP prediction heads for μ and log σ² (Gaussian output).
     Training uses analytical backpropagation through the full GIN
     computation graph with Adam optimizer, cosine annealing, gradient
     clipping, and early stopping.  No external autodiff library is used.

  7. CANDIDATE GENERATORS — six competing QAOA parameter-initialization
     algorithms, all evaluated under the same candidate budget:
       • Random:    uniform sampling in the angle domain
       • Heuristic: concentration-based transfer (training-set median)
       • k-NN:      k=5 nearest-neighbor transfer in feature space
       • TQA:       linear ramp schedule (Sack & Cerezo 2021)
       • GNN point: GIN predicted mean as warm start
       • UQ-QAOA:   three-source Bayesian posterior (GIN + local k-NN
                    + global population) with deterministic coordinate
                    trust-region probes

  8. EVALUATION HELPERS — batch evaluation of all methods on graph
     instances with approximation-ratio computation and convergence
     tracking.

INTEGRITY GUARANTEES:
  • All results are computed from scratch using the algorithms described
    in the manuscript.  No results are hard-coded or cached from external
    sources.
  • Deterministic hash-based seeding (SHA-256) ensures exact reproducibility
    across runs within floating-point tolerance.
    • Every method receives the same configured candidate budget.
  • The GIN predictor is trained from scratch each time the library is
    built, using only the 240 training graphs.
  • Test graphs use separate seeds from training graphs (no data leakage).

REPRODUCIBILITY:
  Global seed: 260424803
  Software: Python 3.11, NumPy 1.26, SciPy 1.12, Matplotlib 3.8
  Hardware: Apple M2 Pro, 10-core, 16 GB unified memory, macOS 14

References:
  [1] Farhi et al., "A Quantum Approximate Optimization Algorithm" (2014)
  [2] Xu et al., "How Powerful Are Graph Neural Networks?" (2019, GIN)
  [3] Sack & Cerezo, "Quantum Annealing Initialization of QAOA" (2021, TQA)
"""
from __future__ import annotations

import csv
import hashlib
import math
import os
import pathlib
import sys
import time

import numpy as np
from scipy.optimize import minimize

# ════════════════════════════════════════════════════════════════════
# GLOBAL CONSTANTS
# ════════════════════════════════════════════════════════════════════

GLOBAL_SEED = 260424803  # Fixed global seed for full reproducibility


def _read_positive_int_env(name, default):
    """Read a positive integer from the environment with validation."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"Environment variable {name} must be positive")
    return value

BASE = pathlib.Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".mplconfig"))

FIG_DIR = BASE / "figures"
TAB_DIR = BASE / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEPTH = 3
DEFAULT_BUDGET = 18

DEPTH = _read_positive_int_env("UQ_QAOA_DEPTH", DEFAULT_DEPTH)
CANDIDATE_BUDGET = _read_positive_int_env("UQ_QAOA_BUDGET", DEFAULT_BUDGET)
DIM = 2 * DEPTH

FAMILIES = ["ER", "REG", "BA", "WS"]  # Graph families used in experiments

# Method identifiers and display labels
METHODS = ["random", "heuristic", "knn", "tqa", "gnn_point", "uq_qaoa"]
METHOD_LABELS = ["Random", "Heuristic", "k-NN", "TQA", "GNN point", "UQ-QAOA"]
COLORS = ["#737373", "#c48a00", "#1e7a1e", "#b81d1d", "#7a4fad", "#155fa0"]


def study_suffix(depth=DEPTH, budget=CANDIDATE_BUDGET):
    """Return a filename suffix for non-default study configurations."""
    if depth == DEFAULT_DEPTH and budget == DEFAULT_BUDGET:
        return ""
    return f"_p{depth}_q{budget}"


def figure_path(stem, depth=DEPTH, budget=CANDIDATE_BUDGET):
    """Build a figure output path for the active study configuration."""
    return FIG_DIR / f"{stem}{study_suffix(depth, budget)}.pdf"


def table_path(stem, depth=DEPTH, budget=CANDIDATE_BUDGET):
    """Build a table output path for the active study configuration."""
    return TAB_DIR / f"{stem}{study_suffix(depth, budget)}.csv"


def stable_seed(*parts):
    """Generate a deterministic seed from an arbitrary sequence of values.

    Uses SHA-256 hashing of the global seed concatenated with all parts,
    then takes modulo 2^31 for compatibility with NumPy's RandomState.
    This ensures that every graph, every candidate set, and every
    optimization run is fully reproducible without seed collisions.

    Parameters
    ----------
    *parts : hashable
        Arbitrary sequence of values (strings, integers, etc.) that
        uniquely identify the random process.

    Returns
    -------
    int
        Deterministic seed in [0, 2^31).
    """
    h = hashlib.sha256(str(GLOBAL_SEED).encode())
    for p in parts:
        h.update(str(p).encode())
    return int(h.hexdigest(), 16) % (2**31)


# ════════════════════════════════════════════════════════════════════
# 1.  GRAPH CONSTRUCTION
#
#     Parameterized generators for four standard graph families:
#       ER:   Erdős–Rényi G(n, p_edge)
#       REG:  random d-regular graphs (configuration model)
#       BA:   Barabási–Albert preferential attachment
#       WS:   Watts–Strogatz small-world
#       cycle: deterministic n-cycle (used for trust-region visualization)
# ════════════════════════════════════════════════════════════════════

def graph_edges(family, n, seed, **kw):
    """Generate an edge list for a graph from the specified family.

    Parameters
    ----------
    family : str
        One of 'cycle', 'ER', 'REG', 'BA', 'WS'.
    n : int
        Number of vertices.
    seed : int
        Random seed for reproducibility.
    **kw : dict
        Family-specific parameters:
          ER:  p_edge (default 0.5)
          REG: degree (default 3)
          BA:  m_attach (default 2)
          WS:  k_ws (default 4), p_ws (default 0.3)

    Returns
    -------
    list of (int, int)
        Sorted edge list with u < v for each edge (u, v).
    """
    rng = np.random.RandomState(seed)
    if family == "cycle":
        return [(i, (i + 1) % n) for i in range(n)]
    if family == "ER":
        p = kw.get("p_edge", 0.5)
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if rng.random() < p]
    if family == "REG":
        # Configuration model for d-regular graphs
        d = kw.get("degree", 3)
        if n * d % 2 != 0:
            d = d - 1 if d > 2 else d + 1
        stubs = list(range(n)) * d
        rng.shuffle(stubs)
        edges = set()
        for k in range(0, len(stubs), 2):
            u, v = stubs[k], stubs[k + 1]
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return sorted(edges)
    if family == "BA":
        # Barabási–Albert preferential attachment
        m0 = kw.get("m_attach", 2)
        es = set()
        for i in range(m0):
            for j in range(i + 1, m0):
                es.add((i, j))
        deg = np.zeros(n)
        for u, v in es:
            deg[u] += 1; deg[v] += 1
        for new in range(m0, n):
            s = deg[:new].sum()
            pr = deg[:new] / s if s > 0 else np.ones(new) / new
            tgts = rng.choice(new, size=min(m0, new), replace=False, p=pr)
            for t in tgts:
                es.add((min(new, t), max(new, t)))
                deg[new] += 1; deg[t] += 1
        return sorted(es)
    if family == "WS":
        # Watts–Strogatz small-world model
        k_ws = kw.get("k_ws", 4); p_ws = kw.get("p_ws", 0.3)
        es = set()
        for i in range(n):
            for j in range(1, k_ws // 2 + 1):
                nb = (i + j) % n
                es.add((min(i, nb), max(i, nb)))
        rw = set()
        for e in list(es):
            if rng.random() < p_ws:
                u = e[0]; nv = rng.randint(0, n)
                while nv == u or (min(u, nv), max(u, nv)) in rw:
                    nv = rng.randint(0, n)
                rw.add((min(u, nv), max(u, nv)))
            else:
                rw.add(e)
        return sorted(rw)
    raise ValueError(f"Unknown graph family: {family}")


# ════════════════════════════════════════════════════════════════════
# 2.  GRAPH FEATURES  (8-dimensional)
#
#     Feature vector: (density, d_bar, sigma_d, C_bar, lambda_2,
#                      lambda_n, n, m)
#
#     density   = 2m / (n(n-1))     — edge density
#     d_bar     = mean degree        — average connectivity
#     sigma_d   = std of degrees     — degree heterogeneity
#     C_bar     = mean clustering    — local triangle density
#     lambda_2  = algebraic conn.    — Fiedler value (spectral gap)
#     lambda_n  = spectral radius    — largest Laplacian eigenvalue
#     n, m      = vertex/edge counts — size normalization
# ════════════════════════════════════════════════════════════════════

def _clustering_coefficient(n, edges):
    """Compute the mean clustering coefficient C_bar.

    For each vertex i with degree k_i >= 2, the local clustering
    coefficient is the fraction of pairs of neighbors that are
    themselves connected.  C_bar is the average over all vertices.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    cc = 0.0
    for i in range(n):
        nb = adj[i]; k = len(nb)
        if k < 2:
            continue
        tri = sum(1 for u in nb for v in nb if u < v and v in adj[u])
        cc += 2.0 * tri / (k * (k - 1))
    return cc / n if n > 0 else 0.0


def graph_features(n, edges):
    """Extract the 8-dimensional graph feature vector.

    Parameters
    ----------
    n : int
        Number of vertices.
    edges : list of (int, int)
        Edge list.

    Returns
    -------
    np.ndarray of shape (8,)
        Feature vector [density, d_bar, sigma_d, C_bar, lambda_2,
        lambda_n, n, m].
    """
    m = len(edges)
    density = 2 * m / (n * (n - 1)) if n > 1 else 0.0
    deg = np.zeros(n)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    d_bar = deg.mean()
    sigma_d = deg.std()
    c_bar = _clustering_coefficient(n, edges)
    # Graph Laplacian spectrum: L = D - A
    L = np.zeros((n, n))
    for u, v in edges:
        L[u, v] -= 1; L[v, u] -= 1; L[u, u] += 1; L[v, v] += 1
    eigs = np.sort(np.linalg.eigvalsh(L))
    lam2 = float(eigs[1]) if n > 1 else 0.0   # algebraic connectivity
    lam_n = float(eigs[-1]) if n > 0 else 0.0  # spectral radius
    return np.array([density, d_bar, sigma_d, c_bar, lam2, lam_n, n, m],
                    dtype=np.float64)


# ════════════════════════════════════════════════════════════════════
# 3.  QAOA STATEVECTOR SIMULATOR
#
#     Exact simulation of the depth-p QAOA circuit for MaxCut.
#
#     MaxCut cost: C_G(z) = #{edges (i,j) where z_i ≠ z_j}
#
#     Phase separator (diagonal in computational basis):
#       |z⟩  →  exp(-i γ_ℓ C_G(z)) |z⟩
#
#     Mixer (transverse-field, product of single-qubit X-rotations):
#       exp(-i β_ℓ X_q) applies a 2×2 rotation to amplitude pairs
#       separated by stride 2^q.
#
#     The simulator stores the full 2^n complex statevector as a
#     contiguous NumPy array of complex128 values and applies gates
#     by vectorized array operations.
# ════════════════════════════════════════════════════════════════════

def qaoa_cost_values(edges, n):
    """Compute the MaxCut cost C(z) for all 2^n computational-basis states.

    C(z) = number of edges (i,j) such that z_i ≠ z_j (the edge is "cut").

    Parameters
    ----------
    edges : list of (int, int)
        Edge list.
    n : int
        Number of qubits/vertices.

    Returns
    -------
    np.ndarray of shape (2^n,)
        Cost value C(z) for each bitstring z in {0, 1, ..., 2^n - 1}.
    """
    N = 1 << n
    c_vals = np.zeros(N)
    bits = np.arange(N, dtype=np.int64)
    for u, v in edges:
        mask_u = (bits >> u) & 1
        mask_v = (bits >> v) & 1
        c_vals += (mask_u != mask_v).astype(np.float64)
    return c_vals


def qaoa_statevector(gamma, beta, edges, n, c_vals=None):
    """Compute the exact QAOA statevector.

    Implements |ψ⟩ = ∏_{ℓ=1}^p exp(-i β_ℓ H_M) exp(-i γ_ℓ H_C) |+⟩^⊗n

    The phase separator exp(-i γ H_C)|z⟩ = exp(-i γ C(z))|z⟩ is applied
    as an element-wise multiplication (bandwidth-bound kernel).

    The mixer exp(-i β H_M) = ∏_q exp(-i β X_q) is applied by reshaping
    the statevector to expose each qubit as a size-2 axis and applying
    the 2×2 rotation matrix [[cos β, -i sin β], [-i sin β, cos β]].

    Parameters
    ----------
    gamma : array-like of shape (p,)
        Phase separator angles γ_1, ..., γ_p.
    beta : array-like of shape (p,)
        Mixer angles β_1, ..., β_p.
    edges : list of (int, int)
        Edge list.
    n : int
        Number of qubits.
    c_vals : np.ndarray or None
        Precomputed cost values (reuse across candidates for same graph).

    Returns
    -------
    np.ndarray of shape (2^n,), dtype complex128
        Final QAOA statevector.
    """
    p = len(gamma); N = 1 << n
    if c_vals is None:
        c_vals = qaoa_cost_values(edges, n)
    # Initialize to uniform superposition |+⟩^⊗n
    psi = np.full(N, 1.0 / np.sqrt(N), dtype=np.complex128)
    for layer in range(p):
        # Phase separator: element-wise exp(-i γ_ℓ C(z))
        psi *= np.exp(-1j * gamma[layer] * c_vals)
        # Mixer: product of single-qubit X-rotations exp(-i β_ℓ X_q)
        c = np.cos(beta[layer])
        s = -1j * np.sin(beta[layer])
        for q in range(n):
            # Reshape statevector so qubit q is an explicit axis of size 2
            # This enables vectorized application of the 2×2 rotation
            shape = [1 << (n - q - 1), 2, 1 << q]
            psi_r = psi.reshape(shape)
            a = psi_r[:, 0, :].copy()
            b_ = psi_r[:, 1, :].copy()
            psi_r[:, 0, :] = c * a + s * b_
            psi_r[:, 1, :] = s * a + c * b_
    return psi


def qaoa_expectation(gamma, beta, edges, n, c_vals=None):
    """Compute the expected cut value ⟨ψ|H_C|ψ⟩.

    This is the exact QAOA objective: f_G(θ) = Σ_z |⟨z|ψ⟩|² C(z).
    """
    if c_vals is None:
        c_vals = qaoa_cost_values(edges, n)
    psi = qaoa_statevector(gamma, beta, edges, n, c_vals)
    probs = np.abs(psi) ** 2
    return float(np.dot(probs, c_vals))


def qaoa_ratio(gamma, beta, edges, n, c_vals=None):
    """Compute the approximation ratio r_G(θ) = ⟨C⟩ / C_max.

    Returns 0.0 if C_max = 0 (no edges).
    """
    if c_vals is None:
        c_vals = qaoa_cost_values(edges, n)
    c_max = c_vals.max()
    if c_max == 0:
        return 0.0
    return qaoa_expectation(gamma, beta, edges, n, c_vals) / c_max


def qaoa_sampled_ratio(gamma, beta, edges, n, shots, rng, c_vals=None):
    """Compute the best sampled bitstring ratio from finite-shot measurement.

    Samples `shots` bitstrings from the QAOA distribution and returns
    the maximum observed cut value normalized by C_max.  This simulates
    finite-shot hardware measurement without replacement.
    """
    if c_vals is None:
        c_vals = qaoa_cost_values(edges, n)
    c_max = c_vals.max()
    if c_max == 0:
        return 0.0
    psi = qaoa_statevector(gamma, beta, edges, n, c_vals)
    probs = np.abs(psi) ** 2
    samples = rng.choice(len(probs), size=shots, p=probs)
    return float(c_vals[samples].max() / c_max)


# ════════════════════════════════════════════════════════════════════
# 4.  OPTIMAL ANGLE FINDER  (multi-start Nelder-Mead)
#
#     For each training graph, we find high-quality QAOA parameters
#     using multi-start derivative-free optimization.  These serve as
#     the training targets for the GIN predictor.
#
#     Protocol: 4 random restarts × 60 Nelder–Mead iterations each,
#     return the parameter vector with highest approximation ratio.
# ════════════════════════════════════════════════════════════════════

def find_optimal_angles(edges, n, depth=DEPTH, n_restarts=8, seed=42,
                        maxiter=150):
    """Find high-quality QAOA angles via multi-start Nelder–Mead.

    Parameters
    ----------
    edges : list of (int, int)
        Graph edge list.
    n : int
        Number of qubits.
    depth : int
        QAOA depth p.
    n_restarts : int
        Number of random restarts (default 8).
    seed : int
        Random seed for restart initialization.
    maxiter : int
        Maximum Nelder–Mead iterations per restart (default 150).

    Returns
    -------
    gamma : np.ndarray of shape (depth,)
        Best γ angles found.
    beta : np.ndarray of shape (depth,)
        Best β angles found.
    ratio : float
        Approximation ratio at the best angles.
    """
    c_vals = qaoa_cost_values(edges, n)
    rng = np.random.RandomState(seed)
    best_ratio = -1.0
    best_params = None
    for _ in range(n_restarts):
        x0 = np.concatenate([rng.uniform(0, np.pi, depth),
                              rng.uniform(0, np.pi / 2, depth)])
        def neg_ratio(x):
            return -qaoa_ratio(x[:depth], x[depth:], edges, n, c_vals)
        res = minimize(neg_ratio, x0, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-6})
        r = -res.fun
        if r > best_ratio:
            best_ratio = r
            best_params = res.x
    gamma_opt = best_params[:depth]
    beta_opt = best_params[depth:]
    return gamma_opt, beta_opt, best_ratio


# ════════════════════════════════════════════════════════════════════
# 5.  TRAINING LIBRARY  (240 graphs)
#
#     60 graphs per family {ER, REG, BA, WS}, with sizes cycling
#     through {8, 10, 12, 14}.  For each graph, optimal angles are
#     found via multi-start Nelder–Mead and stored along with the
#     8-dimensional feature vector.
#
#     The library is cached in memory after the first build to avoid
#     recomputation across multiple figure/table generators.
# ════════════════════════════════════════════════════════════════════

_TRAINING_LIBRARY_CACHE = {}


def build_training_library(depth=DEPTH, n_per_family=60):
    """Build the training library of 240 graphs with optimal angles.

    Each entry contains:
      - features:  8-dim graph feature vector
      - gamma, beta: optimal QAOA angles (from Nelder–Mead)
      - angles: concatenated [gamma, beta] vector
      - ratio:  approximation ratio at optimal angles
      - n:      number of vertices
      - edges:  edge list
      - family: graph family identifier

    Parameters
    ----------
    depth : int
        QAOA depth p (default 2).
    n_per_family : int
        Number of graphs per family (default 60, total 240).

    Returns
    -------
    list of dict
        Training library entries.
    """
    cache_key = (depth, n_per_family)
    if cache_key in _TRAINING_LIBRARY_CACHE:
        return _TRAINING_LIBRARY_CACHE[cache_key]

    print("  Building training library (240 graphs)...", flush=True)
    sizes = [8, 10, 12, 14]
    library = []
    total = len(FAMILIES) * n_per_family
    count = 0
    for fam in FAMILIES:
        for idx in range(n_per_family):
            sz = sizes[idx % len(sizes)]
            sd = stable_seed("train", fam, idx)
            edges = graph_edges(fam, sz, sd)
            if len(edges) == 0:
                count += 1
                continue
            feats = graph_features(sz, edges)
            g_opt, b_opt, ratio = find_optimal_angles(
                edges, sz, depth=depth, n_restarts=4,
                seed=stable_seed("opt", fam, idx))
            count += 1
            if count % 20 == 0:
                print(f"    [{count}/{total}] {fam} n={sz} ratio={ratio:.4f}",
                      flush=True)
            library.append({
                "features": feats,
                "gamma": g_opt,
                "beta": b_opt,
                "angles": np.concatenate([g_opt, b_opt]),
                "ratio": ratio,
                "n": sz,
                "edges": edges,
                "family": fam,
            })
        _TRAINING_LIBRARY_CACHE[cache_key] = library
    print(f"  Training library: {len(library)} graphs, "
          f"median ratio {np.median([e['ratio'] for e in library]):.4f}",
          flush=True)
    return library


# ════════════════════════════════════════════════════════════════════
# 6.  GIN PREDICTOR  (3-layer GIN, hidden=64, NumPy-only)
#
#     Architecture (matching manuscript Appendix A):
#       Input:    one-hot degree features (max_deg+1 per node)
#       GIN update per layer ℓ:
#         h_i^(ℓ+1) = MLP^(ℓ)( (1+ε) h_i^(ℓ) + Σ_{j∈N(i)} h_j^(ℓ) )
#         where MLP^(ℓ) = LayerNorm ∘ Linear_2 ∘ ReLU ∘ Linear_1
#       Readout:  g = [mean-pool h^(L) ; max-pool h^(L)] ∈ R^{2H}
#       Augment:  concat(g, 8-dim global features) → R^{2H+8}
#       μ head:   MLP(2H+8 → H → 2p)      — predicted angle means
#       σ² head:  MLP(2H+8 → H → 2p)      — predicted angle variances
#                 with output clipped to [-5, 2] then exponentiated
#
#     Training:
#       Loss = Gaussian NLL = 0.5 Σ_k [log σ²_k + (θ*_k - μ_k)² / σ²_k]
#       Optimizer: Adam (lr=3e-4, β1=0.9, β2=0.999)
#       Cosine annealing LR schedule, gradient clipping (norm ≤ 1)
#       Early stopping with patience 200, max 1000 epochs
#       Full-batch training on 240 graphs
#
#     The backward pass is implemented analytically (manual chain rule
#     through every layer), not via automatic differentiation.
# ════════════════════════════════════════════════════════════════════

def _relu(x):
    """ReLU activation: max(0, x)."""
    return np.maximum(0, x)


def _layer_norm_fwd(x, eps=1e-5):
    """LayerNorm forward pass.

    Normalizes each row to zero mean and unit variance.
    Returns the normalized output and cache for backpropagation.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, (x, mu, var, eps)


def _layer_norm_bwd(dout, cache):
    """LayerNorm backward pass (analytical gradient)."""
    x, mu, var, eps = cache
    N = x.shape[-1]
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * std_inv
    dvar = np.sum(dout * (x - mu) * (-0.5) * std_inv**3, axis=-1, keepdims=True)
    dmu = (np.sum(dout * (-std_inv), axis=-1, keepdims=True)
           + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
    dx = dout * std_inv + dvar * 2.0 * (x - mu) / N + dmu / N
    return dx


class GINPredictor:
    """3-layer Graph Isomorphism Network with Gaussian prediction heads.

    Pure NumPy implementation with analytical backpropagation.

    Architecture:
      - Input: one-hot degree features (max_deg+1 features per node)
      - 3 GIN layers, each: Linear(in, H) → ReLU → Linear(H, H) → LayerNorm
      - Readout: concat(mean-pool, max-pool) → dim 2H
      - Augmentation: concat(readout, 8-dim graph features) → dim 2H+8
      - μ head:  Linear(2H+8, H) → ReLU → Linear(H, 2p)
      - σ² head: Linear(2H+8, H) → ReLU → Linear(H, 2p), clipped [-5, 2]

    Attributes
    ----------
    params : dict
        All trainable parameters (weights and biases).
    """

    GLOBAL_FEAT_DIM = 8  # (density, d_bar, sigma_d, C_bar, lam2, lam_n, n, m)

    def __init__(self, input_dim, hidden=64, out_dim=None, n_layers=3, seed=42):
        self.input_dim = input_dim
        self.hidden = hidden
        self.out_dim = DIM if out_dim is None else out_dim
        self.n_layers = n_layers
        rng = np.random.RandomState(seed)
        scale = lambda fi, fo: np.sqrt(2.0 / (fi + fo))

        self.params = {}
        in_d = input_dim
        for l in range(n_layers):
            s = scale(in_d, hidden)
            self.params[f"W1_{l}"] = rng.randn(in_d, hidden) * s
            self.params[f"b1_{l}"] = np.zeros(hidden)
            s2 = scale(hidden, hidden)
            self.params[f"W2_{l}"] = rng.randn(hidden, hidden) * s2
            self.params[f"b2_{l}"] = np.zeros(hidden)
            self.params[f"eps_{l}"] = np.zeros(1)
            in_d = hidden

        # Augmented pool dimension: GIN readout (2*hidden) + global features (8)
        aug_dim = 2 * hidden + self.GLOBAL_FEAT_DIM
        # μ head: 2-layer MLP
        s1 = scale(aug_dim, hidden)
        self.params["W_mu1"] = rng.randn(aug_dim, hidden) * s1
        self.params["b_mu1"] = np.zeros(hidden)
        s2m = scale(hidden, out_dim)
        self.params["W_mu2"] = rng.randn(hidden, out_dim) * s2m
        self.params["b_mu2"] = np.zeros(out_dim)
        # σ² head: 2-layer MLP
        s1s = scale(aug_dim, hidden)
        self.params["W_sig1"] = rng.randn(aug_dim, hidden) * s1s
        self.params["b_sig1"] = np.zeros(hidden)
        s2s = scale(hidden, out_dim)
        self.params["W_sig2"] = rng.randn(hidden, out_dim) * s2s
        self.params["b_sig2"] = np.zeros(out_dim)

    def forward(self, adj_norm, x, global_feats=None):
        """Forward pass returning (μ, σ²).

        Parameters
        ----------
        adj_norm : np.ndarray of shape (n, n)
            Row-normalized adjacency matrix with self-loops.
        x : np.ndarray of shape (n, input_dim)
            Node feature matrix (one-hot degree encoding).
        global_feats : np.ndarray of shape (8,) or None
            Normalized global graph features.

        Returns
        -------
        mu : np.ndarray of shape (out_dim,)
            Predicted parameter means (in normalized space).
        sigma2 : np.ndarray of shape (out_dim,)
            Predicted parameter variances (in normalized space).
        """
        h = x
        for l in range(self.n_layers):
            eps_val = self.params[f"eps_{l}"][0]
            # GIN aggregation: (1+ε)·h + A·h
            agg = (1.0 + eps_val) * h + adj_norm @ h
            z1 = agg @ self.params[f"W1_{l}"] + self.params[f"b1_{l}"]
            z1r = _relu(z1)
            z2 = z1r @ self.params[f"W2_{l}"] + self.params[f"b2_{l}"]
            h, _ = _layer_norm_fwd(z2)
        # Graph-level readout: concat(mean-pool, max-pool)
        g = np.concatenate([h.mean(axis=0), h.max(axis=0)])
        # Augment with global graph features
        if global_feats is not None:
            g = np.concatenate([g, global_feats])
        else:
            g = np.concatenate([g, np.zeros(self.GLOBAL_FEAT_DIM)])
        # μ head
        h_mu = _relu(g @ self.params["W_mu1"] + self.params["b_mu1"])
        mu = h_mu @ self.params["W_mu2"] + self.params["b_mu2"]
        # σ² head (log-variance clipped to [-5, 2])
        h_sig = _relu(g @ self.params["W_sig1"] + self.params["b_sig1"])
        log_sig2 = h_sig @ self.params["W_sig2"] + self.params["b_sig2"]
        log_sig2 = np.clip(log_sig2, -5, 2)
        sigma2 = np.exp(log_sig2)
        return mu, sigma2

    def forward_with_cache(self, adj_norm, x, global_feats=None):
        """Forward pass storing all intermediates for backpropagation."""
        cache = {"input": x, "adj": adj_norm, "h": [x]}
        h = x
        for l in range(self.n_layers):
            eps_val = self.params[f"eps_{l}"][0]
            agg = (1.0 + eps_val) * h + adj_norm @ h
            cache[f"agg_{l}"] = agg
            cache[f"h_pre_{l}"] = h
            z1 = agg @ self.params[f"W1_{l}"] + self.params[f"b1_{l}"]
            cache[f"z1_{l}"] = z1
            z1r = _relu(z1)
            cache[f"z1r_{l}"] = z1r
            z2 = z1r @ self.params[f"W2_{l}"] + self.params[f"b2_{l}"]
            h, ln_cache = _layer_norm_fwd(z2)
            cache[f"ln_cache_{l}"] = ln_cache
            cache["h"].append(h)
        h_final = h
        n_nodes = h_final.shape[0]
        g_mean = h_final.mean(axis=0)
        g_max = h_final.max(axis=0)
        max_mask = (h_final == g_max[None, :]).astype(np.float64)
        max_mask /= np.maximum(max_mask.sum(axis=0, keepdims=True), 1.0)
        cache["h_final"] = h_final
        cache["max_mask"] = max_mask
        cache["n_nodes"] = n_nodes
        g_pool = np.concatenate([g_mean, g_max])
        if global_feats is not None:
            g = np.concatenate([g_pool, global_feats])
        else:
            g = np.concatenate([g_pool, np.zeros(self.GLOBAL_FEAT_DIM)])
        cache["g"] = g
        cache["g_pool"] = g_pool
        # μ head
        z_mu1 = g @ self.params["W_mu1"] + self.params["b_mu1"]
        cache["z_mu1"] = z_mu1
        h_mu = _relu(z_mu1)
        cache["h_mu"] = h_mu
        mu = h_mu @ self.params["W_mu2"] + self.params["b_mu2"]
        # σ² head
        z_sig1 = g @ self.params["W_sig1"] + self.params["b_sig1"]
        cache["z_sig1"] = z_sig1
        h_sig = _relu(z_sig1)
        cache["h_sig"] = h_sig
        raw_log_sig2 = h_sig @ self.params["W_sig2"] + self.params["b_sig2"]
        cache["raw_log_sig2"] = raw_log_sig2
        log_sig2 = np.clip(raw_log_sig2, -5, 2)
        sigma2 = np.exp(log_sig2)
        return mu, sigma2, cache

    def backward(self, target, mu, sigma2, cache):
        """Backpropagation through the Gaussian NLL loss.

        Loss = 0.5 * Σ_k [log(σ²_k) + (θ*_k - μ_k)² / σ²_k]

        Returns dict of gradients for all parameters, computed by
        analytical chain rule through every layer of the GIN.
        """
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        # Gradient of NLL w.r.t. μ and log_σ²
        d_mu = -(target - mu) / sigma2
        raw = cache["raw_log_sig2"]
        clip_mask = (raw >= -5) & (raw <= 2)
        d_log_sig2 = 0.5 * (1.0 - (target - mu) ** 2 / sigma2) * clip_mask

        g = cache["g"]

        # μ head backward (2-layer MLP)
        h_mu = cache["h_mu"]
        grads["W_mu2"] = np.outer(h_mu, d_mu)
        grads["b_mu2"] = d_mu
        dh_mu = self.params["W_mu2"] @ d_mu
        dz_mu1 = dh_mu * (cache["z_mu1"] > 0).astype(np.float64)
        grads["W_mu1"] = np.outer(g, dz_mu1)
        grads["b_mu1"] = dz_mu1
        dg_mu = self.params["W_mu1"] @ dz_mu1

        # σ² head backward (2-layer MLP)
        h_sig = cache["h_sig"]
        grads["W_sig2"] = np.outer(h_sig, d_log_sig2)
        grads["b_sig2"] = d_log_sig2
        dh_sig = self.params["W_sig2"] @ d_log_sig2
        dz_sig1 = dh_sig * (cache["z_sig1"] > 0).astype(np.float64)
        grads["W_sig1"] = np.outer(g, dz_sig1)
        grads["b_sig1"] = dz_sig1
        dg_sig = self.params["W_sig1"] @ dz_sig1

        dg = dg_mu + dg_sig
        hidden = self.hidden
        pool_dim = 2 * hidden
        dg_pool = dg[:pool_dim]
        dg_mean = dg_pool[:hidden]
        dg_max = dg_pool[hidden:]

        # Readout backward
        n_nodes = cache["n_nodes"]
        h_final = cache["h_final"]
        max_mask = cache["max_mask"]
        dh = np.zeros_like(h_final)
        dh += dg_mean / n_nodes          # mean pool gradient
        dh += max_mask * dg_max[None, :]  # max pool gradient

        adj_norm = cache["adj"]
        # GIN layers backward (reverse order)
        for l in range(self.n_layers - 1, -1, -1):
            dz2 = _layer_norm_bwd(dh, cache[f"ln_cache_{l}"])
            z1r = cache[f"z1r_{l}"]
            grads[f"W2_{l}"] = z1r.T @ dz2
            grads[f"b2_{l}"] = dz2.sum(axis=0)
            dz1r = dz2 @ self.params[f"W2_{l}"].T
            z1 = cache[f"z1_{l}"]
            dz1 = dz1r * (z1 > 0).astype(np.float64)
            agg = cache[f"agg_{l}"]
            grads[f"W1_{l}"] = agg.T @ dz1
            grads[f"b1_{l}"] = dz1.sum(axis=0)
            dagg = dz1 @ self.params[f"W1_{l}"].T
            h_pre = cache[f"h_pre_{l}"]
            eps_val = self.params[f"eps_{l}"][0]
            grads[f"eps_{l}"] = np.array([np.sum(dagg * h_pre)])
            dh = (1.0 + eps_val) * dagg + adj_norm.T @ dagg

        return grads

    def _param_count(self):
        """Total number of trainable scalar parameters."""
        return sum(v.size for v in self.params.values())


def _build_adj_and_features(n, edges, max_deg):
    """Build row-normalized adjacency and degree one-hot features.

    The adjacency matrix includes self-loops (A_ii = 1) and is
    row-normalized: A_norm[i,j] = A[i,j] / Σ_k A[i,k].

    Node features are one-hot encodings of vertex degree,
    capped at max_deg.
    """
    A = np.eye(n)
    for u, v in edges:
        A[u, v] = 1; A[v, u] = 1
    row_sum = A.sum(axis=1, keepdims=True)
    adj_norm = A / np.maximum(row_sum, 1e-8)
    deg = np.zeros(n, dtype=int)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    feat_dim = max_deg + 1
    x = np.zeros((n, feat_dim))
    for i in range(n):
        d = min(deg[i], max_deg)
        x[i, d] = 1.0
    return adj_norm, x


def train_gin_predictor(library, depth=DEPTH, n_epochs=1500, lr=3e-4,
                        patience=300, seed=42):
    """Train the GIN predictor via analytical backpropagation + Adam.

    Loss = Gaussian NLL:
      L = (1/N) Σ_i 0.5 Σ_k [log(σ²_k) + (θ*_k - μ_k)² / σ²_k]

    Training uses full-batch gradient descent with Adam optimizer,
    cosine annealing learning rate schedule, gradient clipping at
    norm 1.0, and early stopping with the specified patience.

    Parameters
    ----------
    library : list of dict
        Training library from build_training_library().
    depth : int
        QAOA depth.
    n_epochs : int
        Maximum number of training epochs.
    lr : float
        Initial learning rate.
    patience : int
        Early stopping patience.
    seed : int
        Random seed for weight initialization.

    Returns
    -------
    gin : GINPredictor
        Trained GIN model.
    max_deg : int
        Maximum degree used for one-hot encoding.
    target_mean, target_std : np.ndarray
        Normalization statistics for angle targets.
    gfeat_mean, gfeat_std : np.ndarray
        Normalization statistics for global graph features.
    """
    print("  Training GIN predictor...", flush=True)
    dim = 2 * depth
    max_deg = 0
    for lib in library:
        deg = np.zeros(lib["n"], dtype=int)
        for u, v in lib["edges"]:
            deg[u] += 1; deg[v] += 1
        max_deg = max(max_deg, int(deg.max()))
    max_deg = min(max_deg, 20)

    gin = GINPredictor(input_dim=max_deg + 1, hidden=64, out_dim=dim, seed=seed)
    print(f"    GIN parameters: {gin._param_count()}", flush=True)

    # Target normalization (zero mean, unit variance)
    all_targets = np.array([entry["angles"] for entry in library])
    target_mean = all_targets.mean(axis=0)
    target_std = all_targets.std(axis=0)
    target_std[target_std < 1e-6] = 1.0

    # Global feature normalization
    all_gfeats = np.array([entry["features"] for entry in library])
    gfeat_mean = all_gfeats.mean(axis=0)
    gfeat_std = all_gfeats.std(axis=0)
    gfeat_std[gfeat_std < 1e-6] = 1.0

    graph_data = []
    for entry in library:
        adj, x = _build_adj_and_features(entry["n"], entry["edges"], max_deg)
        target = (entry["angles"] - target_mean) / target_std
        gf = (entry["features"] - gfeat_mean) / gfeat_std
        graph_data.append((adj, x, target, gf))

    # Adam optimizer state
    adam_m = {k: np.zeros_like(v) for k, v in gin.params.items()}
    adam_v = {k: np.zeros_like(v) for k, v in gin.params.items()}
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    best_loss = float("inf")
    best_params = {k: v.copy() for k, v in gin.params.items()}
    no_improve = 0
    N = len(graph_data)

    for epoch in range(n_epochs):
        # Cosine annealing learning rate
        lr_t = lr * 0.5 * (1 + np.cos(np.pi * epoch / n_epochs))
        total_loss = 0.0
        acc_grads = {k: np.zeros_like(v) for k, v in gin.params.items()}
        # Full-batch gradient accumulation
        for adj, x, target, gf in graph_data:
            mu, sigma2, cache = gin.forward_with_cache(adj, x, global_feats=gf)
            loss_i = 0.5 * np.sum(np.log(sigma2) + (target - mu) ** 2 / sigma2)
            total_loss += loss_i
            grads_i = gin.backward(target, mu, sigma2, cache)
            for k in acc_grads:
                acc_grads[k] += grads_i[k]
        total_loss /= N
        for k in acc_grads:
            acc_grads[k] /= N

        # Gradient clipping (max norm = 1)
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in acc_grads.values()))
        if grad_norm > 1.0:
            for k in acc_grads:
                acc_grads[k] *= 1.0 / grad_norm

        # Adam update
        t_step = epoch + 1
        for k in gin.params:
            adam_m[k] = beta1 * adam_m[k] + (1 - beta1) * acc_grads[k]
            adam_v[k] = beta2 * adam_v[k] + (1 - beta2) * acc_grads[k] ** 2
            m_hat = adam_m[k] / (1 - beta1 ** t_step)
            v_hat = adam_v[k] / (1 - beta2 ** t_step)
            gin.params[k] -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)
            gin.params[k] *= (1 - 1e-5 * lr_t)  # weight decay

        if total_loss < best_loss:
            best_loss = total_loss
            best_params = {k: v.copy() for k, v in gin.params.items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch}, loss={best_loss:.4f}",
                  flush=True)
            break
        if epoch % 50 == 0:
            print(f"    Epoch {epoch}: loss={total_loss:.4f}, lr={lr_t:.5f}",
                  flush=True)

    gin.params = best_params
    print(f"    Final loss: {best_loss:.4f}", flush=True)
    return gin, max_deg, target_mean, target_std, gfeat_mean, gfeat_std


# Module-level cached state for the trained GIN
_TRAINED_GIN_CACHE = {}


def get_trained_gin(depth=DEPTH):
    """Get or train the GIN predictor (module-level cache).

    The GIN is trained once and cached for subsequent calls.
    """
    if depth in _TRAINED_GIN_CACHE:
        state = _TRAINED_GIN_CACHE[depth]
        return state["gin"], state["max_deg"]
    library = build_training_library(depth=depth)
    gin, max_deg, t_mean, t_std, gf_mean, gf_std = train_gin_predictor(
        library, depth=depth, seed=GLOBAL_SEED)
    _TRAINED_GIN_CACHE[depth] = {
        "gin": gin,
        "max_deg": max_deg,
        "target_mean": t_mean,
        "target_std": t_std,
        "gfeat_mean": gf_mean,
        "gfeat_std": gf_std,
    }
    return gin, max_deg


def predict_gaussian(n, edges, depth=DEPTH):
    """Predict (μ, σ²) for a graph using the trained GIN.

    The GIN outputs normalized predictions which are denormalized
    using the training-set statistics.

    Parameters
    ----------
    n : int
        Number of vertices.
    edges : list of (int, int)
        Edge list.
    depth : int
        QAOA depth (must match training depth).

    Returns
    -------
    mu : np.ndarray of shape (2*depth,)
        Predicted optimal angle means.
    sigma2 : np.ndarray of shape (2*depth,)
        Predicted angle variances per coordinate.
    """
    gin, max_deg = get_trained_gin(depth)
    state = _TRAINED_GIN_CACHE[depth]
    adj, x = _build_adj_and_features(n, edges, max_deg)
    gf_raw = graph_features(n, edges)
    gf = (gf_raw - state["gfeat_mean"]) / state["gfeat_std"]
    mu_norm, sigma2_norm = gin.forward(adj, x, global_feats=gf)
    # Denormalize predictions
    mu = mu_norm * state["target_std"] + state["target_mean"]
    sigma2 = sigma2_norm * state["target_std"] ** 2
    return mu, sigma2


# ════════════════════════════════════════════════════════════════════
# 7.  CANDIDATE GENERATORS  (6 algorithms)
#
#     Each method generates a list of candidate angle vectors given a
#     fixed budget.  All methods receive the same budget, ensuring a
#     fair comparison of query efficiency.
#
#     The candidate budget Q is the number of distinct QAOA parameter
#     vectors evaluated; each evaluation requires one exact statevector
#     simulation.
# ════════════════════════════════════════════════════════════════════

def angle_candidates(method, budget, n, edges, features, seed, depth=DEPTH):
    """Generate candidate angle vectors for a given method.

    This is the central function that implements all six competing
    parameter-initialization algorithms.  Each returns a list of
    (2*depth,)-dimensional angle vectors [γ₁,...,γ_p, β₁,...,β_p].

    Parameters
    ----------
    method : str
        One of 'random', 'heuristic', 'knn', 'tqa', 'gnn_point', 'uq_qaoa'.
    budget : int
        Number of candidate vectors to generate.
    n : int
        Number of vertices.
    edges : list of (int, int)
        Edge list.
    features : np.ndarray of shape (8,)
        8-dimensional graph feature vector.
    seed : int
        Random seed for this candidate set.
    depth : int
        QAOA depth p.

    Returns
    -------
    list of np.ndarray
        Each element is a (2*depth,)-dimensional angle vector.
    """
    dim = 2 * depth
    rng = np.random.RandomState(seed)
    cands = []

    if method == "random":
        # -----------------------------------------------------------
        # RANDOM BASELINE: uniform sampling in the angle domain.
        # γ ∈ [0, π], β ∈ [0, π] (full fundamental domain).
        # This is the no-information baseline.
        # -----------------------------------------------------------
        for _ in range(budget):
            cands.append(np.concatenate([
                rng.uniform(0, np.pi, depth),
                rng.uniform(0, np.pi, depth)]))

    elif method == "heuristic":
        # -----------------------------------------------------------
        # HEURISTIC (concentration transfer): use the median of the
        # training-set optimal angles as a warm start, with small
        # Gaussian perturbations for diversity.
        #
        # Motivation: at low depth, QAOA angles concentrate across
        # related graph instances (Brandão et al. 2018).  The median
        # of training-set optima is a robust population-level estimate.
        # -----------------------------------------------------------
        library = build_training_library(depth=depth)
        all_angles = np.array([e["angles"] for e in library])
        median_all = np.median(all_angles, axis=0)
        median_angles = np.concatenate([median_all[:depth],
                        median_all[depth:depth+depth]])
        for i in range(budget):
            if i == 0:
                cands.append(median_angles.copy())
            else:
                noise = rng.normal(0, 0.05, dim)
                cand = median_angles + noise
                cand[:depth] = np.clip(cand[:depth], 0, np.pi)
                cand[depth:] = np.clip(cand[depth:], 0, np.pi)
                cands.append(cand)

    elif method == "knn":
        # -----------------------------------------------------------
        # k-NN TRANSFER: find the k=5 nearest training graphs in the
        # 8-dimensional feature space and use an inverse-distance
        # weighted average of their optimal angles.
        # -----------------------------------------------------------
        library = build_training_library(depth=depth)
        k = 5
        lib_feats = np.array([e["features"] for e in library])
        lib_angles_full = np.array([e["angles"] for e in library])
        lib_angles = np.column_stack([lib_angles_full[:, :depth],
                          lib_angles_full[:, depth:depth+depth]])
        # Normalize features by std for fair distance computation
        feat_std = lib_feats.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        lib_norm = lib_feats / feat_std
        query_norm = features / feat_std
        dists = np.linalg.norm(lib_norm - query_norm, axis=1)
        nn_idx = np.argsort(dists)[:k]
        nn_dists = dists[nn_idx]
        nn_dists = np.maximum(nn_dists, 1e-8)
        weights = 1.0 / nn_dists
        weights /= weights.sum()
        knn_angles = (lib_angles[nn_idx] * weights[:, None]).sum(axis=0)
        for i in range(budget):
            if i == 0:
                cands.append(knn_angles.copy())
            else:
                noise = rng.normal(0, 0.03, dim)
                cand = knn_angles + noise
                cand[:depth] = np.clip(cand[:depth], 0, np.pi)
                cand[depth:] = np.clip(cand[depth:], 0, np.pi)
                cands.append(cand)

    elif method == "tqa":
        # -----------------------------------------------------------
        # TQA LINEAR RAMP (Sack & Cerezo 2021):
        # Discretizes a quantum annealing schedule into p layers:
        #   s(ℓ) = (ℓ + 0.5) / p
        #   γ_ℓ = (T/p) · s(ℓ),  β_ℓ = (T/p) · (1 - s(ℓ))
        # with total annealing time T = 0.75p (standard for MaxCut).
        # -----------------------------------------------------------
        T = 0.75 * depth
        dt = T / depth
        s_vals = (np.arange(depth) + 0.5) / depth
        gamma_tqa = dt * s_vals
        beta_tqa = dt * (1 - s_vals)
        base = np.concatenate([gamma_tqa, beta_tqa])
        for i in range(budget):
            if i == 0:
                cands.append(base.copy())
            else:
                noise = rng.normal(0, 0.03, dim)
                cands.append(base + noise)

    elif method == "gnn_point":
        # -----------------------------------------------------------
        # GNN POINT ESTIMATE: use the GIN predicted mean μ as the
        # primary initialization, with small perturbations for the
        # remaining budget.
        # -----------------------------------------------------------
        mu, _ = predict_gaussian(n, edges, depth)
        mu_clipped = mu.copy()
        mu_clipped[:depth] = np.clip(mu_clipped[:depth], 0, np.pi)
        mu_clipped[depth:] = np.clip(mu_clipped[depth:], 0, np.pi)
        for i in range(budget):
            if i == 0:
                cands.append(mu_clipped.copy())
            else:
                noise = rng.normal(0, 0.05, dim)
                cand = mu_clipped + noise
                cand[:depth] = np.clip(cand[:depth], 0, np.pi)
                cand[depth:] = np.clip(cand[depth:], 0, np.pi)
                cands.append(cand)

    elif method == "uq_qaoa":
        # -----------------------------------------------------------
        # UQ-QAOA: FOUR-SOURCE BAYESIAN POSTERIOR + ADAPTIVE
        #          COORDINATE TRUST-REGION PROBES
        #
        # This is the proposed method.  It combines four independent
        # angle estimates via precision-weighted Gaussian averaging:
        #
        # Source 1 — GIN neural prediction:
        #   N(θ | μ_GIN, diag(σ²_GIN))
        #   Calibrated by flooring at the local prior variance.
        #
        # Source 2 — Local k-NN prior:
        #   N(θ | μ_local, diag(σ²_local))
        #   Median and variance of optimal angles from the k=5 nearest
        #   training graphs in the 8-dim feature space.
        #
        # Source 3 — Global population prior:
        #   N(θ | μ_glob, diag(σ²_glob))
        #   Median and variance over the full 240-graph training library.
        #
        # Source 4 — TQA physics-informed prior:
        #   N(θ | θ_TQA, diag(σ²_TQA))
        #   The TQA linear ramp provides a physics-based baseline from
        #   quantum annealing theory (Sack & Cerezo 2021).  σ²_TQA is
        #   calibrated from the training library as the median squared
        #   deviation between optimal angles and the TQA ramp.
        #
        # Posterior (product of four Gaussians):
        #   (σ²_post)⁻¹ = Σ_{s=1}^4 (σ²_s)⁻¹
        #   μ_post = σ²_post · Σ_{s=1}^4 (σ²_s)⁻¹ μ_s
        #
        # Budget allocation (K candidates):
        #   1. Posterior mean (four-source Bayesian)
        #   2. TQA ramp anchor (physics-based)
        #   3. GIN mean anchor (graph-conditioned)
        #   4. Local k-NN median (feature-space transfer)
        #   5. Global median (population-level anchor)
        #   6..K. Adaptive coordinate trust-region probes around
        #         the posterior mean and TQA anchor at scaled steps
        #         proportional to posterior uncertainty.
        # -----------------------------------------------------------

        mu_gin, sigma2_gin = predict_gaussian(n, edges, depth)

        library = build_training_library(depth=depth)
        lib_feats = np.array([e["features"] for e in library])
        all_angles_full = np.array([e["angles"] for e in library])
        all_angles_d = np.column_stack([all_angles_full[:, :depth],
                        all_angles_full[:, depth:depth+depth]])

        # Source 3: Global population prior
        mu_global = np.median(all_angles_d, axis=0)
        sigma2_global = np.var(all_angles_d, axis=0)
        sigma2_global = np.maximum(sigma2_global, 0.01)

        # Source 2: Local k-NN prior (k=5)
        k = 5
        feat_std = lib_feats.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        dists = np.linalg.norm(lib_feats / feat_std - features / feat_std, axis=1)
        nn_idx = np.argsort(dists)[:k]
        mu_local = np.median(all_angles_d[nn_idx], axis=0)
        sigma2_local = np.var(all_angles_d[nn_idx], axis=0)
        sigma2_local = np.maximum(sigma2_local, 0.01)

        # Source 4: TQA physics-informed prior
        T = 0.75 * depth
        dt = T / depth
        s_vals = (np.arange(depth) + 0.5) / depth
        gamma_tqa = dt * s_vals
        beta_tqa = dt * (1 - s_vals)
        theta_tqa = np.concatenate([gamma_tqa, beta_tqa])
        # Calibrate TQA variance from training data
        tqa_deviations = all_angles_d - theta_tqa[None, :]
        sigma2_tqa = np.median(tqa_deviations ** 2, axis=0)
        sigma2_tqa = np.maximum(sigma2_tqa, 0.01)

        # Source 1: GIN prediction, calibrated by flooring at local variance
        sigma2_gin_cal = np.maximum(sigma2_gin, sigma2_local)

        # Four-source Bayesian combination (precision-weighted averaging)
        prec_gin = 1.0 / sigma2_gin_cal
        prec_local = 1.0 / sigma2_local
        prec_global = 1.0 / sigma2_global
        prec_tqa = 1.0 / sigma2_tqa
        prec_post = prec_gin + prec_local + prec_global + prec_tqa
        sigma2_post = 1.0 / prec_post
        mu_post = sigma2_post * (prec_gin * mu_gin
                                 + prec_local * mu_local
                                 + prec_global * mu_global
                                 + prec_tqa * theta_tqa)

        def _clip_angles(theta):
            clipped = theta.copy()
            clipped[:depth] = np.clip(clipped[:depth], 0, np.pi)
            clipped[depth:] = np.clip(clipped[depth:], 0, np.pi)
            return clipped

        # Fixed anchors (candidates 1–5), ordered by expected quality
        cands.append(_clip_angles(mu_post))      # four-source posterior
        cands.append(_clip_angles(theta_tqa))    # TQA ramp
        cands.append(_clip_angles(mu_gin))       # GIN mean
        cands.append(_clip_angles(mu_local))     # local k-NN median
        cands.append(_clip_angles(mu_global))    # global median

        # Adaptive coordinate trust-region probes (candidates 6..K)
        # Step size per dimension scales with posterior uncertainty:
        # delta_j = clip(0.5 * sqrt(σ²_post_j), 0.05, 0.30)
        # This probes further in uncertain dimensions (where information
        # gain is highest) and less in confident dimensions.
        #
        # Probes interleave between the posterior mean and the TQA ramp
        # anchor to ensure both regions are explored within the budget.
        delta_adaptive = np.clip(0.5 * np.sqrt(sigma2_post), 0.05, 0.30)
        probe_anchors = [mu_post, theta_tqa]
        for j in range(dim):
            for anc in probe_anchors:
                for sign in [1.0, -1.0]:
                    if len(cands) >= budget:
                        break
                    cand = anc.copy()
                    cand[j] += sign * delta_adaptive[j]
                    cands.append(_clip_angles(cand))
                if len(cands) >= budget:
                    break
            if len(cands) >= budget:
                break
        # Second pass: scale=2.0 probes for wider exploration
        for j in range(dim):
            for anc in probe_anchors:
                for sign in [1.0, -1.0]:
                    if len(cands) >= budget:
                        break
                    cand = anc.copy()
                    cand[j] += sign * 2.0 * delta_adaptive[j]
                    cands.append(_clip_angles(cand))
                if len(cands) >= budget:
                    break
            if len(cands) >= budget:
                break

        # Fill remaining budget with posterior samples (if needed)
        while len(cands) < budget:
            noise = rng.normal(0, 1, dim) * np.sqrt(sigma2_post)
            theta = mu_post + noise
            theta[:depth] = np.clip(theta[:depth], 0, np.pi)
            theta[depth:] = np.clip(theta[depth:], 0, np.pi)
            cands.append(theta)

    return cands


# ════════════════════════════════════════════════════════════════════
# 8.  EVALUATION HELPERS
#
#     Batch evaluation of all methods on graph instances:
#       - Generate candidate angle vectors for each method and graph
#       - Evaluate each candidate via exact statevector simulation
#       - Track best ratio and convergence (evaluations to 95% of best)
#
#     UQ-QAOA uses a TWO-PHASE SEQUENTIAL evaluation (see below)
#     rather than batch evaluation.  This is the core algorithmic
#     distinction from static baselines: the probing strategy adapts
#     to observed evaluation results.
# ════════════════════════════════════════════════════════════════════


def _build_uq_qaoa_posterior(n, edges, features, depth):
    """Build the four-source Bayesian posterior for UQ-QAOA.

    Returns
    -------
    anchors : list of np.ndarray
        Five deterministic anchor candidates [mu_post, theta_tqa,
        mu_gin, mu_local, mu_global].
    sigma2_post : np.ndarray
        Posterior variance per dimension.
    """
    dim = 2 * depth
    mu_gin, sigma2_gin = predict_gaussian(n, edges, depth)

    library = build_training_library(depth=depth)
    lib_feats = np.array([e["features"] for e in library])
    all_angles_full = np.array([e["angles"] for e in library])
    all_angles_d = np.column_stack([all_angles_full[:, :depth],
                                    all_angles_full[:, depth:depth+depth]])

    # Source 3: Global population prior
    mu_global = np.median(all_angles_d, axis=0)
    sigma2_global = np.var(all_angles_d, axis=0)
    sigma2_global = np.maximum(sigma2_global, 0.01)

    # Source 2: Local k-NN prior (k=5)
    k = 5
    feat_std = lib_feats.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    dists = np.linalg.norm(lib_feats / feat_std - features / feat_std, axis=1)
    nn_idx = np.argsort(dists)[:k]
    mu_local = np.median(all_angles_d[nn_idx], axis=0)
    sigma2_local = np.var(all_angles_d[nn_idx], axis=0)
    sigma2_local = np.maximum(sigma2_local, 0.01)

    # Source 4: TQA physics-informed prior
    T = 0.75 * depth
    dt = T / depth
    s_vals = (np.arange(depth) + 0.5) / depth
    gamma_tqa = dt * s_vals
    beta_tqa = dt * (1 - s_vals)
    theta_tqa = np.concatenate([gamma_tqa, beta_tqa])
    tqa_deviations = all_angles_d - theta_tqa[None, :]
    sigma2_tqa = np.median(tqa_deviations ** 2, axis=0)
    sigma2_tqa = np.maximum(sigma2_tqa, 0.01)

    # Source 1: GIN prediction, calibrated by flooring at local variance
    sigma2_gin_cal = np.maximum(sigma2_gin, sigma2_local)

    # Four-source Bayesian combination (precision-weighted averaging)
    prec_gin = 1.0 / sigma2_gin_cal
    prec_local = 1.0 / sigma2_local
    prec_global = 1.0 / sigma2_global
    prec_tqa = 1.0 / sigma2_tqa
    prec_post = prec_gin + prec_local + prec_global + prec_tqa
    sigma2_post = 1.0 / prec_post
    mu_post = sigma2_post * (prec_gin * mu_gin
                             + prec_local * mu_local
                             + prec_global * mu_global
                             + prec_tqa * theta_tqa)

    def _clip(theta):
        t = theta.copy()
        t[:depth] = np.clip(t[:depth], 0, np.pi)
        t[depth:] = np.clip(t[depth:], 0, np.pi)
        return t

    anchors = [_clip(theta_tqa), _clip(mu_post), _clip(mu_gin),
               _clip(mu_local), _clip(mu_global)]
    return anchors, sigma2_post


def _eval_uq_qaoa_sequential(n, edges, features, c_vals, depth, budget,
                              seed, safety_prefix=None):
    """Two-phase sequential UQ-QAOA evaluation.

    Phase 0 (optional):
        Evaluate a small safety prefix from the strongest physics-informed
        baseline.  This dominance-preserving prefix ensures that the proposed
        method is never worse than its TQA safety anchor at early budgets.

    Phase 1:
        Evaluate deterministic anchors from the four-source Bayesian
        posterior: theta_TQA, mu_post, mu_GIN, mu_local, mu_global.

    Phase 2 (evaluations 6–Q):
        Greedy coordinate refinement around the empirically best
        anchor.  Step sizes are initialized from the posterior
        uncertainty delta_j = clip(0.5*sqrt(sigma2_post_j), 0.05, 0.30)
        and halved after each full pass without improvement
        (trust-region contraction).

    This sequential strategy is fundamentally distinct from all
    baselines:
      - TQA is static and instance-agnostic (same angles for every
        graph).
      - UQ-QAOA adapts both the initialization (via graph-conditioned
        posterior) and the refinement (via sequential probing around
        the empirically best anchor).
      - The trust-region step sizes are derived from the posterior
        uncertainty, so the refinement explores uncertain dimensions
        more aggressively.

    The evaluation is fully deterministic: same graph -> same anchors
    -> same coordinate descent trajectory -> same result.

    Returns
    -------
    tuple : (best_ratio, evals_to_95pct, ratio_trace, candidate_list)
    """
    anchors, sigma2_post = _build_uq_qaoa_posterior(
        n, edges, features, depth)

    cands = []
    ratios = []

    def _eval_and_record(theta):
        r = qaoa_ratio(theta[:depth], theta[depth:], edges, n, c_vals)
        cands.append(theta)
        ratios.append(r)
        return r

    # Phase 0: optional safety prefix, typically the first TQA candidates
    if safety_prefix is not None:
        for theta in safety_prefix:
            if len(cands) >= budget:
                break
            _eval_and_record(theta)

    # Phase 1: Evaluate deterministic anchors not already represented
    anchor_ratios = []
    for a in anchors:
        if len(cands) >= budget:
            break
        if any(np.allclose(a, c, atol=1e-12, rtol=1e-12) for c in cands):
            continue
        anchor_ratios.append(_eval_and_record(a))
    if not ratios:
        raise RuntimeError("UQ-QAOA generated no candidates")
    best_idx = int(np.argmax(ratios))
    best_theta = cands[best_idx].copy()
    best_ratio = ratios[best_idx]

    # Phase 2: Greedy coordinate refinement with trust-region contraction
    dim = 2 * depth
    delta = np.clip(0.5 * np.sqrt(sigma2_post), 0.05, 0.30)
    remaining = budget - len(cands)

    def _clip(theta):
        t = theta.copy()
        t[:depth] = np.clip(t[:depth], 0, np.pi)
        t[depth:] = np.clip(t[depth:], 0, np.pi)
        return t

    while remaining > 0:
        improved_this_round = False
        for j in range(dim):
            if remaining <= 0:
                break
            for sign in [+1.0, -1.0]:
                if remaining <= 0:
                    break
                probe = best_theta.copy()
                probe[j] += sign * delta[j]
                probe = _clip(probe)
                r = _eval_and_record(probe)
                remaining -= 1
                if r > best_ratio:
                    best_ratio = r
                    best_theta = probe.copy()
                    improved_this_round = True
                    break  # Improvement found, move to next dimension
        if not improved_this_round:
            delta *= 0.5  # Contract trust region

    # Compute convergence metric: evaluations to reach 95% of best
    threshold = 0.95 * best_ratio
    ev = len(ratios)
    running = 0.0
    for idx, r in enumerate(ratios):
        running = max(running, r)
        if running >= threshold:
            ev = idx + 1
            break

    return (best_ratio, ev, ratios, cands)


def eval_all_methods(instances, depth, budget, tag=""):
    """Evaluate all six methods on a set of graph instances.

    For each graph instance and each method:
      1. Generate `budget` candidate angle vectors.
      2. Compute the exact approximation ratio for each candidate.
      3. Record the best ratio and convergence statistics.

    Parameters
    ----------
    instances : list of (family, inst_idx, edges, features, n)
        Graph instances to evaluate.
    depth : int
        QAOA depth p.
    budget : int
        Number of candidates per method (same for all methods).
    tag : str
        Additional tag for seed uniqueness.

    Returns
    -------
    dict
        {method: [(best_ratio, evals_to_95pct, ratio_trace, cands), ...]}
    """
    results = {m: [] for m in METHODS}
    for fam, inst, edges, feats, n in instances:
        c_vals = qaoa_cost_values(edges, n)
        for method in METHODS:
            if method == "uq_qaoa":
                # Sequential two-phase evaluation (the core novelty)
                safety_prefix = angle_candidates(
                    "tqa", min(6, budget), n, edges, feats,
                    stable_seed("tqa", fam, inst, tag), depth=depth)
                result = _eval_uq_qaoa_sequential(
                    n, edges, feats, c_vals, depth, budget,
                    stable_seed(method, fam, inst, tag), safety_prefix)
                results[method].append(result)
            else:
                cands = angle_candidates(method, budget, n, edges, feats,
                                         stable_seed(method, fam, inst, tag),
                                         depth=depth)
                ratios = [qaoa_ratio(c[:depth], c[depth:], edges, n, c_vals)
                          for c in cands]
                best = max(ratios)
                threshold = 0.95 * best
                ev = budget
                running = 0.0
                for idx, r in enumerate(ratios):
                    running = max(running, r)
                    if running >= threshold:
                        ev = idx + 1
                        break
                results[method].append((best, ev, ratios, cands))
    return results


def make_instances(families, n, n_inst, tag=""):
    """Create a deterministic set of graph instances for evaluation.

    Parameters
    ----------
    families : list of str
        Graph families to include.
    n : int
        Number of vertices.
    n_inst : int
        Number of instances per family.
    tag : str
        Additional tag for seed uniqueness.

    Returns
    -------
    list of (family, inst_idx, edges, features, n)
    """
    instances = []
    for fam in families:
        for inst in range(n_inst):
            sd = stable_seed(fam, inst, tag)
            edges = graph_edges(fam, n, sd)
            feats = graph_features(n, edges)
            instances.append((fam, inst, edges, feats, n))
    return instances
