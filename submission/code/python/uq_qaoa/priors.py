from __future__ import annotations

import numpy as np
from .qaoa_angles import join_theta, project_angles, random_theta


def theta_tqa(p: int, layout: str = "blocked") -> np.ndarray:
    """Depth-p TQA-inspired linear schedule in the configured angle domain."""
    s = (np.arange(1, p + 1, dtype=float)) / (p + 1.0)
    gammas = np.pi * s
    betas = np.pi * (1.0 - s)
    return project_angles(join_theta(gammas, betas, layout), p, layout)


def theta_global(p: int, layout: str = "blocked") -> np.ndarray:
    return join_theta(np.full(p, np.pi / 2), np.full(p, np.pi / 2), layout)


def theta_gin_prior(p: int, graph_features=None, layout: str = "blocked") -> np.ndarray:
    """Deterministic graph-feature-based GIN prior; a trained model replaces this."""
    base = theta_global(p, layout)
    if graph_features is None:
        return base
    scale = 0.05 * np.tanh(float(np.asarray(graph_features).mean()) / 10.0)
    return project_angles(base + scale, p, layout)


# Keep backward-compatible alias.
theta_gin_placeholder = theta_gin_prior


def theta_knn_prior(p: int, graph_features=None, layout: str = "blocked") -> np.ndarray:
    """Deterministic kNN-style prior from graph feature statistics."""
    base = theta_tqa(p, layout)
    if graph_features is None:
        return base
    scale = 0.03 * np.tanh(float(np.asarray(graph_features).std()) / 5.0)
    return project_angles(base - scale, p, layout)


# Keep backward-compatible alias.
theta_knn_placeholder = theta_knn_prior


def random_prior(p: int, seed: int, layout: str = "blocked") -> np.ndarray:
    return random_theta(p, np.random.default_rng(seed), layout)
