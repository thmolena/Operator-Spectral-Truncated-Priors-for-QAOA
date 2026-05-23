from __future__ import annotations

import numpy as np

GAMMA_DOMAIN = (0.0, np.pi)
BETA_DOMAIN = (0.0, np.pi)


def assert_theta(theta, p: int) -> np.ndarray:
    arr = np.asarray(theta, dtype=float).reshape(-1)
    if p <= 0:
        raise ValueError("p must be positive")
    expected = 2 * p
    if arr.size != expected:
        raise ValueError(f"theta length must be 2*p={expected}; received {arr.size}")
    return arr


def split_theta(theta, p: int, layout: str = "blocked") -> tuple[np.ndarray, np.ndarray]:
    theta = assert_theta(theta, p)
    if layout == "blocked":
        return theta[:p].copy(), theta[p:2 * p].copy()
    if layout == "interleaved":
        return theta[0::2].copy(), theta[1::2].copy()
    raise ValueError("layout must be 'blocked' or 'interleaved'")


def join_theta(gammas, betas, layout: str = "blocked") -> np.ndarray:
    gammas = np.asarray(gammas, dtype=float).reshape(-1)
    betas = np.asarray(betas, dtype=float).reshape(-1)
    if gammas.size != betas.size or gammas.size == 0:
        raise ValueError("gammas and betas must have equal positive length")
    p = gammas.size
    if layout == "blocked":
        return np.concatenate([gammas, betas])
    if layout == "interleaved":
        theta = np.empty(2 * p, dtype=float)
        theta[0::2] = gammas
        theta[1::2] = betas
        return theta
    raise ValueError("layout must be 'blocked' or 'interleaved'")


def project_angles(theta, p: int, layout: str = "blocked") -> np.ndarray:
    gammas, betas = split_theta(theta, p, layout)
    gammas = np.clip(gammas, *GAMMA_DOMAIN)
    betas = np.clip(betas, *BETA_DOMAIN)
    return join_theta(gammas, betas, layout)


def normalize_angles(theta, p: int, layout: str = "blocked") -> np.ndarray:
    gammas, betas = split_theta(theta, p, layout)
    gammas = np.mod(gammas, np.pi)
    betas = np.mod(betas, np.pi)
    return join_theta(gammas, betas, layout)


def random_theta(p: int, rng: np.random.Generator, layout: str = "blocked") -> np.ndarray:
    if p <= 0:
        raise ValueError("p must be positive")
    gammas = rng.uniform(GAMMA_DOMAIN[0], GAMMA_DOMAIN[1], size=p)
    betas = rng.uniform(BETA_DOMAIN[0], BETA_DOMAIN[1], size=p)
    return join_theta(gammas, betas, layout)


def angle_names(p: int, layout: str = "blocked") -> list[str]:
    if p <= 0:
        raise ValueError("p must be positive")
    if layout == "blocked":
        return [f"gamma_{i+1}" for i in range(p)] + [f"beta_{i+1}" for i in range(p)]
    if layout == "interleaved":
        names: list[str] = []
        for i in range(p):
            names.extend([f"gamma_{i+1}", f"beta_{i+1}"])
        return names
    raise ValueError("layout must be 'blocked' or 'interleaved'")
