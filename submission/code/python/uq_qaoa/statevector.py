from __future__ import annotations

import numpy as np
from .qaoa_angles import split_theta, assert_theta
from .maxcut import all_cut_values


def apply_mixer_inplace(state: np.ndarray, beta: float, n: int) -> None:
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    size = state.size
    for q in range(n):
        step = 1 << q
        jump = step << 1
        for base in range(0, size, jump):
            a = state[base:base + step].copy()
            b = state[base + step:base + jump].copy()
            state[base:base + step] = c * a + s * b
            state[base + step:base + jump] = s * a + c * b


def qaoa_statevector(n: int, edges, p: int, theta, layout: str = "blocked", dtype=np.complex128, cost_values=None) -> np.ndarray:
    theta = assert_theta(theta, p)
    gammas, betas = split_theta(theta, p, layout)
    if cost_values is None:
        cost_values = all_cut_values(n, edges)
    cost_values = np.asarray(cost_values, dtype=np.float64)
    dim = 1 << n
    if cost_values.size != dim:
        raise ValueError("cost_values length must equal 2**n")
    state = np.full(dim, 1.0 / np.sqrt(dim), dtype=dtype)
    for gamma, beta in zip(gammas, betas):
        state *= np.exp((-1j * gamma * cost_values).astype(dtype, copy=False))
        apply_mixer_inplace(state, float(beta), n)
    return state


def qaoa_expectation(n: int, edges, p: int, theta, layout: str = "blocked", dtype=np.complex128, cost_values=None, return_state: bool = False):
    theta = assert_theta(theta, p)
    if cost_values is None:
        cost_values = all_cut_values(n, edges)
    state = qaoa_statevector(n, edges, p, theta, layout=layout, dtype=dtype, cost_values=cost_values)
    value = float(np.dot(np.abs(state) ** 2, np.asarray(cost_values, dtype=np.float64)))
    if return_state:
        return value, state
    return value
