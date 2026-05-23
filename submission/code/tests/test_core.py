import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import numpy as np
import pytest
from uq_qaoa.config import load_config, dimension_scaled_min_budget, query_curve_values
from uq_qaoa.qaoa_angles import split_theta, join_theta, random_theta, assert_theta
from uq_qaoa.graphs import generate_graph, split_seeds
from uq_qaoa.maxcut import cut_value, exact_maxcut_bruteforce, all_cut_values
from uq_qaoa.statevector import qaoa_expectation
from uq_qaoa.finite_shots import sample_objective
from uq_qaoa.priors import theta_tqa, theta_global
from uq_qaoa.posterior import fuse_diagonal_priors
from uq_qaoa.search_policy import uq_qaoa_search
from uq_qaoa.baselines import tqa_refine

ROOT = Path(__file__).resolve().parents[1]

def test_config_depth_and_budgets():
    cfg = load_config(ROOT / "configs" / "p7_hpc_stress.yaml")
    assert cfg.qaoa_depth == 7
    assert cfg.dim == 14
    assert dimension_scaled_min_budget(7) == 33
    assert 33 in query_curve_values(7)

@pytest.mark.parametrize("p", [3, 6, 7])
@pytest.mark.parametrize("layout", ["blocked", "interleaved"])
def test_angle_roundtrip(p, layout):
    rng = np.random.default_rng(1)
    theta = random_theta(p, rng, layout)
    g, b = split_theta(theta, p, layout)
    assert np.allclose(join_theta(g, b, layout), theta)
    with pytest.raises(ValueError):
        assert_theta(np.zeros(2 * p - 1), p)


def test_graph_determinism_and_splits():
    a = generate_graph("er", 8, 123)
    b = generate_graph("er", 8, 123)
    assert a.edges == b.edges
    s = split_seeds([1,2,3,4,5,6])
    assert not (set(s["train"]) & set(s["validation"]) & set(s["test"]))


def test_maxcut_small():
    edges = [(0,1)]
    assert cut_value(0, edges) == 0
    assert cut_value(1, edges) == 1
    opt, typ = exact_maxcut_bruteforce(2, edges)
    assert opt == 1 and typ == "exact"

@pytest.mark.parametrize("p", [1, 3, 6, 7])
def test_statevector_depth_support(p):
    g = generate_graph("cycle", 4, 1)
    theta = random_theta(p, np.random.default_rng(2), "blocked")
    y1 = qaoa_expectation(g.n, g.edges, p, theta, cost_values=all_cut_values(g.n, g.edges))
    y2 = qaoa_expectation(g.n, g.edges, p, theta, cost_values=all_cut_values(g.n, g.edges))
    assert np.isfinite(y1)
    assert y1 == pytest.approx(y2)
    with pytest.raises(ValueError):
        qaoa_expectation(g.n, g.edges, p, theta[:-1])


def test_finite_shot_reproducible():
    g = generate_graph("cycle", 4, 1)
    theta = theta_tqa(3)
    c = all_cut_values(g.n, g.edges)
    a = sample_objective(g.n, g.edges, 3, theta, 128, np.random.default_rng(7), cost_values=c)
    b = sample_objective(g.n, g.edges, 3, theta, 128, np.random.default_rng(7), cost_values=c)
    assert a["mean"] == b["mean"]


def test_search_budget_accounting():
    p = 3
    calls = []
    def obj(theta):
        calls.append(theta.copy())
        return float(-np.sum((theta - 1.0) ** 2))
    anchors = [("tqa", theta_tqa(p)), ("global", theta_global(p))]
    _, var = fuse_diagonal_priors([theta_tqa(p), theta_global(p)], None, p)
    res = uq_qaoa_search(obj, p, 7, anchors, var)
    assert len(calls) <= 7
    assert len(res.trace) <= 7
    base = tqa_refine(obj, p, 7)
    assert len(base.trace) <= 7
