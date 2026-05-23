"""Configurable-depth UQ-QAOA reference package.

The package intentionally keeps the scientific query policy independent of the
statevector backend.  Depth is always supplied explicitly or via a validated
configuration object; parameter dimension is `2 * p`.
"""

from .config import ExperimentConfig, dimension_scaled_min_budget, query_curve_values, load_config
from .qaoa_angles import assert_theta, split_theta, join_theta, project_angles, random_theta
from .statevector import qaoa_expectation, qaoa_statevector

__all__ = [
    "ExperimentConfig",
    "dimension_scaled_min_budget",
    "query_curve_values",
    "load_config",
    "assert_theta",
    "split_theta",
    "join_theta",
    "project_angles",
    "random_theta",
    "qaoa_expectation",
    "qaoa_statevector",
]
