"""Configurable-depth OST-QAOA reference package.

The package intentionally keeps the scientific query policy independent of the
statevector backend.  Depth is always supplied explicitly or via a validated
configuration object; parameter dimension is `2 * p`.
"""

__version__ = "1.2.0"

from .config import ExperimentConfig, dimension_scaled_min_budget, query_curve_values, load_config
from .qaoa_angles import assert_theta, split_theta, join_theta, project_angles, random_theta
from .statevector import qaoa_expectation, qaoa_statevector
from .operator_spectral import (
    build_operator_library,
    build_training_core,
    diagonal_operator_search,
    effective_dimension,
    evaluate_methods_on_graph,
    operator_prior,
    ost_qaoa_search,
    spectral_operator_matrix,
)

__all__ = [
    "__version__",
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
    "build_operator_library",
    "build_training_core",
    "diagonal_operator_search",
    "effective_dimension",
    "evaluate_methods_on_graph",
    "operator_prior",
    "ost_qaoa_search",
    "spectral_operator_matrix",
]
