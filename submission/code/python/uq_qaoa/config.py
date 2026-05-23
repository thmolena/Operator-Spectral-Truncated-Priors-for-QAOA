from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

VALID_LAYOUTS = {"blocked", "interleaved"}
VALID_BUDGET_MODES = {"fixed", "dimension_scaled", "curve"}


def dimension_scaled_min_budget(p: int, anchors_count: int = 5, signed_coordinate_sweeps: int = 1) -> int:
    """Minimum fair query budget for one signed coordinate sweep in dimension 2p."""
    if p <= 0:
        raise ValueError("qaoa_depth must be positive")
    if anchors_count <= 0 or signed_coordinate_sweeps < 0:
        raise ValueError("invalid query-budget parameters")
    return anchors_count + signed_coordinate_sweeps * 4 * p


def query_curve_values(p: int, extras: list[int] | None = None, anchors_count: int = 5) -> list[int]:
    base = [5, 10, 18, dimension_scaled_min_budget(p, anchors_count), 10 + 8 * p, 64, 100]
    if extras:
        base.extend(int(x) for x in extras)
    return sorted({q for q in base if q >= anchors_count})


@dataclass(frozen=True)
class QueryBudgetConfig:
    mode: str = "curve"
    fixed_Q: int = 18
    curve_extra_values: list[int] = field(default_factory=lambda: [5, 10, 18, 64, 100])
    use_dimension_scaled_minimum: bool = True
    anchors_count: int = 5
    signed_coordinate_sweeps: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "QueryBudgetConfig":
        d = d or {}
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if obj.mode not in VALID_BUDGET_MODES:
            raise ValueError(f"query_budget.mode must be one of {sorted(VALID_BUDGET_MODES)}")
        if obj.fixed_Q <= 0:
            raise ValueError("fixed_Q must be positive")
        return obj

    def values(self, p: int) -> list[int]:
        if self.mode == "fixed":
            return [self.fixed_Q]
        if self.mode == "dimension_scaled":
            return [dimension_scaled_min_budget(p, self.anchors_count, self.signed_coordinate_sweeps)]
        return query_curve_values(p, self.curve_extra_values, self.anchors_count)


@dataclass(frozen=True)
class GraphConfig:
    families: list[str] = field(default_factory=lambda: ["er", "random_regular", "watts_strogatz", "barabasi_albert"])
    sizes: list[int] = field(default_factory=lambda: [8, 10, 12, 14])
    primary_test_n: int = 14
    train_instances_per_family: int = 8
    validation_instances_per_family: int = 4
    test_instances_per_family: int = 4
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1024])

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GraphConfig":
        d = d or {}
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if not obj.families or not obj.sizes or not obj.seeds:
            raise ValueError("graphs.families, graphs.sizes, and graphs.seeds must be non-empty")
        if any(n <= 1 for n in obj.sizes):
            raise ValueError("graph sizes must exceed 1")
        return obj


@dataclass(frozen=True)
class FiniteShotConfig:
    enabled: bool = False
    shots: list[int] = field(default_factory=lambda: [256, 512, 1024, 4096, 8192])
    reevaluate_final_exact: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FiniteShotConfig":
        d = d or {}
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if any(s <= 0 for s in obj.shots):
            raise ValueError("finite_shots.shots must be positive")
        return obj


@dataclass(frozen=True)
class HpcConfig:
    cpp_enabled: bool = True
    openmp_enabled: str | bool = "optional"
    thread_counts: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 8, 10])
    batch_sizes: list[int] = field(default_factory=lambda: [18, 33, 64, 128, 256])
    benchmark_ns: list[int] = field(default_factory=lambda: [14, 16, 18, 20, 22, 24])
    benchmark_ps: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    deterministic_reduction: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "HpcConfig":
        d = d or {}
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        if any(t <= 0 for t in obj.thread_counts):
            raise ValueError("hpc.thread_counts must be positive")
        return obj


@dataclass(frozen=True)
class OutputConfig:
    root: str = "code/results"
    save_traces: bool = True
    save_figures: bool = True
    save_tables: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "OutputConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ExperimentConfig:
    qaoa_depth: int = 3
    theta_layout: str = "blocked"
    dtype: str = "complex128"
    random_seed: int = 42
    backend: str = "python_statevector"
    methods: dict[str, Any] = field(default_factory=lambda: {"include": ["random_search", "tqa", "tqa_refine", "uq_qaoa_full"]})
    query_budget: QueryBudgetConfig = field(default_factory=QueryBudgetConfig)
    graphs: GraphConfig = field(default_factory=GraphConfig)
    finite_shots: FiniteShotConfig = field(default_factory=FiniteShotConfig)
    hpc: HpcConfig = field(default_factory=HpcConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    config_path: str = ""

    @property
    def dim(self) -> int:
        return 2 * self.qaoa_depth

    @property
    def query_budgets(self) -> list[int]:
        return self.query_budget.values(self.qaoa_depth)

    @classmethod
    def from_dict(cls, d: dict[str, Any], config_path: str = "") -> "ExperimentConfig":
        q = QueryBudgetConfig.from_dict(d.get("query_budget"))
        g = GraphConfig.from_dict(d.get("graphs"))
        fs = FiniteShotConfig.from_dict(d.get("finite_shots"))
        hpc = HpcConfig.from_dict(d.get("hpc"))
        out = OutputConfig.from_dict(d.get("outputs"))
        obj = cls(
            qaoa_depth=int(d.get("qaoa_depth", 3)),
            theta_layout=d.get("theta_layout", "blocked"),
            dtype=d.get("dtype", "complex128"),
            random_seed=int(d.get("random_seed", 42)),
            backend=d.get("backend", "python_statevector"),
            methods=d.get("methods", {"include": ["random_search", "tqa", "tqa_refine", "uq_qaoa_full"]}),
            query_budget=q,
            graphs=g,
            finite_shots=fs,
            hpc=hpc,
            outputs=out,
            config_path=config_path,
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.qaoa_depth <= 0 or self.qaoa_depth > 7:
            raise ValueError("qaoa_depth must be in {1,...,7} for this study")
        if self.theta_layout not in VALID_LAYOUTS:
            raise ValueError(f"theta_layout must be one of {sorted(VALID_LAYOUTS)}")
        if self.dtype not in {"complex64", "complex128"}:
            raise ValueError("dtype must be complex64 or complex128")
        if self.dim != 2 * self.qaoa_depth:
            raise AssertionError("internal dimension invariant failed")
        _ = self.query_budgets


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ExperimentConfig.from_dict(data, config_path=str(path))
