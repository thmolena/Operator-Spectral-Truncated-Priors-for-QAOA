# uq-qaoa — Reproduction Artifact

Reference implementation and deterministic reproduction artifact for the
manuscript *Uncertainty-Calibrated Trust Regions for Query-Efficient QAOA
Parameter Search*. The package provides the exact-statevector MaxCut backend,
QAOA angle utilities, configuration handling, and the figure/table generators
that produce every reported quantity. The distribution name on the index is
`uq-qaoa`; the importable top-level package is `uq_qaoa`.

## Installation

```bash
pip install uq-qaoa
```

Installation from the artifact checkout:

```bash
cd submission/code
pip install .
```

The runtime requires Python 3.10 or later. Dependencies are constrained to
tested major versions:

| Dependency   | Constraint        |
|--------------|-------------------|
| numpy        | `>=1.24,<3`       |
| scipy        | `>=1.10,<2`       |
| pandas       | `>=2.0,<4`        |
| pyyaml       | `>=6.0,<7`        |
| networkx     | `>=3.0,<4`        |
| matplotlib   | `>=3.7,<4`        |
| scikit-learn | `>=1.2,<2`        |

The optional `test` extra adds `pytest>=7.0,<9`.

## Reproduction

The package installs a console entry point that regenerates the figures and
tables. From the artifact directory (`submission/code`, which contains
`generate_all.py` and `uq_qaoa_core.py`):

```bash
uqqaoa-reproduce            # publication configuration (p=3, Q=18)
uqqaoa-reproduce --smoke    # fast end-to-end pass at reduced depth (p=2, Q=12)
```

The entry point locates the master generator, fixes the global seed via the
environment, and executes `generate_all.py`. The same regeneration is available
directly:

```bash
python generate_all.py
```

A shell driver exposes named subsets for continuous-integration checks:

```bash
bash reproduce.sh smoke     # fast subset
bash reproduce.sh all       # full validation, figures, and tables
```

The full publication configuration completes in approximately two to five
minutes on the reference platform (Apple M2 Pro), dominated by GIN training and
exact-statevector QAOA evaluation.

## Regenerated figures and tables

Running the entry point writes the following artifacts to `figures/` and
`tables/`. The mapping below follows the reproducibility manifest of the
manuscript.

| Manuscript element                      | Generating script                  | Output |
|-----------------------------------------|------------------------------------|--------|
| Main benchmark (n=14, p=3, Q=18)        | `trace_evaluations.py`             | `tables/table01_computational_efficiency.csv` |
| 48-instance replication                 | `table01_expanded.py`              | `tables/table01_expanded.csv`, `tables/paired_uq_vs_tqa_expanded.csv` |
| Expanded benchmark figure               | `fig05_expanded_benchmark.py`      | `figures/fig05_expanded_benchmark.pdf` |
| Ablation                                | `table03_ablation.py`              | `tables/table03_ablation.csv` |
| Bound verification                      | `table02_bound_verification.py`    | `tables/table02_bound_verification.csv` |
| Cross-size statevector grid             | `fig01_statevector_grid.py`        | `figures/fig01_statevector_grid.pdf` |
| Efficiency-adjusted quality             | `fig02_efficiency_adjusted.py`     | `figures/fig02_efficiency_adjusted.pdf` |
| Held-out quality check                  | `fig03_heldout_quality.py`         | `figures/fig03_heldout_quality.pdf` |
| Trust-region concept                    | `fig04_trust_region_concept.py`    | `figures/fig04_trust_region_concept.pdf` |
| Per-candidate evaluation log            | `trace_evaluations.py`             | `tables/trace_all_evaluations.csv` |

The Python/C++ statevector cross-check (`results/python_cpp_validation.csv`) and
the CPU benchmark tables (`tables/bench_*.csv`) are produced by the C++20 backend
under `cpp/` together with `bench_cpu.py` and `bench_cpu.cpp`.

## Determinism

Every generator is seeded by the fixed global seed `260424803`. Graph
construction, train/test splitting, GIN initialization, and evaluation ordering
are deterministic given this seed, so repeated runs on a fixed platform
reproduce the reported numbers. The entry point also sets `PYTHONHASHSEED` to the
global seed. The QAOA depth and matched query budget default to `p=3` and `Q=18`
and may be overridden through the `UQ_QAOA_DEPTH` and `UQ_QAOA_BUDGET`
environment variables (the `--smoke` flag sets `p=2`, `Q=12`).

## Package layout

```text
python/uq_qaoa/        installable library (statevector backend, QAOA angle
                       utilities, configuration, calibration, search policy,
                       and the uqqaoa-reproduce entry point)
generate_all.py        master deterministic regenerator
uq_qaoa_core.py        training library, GIN predictor, and benchmark routines
fig0*.py, table0*.py   individual figure and table generators
reproduce.sh           shell driver with named subsets
configs/               experiment configurations (p3_main.yaml is primary)
results/, tables/,      generated CSV/TeX/PDF artifacts
  figures/
cpp/                   C++20 reference statevector backend and tests
tests/                 package tests
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
