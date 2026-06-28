"""Console entry point for deterministic manuscript regeneration.

The default path is package-native: after ``pip install .`` or
``pip install uq-qaoa``, ``uqqaoa-reproduce`` regenerates every figure, table,
CSV result, and query trace used by the OST-QAOA manuscript. A legacy switch is
kept for the older flat ``generate_all.py`` artifact, but the installable
package no longer depends on running from the source checkout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

GLOBAL_SEED = 260424803
SMOKE_DEPTH = 2
SMOKE_BUDGET = 12


def _candidate_roots() -> list[Path]:
    """Directories that may contain the master generator and core module."""
    here = Path(__file__).resolve()
    roots: list[Path] = []
    # Installed layout: python/uq_qaoa/reproduce.py -> submission/code
    roots.append(here.parents[2])
    # Source-tree fallbacks for in-place execution.
    roots.append(Path.cwd())
    return roots


def _locate_artifact_root() -> Path:
    """Return the directory holding ``generate_all.py`` and ``uq_qaoa_core.py``."""
    for root in _candidate_roots():
        if (root / "generate_all.py").is_file() and (root / "uq_qaoa_core.py").is_file():
            return root
    searched = "\n".join(f"  - {r}" for r in _candidate_roots())
    raise SystemExit(
        "Unable to locate the reproduction artifact (generate_all.py and "
        "uq_qaoa_core.py).\nSearched:\n"
        f"{searched}\n"
        "Run this command from the submission/code directory of the artifact "
        "checkout, or pass --artifact-root."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uqqaoa-reproduce",
        description=(
            "Deterministically regenerate every figure and table for the "
            "OST-QAOA manuscript (global seed 260424803)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Directory where figures/, tables/, and results/ will be written. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument("--depth", type=int, default=3, help="QAOA depth p.")
    parser.add_argument("--budget", type=int, default=24, help="Matched objective-query budget.")
    parser.add_argument("--rank", type=int, default=4, help="Spectral truncation rank.")
    parser.add_argument("--commutator-weight", type=float, default=4.0, help="Noncommutative commutator interaction weight.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Global deterministic seed.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast end-to-end sanity pass at reduced depth and graph counts."
        ),
    )
    parser.add_argument(
        "--legacy-flat",
        action="store_true",
        help="Run the older source-tree generate_all.py driver instead of the package-native generator.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="Legacy source-tree directory containing generate_all.py and uq_qaoa_core.py.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.legacy_flat:
        import os
        import runpy
        import sys

        root = args.artifact_root.resolve() if args.artifact_root else _locate_artifact_root()
        if not (root / "generate_all.py").is_file():
            raise SystemExit(f"generate_all.py not found under {root}")
        sys.path.insert(0, str(root))
        os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig"))
        os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
        if args.smoke:
            os.environ["UQ_QAOA_DEPTH"] = str(SMOKE_DEPTH)
            os.environ["UQ_QAOA_BUDGET"] = str(SMOKE_BUDGET)
        previous_cwd = Path.cwd()
        os.chdir(root)
        try:
            runpy.run_path(str(root / "generate_all.py"), run_name="__main__")
        finally:
            os.chdir(previous_cwd)
        return 0

    from .paper_artifacts import main as regenerate

    forwarded = [
        "--output-dir",
        str(args.output_dir),
        "--depth",
        str(args.depth),
        "--budget",
        str(args.budget),
        "--rank",
        str(args.rank),
        "--commutator-weight",
        str(args.commutator_weight),
        "--seed",
        str(args.seed),
    ]
    if args.smoke:
        forwarded.append("--quick")
    regenerate(forwarded)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
