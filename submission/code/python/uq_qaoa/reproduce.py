"""Console entry point for deterministic regeneration of the UQ-QAOA artifacts.

The figure and table generators that back the manuscript live alongside the
reproduction artifact (``submission/code/generate_all.py`` and the flat
``uq_qaoa_core`` module). This thin wrapper locates that master generator,
configures the fixed global seed via the environment, and executes it so that
the figures and tables can be regenerated through the installed console script.

The regeneration is deterministic given the global seed ``260424803``; an
optional smoke mode reduces the QAOA depth and query budget for a fast
end-to-end sanity pass.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
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
            "UQ-QAOA manuscript (global seed 260424803)."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Directory containing generate_all.py and uq_qaoa_core.py. "
            "Defaults to the artifact directory bundled with this package."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast sanity pass at reduced depth (p=2) and budget (Q=12) instead "
            "of the publication configuration (p=3, Q=18)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = args.artifact_root.resolve() if args.artifact_root else _locate_artifact_root()
    if not (root / "generate_all.py").is_file():
        raise SystemExit(f"generate_all.py not found under {root}")

    # The flat generator modules import one another by bare name; make the
    # artifact directory importable and the working directory for relative
    # output paths.
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
