"""Run a matched-budget refinement benchmark on the transcriptomic QAOA path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qaoa.transcriptomic import (
    TranscriptomicBenchmarkConfig,
    headline_training_kwargs,
    headline_transcriptomic_benchmark_config,
    run_transcriptomic_matched_budget_benchmark,
)


OUTPUT_DIR = Path("outputs") / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a matched-budget transcriptomic QAOA sweep for heuristic, TQA, and GNN-point initializers. "
            "This runner intentionally excludes UQ-QAOA because the current repository source does not expose "
            "the paper's trust-region policy as executable code."
        )
    )
    parser.add_argument("--depth", type=int, default=2, help="QAOA depth p for the benchmark.")
    parser.add_argument("--benchmark-size", type=int, default=None, help="Optional benchmark graph count override.")
    parser.add_argument("--adaptation-size", type=int, default=None, help="Optional adaptation graph count override.")
    parser.add_argument("--num-starts", type=int, default=None, help="Optional override for classical multistarts.")
    parser.add_argument("--maxiter", type=int, default=None, help="Optional override for classical optimizer iterations.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for GNN training epochs.")
    parser.add_argument("--patience", type=int, default=None, help="Optional override for GNN early stopping patience.")
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[20, 40, 80],
        help="Matched evaluation budgets passed as Nelder-Mead maxfev values.",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default=None,
        help="Optional output filename stem. Defaults to qaoa_transcriptomic_matched_budget_p{depth}.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TranscriptomicBenchmarkConfig:
    base = headline_transcriptomic_benchmark_config()
    return TranscriptomicBenchmarkConfig(
        top_gene_count=base.top_gene_count,
        target_edge_count=base.target_edge_count,
        benchmark_size=base.benchmark_size if args.benchmark_size is None else args.benchmark_size,
        benchmark_seed=base.benchmark_seed,
        adaptation_size=base.adaptation_size if args.adaptation_size is None else args.adaptation_size,
        adaptation_seed=base.adaptation_seed,
        subsample_size=base.subsample_size,
        depth=args.depth,
        num_starts=base.num_starts if args.num_starts is None else args.num_starts,
        maxiter=base.maxiter if args.maxiter is None else args.maxiter,
        seed_offset=base.seed_offset,
        training_seed=base.training_seed,
    )


def build_training_kwargs(args: argparse.Namespace) -> dict[str, object]:
    training_kwargs = headline_training_kwargs()
    if args.epochs is not None:
        training_kwargs["epochs"] = args.epochs
    if args.patience is not None:
        training_kwargs["patience"] = args.patience
    return training_kwargs


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    training_kwargs = build_training_kwargs(args)
    detailed, summary, metadata = run_transcriptomic_matched_budget_benchmark(
        config=config,
        training_kwargs=training_kwargs,
        budgets=args.budgets,
    )

    output_stem = args.output_stem or f"qaoa_transcriptomic_matched_budget_p{config.depth}"
    detailed.to_csv(OUTPUT_DIR / f"{output_stem}_detailed.csv", index=False)
    summary.to_csv(OUTPUT_DIR / f"{output_stem}_summary.csv", index=False)
    metadata_payload = {
        "config": asdict(metadata["config"]),
        "training": metadata["training"],
        "training_kwargs": metadata["training_kwargs"],
        "matched_budget": {
            "budgets": metadata["matched_budget"]["budgets"],
            "tqa_best_total_time": metadata["matched_budget"]["tqa_best_total_time"],
            "tqa_best_mean_ratio": metadata["matched_budget"]["tqa_best_mean_ratio"],
            "note": metadata["matched_budget"]["note"],
        },
    }
    with (OUTPUT_DIR / f"{output_stem}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)
    metadata["matched_budget"]["tqa_grid_scores"].to_csv(
        OUTPUT_DIR / f"{output_stem}_tqa_grid.csv",
        index=False,
    )


if __name__ == "__main__":
    main()