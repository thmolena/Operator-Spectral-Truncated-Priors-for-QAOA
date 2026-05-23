from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import time

TRACE_COLUMNS = [
    "run_id", "timestamp", "git_commit", "config_path", "method", "ablation", "p", "dim", "n", "graph_family", "graph_seed", "graph_id", "edge_count", "weighted", "query_budget_Q", "query_index", "theta_layout", "theta_json", "source", "coordinate_index", "step_sign", "step_size", "eta", "objective_exact", "objective_noisy", "shots", "incumbent_objective", "incumbent_theta_json", "reference_value", "reference_type", "approximation_ratio", "backend", "threads", "dtype", "rng_seed", "elapsed_ms"
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TraceWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TRACE_COLUMNS)
        self._writer.writeheader()

    def write(self, row: dict):
        clean = {k: row.get(k, "") for k in TRACE_COLUMNS}
        self._writer.writerow(clean)
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def theta_to_json(theta) -> str:
    return json.dumps([float(x) for x in theta])
