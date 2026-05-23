from __future__ import annotations

import hashlib
from pathlib import Path
from .traces import git_commit, now_iso


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_metadata(config_path: str, backend: str, threads: int) -> dict[str, str | int]:
    return {"timestamp": now_iso(), "git_commit": git_commit(), "config_path": config_path, "backend": backend, "threads": threads}
