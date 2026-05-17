from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def new_run_id() -> str:
    """Return a filesystem-friendly local timestamp for one experiment run."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(run_root: Path, run_id: str | None = None) -> Path:
    """Create and return runs/<experiment>/<timestamp>."""
    run_root = Path(run_root)
    resolved_run_id = str(run_id) if run_id else new_run_id()
    run_dir = run_root / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def timestamped_run_dirs(run_root: Path) -> list[Path]:
    run_root = Path(run_root)
    if not run_root.exists():
        return []
    return sorted(
        [path for path in run_root.iterdir() if path.is_dir() and RUN_ID_PATTERN.match(path.name)],
        key=lambda path: path.name,
    )


def latest_run_dir(run_root: Path) -> Path:
    """Return the newest timestamped run directory under one experiment root."""
    run_dirs = timestamped_run_dirs(run_root)
    if not run_dirs:
        raise FileNotFoundError(
            f"No timestamped run directory found under {run_root}. "
            "Please train the model first or set run_id explicitly."
        )
    return run_dirs[-1]


def resolve_existing_run_dir(run_root: Path, run_id: str | None = None) -> Path:
    """Resolve one existing run, defaulting to the newest timestamped run."""
    if run_id:
        run_dir = Path(run_root) / str(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir
    return latest_run_dir(Path(run_root))
