"""Repository path helpers shared across Stages pipelines."""

from __future__ import annotations

import os
from pathlib import Path

# Default remote paper tree (override with NIPAH_PROJECT_ROOT if needed).
DEFAULT_PAPER_ROOT = Path(os.environ.get("NIPAH_PAPER_ROOT", "/home/vihaan5/paper"))


def project_root() -> Path:
    """Return the nipahv-rdrp-md repo root (directory containing Stages/Stage 1)."""
    explicit = os.environ.get("NIPAH_PROJECT_ROOT", "").strip()
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if (root / "Stages" / "Stage 1").is_dir():
            return root
        raise RuntimeError(f"NIPAH_PROJECT_ROOT is not a valid repo root: {root}")
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "Stages" / "Stage 1").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root (expected Stages/Stage 1)")
