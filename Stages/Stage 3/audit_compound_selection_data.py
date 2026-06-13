#!/usr/bin/env python3
"""Audit compound-selection inputs and historical artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import atomic_write_json, sha256_file  # noqa: E402

ARTIFACTS = [
    "Stages/Stage 3/data/expanded_library.csv",
    "Stages/Stage 3/data/filtered_library.csv",
    "Stages/Stage 3/results/resistance_screening_results.csv",
    "Stages/data/lake/Broad_Repurposing_Hub.csv",
    "Stages/data/lake/Broad_Repurposing_Samples.csv",
    "Stages/Stage 4/results/verification/verification_results.csv",
    "Stages/Stage 4/results/verification/ghost_clash_verdict.json",
    "Stages/Stage 4/results/verification/admet_results.csv",
    "Stages/Stage 5/results/stage5_verdict.json",
    "Stages/Stage 5/results/focused_100_seed_summary.csv",
]


def _csv_shape(path: Path) -> tuple[int | None, int | None]:
    try:
        with open(path, newline="", encoding="utf-8", errors="ignore") as handle:
            rows = list(csv.reader(handle))
    except Exception:
        return None, None
    if not rows:
        return 0, 0
    return max(0, len(rows) - 1), len(rows[0])


def audit_artifact(path: Path) -> dict:
    row = {
        "artifact_path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "row_count": None,
        "column_count": None,
        "sha256": None,
        "created_by_script": None,
        "used_in_manuscript_table": None,
        "status": "missing",
    }
    if not path.exists():
        return row
    row["sha256"] = sha256_file(path)
    row["status"] = "historical" if "Stage 3" in str(path) and "library_100" not in path.name else "current"
    if path.suffix.lower() == ".csv":
        row["row_count"], row["column_count"] = _csv_shape(path)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit compound-selection data artifacts.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "Stages" / "Stage 3" / "data" / "compound_selection_data_audit.json",
    )
    args = parser.parse_args()

    rows = [audit_artifact(PROJECT_ROOT / artifact) for artifact in ARTIFACTS]
    atomic_write_json(args.output, rows)
    print(f"Wrote audit for {len(rows)} artifacts to {args.output}")


if __name__ == "__main__":
    main()
