#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.append(str(PARENT_DIR))

from stage7_eval_common import load_stage6_cases, resolve_stage6_path, robust_universe
from stage7_production_validation import TIER_TIME_RANGES, validate_trajectory


TIER_LABEL = "10ns_direct"
TOTAL_STEPS = 5_000_000
DCD_INTERVAL = 2_000
EXPECTED_FRAMES = TOTAL_STEPS // DCD_INTERVAL
MIN_REQUIRED_BYTES = {
    "production.dcd": 1,
    "production.log": 1,
    "production_final.pdb": 1,
}
OPTIONAL_FILES = ("production.chk", "TASK_COMPLETE", "_case_manifest.json")
FILE_ATTRS = {
    "production.dcd": "production_dcd",
    "production_final.pdb": "production_final_pdb",
    "production.log": "production_log",
    "production.chk": "production_chk",
    "TASK_COMPLETE": "task_complete",
    "_case_manifest.json": "case_manifest",
}


@dataclass
class ReplicatePaths:
    replicate_dir: Path
    production_dcd: Path
    production_final_pdb: Path
    production_log: Path
    production_chk: Path
    task_complete: Path
    case_manifest: Path


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Stage 7 10ns_direct replicate trajectories for evaluation readiness.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Path to stage7_10ns_direct/results root containing <CASE>/replicate_<N>/ directories.",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Restrict verification to one or more case IDs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write machine-readable manifest JSON to this path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Write human-readable markdown summary to this path.",
    )
    parser.add_argument(
        "--fail-on-not-ready",
        action="store_true",
        help="Exit non-zero if any selected replicate is not evaluation_ready.",
    )
    return parser.parse_args()


def _build_paths(results_root: Path, case_id: str, replicate_id: int) -> ReplicatePaths:
    rep_dir = results_root / case_id / f"replicate_{replicate_id}"
    return ReplicatePaths(
        replicate_dir=rep_dir,
        production_dcd=rep_dir / "production.dcd",
        production_final_pdb=rep_dir / "production_final.pdb",
        production_log=rep_dir / "production.log",
        production_chk=rep_dir / "production.chk",
        task_complete=rep_dir / "TASK_COMPLETE",
        case_manifest=rep_dir / "_case_manifest.json",
    )


def _check_required_files(paths: ReplicatePaths) -> tuple[dict[str, str], list[str], list[str]]:
    checks: dict[str, str] = {}
    failures: list[str] = []
    warnings: list[str] = []

    for filename, min_size in MIN_REQUIRED_BYTES.items():
        path = getattr(paths, FILE_ATTRS[filename])
        if not path.exists():
            checks[f"{filename}_present"] = "fail"
            failures.append(f"missing required file: {filename}")
            continue
        checks[f"{filename}_present"] = "pass"
        size = path.stat().st_size
        if size < min_size:
            checks[f"{filename}_non_empty"] = "fail"
            failures.append(f"required file is empty: {filename}")
        else:
            checks[f"{filename}_non_empty"] = "pass"

    for filename in OPTIONAL_FILES:
        path = getattr(paths, FILE_ATTRS[filename])
        if path.exists():
            checks[f"{filename}_present"] = "pass"
        else:
            checks[f"{filename}_present"] = "warn"
            warnings.append(f"optional file missing: {filename}")

    return checks, failures, warnings


def _resolve_topology_path(case_id: str, stage6_cases: dict[str, dict[str, Any]]) -> Path:
    case_data = stage6_cases[case_id]
    topology_value = case_data.get("topology_pdb")
    if not topology_value:
        raise RuntimeError(f"Stage 6 manifest missing topology_pdb for {case_id}")
    topology_path = Path(resolve_stage6_path(topology_value))
    if not topology_path.exists():
        raise FileNotFoundError(f"Stage 6 topology missing for {case_id}: {topology_path}")
    return topology_path


def _inspect_dcd(topology_path: Path, dcd_path: Path) -> dict[str, Any]:
    universe = robust_universe(str(topology_path), str(dcd_path))
    n_frames = len(universe.trajectory)
    if n_frames <= 0:
        raise RuntimeError("trajectory has zero frames")

    first_time_ps = float(universe.trajectory[0].time)
    last_time_ps = float(universe.trajectory[-1].time)

    return {
        "n_frames": n_frames,
        "first_time_ps": first_time_ps,
        "last_time_ps": last_time_ps,
        "frames_scanned": 2 if n_frames > 1 else 1,
    }


def _run_contract_validation(dcd_path: Path, topology_path: Path) -> None:
    validate_trajectory(str(dcd_path), str(topology_path), TIER_LABEL, TOTAL_STEPS, DCD_INTERVAL)


def inspect_replicate(
    results_root: Path,
    case_id: str,
    replicate_id: int,
    stage6_cases: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paths = _build_paths(results_root, case_id, replicate_id)
    checks, failures, warnings = _check_required_files(paths)

    result: dict[str, Any] = {
        "status": "not_ready",
        "paths": {
            "replicate_dir": str(paths.replicate_dir),
            "production_dcd": str(paths.production_dcd),
            "production_final_pdb": str(paths.production_final_pdb),
            "production_log": str(paths.production_log),
            "production_chk": str(paths.production_chk),
            "task_complete": str(paths.task_complete),
            "case_manifest": str(paths.case_manifest),
        },
        "checks": checks,
        "metrics": {
            "expected_frames": EXPECTED_FRAMES,
            "expected_first_time_ps": TIER_TIME_RANGES[TIER_LABEL][0],
            "expected_last_time_ps": TIER_TIME_RANGES[TIER_LABEL][1],
            "dcd_size_bytes": paths.production_dcd.stat().st_size if paths.production_dcd.exists() else 0,
            "production_final_pdb_size_bytes": paths.production_final_pdb.stat().st_size if paths.production_final_pdb.exists() else 0,
            "production_log_size_bytes": paths.production_log.stat().st_size if paths.production_log.exists() else 0,
        },
        "warnings": warnings,
        "failure_reasons": failures,
    }

    if failures:
        return result

    topology_path = _resolve_topology_path(case_id, stage6_cases)
    result["paths"]["stage6_topology"] = str(topology_path)

    try:
        metrics = _inspect_dcd(topology_path, paths.production_dcd)
        result["checks"]["dcd_readable"] = "pass"
        result["checks"]["topology_load"] = "pass"
        result["checks"]["sequential_scan"] = "pass"
        result["metrics"].update(metrics)
    except Exception as exc:
        result["checks"]["dcd_readable"] = "fail"
        result["checks"]["topology_load"] = "fail"
        result["checks"]["sequential_scan"] = "fail"
        result["failure_reasons"].append(f"DCD inspection failed: {exc}")
        return result

    try:
        _run_contract_validation(paths.production_dcd, topology_path)
        result["checks"]["frame_count"] = "pass"
        result["checks"]["time_span"] = "pass"
    except Exception as exc:
        result["checks"]["frame_count"] = "fail"
        result["checks"]["time_span"] = "fail"
        result["failure_reasons"].append(f"Trajectory contract validation failed: {exc}")
        return result

    result["status"] = "evaluation_ready"
    return result


def verify_results(results_root: Path, selected_cases: list[str] | None = None) -> dict[str, Any]:
    stage6_cases = load_stage6_cases()
    case_ids = selected_cases or sorted(stage6_cases.keys())
    manifest_cases: dict[str, dict[str, Any]] = {}
    summary_counter: Counter[str] = Counter()

    for case_id in case_ids:
        if case_id not in stage6_cases:
            raise KeyError(f"Requested case '{case_id}' not found in Stage 6 manifest")

        case_results: dict[str, Any] = {}
        for replicate_id in range(1, 6):
            rep_key = f"replicate_{replicate_id}"
            rep_result = inspect_replicate(results_root, case_id, replicate_id, stage6_cases)
            summary_counter[rep_result["status"]] += 1
            case_results[rep_key] = rep_result
        manifest_cases[case_id] = case_results

    evaluation_ready_replicates = {
        case_id: [
            rep_key
            for rep_key, rep_result in case_results.items()
            if rep_result["status"] == "evaluation_ready"
        ]
        for case_id, case_results in manifest_cases.items()
    }

    return {
        "campaign": TIER_LABEL,
        "generated_at": _ts(),
        "source_root": str(results_root),
        "tier": TIER_LABEL,
        "cases": manifest_cases,
        "evaluation_ready_replicates": evaluation_ready_replicates,
        "summary": {
            "evaluation_ready": summary_counter.get("evaluation_ready", 0),
            "not_ready": summary_counter.get("not_ready", 0),
        },
    }


def _render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Stage 7 10ns Direct Evaluation-Ready Verifier",
        "",
        f"- Generated: `{manifest['generated_at']}`",
        f"- Source root: `{manifest['source_root']}`",
        f"- Tier: `{manifest['tier']}`",
        "",
        "## Summary",
        "",
        f"- `evaluation_ready`: {manifest['summary']['evaluation_ready']}",
        f"- `not_ready`: {manifest['summary']['not_ready']}",
        "",
        "## Replicates",
        "",
        "| Case | Replicate | Status | Frames | Time Span (ps) | Notes |",
        "|---|---:|---|---:|---|---|",
    ]

    for case_id, case_results in manifest["cases"].items():
        for rep_key, rep_result in sorted(case_results.items()):
            metrics = rep_result["metrics"]
            notes = rep_result["failure_reasons"] or rep_result["warnings"] or ["OK"]
            lines.append(
                "| "
                f"{case_id} | {rep_key.split('_')[-1]} | {rep_result['status']} | "
                f"{metrics.get('n_frames', 'NA')} | "
                f"{metrics.get('first_time_ps', 'NA')} -> {metrics.get('last_time_ps', 'NA')} | "
                f"{'; '.join(notes)} |",
            )

    return "\n".join(lines) + "\n"


def _write_outputs(manifest: dict[str, Any], output_json: Path | None, output_md: Path | None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(manifest, indent=2) + "\n")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_render_markdown(manifest))


def main() -> int:
    args = _parse_args()
    manifest = verify_results(args.results_root, args.cases)
    _write_outputs(manifest, args.output_json, args.output_md)
    print(json.dumps(manifest["summary"], indent=2))

    if args.fail_on_not_ready and manifest["summary"]["not_ready"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
