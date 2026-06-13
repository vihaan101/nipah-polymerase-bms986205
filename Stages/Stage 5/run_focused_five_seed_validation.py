#!/usr/bin/env python3
"""Run five-seed WT/W730A focused validation for locked-100 finalists."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import (  # noqa: E402
    atomic_write_json,
    build_vina_command,
    config_hash,
    load_docking_box,
    parse_affinity,
    resolve_vina_path,
    run_vina_command,
    sha256_file,
    valid_output,
    validate_workers,
    vina_version,
    write_output_meta,
)

STAGE5_RESULTS = PROJECT_ROOT / "Stages" / "Stage 5" / "results"
WT_RECEPTOR = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "9KNZ_W730A.pdbqt"
DEFAULT_SEEDS = [42, 101, 2023, 999, 1234]


def _load_comparators(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = []
    for item in data.get("comparators", []):
        rows.append({
            "library_id": "COMPARATOR_ERDRP",
            "name": item["name"],
            "ligand_pdbqt": item["ligand_pdbqt"],
            "candidate_role": item["role"],
            "rank_mutation_aware": None,
        })
    return rows


def _job(payload: tuple[dict, str, str, int, dict, bool]) -> dict:
    row, receptor_state, receptor_path, seed, run_config, resume = payload
    try:
        out_dir = Path(run_config["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{row['library_id']}_{receptor_state}_seed_{seed}.pdbqt"
        ligand = Path(row["ligand_pdbqt"])
        receptor = Path(receptor_path)
        cfg_hash = config_hash({
            **run_config,
            "seed": seed,
            "receptor_state": receptor_state,
            "library_id": row["library_id"],
            "name": row["name"],
            "ligand_pdbqt": str(ligand),
            "ligand_sha256": sha256_file(ligand),
            "receptor_path": str(receptor),
            "receptor_sha256": sha256_file(receptor),
        })
        if resume and valid_output(output, cfg_hash):
            affinity = parse_affinity(output)
        else:
            cmd = build_vina_command(
                Path(run_config["vina_path"]),
                receptor,
                ligand,
                output,
                run_config["box"],
                seed=seed,
                cpu=1,
            )
            run_vina_command(cmd)
            affinity = parse_affinity(output)
            write_output_meta(output, cfg_hash, {"command": cmd})
        if affinity is None:
            raise RuntimeError(f"no_affinity_in_output:{output}")
        return {
            "library_id": row["library_id"],
            "name": row["name"],
            "candidate_role": row.get("candidate_role", ""),
            "receptor_state": receptor_state,
            "seed": seed,
            "affinity": affinity,
            "output_pdbqt": str(output),
        }
    except Exception as exc:
        return {
            "library_id": row.get("library_id"),
            "name": row.get("name"),
            "receptor_state": receptor_state,
            "seed": seed,
            "error": str(exc),
        }


def summarize(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (library_id, name, receptor_state), group in matrix.groupby(["library_id", "name", "receptor_state"]):
        values = list(group["affinity"])
        rows.append({
            "library_id": library_id,
            "name": name,
            "receptor_state": receptor_state,
            "count": len(values),
            "mean_affinity": statistics.mean(values),
            "sd_affinity": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seed_values": "; ".join(f"{int(row.seed)}:{row.affinity:.3f}" for row in group.itertuples()),
        })
    summary = pd.DataFrame(rows)
    deltas = []
    for (library_id, name), group in summary.groupby(["library_id", "name"]):
        wt = group[group["receptor_state"] == "WT"]
        mut = group[group["receptor_state"] == "W730A"]
        if wt.empty or mut.empty:
            continue
        wt_row = wt.iloc[0]
        mut_row = mut.iloc[0]
        deltas.append({
            "library_id": library_id,
            "name": name,
            "delta_affinity": mut_row["mean_affinity"] - wt_row["mean_affinity"],
            "propagated_error": math.sqrt(wt_row["sd_affinity"] ** 2 + mut_row["sd_affinity"] ** 2),
        })
    if deltas:
        summary = summary.merge(pd.DataFrame(deltas), on=["library_id", "name"], how="left")
    summary["rank_stability"] = summary.groupby("receptor_state")["mean_affinity"].rank(method="min", ascending=True).astype(int)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused five-seed validation.")
    parser.add_argument("--candidates", type=Path, default=STAGE5_RESULTS / "focused_100_validation_candidates.csv")
    parser.add_argument("--comparator-manifest", type=Path, default=STAGE5_RESULTS / "focused_100_comparator_manifest.json")
    parser.add_argument("--vina-path", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates).to_dict("records")
    candidates.extend(_load_comparators(args.comparator_manifest))
    for row in candidates:
        ligand = Path(row["ligand_pdbqt"])
        if not ligand.exists():
            raise FileNotFoundError(f"Focused validation ligand missing for {row['name']}: {ligand}")

    vina_path = resolve_vina_path(args.vina_path)
    run_config = {
        "vina_path": str(vina_path),
        "vina_version": vina_version(vina_path),
        "box": load_docking_box(),
        "out_dir": str(STAGE5_RESULTS / "focused_100_pdbqt"),
        "cpu_per_job": 1,
    }
    jobs = []
    for row in candidates:
        for seed in args.seeds:
            jobs.append((row, "WT", str(WT_RECEPTOR), seed, run_config, args.resume))
            jobs.append((row, "W730A", str(MUT_RECEPTOR), seed, run_config, args.resume))

    successes = []
    failures = []
    workers = validate_workers(args.workers)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_job, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                failures.append(result)
            else:
                successes.append(result)

    matrix = pd.DataFrame(successes).sort_values(["name", "receptor_state", "seed"]) if successes else pd.DataFrame()
    matrix_path = STAGE5_RESULTS / "focused_100_seed_matrix.csv"
    summary_path = STAGE5_RESULTS / "focused_100_seed_summary.csv"
    failures_path = STAGE5_RESULTS / "focused_100_seed_failures.csv"
    matrix.to_csv(matrix_path, index=False)
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    if failures:
        raise RuntimeError(f"Five-seed validation had {len(failures)} failed jobs; see {failures_path}")
    summary = summarize(matrix)
    summary.to_csv(summary_path, index=False)
    atomic_write_json(STAGE5_RESULTS / "focused_100_run_config.json", {
        **run_config,
        "seeds": args.seeds,
        "candidate_count": len(candidates),
        "matrix_csv": str(matrix_path),
        "summary_csv": str(summary_path),
    })
    print(f"Focused five-seed validation complete: {len(successes)} jobs")


if __name__ == "__main__":
    main()
