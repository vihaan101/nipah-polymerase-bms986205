#!/usr/bin/env python3
"""Freeze focused validation candidates from locked-100 mutation-aware ranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import atomic_write_json, sha256_file  # noqa: E402

STAGE3_RESULTS = PROJECT_ROOT / "Stages" / "Stage 3" / "results"
STAGE3_DATA = PROJECT_ROOT / "Stages" / "Stage 3" / "data"
STAGE5_RESULTS = PROJECT_ROOT / "Stages" / "Stage 5" / "results"
BMS_NAMES = {"BMS-986205", "Linrodostat"}
ERDRP_LIGAND = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "ligands" / "ERDRP.pdbqt"


def select_candidates(ranked: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    if ranked.empty:
        raise RuntimeError("Mutation-ranked input is empty")
    ranked = ranked.sort_values("rank_mutation_aware").copy()
    selected = ranked.head(top_n).copy()
    if not selected["name"].isin(BMS_NAMES).any():
        bms = ranked[ranked["name"].isin(BMS_NAMES)]
        if bms.empty:
            raise RuntimeError("BMS-986205/Linrodostat is absent from mutation-ranked results")
        selected = pd.concat([selected, bms.head(1)], ignore_index=True)
    selected = selected.drop_duplicates("library_id", keep="first").sort_values("rank_mutation_aware")
    selected["candidate_role"] = selected["name"].apply(lambda name: "prespecified_bms" if name in BMS_NAMES else "mutation_ranked_finalist")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select focused validation candidates.")
    parser.add_argument("--ranked", type=Path, default=STAGE3_RESULTS / "library_100_mutation_ranked.csv")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", type=Path, default=STAGE5_RESULTS / "focused_100_validation_candidates.csv")
    parser.add_argument("--manifest", type=Path, default=STAGE5_RESULTS / "focused_100_validation_candidates_manifest.json")
    parser.add_argument("--comparator-manifest", type=Path, default=STAGE5_RESULTS / "focused_100_comparator_manifest.json")
    args = parser.parse_args()

    ranked = pd.read_csv(args.ranked)
    selected = select_candidates(ranked, top_n=args.top_n)
    ligand_dir = STAGE3_DATA / "library_100_ligands"
    selected["ligand_pdbqt"] = selected["library_id"].apply(lambda library_id: str(ligand_dir / f"{library_id}.pdbqt"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)
    atomic_write_json(args.manifest, {
        "source_ranked": str(args.ranked),
        "source_ranked_sha256": sha256_file(args.ranked),
        "top_n": args.top_n,
        "candidate_count": int(len(selected)),
        "candidates_csv": str(args.output),
        "frozen_before_five_seed_docking": True,
    })
    atomic_write_json(args.comparator_manifest, {
        "comparators": [
            {
                "name": "ERDRP-0519",
                "role": "external_pocket_relevant_comparator",
                "ligand_pdbqt": str(ERDRP_LIGAND),
                "broad_library_member": False,
            }
        ],
        "erdrp_exclusion_statement": "ERDRP-0519 is outside the locked Broad library count.",
    })
    print(f"Wrote {len(selected)} focused validation candidates to {args.output}")


if __name__ == "__main__":
    main()
