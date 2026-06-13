#!/usr/bin/env python3
"""WT and paired WT/W730A screening for the locked 100-compound library."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import (  # noqa: E402
    add_mutation_composite_scores,
    atomic_write_json,
    build_vina_command,
    config_hash,
    get_ghost_sidechain_coords,
    load_docking_box,
    min_distance_to_ghost,
    parse_affinity,
    pose_center_shift,
    resolve_vina_path,
    run_vina_command,
    sha256_file,
    valid_output,
    validate_workers,
    vina_version,
    write_output_meta,
)

STAGE3_DIR = PROJECT_ROOT / "Stages" / "Stage 3"
STAGE3_DATA = STAGE3_DIR / "data"
STAGE3_RESULTS = STAGE3_DIR / "results"
LOCKED_LIBRARY = STAGE3_DATA / "library_100_locked.csv"
WT_RECEPTOR = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "9KNZ_W730A.pdbqt"
WT_PDB = PROJECT_ROOT / "Stages" / "Stage 1" / "data" / "9KNZ_clean.pdb"


def _require_rdkit():
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise RuntimeError(
            "RDKit and Meeko are required. Run with an environment such as "
            "~/miniforge/envs/md_mac/bin/python."
        ) from exc
    return Chem, AllChem, MoleculePreparation, PDBQTWriterLegacy


def stable_output_name(row: dict) -> str:
    return str(row["library_id"])


def ligand_identity(row: dict, library_sha256: str | None) -> dict:
    return {
        "library_id": row.get("library_id"),
        "name": row.get("name"),
        "smiles": row.get("smiles"),
        "canonical_smiles": row.get("canonical_smiles"),
        "inchikey": row.get("inchikey"),
        "library_sha256": library_sha256,
        "prep_version": "locked100_ligand_prep_v1",
    }


def docking_identity(
    row: dict,
    run_config: dict,
    receptor: Path,
    ligand_hash: str,
    *,
    receptor_state: str,
    mode: str,
) -> dict:
    return {
        "mode": mode,
        "receptor_state": receptor_state,
        "library_id": row.get("library_id"),
        "name": row.get("name"),
        "canonical_smiles": row.get("canonical_smiles"),
        "inchikey": row.get("inchikey"),
        "library_sha256": run_config.get("library_sha256"),
        "ligand_sha256": ligand_hash,
        "receptor_path": str(receptor),
        "receptor_sha256": sha256_file(receptor),
        "vina_path": run_config["vina_path"],
        "vina_version": run_config["vina_version"],
        "seed": run_config["seed"],
        "box": run_config["box"],
        "cpu_per_job": 1,
    }


def prepare_ligand(row: dict, ligand_dir: Path, *, library_sha256: str | None, resume: bool) -> Path:
    Chem, AllChem, MoleculePreparation, PDBQTWriterLegacy = _require_rdkit()
    ligand_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = ligand_dir / f"{stable_output_name(row)}.pdbqt"
    config_id = config_hash(ligand_identity(row, library_sha256))
    if resume and valid_output(ligand_path, config_id):
        return ligand_path
    mol = Chem.MolFromSmiles(str(row["canonical_smiles"] or row["smiles"]))
    if mol is None:
        raise RuntimeError("invalid_smiles")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        raise RuntimeError("3d_embedding_failed")
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    setups = MoleculePreparation().prepare(mol)
    pdbqt = PDBQTWriterLegacy.write_string(setups[0])[0]
    ligand_path.write_text(pdbqt, encoding="utf-8")
    write_output_meta(ligand_path, config_id, {"name": row.get("name"), "library_id": row.get("library_id")})
    return ligand_path


def dock_one(
    row: dict,
    receptor: Path,
    output: Path,
    ligand_dir: Path,
    run_config: dict,
    *,
    receptor_state: str,
    mode: str,
    resume: bool,
) -> dict:
    ligand = prepare_ligand(
        row,
        ligand_dir,
        library_sha256=run_config.get("library_sha256"),
        resume=resume,
    )
    dock_id = config_hash(docking_identity(
        row,
        run_config,
        receptor,
        sha256_file(ligand),
        receptor_state=receptor_state,
        mode=mode,
    ))
    if resume and valid_output(output, dock_id):
        affinity = parse_affinity(output)
        if affinity is not None:
            return {"output": str(output), "affinity": affinity}
    cmd = build_vina_command(
        Path(run_config["vina_path"]),
        receptor,
        ligand,
        output,
        run_config["box"],
        seed=int(run_config["seed"]),
        cpu=1,
    )
    run_vina_command(cmd)
    affinity = parse_affinity(output)
    if affinity is None:
        raise RuntimeError(f"no_affinity_in_output:{output}")
    write_output_meta(output, dock_id, {"command": cmd, "library_id": row.get("library_id")})
    return {"output": str(output), "affinity": affinity}


def _wt_worker(payload: tuple[dict, dict, str, bool]) -> dict:
    row, run_config, out_root, resume = payload
    try:
        output_dir = Path(out_root) / "library_100_wt_pdbqt"
        output_dir.mkdir(parents=True, exist_ok=True)
        ligand_dir = Path(out_root).parent / "data" / "library_100_ligands"
        out = output_dir / f"{stable_output_name(row)}_wt.pdbqt"
        docked = dock_one(
            row,
            WT_RECEPTOR,
            out,
            ligand_dir,
            run_config,
            receptor_state="WT",
            mode="wt",
            resume=resume,
        )
        return {
            "library_id": row["library_id"],
            "name": row["name"],
            "wt_affinity": docked["affinity"],
            "wt_output": docked["output"],
            "bms_inclusion_mode": row.get("bms_inclusion_mode", ""),
            "selection_reason": "",
        }
    except Exception as exc:
        return {"library_id": row.get("library_id"), "name": row.get("name"), "error": str(exc)}


def selection_reason(row: pd.Series, top_ids: set[str], threshold: float) -> str:
    reasons = []
    if row["library_id"] in top_ids:
        reasons.append("top20_wt")
    if row["wt_affinity"] <= threshold:
        reasons.append("wt_threshold")
    if str(row.get("bms_inclusion_mode", "")):
        reasons.append("prespecified_bms")
    return ";".join(reasons)


def run_wt(args: argparse.Namespace, run_config: dict) -> None:
    STAGE3_RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.library)
    workers = validate_workers(args.workers)
    payloads = [(row.to_dict(), run_config, str(STAGE3_RESULTS), args.resume) for _, row in df.iterrows()]
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_wt_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                failures.append(result)
            else:
                rows.append(result)
    result_df = pd.DataFrame(rows).sort_values(["wt_affinity", "name"]) if rows else pd.DataFrame()
    if not result_df.empty:
        top_ids = set(result_df.head(20)["library_id"])
        result_df["selection_reason"] = result_df.apply(selection_reason, axis=1, top_ids=top_ids, threshold=args.wt_threshold)
    result_df.to_csv(STAGE3_RESULTS / "library_100_wt_results.csv", index=False)
    result_df.to_csv(STAGE3_RESULTS / "library_100_wt_ranked.csv", index=False)
    pd.DataFrame(failures).to_csv(STAGE3_RESULTS / "library_100_wt_failures.csv", index=False)
    atomic_write_json(STAGE3_RESULTS / "library_100_wt_config.json", provenance_config(args, run_config))
    if failures:
        raise RuntimeError(
            f"WT screening had {len(failures)} failed compounds; "
            f"see {STAGE3_RESULTS / 'library_100_wt_failures.csv'}"
        )
    if len(rows) != len(df):
        raise RuntimeError(f"WT screening expected {len(df)} result rows, wrote {len(rows)}")
    print(f"WT screening complete: {len(rows)} successes, {len(failures)} failures")


def _paired_worker(payload: tuple[dict, dict, str, bool]) -> dict:
    row, run_config, out_root, resume = payload
    try:
        output_dir = Path(out_root) / "library_100_paired_pdbqt"
        output_dir.mkdir(parents=True, exist_ok=True)
        ligand_dir = Path(out_root).parent / "data" / "library_100_ligands"
        stem = stable_output_name(row)
        wt_out = output_dir / f"{stem}_wt.pdbqt"
        mut_out = output_dir / f"{stem}_mut.pdbqt"
        wt = dock_one(
            row,
            WT_RECEPTOR,
            wt_out,
            ligand_dir,
            run_config,
            receptor_state="WT",
            mode="paired",
            resume=resume,
        )
        mut = dock_one(
            row,
            MUT_RECEPTOR,
            mut_out,
            ligand_dir,
            run_config,
            receptor_state="W730A",
            mode="paired",
            resume=resume,
        )
        ghost_coords = get_ghost_sidechain_coords(WT_PDB)
        wt_dist = min_distance_to_ghost(wt_out, ghost_coords)
        mut_dist = min_distance_to_ghost(mut_out, ghost_coords)
        delta_affinity = mut["affinity"] - wt["affinity"]
        delta_dist = mut_dist - wt_dist
        return {
            "library_id": row["library_id"],
            "name": row["name"],
            "wt_affinity": wt["affinity"],
            "mut_affinity": mut["affinity"],
            "delta_affinity": delta_affinity,
            "wt_dist": wt_dist,
            "mut_dist": mut_dist,
            "delta_dist": delta_dist,
            "ghost_clash_dist": mut_dist,
            "ghost_clash": mut_dist < 2.5,
            "pose_center_shift": pose_center_shift(wt_out, mut_out),
            "selection_reason": row.get("selection_reason", ""),
            "wt_output": wt["output"],
            "mut_output": mut["output"],
        }
    except Exception as exc:
        return {"library_id": row.get("library_id"), "name": row.get("name"), "error": str(exc)}


def carry_forward_rows(args: argparse.Namespace) -> pd.DataFrame:
    wt = pd.read_csv(args.wt_results)
    library = pd.read_csv(args.library)
    merged = wt.merge(library, on=["library_id", "name"], how="left", suffixes=("", "_library"))
    carry = merged[merged["selection_reason"].fillna("") != ""].copy()
    if carry.empty:
        raise RuntimeError("No carry-forward compounds selected from WT results")
    return carry


def run_paired(args: argparse.Namespace, run_config: dict) -> None:
    STAGE3_RESULTS.mkdir(parents=True, exist_ok=True)
    df = carry_forward_rows(args)
    workers = validate_workers(args.workers)
    payloads = [(row.to_dict(), run_config, str(STAGE3_RESULTS), args.resume) for _, row in df.iterrows()]
    rows = []
    failures = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_paired_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                failures.append(result)
            else:
                rows.append(result)
    scored = add_mutation_composite_scores(rows)
    pd.DataFrame(scored).to_csv(STAGE3_RESULTS / "library_100_paired_results.csv", index=False)
    pd.DataFrame(scored).to_csv(STAGE3_RESULTS / "library_100_mutation_ranked.csv", index=False)
    pd.DataFrame(failures).to_csv(STAGE3_RESULTS / "library_100_paired_failures.csv", index=False)
    reconciliation = [
        {
            "library_id": row["library_id"],
            "name": row["name"],
            "ghost_clash_dist": row["ghost_clash_dist"],
            "ghost_warning": row["ghost_warning"],
            "claim_boundary": "warning_only_until_stage3_stage4_standalone_methods_reconciled",
        }
        for row in scored
    ]
    pd.DataFrame(reconciliation).to_csv(STAGE3_RESULTS / "library_100_ghost_reconciliation.csv", index=False)
    if failures:
        raise RuntimeError(
            f"Paired screening had {len(failures)} failed compounds; "
            f"see {STAGE3_RESULTS / 'library_100_paired_failures.csv'}"
        )
    if len(rows) != len(df):
        raise RuntimeError(f"Paired screening expected {len(df)} result rows, wrote {len(rows)}")
    print(f"Paired screening complete: {len(rows)} successes, {len(failures)} failures")


def provenance_config(args: argparse.Namespace, run_config: dict) -> dict:
    data = {
        "mode": args.mode,
        "library": str(args.library),
        "library_sha256": sha256_file(args.library) if args.library.exists() else None,
        "wt_receptor": str(WT_RECEPTOR),
        "mut_receptor": str(MUT_RECEPTOR),
        "wt_pdb": str(WT_PDB),
        "vina_path": run_config["vina_path"],
        "vina_version": run_config["vina_version"],
        "seed": run_config["seed"],
        "workers": args.workers,
        "cpu_per_job": 1,
        "resume": args.resume,
    }
    if args.mode == "wt":
        data["wt_threshold"] = args.wt_threshold
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen locked-100 library.")
    parser.add_argument("--mode", choices=["wt", "paired"], required=True)
    parser.add_argument("--library", type=Path, default=LOCKED_LIBRARY)
    parser.add_argument("--wt-results", type=Path, default=STAGE3_RESULTS / "library_100_wt_ranked.csv")
    parser.add_argument("--vina-path", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wt-threshold", type=float, default=-8.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    STAGE3_RESULTS.mkdir(parents=True, exist_ok=True)
    vina_path = resolve_vina_path(args.vina_path)
    run_config = {
        "vina_path": str(vina_path),
        "vina_version": vina_version(vina_path),
        "seed": args.seed,
        "box": load_docking_box(),
        "library_sha256": sha256_file(args.library),
    }
    if args.mode == "wt":
        run_wt(args, run_config)
    else:
        run_paired(args, run_config)


if __name__ == "__main__":
    main()
