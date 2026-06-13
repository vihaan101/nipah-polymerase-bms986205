#!/usr/bin/env python3
"""Create the locked 100-compound Broad Repurposing Hub library."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import atomic_write_json, sha256_file  # noqa: E402

STAGE3_DIR = PROJECT_ROOT / "Stages" / "Stage 3"
STAGE3_DATA = STAGE3_DIR / "data"
LAKE_DIR = PROJECT_ROOT / "Stages" / "data" / "lake"
BMS_NAMES = {"BMS-986205", "Linrodostat"}
ERDRP_NAMES = {"ERDRP-0519", "ERDRP"}
DOCKABLE_ATOMIC_NUMBERS = {1, 5, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53}
PHASE_ORDER = {
    "launched": 0,
    "phase 4": 1,
    "phase 3": 2,
    "phase 2": 3,
    "phase 1": 4,
    "preclinical": 5,
    "": 6,
}


def _require_rdkit():
    try:
        import rdkit
        from rdkit import Chem
        from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required. Run with an environment such as "
            "~/miniforge/envs/md_mac/bin/python."
        ) from exc
    return rdkit, Chem, AllChem, Crippen, Descriptors, rdMolDescriptors, MaxMinPicker


def clinical_rank(value: object) -> int:
    text = "" if pd.isna(value) else str(value).strip().lower()
    return PHASE_ORDER.get(text, 6)


def stable_library_id(idx: int) -> str:
    return f"LIB100_{idx:04d}"


def _first_present(row: pd.Series, names: list[str]) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return ""


def _normalize_qc(value: object) -> str:
    if pd.isna(value):
        return "0"
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def dockability_failure_reason(Chem, mol) -> str | None:
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) != 1:
        return "multi_fragment_structure"
    unsupported = sorted({atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() not in DOCKABLE_ATOMIC_NUMBERS})
    if unsupported:
        return "unsupported_docking_elements:" + ",".join(unsupported)
    return None


def build_candidates(drugs_path: Path, samples_path: Path, *, strict_qc: bool) -> tuple[pd.DataFrame, list[dict], dict]:
    rdkit, Chem, AllChem, Crippen, Descriptors, rdMolDescriptors, _picker = _require_rdkit()
    drugs = pd.read_csv(drugs_path, sep="\t", comment="!")
    samples = pd.read_csv(samples_path, sep="\t", comment="!")
    diagnostics = {
        "drug_rows": int(len(drugs)),
        "sample_rows": int(len(samples)),
    }
    merged = pd.merge(samples, drugs, on="pert_iname", how="inner", suffixes=("_sample", "_drug"))
    diagnostics["merged_rows"] = int(len(merged))

    rows = []
    failures = []
    for _, row in merged.iterrows():
        name = str(row.get("pert_iname", "")).strip()
        smiles = _first_present(row, ["smiles", "smiles_sample", "smiles_drug"])
        qc = _first_present(row, ["qc_incompatible", "qc_incompatible_sample", "qc_incompatible_drug"])
        if name in ERDRP_NAMES:
            failures.append({"name": name, "reason": "external_comparator_not_broad_library_member"})
            continue
        if strict_qc and _normalize_qc(qc) not in {"0", "False", "false", ""}:
            failures.append({"name": name, "reason": "qc_incompatible"})
            continue
        if pd.isna(smiles) or not str(smiles).strip():
            failures.append({"name": name, "reason": "missing_smiles"})
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            failures.append({"name": name, "reason": "invalid_smiles", "smiles": str(smiles)})
            continue
        dockability_failure = dockability_failure_reason(Chem, mol)
        if dockability_failure:
            failures.append({"name": name, "reason": dockability_failure, "smiles": str(smiles)})
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True)
        inchikey = Chem.MolToInchiKey(mol) or canonical
        mw = Descriptors.MolWt(mol)
        clogp = Crippen.MolLogP(mol)
        rows.append({
            "name": name,
            "pert_iname": name,
            "smiles": str(smiles),
            "canonical_smiles": canonical,
            "inchikey": inchikey,
            "mw": round(float(mw), 4),
            "clogp": round(float(clogp), 4),
            "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
            "hbd": int(rdMolDescriptors.CalcNumHBD(mol)),
            "hba": int(rdMolDescriptors.CalcNumHBA(mol)),
            "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "clinical_phase": _first_present(row, ["clinical_phase", "phase", "clinical_phase_drug"]),
            "moa": _first_present(row, ["moa", "moa_drug"]),
            "indication": _first_present(row, ["indication", "indication_drug"]),
        })
    candidates = pd.DataFrame(rows)
    diagnostics["valid_structure_rows"] = int(len(candidates))
    if candidates.empty:
        return candidates, failures, diagnostics
    candidates["dedupe_key"] = candidates["inchikey"].fillna("")
    candidates.loc[candidates["dedupe_key"] == "", "dedupe_key"] = candidates["canonical_smiles"]
    candidates["clinical_rank"] = candidates["clinical_phase"].apply(clinical_rank)
    candidates = candidates.sort_values(["dedupe_key", "clinical_rank", "name"]).drop_duplicates("dedupe_key", keep="first")
    diagnostics["deduped_rows"] = int(len(candidates))
    filtered = candidates[
        (candidates["mw"] >= 200)
        & (candidates["mw"] <= 650)
        & (candidates["clogp"] >= -1)
        & (candidates["clogp"] <= 7)
    ].copy()
    diagnostics["candidate_pool_rows"] = int(len(filtered))
    return filtered.reset_index(drop=True), failures, diagnostics


def _fingerprints(df: pd.DataFrame):
    _rdkit, Chem, AllChem, _Crippen, _Descriptors, _rdMolDescriptors, _picker = _require_rdkit()
    fps = []
    for smiles in df["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    return fps


def select_locked_library(candidates: pd.DataFrame, *, size: int, seed: int) -> tuple[pd.DataFrame, str, str | None]:
    _rdkit, _Chem, _AllChem, _Crippen, _Descriptors, _rdMolDescriptors, MaxMinPicker = _require_rdkit()
    if len(candidates) < size:
        raise RuntimeError(f"Need at least {size} valid candidates, found {len(candidates)}")
    ordered = candidates.sort_values(["clinical_rank", "name", "inchikey"]).reset_index(drop=True)
    fps = _fingerprints(ordered)
    picker = MaxMinPicker()
    picks = list(picker.LazyBitVectorPick(fps, len(fps), size, seed=seed))
    selected = ordered.iloc[picks].copy()
    bms_mask = selected["name"].isin(BMS_NAMES)
    if bms_mask.any():
        bms_mode = "natural_maxmin"
        removed_name = None
    else:
        bms_rows = ordered[ordered["name"].isin(BMS_NAMES)]
        if bms_rows.empty:
            raise RuntimeError("BMS-986205/Linrodostat was not found in the Broad candidate pool")
        bms_mode = "clinical_hypothesis_override"
        bms_row = bms_rows.sort_values(["clinical_rank", "name"]).head(1)
        selected = selected[~selected["name"].isin(BMS_NAMES)].copy()
        selected = selected.sort_values(["clinical_rank", "name", "inchikey"]).reset_index(drop=True)
        removed = selected.tail(1)
        removed_name = str(removed.iloc[0]["name"])
        selected = pd.concat([selected.iloc[:-1], bms_row], ignore_index=True)
    selected = selected.sort_values(["clinical_rank", "name", "inchikey"]).reset_index(drop=True)
    selected.insert(0, "library_id", [stable_library_id(i) for i in range(1, len(selected) + 1)])
    selected["bms_inclusion_mode"] = ""
    selected.loc[selected["name"].isin(BMS_NAMES), "bms_inclusion_mode"] = bms_mode
    return selected, bms_mode, removed_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Create locked 100-compound Broad-derived library.")
    parser.add_argument("--drugs", type=Path, default=LAKE_DIR / "Broad_Repurposing_Hub.csv")
    parser.add_argument("--samples", type=Path, default=LAKE_DIR / "Broad_Repurposing_Samples.csv")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-qc-missing", action="store_true")
    parser.add_argument("--output", type=Path, default=STAGE3_DATA / "library_100_locked.csv")
    parser.add_argument("--provenance", type=Path, default=STAGE3_DATA / "library_100_provenance.json")
    parser.add_argument("--failures", type=Path, default=STAGE3_DATA / "library_100_failures.csv")
    parser.add_argument("--diagnostics", type=Path, default=STAGE3_DATA / "library_100_diagnostics.csv")
    args = parser.parse_args()

    rdkit, *_ = _require_rdkit()
    candidates, failures, diagnostics = build_candidates(args.drugs, args.samples, strict_qc=not args.allow_qc_missing)
    selected, bms_mode, removed_name = select_locked_library(candidates, size=args.size, seed=args.seed)
    selected.to_csv(args.output, index=False)
    pd.DataFrame(failures).to_csv(args.failures, index=False)
    pd.DataFrame([diagnostics]).to_csv(args.diagnostics, index=False)
    provenance = {
        "input_paths": {"drugs": str(args.drugs), "samples": str(args.samples)},
        "input_hashes": {"drugs": sha256_file(args.drugs), "samples": sha256_file(args.samples)},
        "row_counts": diagnostics,
        "rdkit_version": rdkit.__version__,
        "python_version": platform.python_version(),
        "selection_seed": args.seed,
        "library_size": args.size,
        "bms_inclusion_mode": bms_mode,
        "bms_override_removed_natural_pick": removed_name,
        "erdrp_exclusion_statement": "ERDRP-0519 is retained only as an external comparator.",
        "candidate_filters": {"mw": [200, 650], "clogp": [-1, 7], "qc_incompatible": "0"},
        "final_locked_row_count": int(len(selected)),
    }
    atomic_write_json(args.provenance, provenance)
    print(f"Wrote locked library with {len(selected)} rows to {args.output}")


if __name__ == "__main__":
    main()
