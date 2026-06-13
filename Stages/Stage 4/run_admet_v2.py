
import sys
import json
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
RESULTS_DIR = STAGE4_DIR / "results" / "verification"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical molecular sources
BMS_SMILES = "[H][C@]1(CC[C@@H](CC1)c1ccnc2ccc(F)cc12)[C@@H](C)C(=O)Nc1ccc(Cl)cc1"
ERDRP_SDF = STAGE1_DIR / "data" / "ERDRP_with_bonds.sdf"


def load_molecule(name: str):
    """Load molecule from authoritative source with fail-loud validation."""
    if "BMS" in name:
        mol = Chem.MolFromSmiles(BMS_SMILES)
        if mol is None:
            raise RuntimeError("Failed to parse BMS-986205 SMILES")
        return mol
    else:
        if not ERDRP_SDF.exists():
            raise FileNotFoundError(
                f"FATAL: ERDRP SDF not found at {ERDRP_SDF}"
            )
        suppl = Chem.SDMolSupplier(str(ERDRP_SDF))
        mol = next((m for m in suppl if m is not None), None)
        if mol is None:
            raise RuntimeError(
                f"FATAL: Could not parse any molecule from {ERDRP_SDF}"
            )
        return mol


def compute_admet(mol, name: str) -> dict:
    """Compute Lipinski descriptors and return structured result."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    violations = (int(mw > 500) + int(logp > 5)
                  + int(hbd > 5) + int(hba > 10))
    return {
        "name": name, "mw": round(mw, 2), "logp": round(logp, 2),
        "hbd": hbd, "hba": hba, "lipinski_violations": violations,
        "lipinski_pass": violations == 0,
    }


def apply_override(row: dict, clinical_phase: str) -> dict:
    """Step 4.6: Clinical-context override for Lipinski violations."""
    override_applied = (
        not row["lipinski_pass"]
        and clinical_phase in {"Phase 3", "Launched"}
    )
    row["clinical_phase"] = clinical_phase
    row["override_applied"] = override_applied
    row["final_status"] = "PASS" if (row["lipinski_pass"] or override_applied) else "FAIL"
    if override_applied:
        row["override_rationale"] = (
            f"Retained despite {row['lipinski_violations']} Lipinski violation(s) "
            f"-- compound is in {clinical_phase} clinical trials, "
            f"demonstrating acceptable human pharmacokinetics."
        )
    return row


def main():
    rows = []
    for name, phase in [("BMS-986205", "Phase 3"), ("ERDRP-0519", "Preclinical")]:
        mol = load_molecule(name)
        row = compute_admet(mol, name)
        row = apply_override(row, phase)
        rows.append(row)
        print(f"--- {name} ---")
        print(f"  MW: {row['mw']}, LogP: {row['logp']}, HBD: {row['hbd']}, HBA: {row['hba']}")
        print(f"  Lipinski violations: {row['lipinski_violations']}, Status: {row['final_status']}")

    # Write structured output
    admet_df = pd.DataFrame(rows)
    admet_df.to_csv(RESULTS_DIR / "admet_results.csv", index=False)
    with open(RESULTS_DIR / "admet_results.json", "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nADMET results written to {RESULTS_DIR / 'admet_results.csv'}")


if __name__ == "__main__":
    main()
