
import sys
import json
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from pathlib import Path
from scipy.spatial import distance

# Paths — FAIL-5: canonical project root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE2_DIR = PROJECT_ROOT / "Stages" / "Stage 2"
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"

WT_STRUCTURE = STAGE1_DIR / "data" / "9KNZ_clean.pdb"
DOCKED_LIGAND = STAGE2_DIR / "results" / "verification" / "BMS-986205_rigid_mut.pdbqt"

def get_ghost_atoms(wt_pdb, chain_id="A", res_id=730):
    """Get coordinates of the deleted sidechain atoms (Ghost)"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb))

    ghost_coords = []

    try:
        residue = structure[0][chain_id][res_id]
        # FAIL-5: TRP assertion
        if residue.resname != "TRP":
            raise RuntimeError(f"Expected TRP at {chain_id}:{res_id}, found {residue.resname}")
        # FAIL-5: Exclude backbone AND CB (CB is retained in Ala)
        GHOST_BACKBONE_AND_CB = {"N", "CA", "C", "O", "CB"}
        for atom in residue:
            if atom.name not in GHOST_BACKBONE_AND_CB:
                ghost_coords.append(atom.get_coord())
    except KeyError:
        print(f"FATAL: Residue {res_id} not found in {wt_pdb}.", file=sys.stderr)
        sys.exit(1)

    # FAIL-5: count validation — must be exactly 9 for TRP-730
    if len(ghost_coords) != 9:
        raise RuntimeError(
            f"Expected 9 ghost sidechain atoms for TRP-730, found {len(ghost_coords)}. "
            f"Residue is {residue.resname}, expected TRP."
        )

    return np.array(ghost_coords)

def get_ligand_coords(pdbqt_file):
    coords = []
    if not pdbqt_file.exists():
        return []
    with open(pdbqt_file, 'r') as f:
        capture = False
        for line in f:
            if line.startswith("MODEL 1"): capture = True
            elif line.startswith("ENDMDL"): break
            elif capture and (line.startswith("ATOM") or line.startswith("HETATM")):
                try:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
    return np.array(coords)

def main():
    print("=== Ghost Clash Verification (Vacuum Hole Check) ===")

    # FAIL-5: fail-loud on missing WT structure
    if not WT_STRUCTURE.exists():
        print(f"FATAL: WT structure not found at {WT_STRUCTURE}", file=sys.stderr)
        sys.exit(1)

    # 1. Get Ghost Coordinates (W730 Sidechain)
    ghost_coords = get_ghost_atoms(WT_STRUCTURE)
    print(f"Ghost Residue (W730) Atoms: {len(ghost_coords)}")

    # FAIL-5: fail-loud on missing docked ligand
    if not DOCKED_LIGAND.exists():
        print(f"FATAL: Docked ligand not found at {DOCKED_LIGAND}", file=sys.stderr)
        sys.exit(1)

    # 2. Get Docked Ligand Coordinates (in Mutant)
    ligand_coords = get_ligand_coords(DOCKED_LIGAND)
    print(f"Ligand Atoms: {len(ligand_coords)}")

    # 3. Calculate Clashes
    if len(ghost_coords) == 0 or len(ligand_coords) == 0:
        print("FATAL: Empty coordinates.", file=sys.stderr)
        sys.exit(1)

    # Min distance between any ligand atom and any ghost atom
    dists = distance.cdist(ligand_coords, ghost_coords)
    min_dist = np.min(dists)

    print(f"Minimum Distance to Ghost Sidechain: {min_dist:.2f} A")

    # Verdict — FAIL-5: threshold raised from 1.5 to 2.5
    LIMIT = 2.5  # Van der Waals clash threshold (Angstrom)

    if min_dist < LIMIT:
        print(f"VERDICT: FAIL (Ghost Clash Detected)")
        print(f"The drug is occupying the space of the deleted sidechain ({min_dist:.2f} A < {LIMIT} A).")
        print("It relies on the 'Vacuum Hole'.")
    else:
        print(f"VERDICT: PASS (No Ghost Clash)")
        print(f"The drug respects the volume of the original residue ({min_dist:.2f} A > {LIMIT} A).")
        print("Refuting the Vacuum Hole Fallacy.")

    # FAIL-5: structured JSON output
    verdict_path = STAGE4_DIR / "results" / "verification" / "ghost_clash_verdict.json"
    verdict_path.parent.mkdir(parents=True, exist_ok=True)
    with open(verdict_path, "w") as fh:
        json.dump({
            "compound": "BMS-986205",
            "ghost_clash_dist": float(min_dist),
            "ghost_clash": bool(min_dist < LIMIT),
            "threshold": LIMIT,
            "ghost_atom_count": len(ghost_coords),
        }, fh, indent=2)
    print(f"Verdict written to {verdict_path}")

if __name__ == "__main__":
    main()
