
import sys
import subprocess
import argparse
import json
import os
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import distance

# =============================================================================
# CONFIGURATION
# =============================================================================
# Dynamic loading from canonical Stage 1 JSON
BOX_JSON = Path(__file__).parent.parent / "Stage 1" / "config" / "docking_box.json"
if not BOX_JSON.exists():
    print(f"FATAL: Canonical docking box config not found: {BOX_JSON}", file=sys.stderr)
    sys.exit(1)
with open(BOX_JSON) as f:
    BOX_CONFIG = json.load(f)
BOX_CONFIG.setdefault("num_modes", 1)
BOX_CONFIG.setdefault("exhaustiveness", 16)

# Paths (Relative to this script)
BASE_DIR = Path(__file__).parent.parent
VINA_EXEC = BASE_DIR.parent / "scripts" / "vina"
WT_RECEPTOR = BASE_DIR / "Stage 1" / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR = BASE_DIR / "Stage 1" / "data" / "9KNZ_W730A.pdbqt"
WT_PDB = BASE_DIR / "Stage 1" / "data" / "9KNZ_clean.pdb" # Required for ghost check

# Ghost clash threshold (non-covalent VdW overlap)
GHOST_CLASH_THRESHOLD = 2.5  # Angstrom

# =============================================================================
# FUNCTIONS
# =============================================================================

def check_ghost_clash(ligand_pdbqt, wt_pdb_path=WT_PDB):
    """
    Checks if the ligand occupies the space where the W730 sidechain used to be.
    This prevents the 'Vacuum Hole Fallacy'.
    """
    if not wt_pdb_path.exists():
        raise FileNotFoundError(f"WT PDB missing: {wt_pdb_path}")

    # 1. Get Ghost Atoms (W730 Sidechain)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb_path))
    ghost_coords = []
    chain = structure[0]['A']
    res = None
    for r in chain:
        if r.id[1] == 730:
            res = r
            break

    if res is None or res.resname != "TRP":
        raise RuntimeError(f"Expected TRP at A:730, found {res.resname if res else 'nothing'}")

    for atom in res:
        if atom.name not in ["N", "CA", "C", "O", "CB"]:  # FIXED: exclude CB
            ghost_coords.append(atom.get_coord())

    if len(ghost_coords) != 9:
        raise RuntimeError(f"Expected 9 ghost atoms, found {len(ghost_coords)}")

    # 2. Get Ligand Atoms from PDBQT
    lig_coords = []
    with open(ligand_pdbqt, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    lig_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass

    if not lig_coords:
        raise RuntimeError(f"No ligand coordinates in {ligand_pdbqt}")

    # 3. Calculate Distances
    dists = distance.cdist(lig_coords, ghost_coords)
    return float(np.min(dists))


def check_environment():
    """Verifies that all necessary files exist."""
    required = [VINA_EXEC, WT_RECEPTOR, MUT_RECEPTOR]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        print("Error: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Make sure vina is executable
    if not os.access(VINA_EXEC, os.X_OK):
        print(f"Making {VINA_EXEC.name} executable...")
        try:
            os.chmod(VINA_EXEC, 0o755)
        except Exception as e:
            print(f"Warning: Could not set executable permissions: {e}")

def get_affinity(pdbqt_output):
    """Parses the best affinity from Vina output file."""
    best_affinity = 100.0
    found = False
    try:
        with open(pdbqt_output, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT"):
                    parts = line.split()
                    if len(parts) >= 4:
                        affinity = float(parts[3])
                        if affinity < best_affinity:
                            best_affinity = affinity
                        found = True
    except FileNotFoundError:
        return None

    return best_affinity if found else None

def run_vina(ligand_path, receptor_path, output_path, seed=42):
    """Runs Vina docking."""
    cmd = [
        str(VINA_EXEC),
        "--receptor", str(receptor_path),
        "--ligand", str(ligand_path),
        "--center_x", str(BOX_CONFIG["center_x"]),
        "--center_y", str(BOX_CONFIG["center_y"]),
        "--center_z", str(BOX_CONFIG["center_z"]),
        "--size_x", str(BOX_CONFIG["size_x"]),
        "--size_y", str(BOX_CONFIG["size_y"]),
        "--size_z", str(BOX_CONFIG["size_z"]),
        "--exhaustiveness", str(BOX_CONFIG["exhaustiveness"]),
        "--num_modes", str(BOX_CONFIG["num_modes"]),
        "--scoring", "vinardo",
        "--seed", str(seed),
        "--out", str(output_path),
        "--cpu", "4"
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Error: Docking failed for {receptor_path.name}")
        return False
    return True

def analyze_ligand(ligand_path):
    """Runs the full analysis pipeline for a single ligand."""
    ligand_file = Path(ligand_path)
    if not ligand_file.exists():
        print(f"Error: Ligand file {ligand_file} not found.")
        return

    print(f"\nAnalyzing Candidate: {ligand_file.name}")
    print("-" * 50)

    # Temporary output files
    out_wt = ligand_file.with_suffix('.wt_docked.pdbqt')
    out_mut = ligand_file.with_suffix('.mut_docked.pdbqt')

    # 1. Dock into Wild-Type
    print("  1. Docking into Wild-Type (9KNZ)... ", end="", flush=True)
    if run_vina(ligand_file, WT_RECEPTOR, out_wt):
        wt_score = get_affinity(out_wt)
        print(f"Done. (Affinity: {wt_score} kcal/mol)")
    else:
        print("Failed.")
        return

    # 2. Dock into Mutant (W730A)
    print("  2. Docking into Mutant (W730A)...   ", end="", flush=True)
    if run_vina(ligand_file, MUT_RECEPTOR, out_mut):
        mut_score = get_affinity(out_mut)
        print(f"Done. (Affinity: {mut_score} kcal/mol)")
    else:
        print("Failed.")
        return

    # 3. Calculate Delta
    delta = mut_score - wt_score
    print("-" * 50)
    print(f"  > Resistance Delta (Mutant - WT): {delta:+.2f} kcal/mol")

    # 3b. Ghost Clash Check (Rigorous Physics)
    clash_dist = check_ghost_clash(out_mut)
    print(f"  > Dist to Ghost Sidechain: {clash_dist:.2f} Angstrom", end="")
    if clash_dist < GHOST_CLASH_THRESHOLD:
        print(" [CLASH DETECTED]")
    else:
        print(" [SAFE]")

    # 4. Verdict
    print("-" * 50)

    passed_filters = True

    # Filter 1: Binding Strength
    if wt_score > -6.5:
        print("  VERDICT: REJECTED (Too weak to be a drug)")
        passed_filters = False

    # Filter 2: Resistance Resilience
    elif delta > 0.5:
        print("  VERDICT: REJECTED (Vulnerable to W730A mutation)")
        passed_filters = False

    # Filter 3: Ghost Clash
    elif clash_dist < GHOST_CLASH_THRESHOLD:
        print("  VERDICT: REJECTED (Clashes with Wild-Type structure - Vacuum Hole Fallacy)")
        passed_filters = False

    if passed_filters:
        print("  VERDICT: ACCEPTED (Potent & Resilient Candidate!)")
        print("  Recommendation: Proceed to In-Vitro Validation.")

    # Cleanup
    if out_wt.exists(): out_wt.unlink()
    if out_mut.exists(): out_mut.unlink()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nipah Virus Inhibitor Discovery Pipeline")
    parser.add_argument("ligand", help="Path to the ligand PDBQT file to test")
    args = parser.parse_args()

    check_environment()
    analyze_ligand(args.ligand)
