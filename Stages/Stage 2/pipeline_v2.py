
import sys
import os
import subprocess
import argparse
import json
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import distance


# =============================================================================
# CONFIGURATION
# =============================================================================
# PROJECT_ROOT = vanshaj_workflow/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR   = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE2_DIR   = PROJECT_ROOT / "Stages" / "Stage 2"

# -- Stage 1 artifacts (consumed, never written by Stage 2) --
BOX_CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"
WT_RECEPTOR     = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR    = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"
WT_PDB          = STAGE1_DIR / "data" / "9KNZ_clean.pdb"

# -- Project-level inputs --
VINA_EXEC = PROJECT_ROOT / "scripts" / "vina"

# -- Fail-loud file gate --
def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Stage 2 contract violation: {label} not found at {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Stage 2 contract violation: {label} is empty at {path}")
    return path

# -- Dynamic config loading from Stage 1 --
with open(require_file(BOX_CONFIG_PATH, "Stage 1 docking box config"), "r") as _f:
    BOX_CONFIG = json.load(_f)
for _key in ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"):
    if _key not in BOX_CONFIG:
        raise KeyError(f"Missing `{_key}` in {BOX_CONFIG_PATH}")

# -- Preflight --
for _p in [WT_RECEPTOR, MUT_RECEPTOR, WT_PDB, VINA_EXEC]:
    require_file(_p, _p.name)

# =============================================================================
# FUNCTIONS
# =============================================================================

def check_ghost_clash(ligand_pdbqt, wt_pdb_path=WT_PDB):
    """
    Checks if the ligand occupies the deleted W730 sidechain volume.
    FAILS LOUDLY if the reference structure is missing or malformed.
    """
    require_file(wt_pdb_path, "WT PDB reference for ghost clash check")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb_path))

    if "A" not in structure[0]:
        raise RuntimeError(f"Chain A missing from {wt_pdb_path}")

    chain = structure[0]["A"]
    res = None
    for r in chain:
        if r.id[1] == 730:
            res = r
            break

    if res is None:
        raise RuntimeError(f"Residue 730 not found in chain A of {wt_pdb_path}")
    if res.resname != "TRP":
        raise RuntimeError(f"Residue 730 is {res.resname}, expected TRP in {wt_pdb_path}")

    ghost_coords = []
    for atom in res:
        if atom.name not in ["N", "CA", "C", "O", "CB"]:
            ghost_coords.append(atom.get_coord())

    if not ghost_coords:
        raise RuntimeError(f"No ghost sidechain atoms extracted from W730 in {wt_pdb_path}")

    lig_coords = []
    with open(require_file(Path(ligand_pdbqt), "docked ligand for ghost clash"), "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                lig_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

    if not lig_coords:
        raise RuntimeError(f"No ligand coordinates parsed from {ligand_pdbqt}")

    dists = distance.cdist(lig_coords, ghost_coords)
    return float(np.min(dists))


def check_environment():
    """Ensures Vina binary is executable."""
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
        "--num_modes", "1",
        "--scoring", "vinardo",
        "--seed", str(seed),
        "--out", str(output_path),
        "--cpu", "4" # Use multithreading if available
    ]
    
    # Run quietly
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
    if isinstance(clash_dist, float):
        print(f"  > Dist to Ghost Sidechain: {clash_dist:.2f} Å", end="")
        if clash_dist < 2.5:
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
    elif isinstance(clash_dist, float) and clash_dist < 2.5:
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
