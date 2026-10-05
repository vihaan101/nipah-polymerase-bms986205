
import os
import sys
import time
import subprocess
import json
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser
from scipy.spatial import distance

import argparse

# Paths — canonical project root resolution
PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = PROJECT_ROOT / "Stages" / "common"
sys.path.insert(0, str(COMMON_DIR))
from paths import project_root as _project_root  # noqa: E402

PROJECT_ROOT = _project_root()
STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE3_DIR = PROJECT_ROOT / "Stages" / "Stage 3"
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
RESULTS_DIR = STAGE4_DIR / "results" / "verification"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
VINA_PATH = PROJECT_ROOT / "scripts" / "vina"

# Configs
TARGET_RES_ID = 730
CHAIN_ID = "A"

# FAIL-1: fail-loud config load from canonical Stage 1 path
_box_path = STAGE1_DIR / "config" / "docking_box.json"
if not _box_path.exists():
    raise FileNotFoundError(
        f"Stage 4 contract violation: docking box config missing at {_box_path}"
    )
with open(_box_path) as f:
    BOX_CONFIG = json.load(f)

# FAIL-2: canonical Stage 3 input contract (overridden by CLI)
DEFAULT_INPUT_CSV = STAGE3_DIR / "results" / "resistance_screening_results.csv"
DEFAULT_LIGANDS_DIR = STAGE3_DIR / "data" / "ligands"
LIBRARY_100_INPUT_CSV = STAGE3_DIR / "results" / "library_100_mutation_ranked.csv"
LIBRARY_100_LIGANDS_DIR = STAGE3_DIR / "data" / "library_100_ligands"

# FAIL-3: canonical Stage 1 receptor paths — rigid-rigid only, no flex
WT_RECEPTOR = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"

for _p, _label in [(WT_RECEPTOR, "WT receptor"), (MUT_RECEPTOR, "MUT receptor")]:
    if not _p.exists():
        raise FileNotFoundError(f"Stage 4 contract violation: {_label} missing at {_p}")

# FAIL-4: pipeline-wide resilience threshold
DELTA_AFFINITY_THRESHOLD = 0.5  # kcal/mol

# FAIL-6: ghost sidechain computation constants
GHOST_BACKBONE_AND_CB = {"N", "CA", "C", "O", "CB"}
GHOST_CLASH_THRESHOLD = 2.5  # Angstrom
ALLOSTERIC_MIN_DISTANCE = 5.0  # Angstrom

def _load_ghost_coords():
    """Pre-compute the 9 TRP-730 ghost sidechain coordinates from the WT PDB."""
    wt_pdb = STAGE1_DIR / "data" / "9KNZ_clean.pdb"
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB missing at {wt_pdb}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb))
    residue = structure[0][CHAIN_ID][TARGET_RES_ID]
    if residue.resname != "TRP":
        raise RuntimeError(f"Expected TRP at {CHAIN_ID}:{TARGET_RES_ID}, found {residue.resname}")
    coords = [atom.get_coord() for atom in residue if atom.name not in GHOST_BACKBONE_AND_CB]
    if len(coords) != 9:
        raise RuntimeError(f"Expected 9 ghost sidechain atoms, found {len(coords)}")
    return np.array(coords)

GHOST_COORDS = _load_ghost_coords()

# FAIL-8: lead selection paths
LEAD_MANIFEST = RESULTS_DIR / "lead_selection.json"
DOWNSTREAM_HANDOFF = PROJECT_ROOT / "results" / "verification"

# FAIL-6: distance metric measures against ghost sidechain only
def calculate_min_distance(ligand_pdbqt, ghost_coords):
    """Minimum distance from docked ligand MODEL 1 to ghost sidechain atoms."""
    if not os.path.exists(ligand_pdbqt):
        raise FileNotFoundError(f"Ligand PDBQT missing: {ligand_pdbqt}")
    ligand_coords = []
    with open(ligand_pdbqt, 'r') as f:
        in_model1 = False
        for line in f:
            if line.startswith("MODEL 1"):
                in_model1 = True
            elif line.startswith("ENDMDL"):
                break
            elif in_model1 and line.startswith(("ATOM", "HETATM")):
                try:
                    ligand_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                except ValueError:
                    pass
    if not ligand_coords:
        raise RuntimeError(f"No ligand coordinates found in {ligand_pdbqt}")
    return float(np.min(distance.cdist(np.array(ligand_coords), ghost_coords)))

# FAIL-3: rigid-receptor-only docking
def run_vina(receptor: Path, ligand: Path, output_file: Path) -> bool:
    """Rigid-receptor-only docking per Stage 2 mandate."""
    cmd = [
        str(VINA_PATH),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
        "--center_x", str(BOX_CONFIG["center_x"]),
        "--center_y", str(BOX_CONFIG["center_y"]),
        "--center_z", str(BOX_CONFIG["center_z"]),
        "--size_x", str(BOX_CONFIG["size_x"]),
        "--size_y", str(BOX_CONFIG["size_y"]),
        "--size_z", str(BOX_CONFIG["size_z"]),
        "--exhaustiveness", str(BOX_CONFIG.get("exhaustiveness", 16)),
        "--num_modes", "1",
        "--scoring", "vinardo",
        "--seed", str(BOX_CONFIG.get("seed", 42)),
        "--out", str(output_file),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Docking FAILED: receptor={receptor.name} ligand={ligand.name}\n"
              f"STDERR: {e.stderr}", file=sys.stderr)
        return False

def parse_affinity(pdbqt_file):
    if not os.path.exists(pdbqt_file): return None
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                return float(line.split()[3])
    return None

def verify_compound(payload):
    row, ligands_dir = payload
    start_t = time.time()
    name = row["name"]
    library_id = row.get("library_id")
    if library_id and str(library_id).strip():
        ligand_pdbqt = ligands_dir / f"{library_id}.pdbqt"
    else:
        ligand_pdbqt = ligands_dir / f"{name}.pdbqt"

    if not ligand_pdbqt.exists():
        return {"name": name, "error": "Ligand file missing", "time": 0}

    # 1. Dock WT (Rigid) — FAIL-3
    wt_out = RESULTS_DIR / f"{name}_rigid_wt.pdbqt"
    if not run_vina(WT_RECEPTOR, ligand_pdbqt, wt_out):
        return {"name": name, "error": "WT Rigid Docking failed", "time": time.time() - start_t}

    # 2. Dock MUT (Rigid) — FAIL-3
    mut_out = RESULTS_DIR / f"{name}_rigid_mut.pdbqt"
    if not run_vina(MUT_RECEPTOR, ligand_pdbqt, mut_out):
        return {"name": name, "error": "MUT Rigid Docking failed", "time": time.time() - start_t}

    # 3. Analyze — FAIL-6: ghost coords for distance, ghost clash integration
    wt_aff = parse_affinity(wt_out)
    mut_aff = parse_affinity(mut_out)
    wt_dist = calculate_min_distance(wt_out, GHOST_COORDS)
    mut_dist = calculate_min_distance(mut_out, GHOST_COORDS)
    ghost_clash_dist = mut_dist
    ghost_clash = ghost_clash_dist < GHOST_CLASH_THRESHOLD

    duration = time.time() - start_t

    return {
        "name": name,
        "wt_affinity": wt_aff,
        "mut_affinity": mut_aff,
        "delta_affinity": (mut_aff - wt_aff) if (wt_aff is not None and mut_aff is not None) else None,
        "wt_dist": wt_dist,
        "mut_dist": mut_dist,
        "delta_dist": (mut_dist - wt_dist) if (wt_dist is not None and mut_dist is not None) else None,
        "ghost_clash_dist": ghost_clash_dist,
        "ghost_clash": ghost_clash,
        "time": duration,
    }

# FAIL-8: lead selection handoff
def publish_lead(name: str, pose_src_wt: Path, pose_src_mut: Path):
    """Publish WT/MUT verification poses for Stage 5/6 consumption."""
    DOWNSTREAM_HANDOFF.mkdir(parents=True, exist_ok=True)
    pose_dst_wt = DOWNSTREAM_HANDOFF / f"{name}_best_pose.pdbqt"
    if pose_dst_wt.exists():
        pose_dst_wt.unlink()
    shutil.copy2(pose_src_wt, pose_dst_wt)

    pose_rel_wt = RESULTS_DIR / f"{name}_rigid_wt.pdbqt"
    pose_rel_mut = RESULTS_DIR / f"{name}_rigid_mut.pdbqt"
    if not pose_rel_wt.exists():
        shutil.copy2(pose_src_wt, pose_rel_wt)
    if not pose_rel_mut.exists():
        shutil.copy2(pose_src_mut, pose_rel_mut)

    manifest = {
        "lead_name": name,
        "pose_path": str(pose_dst_wt),
        "pose_path_wt": str(pose_rel_wt.relative_to(STAGE4_DIR / "results" / "verification")),
        "pose_path_mut": str(pose_rel_mut.relative_to(STAGE4_DIR / "results" / "verification")),
        "verification_csv": str(RESULTS_DIR / "verification_results.csv"),
        "admet_csv": str(RESULTS_DIR / "admet_results.csv"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(LEAD_MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Lead selection published: {name} -> WT {pose_dst_wt} (MD uses WT pose for C_BMS_WT)")

def main():
    parser = argparse.ArgumentParser(description="Stage 4 multi-filter hit verification")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Screening results CSV (default: legacy resistance_screening_results.csv)",
    )
    parser.add_argument(
        "--library-100",
        action="store_true",
        help="Use locked-100 ranked CSV and library_100_ligands PDBQT paths",
    )
    args = parser.parse_args()

    if args.library_100:
        input_csv = args.input_csv or LIBRARY_100_INPUT_CSV
        ligands_dir = LIBRARY_100_LIGANDS_DIR
    else:
        input_csv = args.input_csv or DEFAULT_INPUT_CSV
        ligands_dir = DEFAULT_LIGANDS_DIR

    print("=== Stage 4: Multi-Filter Verification ===")
    if not input_csv.exists():
        print(f"FATAL: Stage 3 results not found at {input_csv}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(input_csv)
    if df.empty:
        print(f"FATAL: Stage 3 results are empty at {input_csv}", file=sys.stderr)
        sys.exit(1)
    required_cols = {"name", "wt_affinity", "mut_affinity", "delta_affinity",
                     "delta_dist", "ghost_clash_dist", "ghost_clash"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"FATAL: Stage 3 results missing columns: {sorted(missing)}", file=sys.stderr)
        sys.exit(1)

    # Step 4.1: Baseline potency gate
    hits = df[df["wt_affinity"] <= -7.0]
    print(f"Loaded {len(df)} compounds. {len(hits)} pass potency gate (wt_affinity <= -7.0).")

    if hits.empty:
        print("No hits to verify.")
        return

    print(f"Starting verification of {len(hits)} compounds...")
    start_total = time.time()

    payloads = [(row, ligands_dir) for _, row in hits.iterrows()]
    with Pool(cpu_count()) as p:
        results = p.map(verify_compound, payloads)

    end_total = time.time()
    total_time = end_total - start_total
    avg_time = total_time / len(hits)

    print(f"\n=== Verification Complete ===")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Avg Time per Compound: {avg_time:.2f}s")

    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_DIR / "verification_results.csv", index=False)

    # FAIL-4 + FAIL-6: full 5-gate filter cascade
    if not res_df.empty:
        # Step 4.1: Potency gate
        potent = res_df[res_df['wt_affinity'] <= -7.0]
        # Step 4.2: Resilience gate — FAIL-4
        resilient = potent[potent['delta_affinity'] <= DELTA_AFFINITY_THRESHOLD]
        # Step 4.3a: Absolute allosteric distance — FAIL-6
        allosteric = resilient[resilient['mut_dist'] >= ALLOSTERIC_MIN_DISTANCE]
        # Step 4.3b: Pose stability
        stable = allosteric[allosteric['delta_dist'].abs() < 2.0]
        # Step 4.3c: Ghost clash rejection — FAIL-6
        final_hits = stable[stable['ghost_clash'] == False]

        print(f"\n=== Filter Cascade ===")
        print(f"Potency (wt <= -7.0): {len(potent)}")
        print(f"Resilient (delta <= 0.5): {len(resilient)}")
        print(f"Allosteric (mut_dist >= 5.0): {len(allosteric)}")
        print(f"Pose-stable (|delta_dist| < 2.0): {len(stable)}")
        print(f"Ghost-clear (no clash): {len(final_hits)}")

        print("\n=== CONFIRMED HITS ===")
        if not final_hits.empty:
            print(final_hits[['name', 'wt_affinity', 'delta_affinity', 'delta_dist', 'ghost_clash_dist']])
            final_hits.to_csv(RESULTS_DIR / "confirmed_hits.csv", index=False)
            # FAIL-8: Lead selection — strongest WT affinity wins
            best = final_hits.loc[final_hits['wt_affinity'].idxmin()]
            best_name = best["name"]
            best_pose_wt = RESULTS_DIR / f"{best_name}_rigid_wt.pdbqt"
            best_pose_mut = RESULTS_DIR / f"{best_name}_rigid_mut.pdbqt"
            if best_pose_wt.exists() and best_pose_mut.exists():
                publish_lead(best_name, best_pose_wt, best_pose_mut)
            else:
                print(
                    f"WARNING: Verified poses not found for {best_name} "
                    f"(wt={best_pose_wt}, mut={best_pose_mut})"
                )
        else:
            print("No confirmed hits met all five filter criteria.")

if __name__ == "__main__":
    main()
