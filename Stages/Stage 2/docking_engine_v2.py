
import sys
import subprocess
import json
import shutil
from pathlib import Path

# PROJECT_ROOT = vanshaj_workflow/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR   = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE2_DIR   = PROJECT_ROOT / "Stages" / "Stage 2"

# -- Stage 1 artifacts (consumed, never written by Stage 2) --
BOX_CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"
WT_RIGID   = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RIGID  = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"

# -- Project-level inputs --
BMS_LIGAND = PROJECT_ROOT / "data" / "ligands" / "expanded" / "BMS-986205.pdbqt"
VINA_PATH  = PROJECT_ROOT / "scripts" / "vina"

# -- Stage 2 outputs --
RESULTS_DIR = STAGE2_DIR / "results" / "verification"
STAGE2_HANDOFF_DIR = PROJECT_ROOT / "results" / "verification"
BEST_POSE_PATH     = STAGE2_HANDOFF_DIR / "BMS-986205_best_pose.pdbqt"

# -- Stage 2 intermediate files --
LIGAND_SINGLE   = RESULTS_DIR / "BMS-986205_pure.pdbqt"
OUTPUT_WT_FILE  = RESULTS_DIR / "BMS-986205_rigid_wt.pdbqt"
OUTPUT_MUT_FILE = RESULTS_DIR / "BMS-986205_rigid_mut.pdbqt"

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
for _p in [WT_RIGID, MUT_RIGID, BMS_LIGAND, VINA_PATH]:
    require_file(_p, _p.name)

def extract_pure_ligand(multimodel_path, output_path):
    """Extracts first Vina pose as a rigidified PDBQT (TORSDOF 0)."""
    with open(multimodel_path, 'r') as f_in:
        content = f_in.readlines()

    with open(output_path, 'w') as f_out:
        capture = False
        for line in content:
            if line.startswith("MODEL 1"):
                capture = True
                continue
            if capture:
                if line.startswith("ENDMDL"):
                    break
                if line.startswith("BEGIN_RES"):
                    break
                if line.startswith(("BRANCH", "ENDBRANCH")):
                    continue
                if line.startswith("TORSDOF"):
                    f_out.write("TORSDOF 0\n")
                    continue
                f_out.write(line)

def parse_output(pdbqt_file):
    best_aff = 999.0
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                val = float(line.split()[3])
                if val < best_aff: best_aff = val
    return best_aff if best_aff != 999.0 else None

def run_docking(receptor, ligand, output):
    print(f"Docking into {receptor.name}...")
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
        "--exhaustiveness", str(BOX_CONFIG["exhaustiveness"]),
        "--scoring", "vinardo",
        "--num_modes", "1",
        "--seed", "42",
        "--out", str(output)
    ]
    print("Running command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Vina Output:\n{e.stdout.decode()}")
        print(f"Vina Error:\n{e.stderr.decode()}")
        sys.exit(1)

def publish_best_pose(source_pose: Path) -> Path:
    """Publishes the selected Stage 2 pose for Stage 3 consumption."""
    STAGE2_HANDOFF_DIR.mkdir(parents=True, exist_ok=True)
    if BEST_POSE_PATH.exists():
        BEST_POSE_PATH.unlink()
    shutil.copy2(require_file(source_pose, "Stage 2 best pose"), BEST_POSE_PATH)
    require_file(BEST_POSE_PATH, "Stage 3 handoff artifact")
    print(f"Published best pose to {BEST_POSE_PATH}")
    return BEST_POSE_PATH

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Determining Resistance Profile for BMS-986205 (Rigid vs Rigid Validation) ===")

    # 1. Dock canonical ligand into WT to get multi-model output
    wt_multimodel = RESULTS_DIR / "BMS-986205_flex_wt.pdbqt"
    run_docking(WT_RIGID, BMS_LIGAND, wt_multimodel)

    # 2. Extract pose 1 as rigid (TORSDOF 0, no BRANCH)
    extract_pure_ligand(wt_multimodel, LIGAND_SINGLE)

    # 3. Dock rigid ligand into WT
    run_docking(WT_RIGID, LIGAND_SINGLE, OUTPUT_WT_FILE)

    # 4. Dock rigid ligand into Mutant
    run_docking(MUT_RIGID, LIGAND_SINGLE, OUTPUT_MUT_FILE)

    # 5. Compare
    wt_score = parse_output(OUTPUT_WT_FILE)
    mut_score = parse_output(OUTPUT_MUT_FILE)

    print(f"WT Score (Rigid):     {wt_score} kcal/mol")
    print(f"Mutant Score (Rigid): {mut_score} kcal/mol")

    delta = mut_score - wt_score
    print(f"Delta Affinity:       {delta:.2f} kcal/mol")

    if delta < 0.5:
        print("VERDICT: RESISTANT (Passes Energy Check)")
        print("Confirms that the 7A distance translates to energy independence.")
    else:
        print("VERDICT: SUSCEPTIBLE (Fails Energy Check)")

    # 6. Publish best pose for Stage 3
    publish_best_pose(OUTPUT_WT_FILE)

if __name__ == "__main__":
    main()
