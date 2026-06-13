
import sys
import subprocess
import json
import statistics
import math
from pathlib import Path

# PROJECT_ROOT = vanshaj_workflow/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR   = PROJECT_ROOT / "Stages" / "Stage 1"

# -- Multi-seed configuration --
SEEDS = [42, 101, 2023, 999, 1234]
MIN_SEEDS_FOR_STATS = 2

# -- Stage 1 artifacts (consumed, never written by Stage 5) --
BOX_CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"
WT_RIGID   = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RIGID  = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"

# -- Project-level inputs --
VINA_PATH  = PROJECT_ROOT / "scripts" / "vina"

# -- Stage 4 lead manifest --
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
LEAD_MANIFEST = STAGE4_DIR / "results" / "verification" / "lead_selection.json"

# -- Stage 5 outputs --
STAGE5_DIR = PROJECT_ROOT / "Stages" / "Stage 5"
STAGE5_RESULTS = STAGE5_DIR / "results"
RESULTS_DIR = STAGE5_RESULTS / "verification"

# -- Stage 5 intermediate files --
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

def load_lead_ligand():
    """Loads the canonical BMS ligand path from Stage 4 manifest."""
    if not LEAD_MANIFEST.exists():
        raise FileNotFoundError(
            f"Stage 5 contract violation: Stage 4 lead manifest not found at {LEAD_MANIFEST}. "
            f"Run run_stage4.py first."
        )
    with open(LEAD_MANIFEST, "r") as fh:
        manifest = json.load(fh)
    lead_name = manifest.get("lead_name")
    if lead_name != "BMS-986205":
        raise RuntimeError(
            f"Stage 5 contract violation: expected lead 'BMS-986205', got '{lead_name}'"
        )
    bms_path = PROJECT_ROOT / "Stages" / "Stage 3" / "data" / "ligands" / "BMS-986205.pdbqt"
    require_file(bms_path, "Stage 3 canonical BMS-986205 ligand")
    return bms_path

BMS_LIGAND = load_lead_ligand()

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
    """Extracts first Vina pose as a rigidified PDBQT (TORSDOF 0).

    Collects all ATOM/HETATM records from MODEL 1 and places them
    inside a single ROOT/ENDROOT block with TORSDOF 0, so Vina
    treats the ligand as a fully rigid body.
    """
    with open(multimodel_path, 'r') as f_in:
        content = f_in.readlines()

    # Collect all coordinate lines from MODEL 1
    atom_lines = []
    capture = False
    for line in content:
        if line.startswith("MODEL 1"):
            capture = True
            continue
        if capture:
            if line.startswith("ENDMDL") or line.startswith("BEGIN_RES"):
                break
            if line.startswith(("ATOM", "HETATM")):
                atom_lines.append(line)

    with open(output_path, 'w') as f_out:
        f_out.write("ROOT\n")
        for line in atom_lines:
            f_out.write(line)
        f_out.write("ENDROOT\n")
        f_out.write("TORSDOF 0\n")

def parse_output(pdbqt_file):
    best_aff = 999.0
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                val = float(line.split()[3])
                if val < best_aff: best_aff = val
    return best_aff if best_aff != 999.0 else None

def run_docking(receptor, ligand, output, seed=None):
    if seed is None:
        seed = BOX_CONFIG.get("seed", 42)
    print(f"Docking into {receptor.name} (seed {seed})...")
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
        "--seed", str(seed),
        "--out", str(output)
    ]
    print("Running command: " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Vina Output:\n{e.stdout.decode()}")
        print(f"Vina Error:\n{e.stderr.decode()}")
        sys.exit(1)

def run_docking_multiseed(receptor, ligand, output_template):
    """Run docking across all seeds, return list of scores."""
    scores = []
    for seed in SEEDS:
        out = output_template.parent / f"{output_template.stem}_seed_{seed}.pdbqt"
        run_docking(receptor, ligand, out, seed=seed)
        score = parse_output(out)
        scores.append(score)
    return scores

def stats(vals):
    """Returns (mean, std) or (None, None) if insufficient data."""
    clean = [x for x in vals if x is not None]
    if len(clean) < MIN_SEEDS_FOR_STATS:
        return None, None
    return statistics.mean(clean), statistics.stdev(clean)

def main():
    STAGE5_RESULTS.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Determining Resistance Profile for BMS-986205 (Rigid vs Rigid Validation) ===")

    # 1. Dock canonical ligand into WT to get multi-model output
    wt_multimodel = RESULTS_DIR / "BMS-986205_flex_wt.pdbqt"
    run_docking(WT_RIGID, BMS_LIGAND, wt_multimodel)

    # 2. Extract pose 1 as rigid (TORSDOF 0, no BRANCH)
    extract_pure_ligand(wt_multimodel, LIGAND_SINGLE)

    # 3. Multi-seed dock rigid ligand into WT
    wt_scores = run_docking_multiseed(WT_RIGID, LIGAND_SINGLE, OUTPUT_WT_FILE)

    # 4. Multi-seed dock rigid ligand into Mutant
    mut_scores = run_docking_multiseed(MUT_RIGID, LIGAND_SINGLE, OUTPUT_MUT_FILE)

    # 5. Compute statistics and verdict
    wt_avg, wt_std = stats(wt_scores)
    mut_avg, mut_std = stats(mut_scores)

    if wt_avg is None or mut_avg is None:
        print("ERROR: One or both rigid docking conditions have insufficient seed data (need >= 2).")
        print(f"  WT scores: {wt_scores}")
        print(f"  MUT scores: {mut_scores}")
        print("VERDICT: INCONCLUSIVE (cannot compute delta)")
        sys.exit(1)

    delta = mut_avg - wt_avg
    delta_err = math.sqrt(wt_std**2 + mut_std**2)

    print(f"\nWT Score (Rigid):     {wt_avg:.3f} +/- {wt_std:.3f} kcal/mol ({len([s for s in wt_scores if s is not None])}/{len(SEEDS)} seeds)")
    print(f"Mutant Score (Rigid): {mut_avg:.3f} +/- {mut_std:.3f} kcal/mol ({len([s for s in mut_scores if s is not None])}/{len(SEEDS)} seeds)")
    print(f"Delta Affinity:       {delta:+.3f} +/- {delta_err:.3f} kcal/mol")

    if abs(delta) < 0.5:
        print("VERDICT: RESISTANT (Passes Energy Check)")
        print("Confirms that the 7A distance translates to energy independence.")
    else:
        print("VERDICT: SUSCEPTIBLE (Fails Energy Check)")

if __name__ == "__main__":
    main()
