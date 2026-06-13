
import sys
import subprocess
import json
import statistics
import time
import math
from pathlib import Path
import concurrent.futures

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 101, 2023, 999, 1234]

# PROJECT_ROOT = vanshaj_workflow/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR   = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE2_DIR   = PROJECT_ROOT / "Stages" / "Stage 2"

# -- Stage 1 artifacts (consumed, never written by Stage 2) --
BOX_CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"
WT_RIGID   = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RIGID  = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"
ERDRP_LIGAND = STAGE1_DIR / "data" / "ligands" / "ERDRP.pdbqt"

# -- Project-level inputs --
BMS_LIGAND = PROJECT_ROOT / "data" / "ligands" / "expanded" / "BMS-986205.pdbqt"
VINA_PATH  = PROJECT_ROOT / "scripts" / "vina"

# -- Stage 2 output directories --
RESULTS_DIR = STAGE2_DIR / "results" / "reproducibility"

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
for _p in [WT_RIGID, MUT_RIGID, BMS_LIGAND, ERDRP_LIGAND, VINA_PATH]:
    require_file(_p, _p.name)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_affinity(log_file):
    """Extracts the best binding affinity from a Vina log/pdbqt."""
    best_aff = 999.0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "REMARK VINA RESULT" in line:
                    val = float(line.split()[3])
                    if val < best_aff: best_aff = val
    except FileNotFoundError:
        return None
    return best_aff if best_aff != 999.0 else None

def run_docking_single(job_name, receptor, ligand, seed):
    """Runs a single docking job."""
    out_file = RESULTS_DIR / f"{job_name}_seed_{seed}.pdbqt"
    
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
        "--exhaustiveness", "16",
        "--scoring", "vinardo",
        "--num_modes", "1",
        "--seed", str(seed),
        "--cpu", "1",
        "--out", str(out_file)
    ]
    
    # Run silently to avoid clutter
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    val = parse_affinity(out_file)
    return (job_name, seed, val)

def run_seed_batch(seed):
    """Runs all 4 dockings for a single seed with P01_S02 naming convention."""
    results = []
    # 1. BMS-986205 (C & D)
    # Run C: BMS WT (Success Case)
    results.append(run_docking_single("P01_S02_C_R01", WT_RIGID, BMS_LIGAND, seed))
    # Run D: BMS Mut (Resilience Case)
    results.append(run_docking_single("P01_S02_D_R01", MUT_RIGID, BMS_LIGAND, seed))
    
    # 2. ERDRP-0519 (A & B)
    # Run A: ERDRP WT (Control)
    results.append(run_docking_single("P01_S02_A_R01", WT_RIGID, ERDRP_LIGAND, seed))
    # Run B: ERDRP Mut (Failure Case)
    results.append(run_docking_single("P01_S02_B_R01", MUT_RIGID, ERDRP_LIGAND, seed))
    return results

# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== REPRODUCIBILITY STRESS TEST (PARALLEL) === ")
    print(f"Seeds: {SEEDS}")
    print(f"Receptor WT:  {WT_RIGID.name}")
    print(f"Receptor Mut: {MUT_RIGID.name}")
    print("-" * 60)

    results_db = {
        "BMS": {"WT": [], "Mut": []},
        "ERDRP": {"WT": [], "Mut": []}
    }

    t0 = time.time()
    
    # Run seeds in parallel
    print("Launching parallel jobs...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_seed = {executor.submit(run_seed_batch, seed): seed for seed in SEEDS}
        
        for future in concurrent.futures.as_completed(future_to_seed):
            seed = future_to_seed[future]
            try:
                batch_results = future.result()
                print(f"Seed {seed} Completed.")
                for job_name, s, val in batch_results:
                    if val is None:
                        print(f"WARNING: Job {job_name} returned None")
                        continue
                    if "P01_S02_C_R01" in job_name: # BMS WT
                        results_db["BMS"]["WT"].append(val)
                    elif "P01_S02_D_R01" in job_name: # BMS Mut
                        results_db["BMS"]["Mut"].append(val)
                    elif "P01_S02_A_R01" in job_name: # ERDRP WT
                        results_db["ERDRP"]["WT"].append(val)
                    elif "P01_S02_B_R01" in job_name: # ERDRP Mut
                        results_db["ERDRP"]["Mut"].append(val)
            except Exception as exc:
                print(f"Seed {seed} generated an exception: {exc}")

    total_time = time.time() - t0
    print("-" * 60)
    print(f"Completed {len(SEEDS)} seeds in {total_time:.1f} seconds.")
    print("-" * 60)

    # =============================================================================
    # ANALYSIS & REPORT
    # =============================================================================
    
    print("\n=== FINAL STATISTICAL REPORT ===\n")

    # Function to safe convert to float
    def safe_stats(data):
        clean_data = [x for x in data if x is not None]
        if not clean_data: return 0, 0
        mean = statistics.mean(clean_data)
        std = statistics.stdev(clean_data) if len(clean_data) > 1 else 0
        return mean, std

    # BMS Report
    bms_wt_avg, bms_wt_std = safe_stats(results_db["BMS"]["WT"])
    bms_mut_avg, bms_mut_std = safe_stats(results_db["BMS"]["Mut"])
    bms_delta = bms_mut_avg - bms_wt_avg
    bms_delta_err = math.sqrt(bms_wt_std**2 + bms_mut_std**2)

    print(f"CANDIDATE: BMS-986205")
    print(f"  WT Average:      {bms_wt_avg:.3f} +/- {bms_wt_std:.3f} kcal/mol")
    print(f"  Mutant Average:  {bms_mut_avg:.3f} +/- {bms_mut_std:.3f} kcal/mol")
    print(f"  Delta (Resilience): {bms_delta:.3f} +/- {bms_delta_err:.3f} kcal/mol")
    print("")

    # ERDRP Report
    erdrp_wt_avg, erdrp_wt_std = safe_stats(results_db["ERDRP"]["WT"])
    erdrp_mut_avg, erdrp_mut_std = safe_stats(results_db["ERDRP"]["Mut"])
    erdrp_delta = erdrp_mut_avg - erdrp_wt_avg
    erdrp_delta_err = math.sqrt(erdrp_wt_std**2 + erdrp_mut_std**2)

    print(f"CANDIDATE: ERDRP-0519")
    print(f"  WT Average:      {erdrp_wt_avg:.3f} +/- {erdrp_wt_std:.3f} kcal/mol")
    print(f"  Mutant Average:  {erdrp_mut_avg:.3f} +/- {erdrp_mut_std:.3f} kcal/mol")
    print(f"  Delta (Resilience): {erdrp_delta:.3f} +/- {erdrp_delta_err:.3f} kcal/mol")
    print("")

    # Head-to-Head
    print(f"=== HEAD-TO-HEAD VERDICT ===")
    if bms_delta < 0.5:
        print("  BMS-986205: RESISTANT (CONFIRMED)")
    else:
        print("  BMS-986205: SUSCEPTIBLE (FAILED)")
        
    if erdrp_delta > 0.5:
        print("  ERDRP-0519: SUSCEPTIBLE (CONFIRMED)")
    else:
        print("  ERDRP-0519: RESISTANT (SURPRISINGLY)")

    print(f"\n  Better Absolute Affinity? {'BMS-986205' if bms_wt_avg < erdrp_wt_avg else 'ERDRP-0519'}")

if __name__ == "__main__":
    main()
