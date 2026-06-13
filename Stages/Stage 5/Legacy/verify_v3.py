
import sys
import subprocess
import statistics
import time
import os
import math
import json
from pathlib import Path
import concurrent.futures

# =============================================================================
# CONFIGURATION
# =============================================================================
SEEDS = [42, 101, 2023, 999, 1234]

# PROJECT_ROOT = vanshaj_workflow/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR   = PROJECT_ROOT / "Stages" / "Stage 1"

# -- Stage 1 artifacts (consumed, never written by Stage 5) --
BOX_CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"
WT_RECEPTOR     = STAGE1_DIR / "data" / "9KNZ_clean.pdbqt"
MUT_RECEPTOR    = STAGE1_DIR / "data" / "9KNZ_W730A.pdbqt"
ERDRP_LIGAND    = STAGE1_DIR / "data" / "ligands" / "ERDRP.pdbqt"

# -- Project-level inputs --
VINA_EXEC  = PROJECT_ROOT / "scripts" / "vina"

# -- Stage 4 lead manifest --
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
LEAD_MANIFEST = STAGE4_DIR / "results" / "verification" / "lead_selection.json"

# -- Stage 5 output directories --
STAGE5_DIR = PROJECT_ROOT / "Stages" / "Stage 5"
STAGE5_RESULTS = STAGE5_DIR / "results"
OUT_DIR = STAGE5_RESULTS / "matrix"
STAGE5_VERDICT_PATH = STAGE5_RESULTS / "stage5_verdict.json"

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
for _p in [WT_RECEPTOR, MUT_RECEPTOR, BMS_LIGAND, ERDRP_LIGAND, VINA_EXEC]:
    require_file(_p, _p.name)

# =============================================================================
# HELPERS
# =============================================================================

def check_environment():
    """Ensures output directory exists."""
    STAGE5_RESULTS.mkdir(parents=True, exist_ok=True)
    if not OUT_DIR.exists():
        OUT_DIR.mkdir(parents=True)
    print("Environment verified. All files present.")

def get_affinity(pdbqt_output):
    """Parses best affinity."""
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
    except (FileNotFoundError, ValueError, IndexError):
        return None
    return best_affinity if found else None

def run_dock_job(job_id, receptor, ligand, seed):
    """Execution unit."""
    out_file = OUT_DIR / f"{job_id}_seed_{seed}.pdbqt"

    cmd = [
        str(VINA_EXEC),
        "--receptor", str(receptor),
        "--ligand", str(ligand),
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
        "--out", str(out_file),
        "--cpu", "1" # Single core per job, parallelized by batch
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        score = get_affinity(out_file)
        return score
    except Exception as e:
        print(f"Job {job_id} (Seed {seed}) Failed: {e}")
        return None

def run_seed_batch(seed):
    """Runs the 4 comparisons for a single seed."""
    # 1. BMS vs WT
    bms_wt = run_dock_job("BMS_WT", WT_RECEPTOR, BMS_LIGAND, seed)
    # 2. BMS vs MUT
    bms_mut = run_dock_job("BMS_MUT", MUT_RECEPTOR, BMS_LIGAND, seed)
    # 3. ERDRP vs WT
    erdrp_wt = run_dock_job("ERDRP_WT", WT_RECEPTOR, ERDRP_LIGAND, seed)
    # 4. ERDRP vs MUT
    erdrp_mut = run_dock_job("ERDRP_MUT", MUT_RECEPTOR, ERDRP_LIGAND, seed)

    return {
        "seed": seed,
        "bms_wt": bms_wt,
        "bms_mut": bms_mut,
        "erdrp_wt": erdrp_wt,
        "erdrp_mut": erdrp_mut
    }

def write_verdict(results, bms_delta, bms_delta_err, erdrp_delta, erdrp_delta_err):
    summary = {
        "stage": 5,
        "seeds": SEEDS,
        "conditions": {
            "BMS_WT": {"count": len(results["BMS"]["WT"]), "values": results["BMS"]["WT"]},
            "BMS_MUT": {"count": len(results["BMS"]["MUT"]), "values": results["BMS"]["MUT"]},
            "ERDRP_WT": {"count": len(results["ERDRP"]["WT"]), "values": results["ERDRP"]["WT"]},
            "ERDRP_MUT": {"count": len(results["ERDRP"]["MUT"]), "values": results["ERDRP"]["MUT"]},
        },
        "bms_delta": round(bms_delta, 4),
        "bms_delta_err": round(bms_delta_err, 4),
        "erdrp_delta": round(erdrp_delta, 4),
        "erdrp_delta_err": round(erdrp_delta_err, 4),
        "verdict": {
            "bms": "RESILIENT" if abs(bms_delta) < 0.5 else "VULNERABLE",
            "erdrp": "VULNERABLE" if abs(erdrp_delta) >= 0.5 else "RESILIENT",
        },
    }
    STAGE5_VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGE5_VERDICT_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Stage 5 verdict written to {STAGE5_VERDICT_PATH}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    check_environment()

    print("\n" + "="*60)
    print(" NIPAH VIRUS INHIBITOR: CERTIFICATE OF REPRODUCIBILITY")
    print("="*60)
    print("Objective: Verify BMS-986205 resilience vs ERDRP-0519 failure.")
    print(f"Seeds: {SEEDS}")
    print("Running parallel simulations... (This may take a few minutes)")
    print("-" * 60)

    results = {
        "BMS": {"WT": [], "MUT": []},
        "ERDRP": {"WT": [], "MUT": []}
    }

    # Run in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_seed_batch, seed): seed for seed in SEEDS}

        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                data = future.result()
                print(f"[Finished Seed {seed}]")
                if data["bms_wt"] is not None: results["BMS"]["WT"].append(data["bms_wt"])
                if data["bms_mut"] is not None: results["BMS"]["MUT"].append(data["bms_mut"])
                if data["erdrp_wt"] is not None: results["ERDRP"]["WT"].append(data["erdrp_wt"])
                if data["erdrp_mut"] is not None: results["ERDRP"]["MUT"].append(data["erdrp_mut"])
            except Exception as e:
                print(f"Seed {seed} generated an exception: {e}")

    print("-" * 60)

    # Calculate Statistics
    MIN_SEEDS_FOR_STATS = 2

    def stats(vals):
        """Returns (mean, std) or (None, None) if insufficient data."""
        clean = [x for x in vals if x is not None]
        if len(clean) < MIN_SEEDS_FOR_STATS:
            return None, None
        return statistics.mean(clean), statistics.stdev(clean)

    bms_wt_avg, bms_wt_std = stats(results["BMS"]["WT"])
    bms_mut_avg, bms_mut_std = stats(results["BMS"]["MUT"])

    if bms_wt_avg is None or bms_mut_avg is None:
        print("ERROR: Insufficient seed data for BMS verdict (need >= 2 per condition).")
        print("VERDICT: INCONCLUSIVE")
        sys.exit(1)

    bms_delta = bms_mut_avg - bms_wt_avg
    bms_delta_err = math.sqrt(bms_wt_std**2 + bms_mut_std**2)

    erdrp_wt_avg, erdrp_wt_std = stats(results["ERDRP"]["WT"])
    erdrp_mut_avg, erdrp_mut_std = stats(results["ERDRP"]["MUT"])

    if erdrp_wt_avg is None or erdrp_mut_avg is None:
        print("ERROR: Insufficient seed data for ERDRP verdict (need >= 2 per condition).")
        print("VERDICT: INCONCLUSIVE")
        sys.exit(1)

    erdrp_delta = erdrp_mut_avg - erdrp_wt_avg
    erdrp_delta_err = math.sqrt(erdrp_wt_std**2 + erdrp_mut_std**2)

    print("\n=== FINAL RESULTS ===\n")

    print(f"CANDIDATE 1: BMS-986205 (Our Discovery)")
    print(f"  Wild-Type Affinity: {bms_wt_avg:.3f} +/- {bms_wt_std:.3f} kcal/mol")
    print(f"  Mutant Affinity:    {bms_mut_avg:.3f} +/- {bms_mut_std:.3f} kcal/mol")
    print(f"  RESISTANCE DELTA:   {bms_delta:+.3f} +/- {bms_delta_err:.3f} kcal/mol")
    print(f"  Status: {'RESILIENT (PASS)' if abs(bms_delta) < 0.5 else 'VULNERABLE (FAIL)'}")
    print("")

    print(f"CANDIDATE 2: ERDRP-0519 (Control/Fail)")
    print(f"  Wild-Type Affinity: {erdrp_wt_avg:.3f} +/- {erdrp_wt_std:.3f} kcal/mol")
    print(f"  Mutant Affinity:    {erdrp_mut_avg:.3f} +/- {erdrp_mut_std:.3f} kcal/mol")
    print(f"  RESISTANCE DELTA:   {erdrp_delta:+.3f} +/- {erdrp_delta_err:.3f} kcal/mol")
    print(f"  Status: {'RESILIENT (PASS)' if abs(erdrp_delta) < 0.5 else 'VULNERABLE (FAIL)'}")

    write_verdict(results, bms_delta, bms_delta_err, erdrp_delta, erdrp_delta_err)

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
