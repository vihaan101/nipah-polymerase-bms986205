#!/usr/bin/env python3
"""
Stage 3 Runner — executes the Stage 3 library screening pipeline in order.

Order:
  1. select_expanded_library_v2.py  — assemble Broad Hub library + MaxMin diversity pick (50 compounds)
  2. select_library_v2.py           — Lipinski drug-likeness pre-filter (bRo5 caps)
  3. screen_library_v2.py           — SMILES→PDBQT, dual-receptor docking, ghost clash, filtering

Note: pipeline_v2.py is a single-ligand diagnostic CLI tool and is NOT part of
the batch pipeline. Run it separately with: python pipeline_v2.py <ligand.pdbqt>

Usage:
  python run_stage3.py                   # run all 3 steps
  python run_stage3.py --step 1          # run only library assembly
  python run_stage3.py --step 2 3        # run only filter + screen
  python run_stage3.py --from-step 2     # run from filtering onward
  python run_stage3.py --locked-100      # run the locked 100-compound workflow
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

STAGE3_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "step": 1,
        "name": "Library Assembly + Diversity Selection",
        "script": "select_expanded_library_v2.py",
        "args": [],
        "description": "Reads Broad Repurposing Hub, applies pre-filters, selects 50 diverse candidates via MaxMin Tanimoto picking",
    },
    {
        "step": 2,
        "name": "Lipinski Drug-Likeness Pre-Filter",
        "script": "select_library_v2.py",
        "args": [],
        "description": "Applies bounded relaxation (MW <= 600, LogP <= 5.0) to produce docking-ready library",
    },
    {
        "step": 3,
        "name": "Batch Screening Pipeline",
        "script": "screen_library_v2.py",
        "args": [],
        "description": "SMILES->PDBQT (MMFF minimized, pH 7.4), rigid-rigid docking, ghost clash + allosteric distance filtering",
    },
]

LOCKED_100_STEPS = [
    {
        "step": 1,
        "name": "Compound Selection Data Audit",
        "script": "audit_compound_selection_data.py",
        "args": [],
        "description": "Hashes and records historical/current compound-selection artifacts",
    },
    {
        "step": 2,
        "name": "Locked 100-Compound Library",
        "script": "select_library_100_locked.py",
        "args": [],
        "description": "Builds deterministic Broad-derived locked-100 library with provenance",
    },
    {
        "step": 3,
        "name": "Locked 100 WT Screen",
        "script": "screen_library_100.py",
        "args": ["--mode", "wt", "--resume"],
        "description": "Docks all locked compounds against WT receptor",
    },
    {
        "step": 4,
        "name": "Locked 100 Paired WT/W730A Screen",
        "script": "screen_library_100.py",
        "args": ["--mode", "paired", "--resume"],
        "description": "Docks carry-forward compounds against WT and W730A and ranks mutation-aware response",
    },
]


def run_step(step_info, python_exe):
    """Runs a single Stage 3 step as a subprocess."""
    script_path = STAGE3_DIR / step_info["script"]
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found")
        return False

    cmd = [python_exe, str(script_path)] + step_info["args"]

    print(f"\n{'='*70}")
    print(f"  STEP {step_info['step']}: {step_info['name']}")
    print(f"  Script: {step_info['script']}")
    print(f"  {step_info['description']}")
    print(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  STEP {step_info['step']} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\n  Step {step_info['step']} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3 Runner: execute the library screening pipeline in order"
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[1, 2, 3, 4],
        help="Run only specific step(s), e.g. --step 1 2"
    )
    parser.add_argument(
        "--from-step", type=int, choices=[1, 2, 3, 4],
        help="Run from this step onward, e.g. --from-step 2 runs steps 2 and 3"
    )
    parser.add_argument(
        "--locked-100", action="store_true",
        help="Run the locked 100-compound workflow instead of the historical Stage 3 workflow"
    )
    parser.add_argument(
        "--workers", type=int,
        help="Worker count to pass to locked-100 docking steps"
    )
    parser.add_argument(
        "--vina-path",
        help="Vina executable path to pass to locked-100 docking steps"
    )
    args = parser.parse_args()

    # Determine which steps to run
    active_steps = LOCKED_100_STEPS if args.locked_100 else STEPS
    if args.locked_100:
        for step in active_steps:
            if step["script"] == "screen_library_100.py":
                if args.workers:
                    step["args"].extend(["--workers", str(args.workers)])
                if args.vina_path:
                    step["args"].extend(["--vina-path", args.vina_path])

    if args.step:
        steps_to_run = [s for s in active_steps if s["step"] in args.step]
    elif args.from_step:
        steps_to_run = [s for s in active_steps if s["step"] >= args.from_step]
    else:
        steps_to_run = active_steps

    python_exe = sys.executable

    print("=" * 70)
    title = "Locked 100-Compound Screening" if args.locked_100 else "Ligand Library Preparation & Primary Screening"
    print(f"  STAGE 3 EXECUTION — {title}")
    print(f"  Steps to run: {[s['step'] for s in steps_to_run]}")
    print(f"  Python: {python_exe}")
    print("=" * 70)

    t_total = time.time()
    passed = 0
    failed = 0

    for step_info in steps_to_run:
        ok = run_step(step_info, python_exe)
        if ok:
            passed += 1
        else:
            failed += 1
            print(f"\n  Aborting: step {step_info['step']} failed.")
            break

    total_elapsed = time.time() - t_total

    print(f"\n{'='*70}")
    print(f"  STAGE 3 SUMMARY")
    print(f"  Passed: {passed}  Failed: {failed}  Total time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
