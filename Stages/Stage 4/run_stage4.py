#!/usr/bin/env python3
"""
Stage 4 Runner -- executes all Stage 4 scripts in the correct order.

Order:
  1. verify_hits_v2.py       -- multi-filter verification (potency + resilience + distance + ghost clash)
  2. run_admet_v2.py          -- ADMET/Lipinski screening with clinical override
  3. verify_ghost_clash_v2.py -- standalone ghost clash diagnostic (independent check)

Usage:
  python run_stage4.py                   # run all 3 steps
  python run_stage4.py --step 1          # run only verification
  python run_stage4.py --step 2 3        # run only ADMET + ghost clash
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

STAGE4_DIR = Path(__file__).resolve().parent

STEPS = [
    {
        "step": 1,
        "name": "Multi-Filter Verification",
        "script": "verify_hits_v2.py",
        "args": [],
        "description": "5-gate filter cascade: potency, resilience, allosteric distance, pose stability, ghost clash",
    },
    {
        "step": 2,
        "name": "ADMET / Lipinski Screening",
        "script": "run_admet_v2.py",
        "args": [],
        "description": "Lipinski Rule of 5 with clinical-phase override logic",
    },
    {
        "step": 3,
        "name": "Ghost Clash Diagnostic",
        "script": "verify_ghost_clash_v2.py",
        "args": [],
        "description": "Standalone ghost clash verification against BMS-986205",
    },
]


def run_step(step_info, python_exe):
    script_path = STAGE4_DIR / step_info["script"]
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
        description="Stage 4 Runner: execute all verification scripts in order"
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[1, 2, 3],
        help="Run only specific step(s), e.g. --step 1 2"
    )
    args = parser.parse_args()

    steps_to_run = [s for s in STEPS if s["step"] in args.step] if args.step else STEPS

    python_exe = sys.executable

    print("=" * 70)
    print("  STAGE 4 EXECUTION -- Multi-Filter Verification & ADMET")
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
    print(f"  STAGE 4 SUMMARY")
    print(f"  Passed: {passed}  Failed: {failed}  Total time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
