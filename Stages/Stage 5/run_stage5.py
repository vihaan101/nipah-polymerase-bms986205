#!/usr/bin/env python3
"""
Stage 5 Runner -- executes all Stage 5 scripts in the correct order.

Order:
  1. verify_v3.py          -- multi-seed statistical validation (5 seeds x 4 conditions)
  2. docking_engine_v3.py  -- rigid-vs-rigid multi-seed validation

Usage:
  python run_stage5.py                   # run all 2 steps
  python run_stage5.py --step 1          # run only multi-seed verification
  python run_stage5.py --step 2          # run only rigid validation
  python run_stage5.py --focused-100     # run locked-100 focused validation
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

STAGE5_DIR = Path(__file__).resolve().parent
STAGE5_RESULTS = STAGE5_DIR / "results"

STEPS = [
    {
        "step": 1,
        "name": "Multi-Seed Verification",
        "script": "verify_v3.py",
        "args": [],
        "description": "5-seed statistical validation across BMS-986205 and ERDRP-0519",
    },
    {
        "step": 2,
        "name": "Rigid Validation",
        "script": "docking_engine_v3.py",
        "args": [],
        "description": "Rigid-vs-rigid multi-seed re-docking for BMS-986205",
    },
]

FOCUSED_100_STEPS = [
    {
        "step": 1,
        "name": "Focused 100 Candidate Freeze",
        "script": "select_focused_validation_candidates.py",
        "args": [],
        "description": "Freezes top mutation-aware finalists plus prespecified BMS and ERDRP comparator manifest",
    },
    {
        "step": 2,
        "name": "Focused 100 Five-Seed Validation",
        "script": "run_focused_five_seed_validation.py",
        "args": ["--resume"],
        "description": "Runs five-seed WT/W730A validation for frozen locked-100 finalists",
    },
]


def run_step(step_info, python_exe):
    script_path = STAGE5_DIR / step_info["script"]
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
        description="Stage 5 Runner: execute all reproducibility and validation scripts in order"
    )
    parser.add_argument(
        "--step", type=int, nargs="+", choices=[1, 2],
        help="Run only specific step(s), e.g. --step 1"
    )
    parser.add_argument(
        "--focused-100", action="store_true",
        help="Run locked-100 focused validation instead of the historical Stage 5 workflow"
    )
    parser.add_argument(
        "--workers", type=int,
        help="Worker count to pass to focused-100 docking"
    )
    parser.add_argument(
        "--vina-path",
        help="Vina executable path to pass to focused-100 docking"
    )
    args = parser.parse_args()

    active_steps = FOCUSED_100_STEPS if args.focused_100 else STEPS
    if args.focused_100:
        for step in active_steps:
            if step["script"] == "run_focused_five_seed_validation.py":
                if args.workers:
                    step["args"].extend(["--workers", str(args.workers)])
                if args.vina_path:
                    step["args"].extend(["--vina-path", args.vina_path])

    steps_to_run = [s for s in active_steps if s["step"] in args.step] if args.step else active_steps

    # Ensure results directory exists
    STAGE5_RESULTS.mkdir(parents=True, exist_ok=True)

    python_exe = sys.executable

    print("=" * 70)
    title = "Focused 100 Validation" if args.focused_100 else "Reproducibility & Adversarial Audit"
    print(f"  STAGE 5 EXECUTION -- {title}")
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
    print(f"  STAGE 5 SUMMARY")
    print(f"  Passed: {passed}  Failed: {failed}  Total time: {total_elapsed:.1f}s")
    print(f"{'='*70}")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
