#!/usr/bin/env python3
"""
Stage 3 Runner — executes the library screening pipeline in order.

Default: locked 100-compound workflow (audit → library → WT screen → paired screen).

Legacy 50-compound path lives under Legacy/ and is selected with --legacy.

Usage:
  python run_stage3.py                   # locked-100 (default)
  python run_stage3.py --legacy          # historical 3-step workflow
  python run_stage3.py --list            # print active step scripts
  python run_stage3.py --step 2          # run specific locked-100 step(s)
"""

import sys
import subprocess
import time
import argparse
from pathlib import Path

STAGE3_DIR = Path(__file__).resolve().parent
LEGACY_DIR = STAGE3_DIR / "Legacy"

LEGACY_STEPS = [
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
        "description": "SMILES->PDBQT, rigid-rigid docking, ghost clash + allosteric distance filtering",
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


def run_step(step_info, python_exe, script_root: Path):
    """Runs a single Stage 3 step as a subprocess."""
    script_path = script_root / step_info["script"]
    if not script_path.exists():
        print(f"  ERROR: {script_path} not found")
        return False

    cmd = [python_exe, str(script_path)] + step_info["args"]

    print(f"\n{'='*70}")
    print(f"  STEP {step_info['step']}: {step_info['name']}")
    print(f"  Script: {script_path}")
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
        "--legacy", action="store_true",
        help="Run the historical 50-compound Legacy workflow instead of locked-100"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List step numbers and scripts for the active workflow, then exit"
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

    active_steps = LEGACY_STEPS if args.legacy else LOCKED_100_STEPS
    script_root = LEGACY_DIR if args.legacy else STAGE3_DIR

    if not args.legacy:
        for step in active_steps:
            if step["script"] == "screen_library_100.py":
                if args.workers:
                    step["args"].extend(["--workers", str(args.workers)])
                if args.vina_path:
                    step["args"].extend(["--vina-path", args.vina_path])

    if args.list:
        mode = "legacy" if args.legacy else "locked-100"
        print(f"Stage 3 workflow: {mode} (scripts under {script_root})")
        for step in active_steps:
            print(f"  {step['step']}: {step['script']} — {step['name']}")
        sys.exit(0)

    if args.step:
        steps_to_run = [s for s in active_steps if s["step"] in args.step]
    elif args.from_step:
        steps_to_run = [s for s in active_steps if s["step"] >= args.from_step]
    else:
        steps_to_run = active_steps

    python_exe = sys.executable

    print("=" * 70)
    title = "Legacy 50-Compound Screening" if args.legacy else "Locked 100-Compound Screening"
    print(f"  STAGE 3 EXECUTION — {title}")
    print(f"  Steps to run: {[s['step'] for s in steps_to_run]}")
    print(f"  Python: {python_exe}")
    print("=" * 70)

    t_total = time.time()
    passed = 0
    failed = 0

    for step_info in steps_to_run:
        ok = run_step(step_info, python_exe, script_root)
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
