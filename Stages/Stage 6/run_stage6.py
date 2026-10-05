#!/usr/bin/env python3
"""
Stage 6 Runner -- executes the MD-Ready System Reconstruction pipeline.

Uses a single GPU (default physical GPU 0 via CUDA_VISIBLE_DEVICES; override with
NIPAH_CUDA_DEVICE). With --parallel N>1, concurrent cases share that GPU through
NVIDIA MPS.

Usage:
  python run_stage6.py                          # all 4 cases, sequential
  python run_stage6.py --case C_BMS_WT          # single case
  python run_stage6.py --parallel 4             # 4-way MPS on one GPU
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path

STAGE6_DIR = Path(__file__).resolve().parent
COMMON_DIR = STAGE6_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))
from docking_utils import (  # noqa: E402
    cuda_driver_available,
    openmm_cpu_threads,
    openmm_cuda_usable,
    openmm_subprocess_env,
    resolve_openmm_platform,
    resolve_stage6_cuda_device,
)
STAGE6_RESULTS = STAGE6_DIR / "results"
RECONSTRUCTION_SCRIPT = STAGE6_DIR / "reconstruct_system_from_scratch_v2.py"

ALL_CASE_IDS = ["A_ERDRP_WT", "B_ERDRP_MUT", "C_BMS_WT", "D_BMS_MUT"]


def _start_mps():
    """Start NVIDIA MPS daemon if not already running. No-op on non-CUDA systems."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True)
    print("  MPS daemon started.")
    return True


def _stop_mps():
    """Stop NVIDIA MPS daemon."""
    subprocess.run(
        ["bash", "-c", "echo quit | nvidia-cuda-mps-control"],
        capture_output=True,
    )
    print("  MPS daemon stopped.")


def _case_subprocess_env(parallel_cases: int) -> dict:
    return openmm_subprocess_env(parallel_cases)


def _run_case(case_id, python_exe, env):
    """Launch a single --case subprocess. Returns (case_id, Popen)."""
    cmd = [python_exe, str(RECONSTRUCTION_SCRIPT), "--case", case_id]
    log_path = STAGE6_RESULTS / f"{case_id}.log"
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
    return case_id, proc, log_fh


def _run_batch(batch, python_exe, env):
    """Run a batch of cases in parallel. Returns list of (case_id, returncode, elapsed)."""
    results = []
    handles = []
    t0 = time.time()
    for case_id in batch:
        cid, proc, fh = _run_case(case_id, python_exe, env)
        handles.append((cid, proc, fh))
        print(f"    Launched {cid} (pid {proc.pid})")

    for cid, proc, fh in handles:
        proc.wait()
        fh.close()
        elapsed = time.time() - t0
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"    {cid}: {status}  [{elapsed:.1f}s]")
        results.append((cid, proc.returncode, elapsed))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6 Runner: execute MD-Ready System Reconstruction"
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Run only this case (e.g., 'C_BMS_WT')."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run N cases concurrently on one GPU via MPS when N>1 (default: 1). "
             "CPU thread caps still apply to system-build phases.",
    )
    args = parser.parse_args()

    if args.case:
        case_ids = [args.case]
    else:
        case_ids = list(ALL_CASE_IDS)

    parallel = max(1, args.parallel)
    python_exe = sys.executable
    case_parallel = parallel if not args.case else 1
    openmm_threads = openmm_cpu_threads(case_parallel)
    case_env = _case_subprocess_env(case_parallel)

    STAGE6_RESULTS.mkdir(parents=True, exist_ok=True)

    platform = resolve_openmm_platform()
    cuda_device = resolve_stage6_cuda_device()
    use_gpu = cuda_device is not None and platform in ("CUDA", "OpenCL")
    use_mps = parallel > 1 and platform == "CUDA" and openmm_cuda_usable()

    print("=" * 70)
    print("  STAGE 6 EXECUTION -- MD-Ready System Reconstruction")
    print(f"  Cases: {case_ids}")
    if parallel > 1:
        par_label = " (MPS, single GPU)" if use_mps else " (CPU OpenMM, no MPS)"
    else:
        par_label = " (single GPU)" if use_gpu else " (sequential CPU)"
    print(f"  Parallelism: {parallel}-way{par_label}")
    if use_gpu:
        print(f"  GPU: CUDA_VISIBLE_DEVICES={cuda_device} (override: NIPAH_CUDA_DEVICE)")
    if cuda_driver_available() and not openmm_cuda_usable() and platform == "OpenCL":
        print(
            "  Note: OpenMM CUDA unavailable on this host (PTX/driver mismatch); "
            "using OpenCL on the GPU."
        )
    print(f"  OPENMM_CPU_THREADS per case: {openmm_threads}")
    max_total = os.environ.get("NIPAH_OPENMM_MAX_TOTAL", "").strip()
    if max_total:
        print(f"  NIPAH_OPENMM_MAX_TOTAL: {max_total}")
    print(f"  OpenMM platform (subprocess): {platform}")
    print("  (BLAS/OpenMP threads per case match OPENMM_CPU_THREADS)")
    print(f"  Python: {python_exe}")
    print("=" * 70)

    if use_mps:
        _start_mps()

    t_total = time.time()
    passed = 0
    failed = 0
    failed_cases = []

    if parallel <= 1:
        # Sequential: run reconstruct script with all cases (or --case filter)
        cmd = [python_exe, str(RECONSTRUCTION_SCRIPT)]
        if args.case:
            cmd += ["--case", args.case]
        print(f"\n  Running sequentially...")
        result = subprocess.run(cmd, env=case_env)
        if result.returncode == 0:
            passed = len(case_ids)
        else:
            failed = len(case_ids)
            failed_cases = case_ids
    else:
        # Parallel: batch cases into groups of N, run each batch concurrently
        batches = [
            case_ids[i:i + parallel]
            for i in range(0, len(case_ids), parallel)
        ]
        for batch_idx, batch in enumerate(batches):
            print(f"\n  Batch {batch_idx + 1}/{len(batches)}: {batch}")
            batch_results = _run_batch(batch, python_exe, case_env)
            for cid, rc, _ in batch_results:
                if rc == 0:
                    passed += 1
                else:
                    failed += 1
                    failed_cases.append(cid)

        if failed == 0:
            print("\n  Merging per-case manifests into unified stage6_manifest.json...")
            _merge_stage6_manifests(case_ids)

    if use_mps:
        _stop_mps()

    total_elapsed = time.time() - t_total

    print(f"\n{'='*70}")
    print(f"  STAGE 6 SUMMARY")
    print(f"  Passed: {passed}  Failed: {failed}  Total time: {total_elapsed:.1f}s")
    if failed_cases:
        print(f"  Failed cases: {failed_cases}")
        print(f"  Logs: {STAGE6_RESULTS}/{{case_id}}.log")
    print(f"{'='*70}")

    sys.exit(1 if failed > 0 else 0)


def _merge_stage6_manifests(case_ids):
    """Merge per-case _case_manifest.json fragments into a unified stage6_manifest.json.

    Each --case subprocess writes results/{case_id}/_case_manifest.json with its
    case data. This function loads them all and produces the unified manifest.
    """
    import json
    from datetime import datetime, timezone

    merged_cases = {}
    raw_receptor_source = ""

    for cid in case_ids:
        case_dir = STAGE6_RESULTS / cid
        fragment = case_dir / "_case_manifest.json"
        if not fragment.exists():
            print(f"    WARNING: {fragment} not found, skipping {cid}")
            continue
        with open(fragment) as fh:
            merged_cases[cid] = json.load(fh)

    # Read raw_receptor_source from any existing top-level manifest
    existing_manifest = STAGE6_RESULTS / "stage6_manifest.json"
    if existing_manifest.exists():
        with open(existing_manifest) as fh:
            raw_receptor_source = json.load(fh).get("raw_receptor_source", "")

    manifest = {
        "stage": 6,
        "raw_receptor_source": raw_receptor_source,
        "cases": merged_cases,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }
    with open(existing_manifest, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"    Unified manifest written: {existing_manifest}")
    print(f"    Cases: {list(merged_cases.keys())}")


if __name__ == "__main__":
    main()
