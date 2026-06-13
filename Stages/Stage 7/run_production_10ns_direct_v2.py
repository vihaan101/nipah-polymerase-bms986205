#!/usr/bin/env python3
"""
Production MD Protocol (0→10 ns direct bootstrap) -- Single Replicate Mode
===========================================================================
Runs 5,000,000 steps (10 ns at 2 fs/step) per replicate,
starting directly from Stage 6 equilibrated_state_xml with NO prior-tier
checkpoint dependency.

Key differences from run_production_10ns_v2.py:
  - Bootstraps from equilibrated_state_xml (not a 2ns production.chk)
  - Operates on a single (case_id, replicate_id) pair per invocation
  - Outputs to <results-root>/<case_id>/replicate_<N>/
  - Per-replicate Langevin seed: sha256("{case}_rep{N}") % 2^31
  - Writes TASK_COMPLETE sentinel on success
  - Supports --resume for Spot VM eviction recovery

Usage:
  python run_production_10ns_direct_v2.py \
    --case A_ERDRP_WT --replicate 1 \
    --results-root /path/to/stage7_10ns_direct/results \
    --stage6-results "/path/to/Stage 6/results" \
    [--resume]
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openmm as mm
from openmm import app, unit

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIER_LABEL    = "10ns_direct"
TOTAL_STEPS   = 5_000_000        # 10 ns @ 2 fs/step
CHUNK_SIZE    = 500_000           # ~1 ns per chunk; checkpoint every chunk
TEMPERATURE   = 300 * unit.kelvin
PRESSURE      = 1.0 * unit.bar
TIMESTEP      = 0.002 * unit.picoseconds
FRICTION      = 1 / unit.picosecond
BAROSTAT_FREQ = 250
LOG_INTERVAL  = 1000
DCD_INTERVAL  = 2000
CHK_INTERVAL  = 10_000           # Save checkpoint every 10k steps within a chunk

VALID_CASES = frozenset(["A_ERDRP_WT", "B_ERDRP_MUT", "C_BMS_WT", "D_BMS_MUT"])

SENTINEL_FILE = "TASK_COMPLETE"   # Written in the replicate output dir on success


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"[{TIER_LABEL}] Contract violation: {label} not found at {path}"
        )
    if path.stat().st_size == 0:
        raise RuntimeError(
            f"[{TIER_LABEL}] Contract violation: {label} is empty at {path}"
        )
    return path


def _completed_step_from_sentinel(sentinel: Path, resume: bool) -> int | None:
    """Return the recorded completed step count if this run is allowed to resume."""
    if not sentinel.exists():
        return None
    if not resume:
        print(
            f"  WARNING: ignoring local sentinel at {sentinel} because --resume "
            "was not requested."
        )
        return None
    try:
        with sentinel.open() as fh:
            payload = json.load(fh)
    except Exception:
        print("  Found sentinel, but could not verify steps. Resuming for safety.")
        return None
    return int(payload.get("final_step", 0))


def _should_resume_from_checkpoint(checkpoint_path: Path, resume: bool) -> bool:
    """Only trust a local checkpoint when resume mode was explicitly requested."""
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size <= 0:
        return False
    if not resume:
        print(
            f"  WARNING: ignoring local checkpoint at {checkpoint_path} because "
            "--resume was not requested."
        )
        return False
    return True


def _resolve_stage6_path(manifest_path: str, stage6_results: Path) -> Path:
    """Resolve a manifest path to the local Stage 6 results directory.

    The manifest was written on the Stage 6 VM with absolute paths; extract the
    relative tail after the last 'results' segment and resolve against the
    local stage6_results dir.
    """
    p = Path(manifest_path)
    parts = p.parts
    last_results_idx = None
    for i, part in enumerate(parts):
        if part == "results":
            last_results_idx = i
    if last_results_idx is not None and last_results_idx + 1 < len(parts):
        tail = Path(*parts[last_results_idx + 1:])
    else:
        tail = Path(p.name)
    local = stage6_results / tail
    if local.exists():
        return local
    if p.exists():
        return p
    return local  # return even if missing so require_file gives a clear error


def _clear_run_outputs(case_results_dir: Path) -> None:
    """Remove stale local outputs before replaying a replicate from Stage 6."""
    for name in (
        "production.chk",
        "production.dcd",
        "production.log",
        "production_final.pdb",
        SENTINEL_FILE,
        "_case_manifest.json",
    ):
        path = case_results_dir / name
        if path.exists():
            path.unlink()


def _validate_completed_outputs(case_results_dir: Path, topology_pdb_path: Path) -> tuple[bool, str | None]:
    """Verify that an already-completed local replicate really has a valid trajectory."""
    production_dcd = case_results_dir / "production.dcd"
    if not production_dcd.exists():
        return False, f"missing completed trajectory at {production_dcd}"
    try:
        _validate(production_dcd, topology_pdb_path)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _bootstrap_from_stage6_state(simulation, state_xml_path: Path) -> None:
    """Restore positions, velocities, and box vectors from the Stage 6 state XML."""
    print("  Bootstrapping from Stage 6 equilibrated_state_xml...")
    with state_xml_path.open() as f:
        state = mm.XmlSerializer.deserialize(f.read())
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    print("  State loaded (positions, velocities, box vectors restored)")


def _create_simulation(system, topology, case_seed: int):
    """Build a fresh simulation so replay starts from a clean OpenMM step counter."""
    for platform_name in ["CUDA", "OpenCL"]:
        integrator = mm.LangevinMiddleIntegrator(TEMPERATURE, FRICTION, TIMESTEP)
        integrator.setRandomNumberSeed(case_seed)
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            props = {"Precision": "mixed"}
            simulation = app.Simulation(topology, system, integrator, platform, props)
            print(f"  Using {platform_name} platform")
            return simulation
        except Exception:
            continue
    raise RuntimeError(
        "No GPU platform available (tried CUDA, OpenCL). "
        "Production MD on CPU is not permitted."
    )


def load_stage6_manifest(stage6_results: Path) -> dict:
    manifest_path = stage6_results / "stage6_manifest.json"
    require_file(manifest_path, "Stage 6 manifest")
    with manifest_path.open() as fh:
        manifest = json.load(fh)
    cases = manifest.get("cases")
    if not cases:
        raise RuntimeError("[10ns_direct] Stage 6 manifest has no 'cases' dict")
    required = ["system_xml", "topology_pdb", "equilibrated_state_xml"]
    for cid, cdata in cases.items():
        missing = [k for k in required if not cdata.get(k)]
        if missing:
            raise RuntimeError(
                f"[10ns_direct] Case '{cid}' missing Stage 6 keys: {missing}"
            )
    return manifest


# ---------------------------------------------------------------------------
# Core production runner
# ---------------------------------------------------------------------------

def run_production(
    case_id: str,
    case_data: dict,
    replicate: int,
    results_root: Path,
    stage6_results: Path,
    resume: bool = False,
) -> dict:
    """Run a 0→10 ns direct production MD for one (case, replicate) pair.

    Returns a dict of output paths on success.
    """
    case_results_dir = results_root / case_id / f"replicate_{replicate}"

    print(f"\n{'='*80}")
    print(f"  PRODUCTION MD [{TIER_LABEL}]: {case_id}  replicate={replicate}")
    print(f"  Output dir: {case_results_dir}")
    print(f"  {_ts()}")
    print(f"{'='*80}")

    # Fast-exit only for an explicit resume. Fresh runs must ignore stale local
    # sentinels left behind in a golden image or reused workspace.
    sentinel = case_results_dir / SENTINEL_FILE
    completed_step = _completed_step_from_sentinel(sentinel, resume)

    # ------------------------------------------------------------------ #
    # [1/4] Load system from Stage 6
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading system from Stage 6...")
    system_xml_path  = require_file(
        _resolve_stage6_path(case_data["system_xml"], stage6_results),
        f"{case_id} system.xml"
    )
    topology_pdb_path = require_file(
        _resolve_stage6_path(case_data["topology_pdb"], stage6_results),
        f"{case_id} topology.pdb"
    )
    state_xml_path = require_file(
        _resolve_stage6_path(case_data["equilibrated_state_xml"], stage6_results),
        f"{case_id} equilibrated_state_xml"
    )

    with system_xml_path.open() as f:
        system = mm.XmlSerializer.deserialize(f.read())

    print("  Scanning forces...")
    forces = system.getForces()
    restraint_indices = []
    barostat_indices  = []
    for i, force in enumerate(forces):
        if isinstance(force, mm.CustomExternalForce):
            restraint_indices.append(i)
        elif isinstance(force, mm.MonteCarloBarostat):
            barostat_indices.append(i)

    if len(restraint_indices) > 1:
        raise RuntimeError(
            f"[{TIER_LABEL}] Contract violation: expected <=1 restraint force, "
            f"found {len(restraint_indices)}"
        )
    for idx in reversed(restraint_indices):
        system.removeForce(idx)
        print("  Removed serialized restraint force")

    if len(barostat_indices) > 1:
        raise RuntimeError(
            f"[{TIER_LABEL}] Contract violation: expected <=1 barostat, "
            f"found {len(barostat_indices)}"
        )
    if not barostat_indices:
        print(f"  + Adding MonteCarloBarostat ({PRESSURE}, {TEMPERATURE}, freq={BAROSTAT_FREQ})")
        system.addForce(mm.MonteCarloBarostat(PRESSURE, TEMPERATURE, BAROSTAT_FREQ))
    else:
        print("  Barostat already present")

    # ------------------------------------------------------------------ #
    # [2/4] Load topology
    # ------------------------------------------------------------------ #
    pdb = app.PDBFile(str(topology_pdb_path))

    if completed_step is not None:
        if completed_step >= TOTAL_STEPS:
            outputs_valid, validation_error = _validate_completed_outputs(
                case_results_dir, topology_pdb_path
            )
            if outputs_valid:
                print(
                    f"  SKIP: {sentinel} exists and completed outputs revalidated "
                    f"({completed_step} steps)."
                )
                return _build_output_paths(case_results_dir)
            print(
                "  WARNING: completed sentinel exists but outputs are invalid: "
                f"{validation_error}"
            )
            if resume:
                print("  Clearing completed resume artifacts and rebuilding from Stage 6.")
                _clear_run_outputs(case_results_dir)
            completed_step = None
        if completed_step is not None:
            print(
                f"  Existing sentinel is for {completed_step} steps. Extending to "
                f"{TOTAL_STEPS}."
            )

    # ------------------------------------------------------------------ #
    # [3/4] Create simulation
    # ------------------------------------------------------------------ #
    print("\n[3/4] Creating simulation...")
    # Per-replicate deterministic seed (different trajectory for each replicate)
    seed_str  = f"{case_id}_rep{replicate}"
    case_seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2 ** 31)
    print(f"  Langevin seed: {case_seed} (derived from '{seed_str}')")
    simulation = _create_simulation(system, pdb.topology, case_seed)

    case_results_dir.mkdir(parents=True, exist_ok=True)
    production_chk = case_results_dir / "production.chk"
    start_step = 0

    # Resume only when explicitly requested by the worker/orchestrator.
    if _should_resume_from_checkpoint(production_chk, resume):
        print(f"  Found existing checkpoint: {production_chk}. Resuming...")
        simulation.loadCheckpoint(str(production_chk))
        start_step = simulation.currentStep
        print(f"  Resumed at step {start_step}")
    else:
        # Fresh start: bootstrap from Stage 6 equilibrated state
        _bootstrap_from_stage6_state(simulation, state_xml_path)

    remaining_steps = TOTAL_STEPS - start_step
    if remaining_steps <= 0:
        outputs_valid, validation_error = _validate_completed_outputs(
            case_results_dir, topology_pdb_path
        )
        if outputs_valid:
            print(f"  Already completed {TOTAL_STEPS} steps. Existing outputs validated.")
            _write_sentinel(case_results_dir, case_id, replicate, TOTAL_STEPS)
            return _build_output_paths(case_results_dir)
        if not resume:
            raise RuntimeError(
                f"[{TIER_LABEL}] completed checkpoint outputs invalid outside resume mode: "
                f"{validation_error}"
            )
        print(
            "  WARNING: completed checkpoint outputs are invalid. "
            f"Rebuilding fresh from Stage 6: {validation_error}"
        )
        _clear_run_outputs(case_results_dir)
        simulation = None
        gc.collect()
        print("  Recreating simulation to reset OpenMM step state before replay.")
        simulation = _create_simulation(system, pdb.topology, case_seed)
        _bootstrap_from_stage6_state(simulation, state_xml_path)
        start_step = 0
        remaining_steps = TOTAL_STEPS

    # ------------------------------------------------------------------ #
    # [4/4] Run production in chunks with dense checkpointing
    # ------------------------------------------------------------------ #
    production_log = case_results_dir / "production.log"
    production_dcd = case_results_dir / "production.dcd"
    production_pdb = case_results_dir / "production_final.pdb"

    # Append if we are resuming from a non-zero step
    append_mode = start_step > 0
    # If resuming but DCD was lost (e.g. VM evicted before final_sync), create fresh DCD
    dcd_append_mode = append_mode and production_dcd.exists()
    if append_mode and not dcd_append_mode:
        print(f"  WARNING: Resume mode (step={start_step}) but DCD missing; creating fresh DCD")
    print(f"  Setting up reporters (append={append_mode}, dcd_append={dcd_append_mode})...")
    simulation.reporters.append(app.StateDataReporter(
        str(production_log), LOG_INTERVAL,
        step=True, time=True, potentialEnergy=True,
        temperature=True, progress=True, remainingTime=True,
        speed=True, totalSteps=TOTAL_STEPS, append=append_mode
    ))
    simulation.reporters.append(app.DCDReporter(
        str(production_dcd), DCD_INTERVAL, append=dcd_append_mode
    ))
    simulation.reporters.append(app.CheckpointReporter(str(production_chk), CHK_INTERVAL))

    n_chunks  = remaining_steps // CHUNK_SIZE   # do NOT use max(1,...) — that forces a full chunk overrun when remaining < CHUNK_SIZE
    leftover  = remaining_steps % CHUNK_SIZE
    print(
        f"\n[4/4] Running {remaining_steps} steps in {n_chunks} chunks of {CHUNK_SIZE}"
        f"{f' + {leftover} leftover' if leftover else ''}"
    )
    print(f"  {_ts()}")

    run_start = time.time()
    for chunk_i in range(n_chunks):
        simulation.step(CHUNK_SIZE)
        # Flush reporters to ensure data is on disk before next chunk/eviction
        for reporter in simulation.reporters:
            if hasattr(reporter, "flush"):
                try: reporter.flush()
                except: pass

        elapsed = time.time() - run_start
        current = simulation.currentStep
        pct     = current / TOTAL_STEPS * 100
        print(
            f"  [{_ts()}] Chunk {chunk_i+1}/{n_chunks}: "
            f"step {current:,}/{TOTAL_STEPS:,} ({pct:.1f}%) [{elapsed:.0f}s]"
        )
        simulation.saveCheckpoint(str(production_chk))

    if leftover > 0:
        simulation.step(leftover)
        simulation.saveCheckpoint(str(production_chk))

    elapsed_min = (time.time() - run_start) / 60
    print(f"\n{'='*80}")
    print(f"  PRODUCTION COMPLETE [{TIER_LABEL}]: {case_id} rep{replicate}")
    print(f"  Time: {elapsed_min:.2f} min  |  Final step: {simulation.currentStep:,}")
    print(f"  {_ts()}")
    print(f"{'='*80}")

    # Write final PDB
    with production_pdb.open('w') as f:
        app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True).getPositions(),
            f
        )
    print(f"  Saved {production_pdb}")

    # Validate trajectory (imports MDAnalysis if available)
    _validate(production_dcd, topology_pdb_path)

    # Write TASK_COMPLETE sentinel
    _write_sentinel(case_results_dir, case_id, replicate, simulation.currentStep)

    outputs = _build_output_paths(case_results_dir)

    # Write per-replicate manifest
    manifest_path = case_results_dir / "_case_manifest.json"
    with manifest_path.open("w") as fh:
        json.dump({"case_id": case_id, "replicate_id": replicate,
                   "tier": TIER_LABEL, **outputs,
                   "completed_at": _ts()}, fh, indent=2)

    del simulation
    gc.collect()

    return outputs


# ---------------------------------------------------------------------------
# Helpers for outputs and validation
# ---------------------------------------------------------------------------

def _build_output_paths(case_results_dir: Path) -> dict:
    return {
        "production_log":        str(case_results_dir / "production.log"),
        "production_dcd":        str(case_results_dir / "production.dcd"),
        "production_checkpoint": str(case_results_dir / "production.chk"),
        "production_final_pdb":  str(case_results_dir / "production_final.pdb"),
    }


def _write_sentinel(case_results_dir: Path, case_id: str, replicate: int, final_step: int):
    sentinel = case_results_dir / SENTINEL_FILE
    data = {
        "case_id":    case_id,
        "replicate":  replicate,
        "tier":       TIER_LABEL,
        "final_step": final_step,
        "completed_at": _ts(),
    }
    tmp = sentinel.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(sentinel)
    print(f"  Sentinel written: {sentinel}")


def _validate(dcd_path: Path, topology_path: Path):
    """Run H3 trajectory validation; soft-fail if MDAnalysis unavailable."""
    try:
        from stage7_production_validation import validate_trajectory
        validate_trajectory(
            str(dcd_path), str(topology_path),
            TIER_LABEL, TOTAL_STEPS, DCD_INTERVAL
        )
    except ImportError:
        print("  [H3] WARNING: stage7_production_validation not found. Skipping.")
    except Exception as e:
        raise RuntimeError(f"[{TIER_LABEL}] Trajectory validation FAILED: {e}") from e


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Stage 7: Production MD 0→10 ns direct bootstrap ({TIER_LABEL})"
    )
    parser.add_argument(
        "--case", required=True,
        choices=sorted(VALID_CASES),
        help="Case ID, e.g. A_ERDRP_WT"
    )
    parser.add_argument(
        "--replicate", required=True, type=int,
        help="Replicate number (1-based, e.g. 1..5)"
    )
    parser.add_argument(
        "--results-root", type=Path, default=None,
        help=(
            "Root directory for output. Outputs go to <results-root>/<case>/"
            "replicate_<N>/. Defaults to <script_dir>/stage7_10ns_direct/results"
        )
    )
    parser.add_argument(
        "--stage6-results", type=Path, default=None,
        help=(
            "Path to Stage 6 results directory containing stage6_manifest.json. "
            "Defaults to <project_root>/Stages/Stage 6/results"
        )
    )
    parser.add_argument(
        "--bootstrap-from-stage6", action="store_true", default=True,
        help="(Default) Bootstrap from Stage 6 equilibrated_state_xml. Always on."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from this tier's own checkpoint if available (spot eviction recovery)."
    )
    args = parser.parse_args()

    if args.replicate < 1:
        print(f"FATAL: --replicate must be >= 1, got {args.replicate}")
        sys.exit(1)

    # Resolve paths
    script_dir   = Path(__file__).resolve().parent
    project_root = script_dir.parents[1]

    stage6_results = args.stage6_results or (
        project_root / "Stages" / "Stage 6" / "results"
    )
    results_root = args.results_root or (
        script_dir / "stage7_10ns_direct" / "results"
    )

    print(f"[{TIER_LABEL}] Stage 6 results:  {stage6_results}")
    print(f"[{TIER_LABEL}] Output root:      {results_root}")
    print(f"[{TIER_LABEL}] Case: {args.case}  Replicate: {args.replicate}")
    print(f"[{TIER_LABEL}] Resume: {args.resume}")

    manifest = load_stage6_manifest(stage6_results)
    cases    = manifest["cases"]

    if args.case not in cases:
        print(f"FATAL: Case '{args.case}' not found in Stage 6 manifest. "
              f"Valid: {sorted(cases.keys())}")
        sys.exit(1)

    try:
        outputs = run_production(
            case_id      = args.case,
            case_data    = cases[args.case],
            replicate    = args.replicate,
            results_root = results_root,
            stage6_results = stage6_results,
            resume       = args.resume,
        )
    except Exception as e:
        print(f"\nFATAL [{args.case} rep{args.replicate}]: {e}")
        sys.exit(1)

    print(f"\nOutput files:")
    for k, v in outputs.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
