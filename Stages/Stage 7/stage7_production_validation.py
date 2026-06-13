#!/usr/bin/env python3
"""
Stage 7 -- Shared production validation utilities (H2, H2b, H3).
Imported by run_production_{tier}_v2.py scripts for checkpoint integrity,
lineage verification, and trajectory continuity validation.
"""

# Constants: tier progression for lineage verification
# When loading the previous tier's checkpoint, currentStep should match
# the previous tier's TOTAL_STEPS.
TIER_EXPECTED_STEPS = {
    "2ns":  1_000_000,   # previous tier for 10ns
    "10ns": 4_000_000,   # previous tier for 20ns
    "20ns": 5_000_000,   # previous tier for 50ns
}

# Tier time ranges (within each tier's own DCD, times start near 0)
TIER_TIME_RANGES = {
    "2ns":        (4.0, 2000.0),    # 2ns tier: first DCD frame at 4ps, last at 2000ps
    "10ns":       (4.0, 8000.0),    # 10ns tier (continuation): first at 4ps, last at 8000ps (8ns duration)
    "10ns_direct":(4.0, 10000.0),   # 0→10ns direct bootstrap: 10ns duration (5M steps × 2fs), starts from Stage 6 state
    "20ns":       (4.0, 10000.0),   # 20ns tier: first DCD frame at 4ps, last at 10000ps (10ns duration)
    "50ns":       (4.0, 30000.0),   # 50ns tier: first DCD frame at 4ps, last at 30000ps (30ns duration)
}

MIN_CHECKPOINT_SIZE = 1_000_000  # 1MB minimum; real H100 checkpoints are ~50MB
TOLERANCE_PS = 4.0  # 1 DCD frame interval = 4ps (DCD_INTERVAL=2000 * TIMESTEP=0.002ps)


def validate_checkpoint_integrity(checkpoint_path, tier_label):
    """H2: Verify checkpoint file exists and is not truncated.

    Args:
        checkpoint_path: Path to the production.chk file
        tier_label: Current tier (e.g., "10ns") for error messages

    Raises:
        RuntimeError if checkpoint is too small (likely truncated from Spot eviction)
    """
    from pathlib import Path
    chk = Path(checkpoint_path)
    if not chk.exists():
        raise FileNotFoundError(
            f"Stage 7 [{tier_label}] checkpoint integrity FAIL: "
            f"{chk} does not exist."
        )
    chk_size = chk.stat().st_size
    if chk_size < MIN_CHECKPOINT_SIZE:
        raise RuntimeError(
            f"Stage 7 [{tier_label}] checkpoint integrity FAIL: "
            f"{chk} is only {chk_size:,} bytes (expected > {MIN_CHECKPOINT_SIZE:,}). "
            f"Typical H100 checkpoint is ~50MB. This checkpoint is likely "
            f"truncated from a Spot VM eviction."
        )
    print(f"  [H2] Checkpoint integrity OK: {chk} ({chk_size:,} bytes)")


def validate_checkpoint_lineage(simulation, prev_tier_label, tier_label):
    """H2b: Verify loaded checkpoint matches expected previous tier.

    MUST be called AFTER loadCheckpoint() and BEFORE simulation.currentStep = 0.
    The checkpoint stores the previous tier's final step count; after reset it's lost.

    Args:
        simulation: OpenMM Simulation object after loadCheckpoint()
        prev_tier_label: The tier whose checkpoint was loaded (e.g., "2ns" when running 10ns)
        tier_label: Current tier for error messages

    Raises:
        RuntimeError if step count doesn't match expected
    """
    loaded_step = simulation.currentStep
    expected_steps = TIER_EXPECTED_STEPS.get(prev_tier_label)

    if expected_steps is None:
        print(f"  [H2b] WARNING: No expected step count for prev tier '{prev_tier_label}'. Skipping lineage check.")
        return

    # Allow some tolerance: checkpoint might be from a resumed run
    # where total steps slightly differ due to chunk boundaries
    if loaded_step != expected_steps:
        raise RuntimeError(
            f"Stage 7 [{tier_label}] checkpoint lineage FAIL: "
            f"Loaded checkpoint has step={loaded_step:,}, but previous tier "
            f"'{prev_tier_label}' should have completed at step={expected_steps:,}. "
            f"This checkpoint may be from a different tier, a corrupted run, "
            f"or a partially-completed evicted run."
        )

    print(f"  [H2b] Checkpoint lineage OK: step {loaded_step:,} matches {prev_tier_label} ({expected_steps:,} expected)")


def validate_trajectory(dcd_path, topology_path, tier_label, total_steps, dcd_interval):
    """H3: Post-production trajectory validation using MDAnalysis.

    Reads the tier's DCD file and verifies frame count and time stamps.

    Args:
        dcd_path: Path to production.dcd
        topology_path: Path to topology PDB
        tier_label: Current tier (e.g., "10ns")
        total_steps: TOTAL_STEPS for this tier
        dcd_interval: DCD_INTERVAL (typically 2000)

    Raises:
        RuntimeError if validation fails
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        print(f"  [H3] WARNING: MDAnalysis not installed. Skipping trajectory validation.")
        return

    from pathlib import Path
    dcd = Path(dcd_path)
    if not dcd.exists() or dcd.stat().st_size == 0:
        raise RuntimeError(
            f"Stage 7 [{tier_label}] trajectory validation FAIL: "
            f"DCD file {dcd} missing or empty."
        )

    expected_frames = total_steps // dcd_interval
    expected_start, expected_end = TIER_TIME_RANGES[tier_label]

    print(f"\n  [H3] Validating trajectory for {tier_label}...")
    print(f"       Expected: {expected_frames} frames, {expected_start:.0f}-{expected_end:.0f} ps")

    u = mda.Universe(str(topology_path), str(dcd_path))
    n_frames = len(u.trajectory)

    issues = []

    # Check frame count (within 1% tolerance)
    frame_tolerance = max(1, int(expected_frames * 0.01))
    if abs(n_frames - expected_frames) > frame_tolerance:
        issues.append(
            f"Frame count {n_frames} vs expected {expected_frames} "
            f"(tolerance: +/-{frame_tolerance})"
        )

    # Check first frame time
    first_time = u.trajectory[0].time
    if abs(first_time - expected_start) > TOLERANCE_PS:
        issues.append(
            f"First frame time {first_time:.1f}ps vs expected {expected_start:.1f}ps"
        )

    # Check last frame time
    last_time = u.trajectory[-1].time
    if abs(last_time - expected_end) > TOLERANCE_PS:
        issues.append(
            f"Last frame time {last_time:.1f}ps vs expected {expected_end:.1f}ps"
        )

    if issues:
        msg = "; ".join(issues)
        raise RuntimeError(
            f"Stage 7 [{tier_label}] trajectory validation FAIL: {msg}"
        )

    print(f"  [H3] PASS: {n_frames} frames, {first_time:.1f}-{last_time:.1f} ps")
