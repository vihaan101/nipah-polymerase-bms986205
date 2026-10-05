"""
Stage 7 — Shared evaluation utilities for all post-MD analysis scripts.
Provides manifest-driven path resolution, file validation, and case metadata.

Path defaults assume a repo checkout under ``NIPAH_PROJECT_ROOT`` (or auto-detect via ``Stages/Stage 1``).
Override trajectory and eval locations with ``STAGE7_*`` environment variables (see module constants below).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from paths import project_root

PROJECT_ROOT = project_root()

# ── Path resolution for Stage 6 artifacts ───────────────────────────
_DEFAULT_STAGE6 = PROJECT_ROOT / "Stages" / "Stage 6" / "results"
STAGE6_RESULTS = Path(
    os.environ.get("STAGE7_STAGE6_RESULTS_DIR", str(_DEFAULT_STAGE6))
)
STAGE6_MANIFEST = Path(
    os.environ.get(
        "STAGE7_STAGE6_MANIFEST",
        str(STAGE6_RESULTS / "stage6_manifest.json"),
    )
)

STAGE7_LOCAL_20NS = PROJECT_ROOT / "stage7_results_20ns"
STAGE7_CANONICAL_20NS = PROJECT_ROOT / "Stages" / "Stage 7" / "results_20ns"

_DEFAULT_DIRECT_RESULTS = PROJECT_ROOT / "stage7_50ns_direct" / "results"
STAGE7_DIRECT_RESULTS_ROOT = Path(
    os.environ.get("STAGE7_DIRECT_RESULTS_ROOT", str(_DEFAULT_DIRECT_RESULTS))
)

_DEFAULT_ANALYSIS_WORK = PROJECT_ROOT / "Stages" / "Stage 7" / "analysis_50ns_direct"


def _resolved_eval_base() -> Path:
    """Directory that holds per-metric subdirs (writes and cross-metric reads)."""
    if os.environ.get("STAGE7_EVAL_WORK_DIR"):
        return Path(os.environ["STAGE7_EVAL_WORK_DIR"])
    if os.environ.get("STAGE7_EVAL_50NS_DIR"):
        return Path(os.environ["STAGE7_EVAL_50NS_DIR"])
    return _DEFAULT_ANALYSIS_WORK


EVAL_50NS_DIR = Path(
    os.environ.get("STAGE7_EVAL_50NS_DIR", str(_resolved_eval_base()))
)

RMSD_50NS_DIR = EVAL_50NS_DIR / "rmsd_50ns_direct"
MMGBSA_50NS_DIR = EVAL_50NS_DIR / "mmgbsa_50ns_direct"

DEFAULT_FRAME_INTERVAL_PS = 4.0


def eval_work_dir() -> Path:
    """Base directory for per-metric analyzer outputs."""
    override = os.environ.get("STAGE7_EVAL_WORK_DIR")
    if override:
        return Path(override)
    if os.environ.get("STAGE7_EVAL_50NS_DIR"):
        return Path(os.environ["STAGE7_EVAL_50NS_DIR"])
    return _DEFAULT_ANALYSIS_WORK


def eval_metric_dir(metric_subdir: str) -> Path:
    """Return (and create) the output directory for a metric under the eval work dir."""
    out = eval_work_dir() / metric_subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _valid_box_lengths(box_lengths):
    if box_lengths is None:
        return None
    box = np.asarray(box_lengths, dtype=float).reshape(-1)
    if box.size < 3:
        return None
    box = box[:3]
    if not np.all(np.isfinite(box)) or np.any(box <= 0):
        return None
    return box


def minimum_image_displacements(vectors, box_lengths):
    """Apply an orthorhombic minimum-image convention when box lengths are available."""
    box = _valid_box_lengths(box_lengths)
    arr = np.asarray(vectors, dtype=float)
    if box is None or arr.size == 0:
        return arr
    return arr - box * np.round(arr / box)


def minimum_image_distance(point_a, point_b, box_lengths):
    """Return the minimum-image distance between two points."""
    delta = minimum_image_displacements(
        np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float),
        box_lengths,
    )
    return float(np.linalg.norm(delta))


def require_file(path, label):
    """Fail loudly if a required input file is missing or empty."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stage 7 eval contract violation: {label} not found at {p}")
    if p.stat().st_size == 0:
        raise RuntimeError(f"Stage 7 eval contract violation: {label} is empty at {p}")
    return p


def load_stage6_cases():
    """Load and validate the Stage 6 manifest."""
    with require_file(STAGE6_MANIFEST, "Stage 6 manifest").open() as fh:
        manifest = json.load(fh)
    cases = manifest.get("cases")
    if not cases:
        raise RuntimeError("Stage 7 eval contract violation: Stage 6 manifest has no 'cases'")

    target_case = os.environ.get("STAGE7_TARGET_CASE")
    if target_case:
        if target_case in cases:
            return {target_case: cases[target_case]}
        raise ValueError(f"Target case '{target_case}' not found in manifest")

    return cases


def resolve_stage6_path(manifest_value):
    """Resolve a manifest path to a file under STAGE6_RESULTS."""
    p = Path(manifest_value)

    if p.exists():
        return p

    if "results" in p.parts:
        idx = max(i for i, part in enumerate(p.parts) if part == "results")
        tail = Path(*p.parts[idx + 1 :])
        candidate = STAGE6_RESULTS / tail
        if candidate.exists():
            return candidate
    return p


def resolve_stage6_case_path(case_id, filename):
    """Resolve a conventional Stage 6 per-case artifact location."""
    candidate = STAGE6_RESULTS / "results" / case_id / filename
    if not candidate.exists():
        candidate = STAGE6_RESULTS / case_id / filename
    return require_file(candidate, f"{filename} for {case_id}")


def _direct_results_roots() -> list[Path]:
    return [STAGE7_DIRECT_RESULTS_ROOT]


def find_trajectory(case_id, filename="production.dcd"):
    """Find trajectory in the 50ns direct results (prefers replicate 1)."""
    roots = _direct_results_roots()
    for root in roots:
        candidate = root / case_id / "replicate_1" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Stage 7 eval contract violation: {filename} for {case_id} not found "
        f"under {roots} (replicate_1)"
    )


def get_replicate_trajectory(case_id, replicate_id, filename="production.dcd"):
    """Get the trajectory for a specific 50ns direct replicate."""
    roots = _direct_results_roots()
    for root in roots:
        candidate = root / case_id / f"replicate_{replicate_id}" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Trajectory broken: {filename} for {case_id} rep{replicate_id} not found. "
        f"Searched roots: {roots}"
    )


def find_trajectory_chain(case_id, filename="production.dcd"):
    """Return ordered list of DCD paths across the 5 replicates for MDAnalysis ChainReader."""
    root = STAGE7_DIRECT_RESULTS_ROOT
    chain = []
    for rep_id in range(1, 6):
        candidate = root / case_id / f"replicate_{rep_id}" / filename
        if not candidate.exists():
            raise FileNotFoundError(
                f"Trajectory chain broken: {filename} for {case_id} rep{rep_id} not found "
                f"at {candidate}. Verify STAGE7_DIRECT_RESULTS_ROOT."
            )
        chain.append(str(candidate))
    return chain


_universe_factory: Callable | None = None


def set_universe_factory(factory: Callable | None) -> None:
    """Inject MDAnalysis Universe constructor (for tests)."""
    global _universe_factory
    _universe_factory = factory


def _default_universe_factory():
    import MDAnalysis as mda

    return mda.Universe


def robust_universe(topology, trajectory=None, attempts=5, delay=2.0, **kwargs):
    """Initialize an MDAnalysis Universe with retries to handle transient I/O errors."""
    factory = _universe_factory or _default_universe_factory()
    last_err = None
    for i in range(attempts):
        try:
            if trajectory:
                return factory(topology, trajectory, **kwargs)
            return factory(topology, **kwargs)
        except Exception as e:
            last_err = e
            print(f"  WARNING: Universe initialization failed (attempt {i+1}/{attempts}): {e}")
            if "Reading DCD header failed" in str(e) or "StopIteration" in str(e):
                time.sleep(delay * (i + 1))
            else:
                time.sleep(0.5)
    raise last_err


def cumulative_time_ns_from_raw_times(raw_times_ps):
    """Repair tier-reset timestamps into a monotonic cumulative nanosecond axis."""
    raw = np.asarray(raw_times_ps, dtype=float)
    if raw.size == 0:
        return np.array([], dtype=float)

    if raw.size > 1:
        diffs = np.diff(raw)
        positive_diffs = diffs[diffs > 0]
        frame_interval_ps = float(positive_diffs[0]) if positive_diffs.size else DEFAULT_FRAME_INTERVAL_PS
    else:
        frame_interval_ps = DEFAULT_FRAME_INTERVAL_PS

    repaired = []
    prev_last = None
    for time_ps in raw:
        adjusted = float(time_ps)
        if prev_last is not None and adjusted <= prev_last:
            adjusted += (prev_last + frame_interval_ps) - adjusted
        repaired.append(adjusted)
        prev_last = adjusted

    return np.asarray(repaired, dtype=float) / 1000.0


def sampled_cumulative_time_ns(universe, stride):
    """Return cumulative sampled times for a chained trajectory using frame order."""
    raw_times_ps = [ts.time for ts in universe.trajectory[::stride]]
    return cumulative_time_ns_from_raw_times(raw_times_ps)


CASE_META = {
    "A_ERDRP_WT": {
        "drug": "ERDRP-0519",
        "receptor": "WT",
        "label": "A: ERDRP-0519 / WT (Control)",
        "color": "#0072B2",
        "linestyle": "-",
        "marker": "o",
    },
    "B_ERDRP_MUT": {
        "drug": "ERDRP-0519",
        "receptor": "W730A",
        "label": "B: ERDRP-0519 / W730A (Failure)",
        "color": "#E69F00",
        "linestyle": "--",
        "marker": "s",
    },
    "C_BMS_WT": {
        "drug": "BMS-986205",
        "receptor": "WT",
        "label": "C: BMS-986205 / WT (Success)",
        "color": "#009E73",
        "linestyle": "-.",
        "marker": "^",
    },
    "D_BMS_MUT": {
        "drug": "BMS-986205",
        "receptor": "W730A",
        "label": "D: BMS-986205 / W730A (Resistance)",
        "color": "#CC79A7",
        "linestyle": ":",
        "marker": "D",
    },
}

from .plotting import MARKER_EVERY, PUB_DPI, apply_pub_style, save_pub_figure  # noqa: E402
