"""
Stage 7 — Shared evaluation utilities for all post-MD analysis scripts.
Provides manifest-driven path resolution, file validation, and case metadata.
"""

import os
from pathlib import Path
import json

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AZURE_ONLY = os.environ.get("STAGE7_AZURE_ONLY", "").lower() in {"1", "true", "yes", "on"}
AZURE_MOUNT = Path(os.environ.get("STAGE7_AZURE_MOUNT", "/mnt/biomni"))
EXCLUDE_50NS_DCD = os.environ.get("STAGE7_EXCLUDE_50NS_DCD", "").lower() in {"1", "true", "yes", "on"}

# ── Path resolution for Stage 6 artifacts ───────────────────────────
# Azure Share verified path: /mnt/biomni/stage6/
STAGE6_RESULTS = Path(
    os.environ.get(
        "STAGE7_STAGE6_RESULTS_DIR",
        str(AZURE_MOUNT / "stage6"),
    )
)
STAGE6_MANIFEST = Path(
    os.environ.get(
        "STAGE7_STAGE6_MANIFEST",
        str(STAGE6_RESULTS / "stage6_manifest.json"),
    )
)

STAGE7_LOCAL_20NS = PROJECT_ROOT / "stage7_results_20ns"
STAGE7_CANONICAL_20NS = PROJECT_ROOT / "Stages" / "Stage 7" / "results_20ns"

# Azure Share verified path: /mnt/biomni/stage7_10ns_direct/results/
STAGE7_DIRECT_RESULTS_AZURE = AZURE_MOUNT / "stage7_10ns_direct" / "results"
STAGE7_DIRECT_RESULTS_LOCAL = PROJECT_ROOT / "stage7_10ns_direct" / "results"

TRAJECTORY_TIERS = [
    ("10ns_rep1", STAGE7_DIRECT_RESULTS_AZURE),
    ("10ns_rep2", STAGE7_DIRECT_RESULTS_AZURE),
    ("10ns_rep3", STAGE7_DIRECT_RESULTS_AZURE),
    ("10ns_rep4", STAGE7_DIRECT_RESULTS_AZURE),
    ("10ns_rep5", STAGE7_DIRECT_RESULTS_AZURE),
]

# ── Baseline eval results directories (Azure-aware) ──────────────────
_DEFAULT_EVAL_DIR = AZURE_MOUNT / "stage7_evals_10ns_direct" if AZURE_ONLY else PROJECT_ROOT / "stage7_results_eval_10ns_direct"
EVAL_10NS_DIR = Path(os.environ.get("STAGE7_EVAL_10NS_DIR", str(_DEFAULT_EVAL_DIR)))

# 10ns_direct output subdirs
RMSD_10NS_DIR = EVAL_10NS_DIR / "rmsd_10ns_direct"
MMGBSA_10NS_DIR = EVAL_10NS_DIR / "mmgbsa_10ns_direct"

# Backwards compatibility for scripts using 50ns-named variables
RMSD_50NS_DIR = RMSD_10NS_DIR
MMGBSA_50NS_DIR = MMGBSA_10NS_DIR

DEFAULT_FRAME_INTERVAL_PS = 4.0


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
    delta = minimum_image_displacements(np.asarray(point_a, dtype=float) - np.asarray(point_b, dtype=float), box_lengths)
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
        else:
            raise ValueError(f"Target case '{target_case}' not found in manifest")

    return cases


def resolve_stage6_path(manifest_value):
    """Resolve an Azure-absolute manifest path to the local filesystem."""
    p = Path(manifest_value)
    
    if AZURE_ONLY:
        # Match verified Azure structure: /mnt/biomni/stage6/<case_id>/<filename>
        # (Manifest was written on Stage 6 VM with local paths)
        if "results" in p.parts:
            idx = max(i for i, part in enumerate(p.parts) if part == "results")
            tail = Path(*p.parts[idx+1:])
            # Share has stage6/<case_id>/<filename>
            return AZURE_MOUNT / "stage6" / tail
        # Fallback
        return AZURE_MOUNT / "stage6" / p.name

    if p.exists():
        return p
    
    if "results" in p.parts:
        idx = max(i for i, part in enumerate(p.parts) if part == "results")
        tail = Path(*p.parts[idx + 1:])
        candidate = STAGE6_RESULTS / tail
        if candidate.exists():
            return candidate
    return p


def resolve_stage6_case_path(case_id, filename):
    """Resolve a conventional Stage 6 per-case artifact location."""
    if AZURE_ONLY:
        # Match verified Azure structure: /mnt/biomni/stage6/results/<case_id>/<filename>
        # Fallback to /mnt/biomni/stage6/<case_id>/<filename>
        candidate = AZURE_MOUNT / "stage6" / "results" / case_id / filename
        if candidate.exists():
            return candidate
        return require_file(AZURE_MOUNT / "stage6" / case_id / filename, f"{filename} for {case_id}")
    candidate = STAGE6_RESULTS / "results" / case_id / filename
    if not candidate.exists():
         candidate = STAGE6_RESULTS / case_id / filename
    return require_file(candidate, f"{filename} for {case_id}")


def find_trajectory(case_id, filename="production.dcd"):
    """Find trajectory in the 10ns direct results (prefers replicate 1)."""
    roots = [STAGE7_DIRECT_RESULTS_AZURE] if AZURE_ONLY else [STAGE7_DIRECT_RESULTS_LOCAL, STAGE7_DIRECT_RESULTS_AZURE]
    for root in roots:
        # For 10ns direct, trajectories are in results/<case>/replicate_<N>/
        candidate = root / case_id / "replicate_1" / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Stage 7 eval contract violation: {filename} for {case_id} not found "
        f"under {roots} (replicate_1)"
    )


def get_replicate_trajectory(case_id, replicate_id, filename="production.dcd"):
    """Get the trajectory for a specific 10ns direct replicate."""
    roots = [STAGE7_DIRECT_RESULTS_AZURE] if AZURE_ONLY else [STAGE7_DIRECT_RESULTS_LOCAL, STAGE7_DIRECT_RESULTS_AZURE]
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
    chain = []
    root = STAGE7_DIRECT_RESULTS_AZURE if AZURE_ONLY else (STAGE7_DIRECT_RESULTS_LOCAL if STAGE7_DIRECT_RESULTS_LOCAL.exists() else STAGE7_DIRECT_RESULTS_AZURE)
    
    for rep_id in range(1, 6):
        candidate = root / case_id / f"replicate_{rep_id}" / filename
        if not candidate.exists():
             raise FileNotFoundError(
                f"Trajectory chain broken: {filename} for {case_id} rep{rep_id} not found "
                f"at {candidate}. Verify data location."
            )
        chain.append(str(candidate))
    return chain


def robust_universe(topology, trajectory=None, attempts=5, delay=2.0, **kwargs):
    """Initialize an MDAnalysis Universe with retries to handle transient I/O errors."""
    import MDAnalysis as mda
    import time
    last_err = None
    for i in range(attempts):
        try:
            if trajectory:
                return mda.Universe(topology, trajectory, **kwargs)
            return mda.Universe(topology, **kwargs)
        except Exception as e:
            last_err = e
            # Log the failure and wait before retrying
            print(f"  WARNING: Universe initialization failed (attempt {i+1}/{attempts}): {e}")
            if "Reading DCD header failed" in str(e) or "StopIteration" in str(e):
                time.sleep(delay * (i + 1))
            else:
                # If it's a different error (e.g. file truly missing), don't retry as long
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
    """Return cumulative sampled times for a chained trajectory using frame order, not raw ChainReader clock."""
    raw_times_ps = [ts.time for ts in universe.trajectory[::stride]]
    return cumulative_time_ns_from_raw_times(raw_times_ps)


CASE_META = {
    "A_ERDRP_WT":  {"drug": "ERDRP-0519", "receptor": "WT",    "label": "A: ERDRP-0519 / WT (Control)",
                     "color": "#0072B2", "linestyle": "-",  "marker": "o"},
    "B_ERDRP_MUT": {"drug": "ERDRP-0519", "receptor": "W730A", "label": "B: ERDRP-0519 / W730A (Failure)",
                     "color": "#E69F00", "linestyle": "--", "marker": "s"},
    "C_BMS_WT":    {"drug": "BMS-986205",  "receptor": "WT",    "label": "C: BMS-986205 / WT (Success)",
                     "color": "#009E73", "linestyle": "-.", "marker": "^"},
    "D_BMS_MUT":   {"drug": "BMS-986205",  "receptor": "W730A", "label": "D: BMS-986205 / W730A (Resistance)",
                     "color": "#CC79A7", "linestyle": ":",  "marker": "D"},
}


# ── Publication-quality plotting helpers ─────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PUB_DPI = 300
MARKER_EVERY = 20


def apply_pub_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": PUB_DPI,
        "savefig.dpi": PUB_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pub_figure(fig, path_stem):
    """Save figure as PNG (300 DPI) + PDF (vector), then close.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path_stem : str
        Output path WITHOUT extension.  Produces path_stem.png and path_stem.pdf.
    """
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem + ext, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)
