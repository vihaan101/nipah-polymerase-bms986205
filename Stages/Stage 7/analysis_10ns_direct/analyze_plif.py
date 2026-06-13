#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 — Protein-Ligand Interaction Fingerprints (10ns_direct production)
Quantifies ALL interaction types between drug and protein: H-bonds,
hydrophobic contacts, pi-stacking, pi-cation, salt bridges, VdW.

Primary method: ProLIF library (pip install prolif).
Fallback: Manual MDAnalysis distance-based detection if ProLIF unavailable.

D_BMS_MUT anomaly filtering: YES — ligand-protein interactions are invalid
during PBC wrapping artifact frames.
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from stage7_eval_common import (
    get_replicate_trajectory,
    CASE_META, load_stage6_cases, resolve_stage6_path,
    require_file, PROJECT_ROOT,
    apply_pub_style, save_pub_figure,
    RMSD_50NS_DIR,
    robust_universe,
)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    STATUS_NOT_ASSESSED,
    criterion_result,
    make_metric_summary,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "plif_10ns_direct")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "_checkpoints")

STRIDE = 10
CHECKPOINT_EVERY_FRAMES = max(
    1, int(os.environ.get("STAGE7_PLIF_CHECKPOINT_EVERY_FRAMES", "20")))
LIVE_SYNC_ROOT = os.environ.get("STAGE7_AZURE_SYNC_DIR", "").strip()
PROLIF_N_JOBS = max(1, int(os.environ.get("STAGE7_PLIF_N_JOBS", os.environ.get("PROLIF_N_JOBS", "10"))))
PROLIF_PARALLEL_STRATEGY = os.environ.get("STAGE7_PLIF_PARALLEL_STRATEGY", "queue").strip() or "queue"

# D_BMS_MUT anomaly filtering (Azure-aware via stage7_eval_common)
RMSD_DIR = str(RMSD_50NS_DIR)
RMSD_THRESHOLD = 10.0  # Angstrom

# Attempt ProLIF import
PROLIF_AVAILABLE = False
PROLIF_VERSION = "N/A"
try:
    import prolif as plf
    PROLIF_AVAILABLE = True
    PROLIF_VERSION = getattr(plf, "__version__", "unknown")
except ImportError:
    pass


def _case_output_paths(case_id):
    return {
        "freq_csv": os.path.join(OUT_DIR, f"{case_id}_plif_frequencies.csv"),
        "ts_csv": os.path.join(OUT_DIR, f"{case_id}_plif_timeseries.csv"),
        "summary_json": os.path.join(OUT_DIR, f"{case_id}_plif_summary.json"),
    }


def _case_checkpoint_paths(case_id):
    case_dir = os.path.join(CHECKPOINT_DIR, case_id)
    return {
        "case_dir": case_dir,
        "state_json": os.path.join(case_dir, "state.json"),
        "chunks_dir": os.path.join(case_dir, "chunks"),
        "manual_state_json": os.path.join(case_dir, "manual_state.json"),
    }


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def _copy_tree_no_metadata(src, dest):
    if not os.path.exists(src):
        return
    for root, _, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        dest_root = dest if rel_root == "." else os.path.join(dest, rel_root)
        os.makedirs(dest_root, exist_ok=True)
        for name in files:
            shutil.copyfile(os.path.join(root, name), os.path.join(dest_root, name))


def _sync_live_outputs(reason):
    if not LIVE_SYNC_ROOT:
        return
    dest = os.path.join(LIVE_SYNC_ROOT, os.path.basename(OUT_DIR))
    try:
        os.makedirs(dest, exist_ok=True)
        _copy_tree_no_metadata(OUT_DIR, dest)
        print(f"  [CHECKPOINT SYNC] {reason} -> {dest}")
    except Exception as exc:
        print(f"  [CHECKPOINT SYNC WARNING] {reason}: {exc}")


def _load_completed_case_result(case_id):
    paths = _case_output_paths(case_id)
    if not (os.path.exists(paths["freq_csv"]) and os.path.exists(paths["summary_json"])):
        return None

    summary_payload = None
    try:
        with open(paths["summary_json"]) as fh:
            summary_payload = json.load(fh)
    except Exception:
        return None

    freq_data = []
    with open(paths["freq_csv"]) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("residue,") or not line.strip():
                continue
            residue, interaction_type, frequency_pct = line.strip().split(",")
            freq_data.append({
                "residue": residue,
                "interaction_type": interaction_type,
                "frequency_pct": float(frequency_pct),
            })

    primary = summary_payload.get("primary_output", {})
    return {
        "label": CASE_META[case_id]["label"],
        "freq_data": freq_data,
        "method": primary.get("method", "unknown"),
        "n_frames": int(primary.get("n_frames", 0)),
        "summary_path": paths["summary_json"],
    }


def _load_prolif_chunk(path):
    df = pd.read_pickle(path)
    if isinstance(df, pd.DataFrame):
        return df
    raise ValueError(f"Unexpected chunk payload in {path}")


def _compute_freq_data_from_raw_df(raw_df):
    n_frames = len(raw_df)
    freq_data = []
    if n_frames == 0:
        return freq_data, 0

    for col in raw_df.columns:
        if isinstance(col, tuple) and len(col) == 3:
            _, prot_res, interaction = col
            occ = raw_df[col].sum() / n_frames * 100.0
            if occ > 0:
                freq_data.append({
                    "residue": str(prot_res),
                    "interaction_type": interaction,
                    "frequency_pct": occ,
                })
    return freq_data, n_frames


def _get_artifact_frame_indices(case_id, stride):
    """Get trajectory frame indices that correspond to PBC artifacts for D_BMS_MUT.

    Returns set of frame indices (in the strided trajectory) to skip.
    """
    if case_id != "D_BMS_MUT":
        return set()

    rmsd_csv = os.path.join(RMSD_DIR, "D_BMS_MUT_ligand_rmsd.csv")
    if not os.path.exists(rmsd_csv):
        print("  WARNING: D_BMS_MUT RMSD CSV not found, no artifact filtering")
        return set()

    df = pd.read_csv(rmsd_csv)
    # RMSD CSV uses stride=10. Our stride is also 10, so row indices match directly.
    # If strides differ, we'd need: artifact_frame = rmsd_row * RMSD_STRIDE / stride
    rmsd_stride = 10
    rmsd_col = "ligand_rmsd_A_mean" if "ligand_rmsd_A_mean" in df.columns else "ligand_rmsd_A"
    artifact_rows = df.index[df[rmsd_col] > RMSD_THRESHOLD].tolist()

    artifact_frames = set()
    for rmsd_row in artifact_rows:
        frame_idx = int(rmsd_row * rmsd_stride / stride)
        artifact_frames.add(frame_idx)

    if artifact_frames:
        print(f"  D_BMS_MUT: will skip {len(artifact_frames)} artifact frames")

    return artifact_frames


def _load_reference_fingerprints():
    path = os.environ.get("STAGE7_PLIF_REFERENCE_JSON", "").strip()
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"  WARNING: PLIF reference file not found: {path}")
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _compute_plif_prolif(u, case_id, meta, stride, artifact_frames):
    """Compute PLIF using ProLIF library."""
    print("  Using ProLIF method")
    print(f"  ProLIF parallel config: n_jobs={PROLIF_N_JOBS}, strategy={PROLIF_PARALLEL_STRATEGY}, checkpoint_every={CHECKPOINT_EVERY_FRAMES}")

    ckpt_paths = _case_checkpoint_paths(case_id)
    os.makedirs(ckpt_paths["chunks_dir"], exist_ok=True)

    # Create frame list excluding artifacts.
    all_frames = list(range(0, len(u.trajectory), stride))
    clean_frames = [f for i, f in enumerate(all_frames) if i not in artifact_frames]

    # ProLIF 2.x API: pass MDAnalysis AtomGroups directly to fp.run()
    # (plf.Molecule.from_mda() pre-building is no longer needed and breaks 2.x)
    protein_ag = u.select_atoms("protein")
    ligand_ag = u.select_atoms("resname UNK")

    next_clean_index = 0
    if os.path.exists(ckpt_paths["state_json"]):
        with open(ckpt_paths["state_json"]) as fh:
            state = json.load(fh)
        if state.get("mode") == "prolif" and int(state.get("stride", stride)) == stride:
            next_clean_index = int(state.get("next_clean_index", 0))
            if next_clean_index:
                print(f"  Resuming ProLIF from clean frame offset {next_clean_index}/{len(clean_frames)}")

    for clean_index in range(next_clean_index, len(clean_frames), CHECKPOINT_EVERY_FRAMES):
        chunk_frames = clean_frames[clean_index:clean_index + CHECKPOINT_EVERY_FRAMES]
        chunk_path = os.path.join(
            ckpt_paths["chunks_dir"],
            f"chunk_{clean_index:06d}_{clean_index + len(chunk_frames) - 1:06d}.pkl",
        )
        if os.path.exists(chunk_path):
            continue

        fp = plf.Fingerprint()
        try:
            fp.run(
                u.trajectory[chunk_frames],
                ligand_ag,
                protein_ag,
                n_jobs=PROLIF_N_JOBS,
            )
        except Exception as e:
            print(f"  ERROR: ProLIF fingerprint failed: {e}")
            print("  Falling back to manual method")
            return None

        df_chunk = fp.to_dataframe()
        df_chunk.to_pickle(chunk_path)
        completed = clean_index + len(chunk_frames)
        _write_json_atomic(
            ckpt_paths["state_json"],
            {
                "case_id": case_id,
                "mode": "prolif",
                "stride": stride,
                "checkpoint_every_frames": CHECKPOINT_EVERY_FRAMES,
                "next_clean_index": completed,
                "clean_frames_total": len(clean_frames),
                "artifact_frame_count": len(artifact_frames),
            },
        )
        print(f"    Checkpointed ProLIF chunk ending at {completed}/{len(clean_frames)} clean frames")
        _sync_live_outputs(f"{case_id} prolif chunk {completed}/{len(clean_frames)}")

    chunk_paths = sorted(
        os.path.join(ckpt_paths["chunks_dir"], name)
        for name in os.listdir(ckpt_paths["chunks_dir"])
        if name.endswith(".pkl")
    )
    raw_df = pd.concat([_load_prolif_chunk(path) for path in chunk_paths], axis=0) if chunk_paths else pd.DataFrame()
    freq_data, n_frames = _compute_freq_data_from_raw_df(raw_df)
    print(f"  ProLIF processed {n_frames} frames")
    return freq_data, raw_df, n_frames


def _compute_plif_manual(u, case_id, meta, stride, artifact_frames):
    """Manual fallback PLIF using MDAnalysis distance calculations."""
    print("  Using MANUAL fallback method (ProLIF unavailable)")

    ckpt_paths = _case_checkpoint_paths(case_id)
    os.makedirs(ckpt_paths["case_dir"], exist_ok=True)

    protein = u.select_atoms("protein")
    ligand = u.select_atoms("resname UNK")

    if len(ligand) == 0:
        print("  WARNING: no ligand found")
        return [], None, 0

    # Get protein residues near ligand at frame 0
    u.trajectory[0]
    nearby = u.select_atoms(
        "protein and (around 6.0 (resname UNK and not name H*))")
    nearby_resids = sorted(set(nearby.residues.resids))

    # Per-residue interaction tracking
    interactions = defaultdict(lambda: defaultdict(int))
    n_frames = 0
    next_clean_index = 0
    all_frames = list(range(0, len(u.trajectory), stride))
    clean_entries = [(i, frame_no) for i, frame_no in enumerate(all_frames) if i not in artifact_frames]

    if os.path.exists(ckpt_paths["manual_state_json"]):
        with open(ckpt_paths["manual_state_json"]) as fh:
            state = json.load(fh)
        if state.get("mode") == "manual" and int(state.get("stride", stride)) == stride:
            next_clean_index = int(state.get("next_clean_index", 0))
            n_frames = int(state.get("n_frames", 0))
            for residue, interaction_counts in state.get("interactions", {}).items():
                for interaction_type, count in interaction_counts.items():
                    interactions[residue][interaction_type] = int(count)
            if next_clean_index:
                print(f"  Resuming manual PLIF from clean frame offset {next_clean_index}/{len(clean_entries)}")

    for clean_index, (_, frame_no) in enumerate(clean_entries[next_clean_index:], start=next_clean_index):
        ts = u.trajectory[frame_no]
        n_frames += 1

        lig_pos = ligand.positions

        for resid in nearby_resids:
            res_atoms = u.select_atoms(f"resid {resid} and protein")
            res_pos = res_atoms.positions

            if len(res_pos) == 0:
                continue

            # Current frame box for PBC-aware distance computation.
            # Without this, a ligand in a different periodic image would appear
            # ~box_length away from every residue → zero contacts for that frame.
            box = ts.dimensions

            # Distance matrix between residue and ligand
            dists = distance_array(res_pos, lig_pos, box=box)
            min_dist = dists.min()

            resname = res_atoms[0].resname
            res_label = f"{resname}{resid}"

            # Hydrophobic: C-C contacts < 4.0 A
            res_c = u.select_atoms(f"resid {resid} and protein and name C*")
            lig_c = u.select_atoms("resname UNK and name C*")
            if len(res_c) > 0 and len(lig_c) > 0:
                cc_dists = distance_array(res_c.positions, lig_c.positions, box=box)
                if cc_dists.min() < 4.0:
                    interactions[res_label]["Hydrophobic"] += 1

            # VdW contact: any heavy atom < 4.5 A
            res_heavy = u.select_atoms(f"resid {resid} and protein and not name H*")
            lig_heavy = u.select_atoms("resname UNK and not name H*")
            if len(res_heavy) > 0 and len(lig_heavy) > 0:
                heavy_dists = distance_array(res_heavy.positions, lig_heavy.positions, box=box)
                if heavy_dists.min() < 4.5:
                    interactions[res_label]["VdWContact"] += 1

            # H-bond proxy: N/O within 3.5 A of ligand N/O
            res_donor = u.select_atoms(
                f"resid {resid} and protein and (name N* or name O*)")
            lig_acceptor = u.select_atoms("resname UNK and (name N* or name O*)")
            if len(res_donor) > 0 and len(lig_acceptor) > 0:
                hb_dists = distance_array(res_donor.positions, lig_acceptor.positions, box=box)
                if hb_dists.min() < 3.5:
                    interactions[res_label]["HBond"] += 1

        completed = clean_index + 1
        if completed % CHECKPOINT_EVERY_FRAMES == 0 or completed == len(clean_entries):
            _write_json_atomic(
                ckpt_paths["manual_state_json"],
                {
                    "case_id": case_id,
                    "mode": "manual",
                    "stride": stride,
                    "checkpoint_every_frames": CHECKPOINT_EVERY_FRAMES,
                    "next_clean_index": completed,
                    "clean_frames_total": len(clean_entries),
                    "n_frames": n_frames,
                    "interactions": {
                        residue: dict(interaction_counts)
                        for residue, interaction_counts in interactions.items()
                    },
                },
            )
            print(f"    Checkpointed manual PLIF at {completed}/{len(clean_entries)} clean frames")
            _sync_live_outputs(f"{case_id} manual chunk {completed}/{len(clean_entries)}")

    # Convert to frequency data
    freq_data = []
    for res_label, itypes in interactions.items():
        for itype, count in itypes.items():
            occ = count / n_frames * 100.0 if n_frames > 0 else 0
            if occ > 0:
                freq_data.append({
                    "residue": res_label,
                    "interaction_type": itype,
                    "frequency_pct": occ,
                })

    return freq_data, None, n_frames


def compute_plif(case_id, meta, stride, stage6_cases, reference_data):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))
    
    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_freq_dicts = []
    total_frames = 0
    method_used = "manual"

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        n_frames = len(u.trajectory)

        ligand = u.select_atoms("resname UNK")
        if len(ligand) == 0:
            continue

        artifact_frames = _get_artifact_frame_indices(case_id, stride)

        if PROLIF_AVAILABLE:
            result = _compute_plif_prolif(u, case_id, meta, stride, artifact_frames)
            if result is not None:
                freq_data, raw_df, n_analyzed = result
                method_used = f"prolif-{PROLIF_VERSION}"
            else:
                freq_data, raw_df, n_analyzed = _compute_plif_manual(u, case_id, meta, stride, artifact_frames)
        else:
            freq_data, raw_df, n_analyzed = _compute_plif_manual(u, case_id, meta, stride, artifact_frames)

        freq_dict = {(d["residue"], d["interaction_type"]): d["frequency_pct"] for d in freq_data}
        all_freq_dicts.append(freq_dict)
        total_frames += n_analyzed

    if not all_freq_dicts:
        return None

    from collections import defaultdict
    combined_freqs = defaultdict(list)
    for rep_dict in all_freq_dicts:
        for key, val in rep_dict.items():
            combined_freqs[key].append(val)
            
    num_reps = len(all_freq_dicts)
    for key in combined_freqs:
        while len(combined_freqs[key]) < num_reps:
            combined_freqs[key].append(0.0)

    final_freq_data = []
    for key, vals in combined_freqs.items():
        avg_freq = sum(vals) / num_reps
        if avg_freq > 0:
            final_freq_data.append({
                "residue": key[0],
                "interaction_type": key[1],
                "frequency_pct": avg_freq,
            })

    csv_path = os.path.join(OUT_DIR, f"{case_id}_plif_frequencies.csv")
    with open(csv_path, "w") as f:
        f.write(f"# method={method_used}\n")
        f.write("residue,interaction_type,frequency_pct\n")
        for entry in sorted(final_freq_data, key=lambda x: -x["frequency_pct"]):
            f.write(f"{entry['residue']},{entry['interaction_type']},{entry['frequency_pct']:.2f}\n")

    summary_payload = make_metric_summary(
        "plif",
        case_id,
        {
            "method": method_used,
            "n_frames": total_frames,
            "n_interaction_entries": len(final_freq_data),
        },
        derived_metrics={"top_interactions": sorted(final_freq_data, key=lambda row: -row["frequency_pct"])[:15]},
        interpretation=[],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_plif_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "freq_data": final_freq_data,
        "method": method_used,
        "n_frames": total_frames,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(
        description="Protein-Ligand Interaction Fingerprints — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None,
                        help="Run only this case ID (e.g. 'C_BMS_WT')")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Frame stride (default: {STRIDE})")
    args = parser.parse_args()
    if args.case is None:
        args.case = os.environ.get("STAGE7_TARGET_CASE") or None

    stride = args.stride
    cases_to_run = CASE_META
    if args.case:
        if args.case not in CASE_META:
            print(f"FATAL: Unknown case '{args.case}'. Valid: {list(CASE_META.keys())}")
            sys.exit(1)
        cases_to_run = {args.case: CASE_META[args.case]}

    os.makedirs(OUT_DIR, exist_ok=True)
    apply_pub_style()
    stage6_cases = load_stage6_cases()

    if PROLIF_AVAILABLE:
        print(f"ProLIF {PROLIF_VERSION} detected")
    else:
        print("ProLIF NOT available, will use manual fallback")
    reference_data = _load_reference_fingerprints()

    results = {}
    skipped = []
    # Run sequentially (ProLIF may not be thread-safe)
    for case_id, meta in cases_to_run.items():
        completed = _load_completed_case_result(case_id)
        if completed is not None:
            print(f"  Reusing completed PLIF outputs for {case_id}")
            results[case_id] = completed
            continue
        try:
            data = compute_plif(case_id, meta, stride, stage6_cases, reference_data)
            if data is not None:
                results[case_id] = data
        except Exception as e:
            print(f"  ERROR: {case_id} skipped: {e}")
            skipped.append((case_id, str(e)))

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Heatmap: 2x2 grid (residues x interaction types) ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes_flat = axes.flatten()

    for idx, case_id in enumerate(CASE_META):
        ax = axes_flat[idx]
        if case_id not in results:
            ax.text(0.5, 0.5, f"{case_id}\n(no data)", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(CASE_META[case_id]["label"], fontsize=7)
            continue

        freq_data = results[case_id]["freq_data"]
        if not freq_data:
            ax.text(0.5, 0.5, "No interactions", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(CASE_META[case_id]["label"], fontsize=7)
            continue

        df = pd.DataFrame(freq_data)
        pivot = df.pivot_table(
            index="residue", columns="interaction_type",
            values="frequency_pct", fill_value=0)

        # Sort by total frequency (descending), show top 15
        pivot["_total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("_total", ascending=False).head(15)
        pivot = pivot.drop(columns=["_total"])

        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                           vmin=0, vmax=100)
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=5)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index, fontsize=5)
        else:
            ax.text(0.5, 0.5, "Empty pivot", ha="center",
                    va="center", transform=ax.transAxes)

        ax.set_title(CASE_META[case_id]["label"], fontsize=7)

    fig.suptitle("Protein-Ligand Interaction Fingerprints", fontsize=8,
                 fontweight="bold")
    fig.tight_layout()
    hm_stem = os.path.join(OUT_DIR, "plif_heatmap")
    save_pub_figure(fig, hm_stem)
    print(f"\nHeatmap saved: {hm_stem}.png / .pdf")

    # ── Resid 591 comparison bar chart (paper-specific hook only) ───────────
    fig, ax = plt.subplots(figsize=(5, 3.5))

    resid591_data = {}
    for case_id in CASE_META:
        if case_id not in results:
            continue
        freq_data = results[case_id]["freq_data"]
        r591 = [e for e in freq_data
                if "591" in str(e["residue"])]
        if r591:
            resid591_data[case_id] = {e["interaction_type"]: e["frequency_pct"]
                                      for e in r591}

    if resid591_data:
        all_itypes = sorted(set(
            itype for d in resid591_data.values() for itype in d))
        n_types = len(all_itypes)
        n_cases = len(resid591_data)
        case_ids_591 = list(resid591_data.keys())

        x = np.arange(n_types)
        width = 0.8 / max(n_cases, 1)

        for i, case_id in enumerate(case_ids_591):
            vals = [resid591_data[case_id].get(it, 0) for it in all_itypes]
            ax.bar(x + i * width, vals, width,
                   color=CASE_META[case_id]["color"],
                   label=CASE_META[case_id]["label"], edgecolor="black",
                   linewidth=0.3)

        ax.set_xticks(x + width * (n_cases - 1) / 2)
        ax.set_xticklabels(all_itypes, rotation=30, ha="right", fontsize=6)
        ax.set_ylabel("Frequency (%)")
        ax.set_title("Resid 591 (paper-specific comparison)", fontsize=8)
        ax.legend(fontsize=5, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    else:
        ax.text(0.5, 0.5, "No resid 591 interactions detected",
                ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    r591_stem = os.path.join(OUT_DIR, "plif_resid591_comparison")
    save_pub_figure(fig, r591_stem)
    print(f"Resid 591 chart saved: {r591_stem}.png / .pdf")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Case':<18} {'Method':<20} {'N_frames':>8} {'Interactions':>12}")
    print(f"{'-'*70}")
    for case_id in CASE_META:
        if case_id in results:
            r = results[case_id]
            print(f"{case_id:<18} {r['method']:<20} {r['n_frames']:>7}  "
                  f"{len(r['freq_data']):>11}")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "plif_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "plif",
            "case_summaries": {
                case_id: results[case_id]["summary_path"] for case_id in results
            },
        },
    )
    print(f"Metric index saved: {overview_path}")

    if skipped:
        print(f"\n{len(skipped)} case(s) skipped:")
        for cid, reason in skipped:
            print(f"  {cid}: {reason}")
        sys.exit(1)


if __name__ == "__main__":
    main()