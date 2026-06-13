#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\nStage 7 — Binding pocket volume over time (10ns_direct production)\nTracks the ConvexHull volume of binding pocket heavy atoms per frame.\nWhen TRP (204 Da) mutates to ALA (89 Da), the pocket should expand,\ngiving the drug more room → higher conformational entropy penalty.\n\nD_BMS_MUT anomaly filtering: NO — pocket volume measures protein-only\natoms, unaffected by ligand PBC wrapping.\n"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

from scipy.spatial import ConvexHull, QhullError

from stage7_eval_common import (robust_universe,
    CASE_META, load_stage6_cases, resolve_stage6_path,
    get_replicate_trajectory, require_file,
    apply_pub_style, save_pub_figure, MARKER_EVERY,
)
from stage7_success_criteria import (
    STATUS_CONCERN,
    STATUS_DESCRIPTIVE,
    STATUS_NOT_ASSESSED,
    STATUS_SUPPORTS,
    criterion_result,
    late_window_metrics,
    make_metric_summary,
    series_summary,
    split_window_drift,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "pocket_volume_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h per case; ConvexHull over full 10ns_direct trajectory


def compute_pocket_volume(case_id, meta, stride, stage6_cases):
    """Compute binding pocket ConvexHull volume per frame, averaged across 5 replicates."""
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))
    require_file(topology, f"topology.pdb for {case_id}")

    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    # Detect pocket residues from replicate 1, frame 0
    try:
        rep1_path = str(get_replicate_trajectory(case_id, 1))
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None
    u0 = robust_universe(topology, rep1_path)
    u0.trajectory[0]
    ligand = u0.select_atoms("resname UNK")
    if len(ligand) == 0:
        print(f"  WARNING: no ligand found, skipping.")
        return None
    pocket_atoms = u0.select_atoms("protein and (around 5.0 (resname UNK and not name H*))")
    pocket_resids = sorted(set(pocket_atoms.residues.resids))
    if len(pocket_resids) == 0:
        print(f"  WARNING: no pocket residues found, skipping.")
        return None
    print(f"  Pocket residues detected: {len(pocket_resids)}")
    resid_sel = " or ".join(f"resid {r}" for r in pocket_resids)

    # Per-replicate volume computation
    all_rep_volumes = []
    ref_times_ns = None
    n_qhull_errors_total = 0

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING rep{rep_id}: {e}")
            continue

        u = robust_universe(topology, traj_path)
        pocket_heavy = u.select_atoms(f"protein and ({resid_sel}) and not name H*")
        if len(pocket_heavy) < 4:
            print(f"  WARNING rep{rep_id}: <4 pocket heavy atoms, skipping rep.")
            continue

        n_frames = len(u.trajectory)
        print(f"  rep{rep_id}: {n_frames} raw frames | stride={stride} | analysed={n_frames // stride}")

        volumes = []
        times_ps = []
        n_qhull_errors = 0
        for frame_i, ts in enumerate(u.trajectory[::stride]):
            try:
                hull = ConvexHull(pocket_heavy.positions)
                volumes.append(hull.volume)
            except QhullError:
                volumes.append(float("nan"))
                n_qhull_errors += 1
            times_ps.append(ts.time)
            if (frame_i + 1) % 50 == 0:
                print(f"    rep{rep_id} frame {frame_i + 1}/{n_frames // stride}: "
                      f"volume = {volumes[-1]:.1f} A^3")

        n_qhull_errors_total += n_qhull_errors
        if n_qhull_errors:
            print(f"    rep{rep_id}: {n_qhull_errors} QhullError frames (set to NaN)")

        # Time axis: raw ps from a single replicate → ns, no cumulative patching needed
        times_ns_rep = np.array(times_ps) / 1000.0
        all_rep_volumes.append(np.array(volumes))
        if ref_times_ns is None:
            ref_times_ns = times_ns_rep

    if not all_rep_volumes:
        print(f"  ERROR: no replicate data for {case_id}")
        return None

    # Align to minimum length (in case replicates differ by 1-2 frames)
    min_len = min(len(v) for v in all_rep_volumes)
    vols_arr = np.array([v[:min_len] for v in all_rep_volumes])  # (n_reps, n_frames)
    times_ns = ref_times_ns[:min_len]

    mean_vol = np.nanmean(vols_arr, axis=0)
    std_vol = np.nanstd(vols_arr, axis=0)

    # Scalar stats across all frames and reps
    all_valid = vols_arr[~np.isnan(vols_arr)]
    mean_scalar = float(np.nanmean(mean_vol))
    std_scalar = float(np.nanmean(std_vol))
    print(f"  Volume: mean={mean_scalar:.1f} A^3, pooled_std={std_scalar:.1f}")
    if n_qhull_errors_total:
        print(f"  Total QhullError frames: {n_qhull_errors_total}")
    if len(all_valid) > 0 and all_valid.min() <= 0:
        print(f"  WARNING: some volumes <= 0 ({all_valid.min():.1f})")

    # Write CSV: per-timepoint mean ± std across replicates (time 0–10 ns)
    csv_path = os.path.join(OUT_DIR, f"{case_id}_pocket_volume.csv")
    with open(csv_path, "w") as f:
        f.write("time_ns,volume_A3_mean,volume_A3_std\n")
        for t, m, s in zip(times_ns, mean_vol, std_vol):
            f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")
    print(f"  Saved: {csv_path}")

    # Use mean_vol/std_vol as proxy for downstream stats
    volumes = mean_vol  # for compatibility with summary functions below
    n_qhull_errors = n_qhull_errors_total

    valid = volumes[~np.isnan(volumes)]
    split_stats = split_window_drift(valid)
    late_stats = late_window_metrics(times_ns[~np.isnan(volumes)], valid) if len(valid) else {"n_points": 0}
    if len(valid) < 8:
        conv_status = STATUS_NOT_ASSESSED
    else:
        mean_shift = split_stats.get("mean_shift")
        pooled_std = split_stats.get("pooled_std")
        conv_status = (
            STATUS_SUPPORTS
            if mean_shift is not None and pooled_std is not None and abs(mean_shift) <= max(pooled_std, 25.0)
            else STATUS_CONCERN
        )
    interpretation = [
        criterion_result(
            "pocket_volume_convergence",
            conv_status,
            "Pocket volume remains descriptive. Early/late distribution drift is reported as a heuristic convergence proxy, while nonpositive values remain QC-only concerns.",
            extra={
                "split_window": split_stats,
                "late_window": late_stats,
                "n_qhull_errors": n_qhull_errors,
                "positive_volume_qc": bool(len(valid) > 0 and valid.min() > 0),
            },
            threshold={"split_window_mean_shift_vs_pooled_std": "<= 1.0"},
            threshold_note="Heuristic convergence proxy for a descriptive pocket-volume metric.",
        )
    ]
    summary_payload = make_metric_summary(
        "pocket_volume",
        case_id,
        series_summary(valid, times=times_ns[~np.isnan(volumes)] if len(valid) else None),
        derived_metrics={
            "split_window": split_stats,
            "late_window": late_stats,
            "n_qhull_errors": n_qhull_errors,
            "n_nonpositive_values": int(np.sum(valid <= 0)) if len(valid) else 0,
            "pocket_resids": pocket_resids,
        },
        interpretation=interpretation,
        notes=["Positive volume is treated as a QC sanity check, not as a scientific success criterion."],
        limitations=["No reference holo/apo pocket volume distribution is available for calibration."],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_pocket_volume_summary.json")
    write_json(summary_path, summary_payload)
    print(f"  Saved: {summary_path}")

    return {
        "label": meta["label"],
        "times_ns": times_ns,
        "volumes": mean_vol,
        "volumes_std": std_vol,
        "mean": mean_scalar,
        "std": std_scalar,
        "pocket_resids": pocket_resids,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Binding pocket volume — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None,
                        help="Run only this case ID (e.g. 'C_BMS_WT')")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Frame stride (default: {STRIDE})")
    args = parser.parse_args()

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

    # Run each case (ProcessPoolExecutor for parallelism)
    results = {}
    skipped = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(compute_pocket_volume, case_id, meta, stride, stage6_cases): case_id
            for case_id, meta in cases_to_run.items()
        }
        try:
            for future in as_completed(futures, timeout=WORKER_TIMEOUT_S):
                case_id = futures[future]
                try:
                    data = future.result()
                    if data is not None:
                        results[case_id] = data
                except Exception as e:
                    print(f"  ERROR: {case_id} skipped: {e}", flush=True)
                    skipped.append((case_id, str(e)))
        except FutureTimeoutError:
            for future, case_id in futures.items():
                if not future.done():
                    print(f"  TIMEOUT: {case_id} worker exceeded {WORKER_TIMEOUT_S}s — cancelling", flush=True)
                    skipped.append((case_id, f"worker timeout after {WORKER_TIMEOUT_S}s"))
                    future.cancel()

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Time-series overlay plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for case_id in CASE_META:
        if case_id not in results:
            continue
        data = results[case_id]
        meta = CASE_META[case_id]
        mean = data["volumes"]
        std = data["volumes_std"]
        ax.plot(data["times_ns"], mean,
                color=meta["color"], linestyle=meta["linestyle"],
                marker=meta["marker"], markevery=MARKER_EVERY, markersize=3,
                lw=1.2, label=data["label"])
        ax.fill_between(data["times_ns"], mean - std, mean + std,
                        color=meta["color"], alpha=0.15)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Pocket Volume ($\AA^3$)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)

    fig.tight_layout()
    ts_stem = os.path.join(OUT_DIR, "pocket_volume_overlay")
    save_pub_figure(fig, ts_stem)
    print(f"\nOverlay plot saved: {ts_stem}.png / .pdf")

    # ── Bar chart: mean +/- std ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3))

    case_ids = [c for c in CASE_META if c in results]
    means = [results[c]["mean"] for c in case_ids]
    stds = [results[c]["std"] for c in case_ids]
    colors = [CASE_META[c]["color"] for c in case_ids]
    labels = [results[c]["label"] for c in case_ids]

    bars = ax.bar(range(len(case_ids)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"Mean Pocket Volume ($\AA^3$)")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 20, f"{m:.0f}", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    bar_stem = os.path.join(OUT_DIR, "pocket_volume_bar")
    save_pub_figure(fig, bar_stem)
    print(f"Bar chart saved: {bar_stem}.png / .pdf")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Case':<18} {'Mean (A^3)':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    for case_id in CASE_META:
        if case_id in results:
            v = results[case_id]["volumes"]
            valid = v[~np.isnan(v)]
            if len(valid) > 0:
                print(f"{case_id:<18} {valid.mean():>10.1f}  {valid.std():>9.1f}  "
                      f"{valid.min():>9.1f}  {valid.max():>9.1f}")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "pocket_volume_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "pocket_volume",
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