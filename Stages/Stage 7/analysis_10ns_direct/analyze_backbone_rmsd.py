"""\nStage 7 — Protein backbone RMSD overlay (10ns_direct production trajectories)\nBackbone (N CA C O) RMSD vs frame 0 for all 4 cases.\n"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms

from stage7_eval_common import (robust_universe, get_replicate_trajectory, CASE_META, load_stage6_cases, resolve_stage6_path, find_trajectory, get_replicate_trajectory, require_file, apply_pub_style, save_pub_figure, MARKER_EVERY, sampled_cumulative_time_ns)
from stage7_success_criteria import (
    STATUS_CONCERN,
    STATUS_NOT_ASSESSED,
    STATUS_SUPPORTS,
    criterion_result,
    late_window_metrics,
    make_metric_summary,
    rolling_stability,
    series_summary,
    split_window_drift,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "rmsd_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h per case; hangs if GPU/CIFS stalls


def _plateau_interpretation(times_ns, rmsd_A):
    late = late_window_metrics(times_ns, rmsd_A)
    rolling = rolling_stability(rmsd_A)
    drift = split_window_drift(rmsd_A)
    summary = {
        "late_window": late,
        "rolling_window": rolling,
        "split_window": drift,
    }

    if len(rmsd_A) < 8:
        status = STATUS_NOT_ASSESSED
        detail = "Too few RMSD samples for a late-window plateau heuristic."
    else:
        slope = late.get("slope_per_ns")
        mean_shift = drift.get("mean_shift")
        pooled_std = drift.get("pooled_std")
        rolling_range = rolling.get("rolling_mean_range")
        supports = (
            slope is not None
            and abs(slope) <= 0.05
            and mean_shift is not None
            and pooled_std is not None
            and abs(mean_shift) <= max(pooled_std, 0.25)
            and rolling_range is not None
            and rolling_range <= max(1.0, 2.0 * (late.get("std") or 0.0))
        )
        status = STATUS_SUPPORTS if supports else STATUS_CONCERN
        detail = (
            "Heuristic plateau proxy based on late-window slope, rolling-window stability, "
            "and split-window drift. This approximates the literature-backed plateau concept "
            "without claiming a universal RMSD cutoff."
        )

    interpretation = criterion_result(
        "backbone_plateau",
        status,
        detail,
        extra=summary,
        threshold={
            "late_window_slope_abs_A_per_ns": 0.05,
            "split_window_mean_shift_vs_pooled_std": "<= 1.0",
        },
        threshold_note="Heuristic proxy for a qualitative gold-standard plateau criterion; not a universal cutoff.",
    )
    return summary, interpretation



def compute_backbone_rmsd(case_id, meta, stride, stage6_cases):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))

    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_bb_rmsd = []
    all_pocket_rmsd = []
    all_times = []

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        backbone = u.select_atoms("backbone")
        
        pocket_atoms = None
        lig = u.select_atoms("resname UNK")
        if len(lig) > 0:
            pocket_bb = u.select_atoms("backbone and (around 10.0 resname UNK)")
            pocket_resids = sorted(set(pocket_bb.residues.resids))
            if len(pocket_resids) > 0:
                resid_sel = "backbone and resid " + " ".join(str(r) for r in pocket_resids)
                pocket_atoms = u.select_atoms(resid_sel)

        u.trajectory[0]
        bb_ref = backbone.positions.copy()
        pocket_ref = pocket_atoms.positions.copy() if pocket_atoms is not None else None

        times_list = []
        bb_rmsd_list = []
        pocket_rmsd_list = []

        sampled_times_ns = sampled_cumulative_time_ns(u, stride)
        for time_ns, ts in zip(sampled_times_ns, u.trajectory[::stride]):
            times_list.append(time_ns)
            bb_rmsd_list.append(rms.rmsd(backbone.positions, bb_ref, center=True, superposition=True))
            if pocket_ref is not None:
                pocket_rmsd_list.append(rms.rmsd(pocket_atoms.positions, pocket_ref, center=True, superposition=True))

        all_times.append(times_list)
        all_bb_rmsd.append(bb_rmsd_list)
        if pocket_ref is not None:
            all_pocket_rmsd.append(pocket_rmsd_list)

    if not all_bb_rmsd:
        return None

    min_len = min(len(t) for t in all_times)
    times_ns = np.array(all_times[0][:min_len])
    
    bb_rmsd_arr = np.array([r[:min_len] for r in all_bb_rmsd])
    bb_rmsd_mean = bb_rmsd_arr.mean(axis=0)
    bb_rmsd_std = bb_rmsd_arr.std(axis=0)

    pocket_rmsd_mean = None
    pocket_rmsd_std = None
    if all_pocket_rmsd:
        pocket_rmsd_arr = np.array([r[:min_len] for r in all_pocket_rmsd])
        pocket_rmsd_mean = pocket_rmsd_arr.mean(axis=0)
        pocket_rmsd_std = pocket_rmsd_arr.std(axis=0)

    mean_val = bb_rmsd_mean.mean()
    std_val = bb_rmsd_mean.std()
    print(f"  Backbone RMSD — mean: {mean_val:.2f} A  std: {std_val:.2f} A  max: {bb_rmsd_mean.max():.2f} A  final: {bb_rmsd_mean[-1]:.2f} A")

    csv_path = os.path.join(OUT_DIR, f"{case_id}_backbone_rmsd.csv")
    with open(csv_path, "w") as f:
        f.write("time_ns,backbone_rmsd_A_mean,backbone_rmsd_A_std\n")
        for t, m, s in zip(times_ns, bb_rmsd_mean, bb_rmsd_std):
            f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")

    if pocket_rmsd_mean is not None:
        pocket_csv = os.path.join(OUT_DIR, f"{case_id}_pocket_rmsd.csv")
        with open(pocket_csv, "w") as f:
            f.write("time_ns,pocket_backbone_rmsd_A_mean,pocket_backbone_rmsd_A_std\n")
            for t, m, s in zip(times_ns, pocket_rmsd_mean, pocket_rmsd_std):
                f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")

    derived_metrics, interpretation = _plateau_interpretation(times_ns, bb_rmsd_mean)
    summary_payload = make_metric_summary(
        "backbone_rmsd",
        case_id,
        series_summary(bb_rmsd_mean, times=times_ns),
        derived_metrics={
            **derived_metrics,
            "pocket_series_summary": series_summary(pocket_rmsd_mean, times=times_ns) if pocket_rmsd_mean is not None else None,
        },
        interpretation=[interpretation],
        notes=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_backbone_rmsd_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "times_ns": times_ns,
        "bb_rmsd": bb_rmsd_mean,
        "bb_rmsd_std": bb_rmsd_std,
        "mean": mean_val,
        "std": std_val,
        "pocket_rmsd": pocket_rmsd_mean,
        "pocket_rmsd_std": pocket_rmsd_std,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(description="Backbone RMSD overlay — 10ns_direct production")
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

    results = {}
    skipped = []
    with ProcessPoolExecutor(max_workers=min(4, len(cases_to_run))) as pool:
        futures = {pool.submit(compute_backbone_rmsd, case_id, meta, stride, stage6_cases): case_id
                   for case_id, meta in cases_to_run.items()}
        try:
            for future in as_completed(futures, timeout=WORKER_TIMEOUT_S):
                case_id = futures[future]
                try:
                    data = future.result()
                except Exception as e:
                    print(f"  ERROR: {case_id} skipped: {e}", flush=True)
                    skipped.append((case_id, str(e)))
                    continue
                if data is not None:
                    results[case_id] = data
        except FutureTimeoutError:
            for future, case_id in futures.items():
                if not future.done():
                    print(f"  TIMEOUT: {case_id} worker exceeded {WORKER_TIMEOUT_S}s — cancelling", flush=True)
                    skipped.append((case_id, f"worker timeout after {WORKER_TIMEOUT_S}s"))
                    future.cancel()

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for case_id, data in results.items():
        meta = CASE_META[case_id]
        ax.plot(data["times_ns"], data["bb_rmsd"],
                color=meta["color"], linestyle=meta["linestyle"],
                marker=meta["marker"], markevery=MARKER_EVERY, markersize=3,
                lw=1.2, label=data["label"])
        ax.fill_between(data["times_ns"], data["bb_rmsd"] - data["bb_rmsd_std"], data["bb_rmsd"] + data["bb_rmsd_std"], color=meta["color"], alpha=0.2)

    pocket_legend_added = False
    for case_id, data in results.items():
        if data.get("pocket_rmsd") is not None:
            pocket_label = "Pocket-local backbone RMSD" if not pocket_legend_added else None
            ax.plot(data["times_ns"], data["pocket_rmsd"],
                    color=CASE_META[case_id]["color"], lw=0.8,
                    linestyle="--", label=pocket_label)
            ax.fill_between(data["times_ns"], data["pocket_rmsd"] - data["pocket_rmsd_std"], data["pocket_rmsd"] + data["pocket_rmsd_std"], color=CASE_META[case_id]["color"], alpha=0.1)
            pocket_legend_added = True

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Backbone RMSD (Å)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_stem = os.path.join(OUT_DIR, "backbone_rmsd_overlay")
    save_pub_figure(fig, plot_stem)
    print(f"\nPlot saved: {plot_stem}.png / .pdf")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Case':<18} {'Mean':>8} {'Std':>8} {'Max':>8} {'Final':>8}")
    print(f"{'-'*70}")
    for case_id, data in results.items():
        r = data["bb_rmsd"]
        print(f"{case_id:<18} {r.mean():>7.2f}Å {r.std():>7.2f}Å {r.max():>7.2f}Å {r[-1]:>7.2f}Å")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "backbone_plateau_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "backbone_rmsd",
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