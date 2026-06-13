"""\nStage 7 — Ligand RMSD overlay (10ns_direct production trajectories)\nBackbone-aligned ligand heavy-atom RMSD vs frame 0 for all 4 cases.\n"""

import argparse
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align

from stage7_eval_common import (robust_universe, 
    get_replicate_trajectory,
    CASE_META,
    MARKER_EVERY,
    apply_pub_style,
    get_replicate_trajectory,
    load_stage6_cases,
    require_file,
    resolve_stage6_path,
    save_pub_figure,
    minimum_image_displacements,
)
from stage7_success_criteria import (
    STATUS_CONCERN,
    STATUS_DESCRIPTIVE,
    STATUS_NOT_ASSESSED,
    STATUS_SUPPORTS,
    criterion_result,
    excursion_metrics,
    late_window_metrics,
    make_metric_summary,
    series_summary,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "rmsd_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h total; with 4 parallel workers this is ~3 h per case
DEFAULT_FRAME_INTERVAL_PS = 4.0


def _ligand_pose_interpretation(times_ns, rmsd_A, com_dist_A):
    late = late_window_metrics(times_ns, rmsd_A)
    excursion_2A = excursion_metrics(rmsd_A, 2.0, times=times_ns)
    excursion_4A = excursion_metrics(rmsd_A, 4.0, times=times_ns)
    pocket_cutoff = max(5.0, float(np.nanpercentile(com_dist_A, 10)) + 2.0) if len(com_dist_A) else None
    residence_fraction = float(np.mean(com_dist_A <= pocket_cutoff)) if pocket_cutoff is not None else None
    relocation_candidate = bool(
        len(com_dist_A)
        and com_dist_A[-1] > pocket_cutoff
        and excursion_4A.get("longest_run_time_ns") is not None
        and excursion_4A["longest_run_time_ns"] >= 2.0
    )

    derived = {
        "late_window": late,
        "heuristic_excursions": {
            "above_2A": excursion_2A,
            "above_4A": excursion_4A,
        },
        "pocket_residence_proxy": {
            "cutoff_A": pocket_cutoff,
            "residence_fraction": residence_fraction,
            "mean_distance_A": float(np.mean(com_dist_A)) if len(com_dist_A) else None,
            "max_distance_A": float(np.max(com_dist_A)) if len(com_dist_A) else None,
            "final_distance_A": float(com_dist_A[-1]) if len(com_dist_A) else None,
            "alternative_pocket_candidate": relocation_candidate,
        },
    }

    if len(rmsd_A) < 8:
        pose_status = STATUS_NOT_ASSESSED
        pose_detail = "Too few samples for a cautious pose-retention heuristic."
    else:
        supports = (
            late.get("slope_per_ns") is not None
            and abs(late["slope_per_ns"]) <= 0.05
            and (excursion_4A.get("longest_run_time_ns") or 0.0) < 2.0
            and (residence_fraction is None or residence_fraction >= 0.75)
            and not relocation_candidate
        )
        pose_status = STATUS_SUPPORTS if supports else STATUS_CONCERN
        pose_detail = (
            "Heuristic proxy for the qualitative literature criterion of remaining in the starting pocket "
            "without sustained relocation. This is not a universal ligand RMSD cutoff."
        )

    return derived, [
        criterion_result(
            "ligand_pose_retention",
            pose_status,
            pose_detail,
            extra=derived,
            threshold={
                "late_window_slope_abs_A_per_ns": 0.05,
                "sustained_excursion_A": 4.0,
                "sustained_excursion_time_ns": 2.0,
                "pocket_residence_fraction": 0.75,
            },
            threshold_note="Heuristic proxy for a qualitative gold-standard pose-retention criterion.",
        ),
        criterion_result(
            "ligand_rmsd_2a_reference",
            STATUS_DESCRIPTIVE,
            "Reported as retrospective reference context only; the 2 Å band is not treated as a universal success threshold.",
            extra={"excursion_metrics": excursion_2A},
            threshold=2.0,
            threshold_note="Retrospective BPMD/pose-retention reference band, not a universal pass/fail rule.",
        ),
    ]


def compute_ligand_rmsd(case_id, meta, stride, stage6_cases):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))

    print(f"\n{'=' * 60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'=' * 60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_lig_rmsds = []
    all_com_distances = []
    all_times = []

    ref = None

    for rep_id in range(1, 6):
        try:
            tier_dcd = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        if ref is None:
            ref = robust_universe(topology, tier_dcd)
            ref.trajectory[0]
            lig_ref = ref.select_atoms("resname UNK and not name H*")
            pocket_ref = ref.select_atoms("protein and (around 5.0 (resname UNK and not name H*)) and not name H*")
            pocket_com_ref = pocket_ref.center_of_mass() if len(pocket_ref) else None

        u = robust_universe(topology, tier_dcd)
        ligand = u.select_atoms("resname UNK and not name H*")
        backbone = u.select_atoms("backbone")
        
        if len(ligand) == 0:
            print(f"  WARNING: no ligand found, skipping rep {rep_id}.")
            continue

        lig_rmsds = []
        times_ns = []
        ligand_com_distances = []

        sampled_times_ns = np.arange(0, len(u.trajectory), stride, dtype=float) * (DEFAULT_FRAME_INTERVAL_PS / 1000.0)

        sample_idx = 0
        for frame_idx, _ in enumerate(u.trajectory):
            if frame_idx % stride != 0:
                continue

            if sample_idx >= len(sampled_times_ns):
                break

            time_ns = sampled_times_ns[sample_idx]
            sample_idx += 1

            box_lengths = u.dimensions[:3] if u.dimensions is not None else None

            if box_lengths is not None and len(backbone) > 0:
                bb_com = backbone.center_of_mass()
                lig_com = ligand.center_of_mass()
                raw_delta = lig_com - bb_com
                delta_mic = minimum_image_displacements(raw_delta.reshape(1, 3), box_lengths).flatten()
                shift = delta_mic - raw_delta
                if np.any(np.abs(shift) > 1e-6):
                    ligand.positions = ligand.positions + shift

            align.alignto(u, ref, select="backbone")

            diff = ligand.positions - lig_ref.positions
            lig_rmsds.append(np.sqrt((diff ** 2).sum(axis=1).mean()))
            times_ns.append(time_ns)
            if pocket_com_ref is not None:
                ligand_com_distances.append(float(np.linalg.norm(ligand.center_of_mass() - pocket_com_ref)))
        
        all_lig_rmsds.append(lig_rmsds)
        all_times.append(times_ns)
        if ligand_com_distances:
            all_com_distances.append(ligand_com_distances)

    if not all_lig_rmsds:
        return None

    min_len = min(len(t) for t in all_times)
    times_ns = np.array(all_times[0][:min_len])
    
    lig_rmsd_arr = np.array([r[:min_len] for r in all_lig_rmsds])
    lig_rmsds_mean = lig_rmsd_arr.mean(axis=0)
    lig_rmsds_std = lig_rmsd_arr.std(axis=0)

    com_dist_mean = None
    com_dist_std = None
    if all_com_distances:
        com_dist_arr = np.array([d[:min_len] for d in all_com_distances])
        com_dist_mean = com_dist_arr.mean(axis=0)
        com_dist_std = com_dist_arr.std(axis=0)

    mean_val = lig_rmsds_mean.mean()
    std_val = lig_rmsds_mean.std()

    csv_path = os.path.join(OUT_DIR, f"{case_id}_ligand_rmsd.csv")
    with open(csv_path, "w") as f:
        f.write("time_ns,ligand_rmsd_A_mean,ligand_rmsd_A_std\n")
        for t, m, s in zip(times_ns, lig_rmsds_mean, lig_rmsds_std):
            f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")

    if com_dist_mean is not None:
        com_csv_path = os.path.join(OUT_DIR, f"{case_id}_ligand_pocket_com.csv")
        with open(com_csv_path, "w") as f:
            f.write("time_ns,ligand_com_to_starting_pocket_A_mean,ligand_com_to_starting_pocket_A_std\n")
            for t, m, s in zip(times_ns, com_dist_mean, com_dist_std):
                f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")

    derived_metrics, interpretation = _ligand_pose_interpretation(times_ns, lig_rmsds_mean, com_dist_mean if com_dist_mean is not None else np.array([]))
    summary_payload = make_metric_summary(
        "ligand_rmsd",
        case_id,
        series_summary(lig_rmsds_mean, times=times_ns),
        derived_metrics=derived_metrics,
        interpretation=interpretation,
        notes=[],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_ligand_rmsd_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "times_ns": times_ns,
        "lig_rmsd": lig_rmsds_mean,
        "lig_rmsd_std": lig_rmsds_std,
        "mean": mean_val,
        "std": std_val,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(description="Ligand RMSD overlay — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None, help="Run only this case ID (e.g. 'C_BMS_WT')")
    parser.add_argument("--stride", type=int, default=STRIDE, help=f"Frame stride (default: {STRIDE})")
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
        futures = {
            pool.submit(compute_ligand_rmsd, case_id, meta, stride, stage6_cases): case_id
            for case_id, meta in cases_to_run.items()
        }
        try:
            for future in as_completed(futures, timeout=WORKER_TIMEOUT_S):
                case_id = futures[future]
                try:
                    data = future.result()
                except Exception as e:
                    print(f"  ERROR: {case_id} skipped: {e}", flush=True)
                    traceback.print_exc()
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

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.axvspan(0, 2, color="gray", alpha=0.10, label="Equilibration window (0–2 ns)")
    ax.axhline(
        y=2.0,
        color="black",
        linestyle="--",
        linewidth=1.0,
        alpha=0.6,
        label="2 Å heuristic reference (not universal)",
    )

    for case_id, data in results.items():
        meta = CASE_META[case_id]
        ax.plot(
            data["times_ns"],
            data["lig_rmsd"],
            color=meta["color"],
            linestyle=meta["linestyle"],
            marker=meta["marker"],
            markevery=MARKER_EVERY,
            markersize=3,
            lw=1.2,
            label=data["label"],
        )
        ax.fill_between(data["times_ns"], data["lig_rmsd"] - data["lig_rmsd_std"], data["lig_rmsd"] + data["lig_rmsd_std"], color=meta["color"], alpha=0.2)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ligand RMSD (Å)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_stem = os.path.join(OUT_DIR, "ligand_rmsd_overlay")
    save_pub_figure(fig, plot_stem)
    print(f"\nPlot saved: {plot_stem}.png / .pdf")

    print(f"\n{'=' * 70}")
    print(f"{'Case':<18} {'Mean':>8} {'Std':>8} {'Max':>8} {'Final':>8}")
    print(f"{'-' * 70}")
    for case_id, data in results.items():
        r = data["lig_rmsd"]
        print(f"{case_id:<18} {r.mean():>7.2f}Å {r.std():>7.2f}Å {r.max():>7.2f}Å {r[-1]:>7.2f}Å")
    print(f"{'=' * 70}")
    print(f"\nAll outputs in: {OUT_DIR}")

    # Build the metric index by scanning the output directory for completed summary
    # files rather than relying solely on the in-memory results dict. This ensures
    # cases that wrote their summary file but raised an exception before returning
    # (e.g. worker post-processing failures) are still included in the index.
    case_summaries = {}
    for case_id in CASE_META:
        candidate = os.path.join(OUT_DIR, f"{case_id}_ligand_rmsd_summary.json")
        if os.path.exists(candidate):
            case_summaries[case_id] = candidate
    # In-memory results take precedence for paths (they match the current run)
    for case_id, data in results.items():
        case_summaries[case_id] = data["summary_path"]

    overview_path = os.path.join(OUT_DIR, "ligand_rmsd_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "ligand_rmsd",
            "case_summaries": case_summaries,
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