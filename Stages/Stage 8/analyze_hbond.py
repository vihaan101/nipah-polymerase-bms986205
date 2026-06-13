"""
Stage 7 — Hydrogen bond persistence analysis (10ns_direct production)
Ligand (UNK) <-> protein H-bonds via MDAnalysis HydrogenBondAnalysis.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis

from stage7_eval_common import (
    get_replicate_trajectory,
    CASE_META,
    apply_pub_style,
    load_stage6_cases,
    require_file,
    resolve_stage6_path,
    save_pub_figure,
    robust_universe,
)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    criterion_result,
    make_metric_summary,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "hbond_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h; HydrogenBondAnalysis iterates full trajectory


def _bond_label(u, donor_idx, acceptor_idx):
    d = u.atoms[donor_idx]
    a = u.atoms[acceptor_idx]
    return f"{d.resname}{d.resid}:{d.name}-{a.resname}{a.resid}:{a.name}"


def _window_occupancy(frame_hits, frame_ids):
    if not frame_ids:
        return {"overall_pct": 0.0, "early_pct": 0.0, "late_pct": 0.0}
    frame_ids = sorted(frame_ids)
    split = max(1, len(frame_ids) // 2)
    early = set(frame_ids[:split])
    late = set(frame_ids[split:])
    hits = set(frame_hits)
    return {
        "overall_pct": 100.0 * len(hits) / len(frame_ids),
        "early_pct": 100.0 * len(hits & early) / len(early) if early else 0.0,
        "late_pct": 100.0 * len(hits & late) / len(late) if late else 0.0,
    }


def _load_hbond_config():
    path = os.environ.get("STAGE7_CRITICAL_HBONDS_JSON", "").strip()
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"  WARNING: configured critical H-bond file not found: {path}")
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def compute_hbonds(case_id, meta, stride, stage6_cases, config):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))
    
    print(f"\n{'=' * 60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'=' * 60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_occupancies = []

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        n_frames = len(u.trajectory)
        analyzed_frame_ids = list(range(0, n_frames, stride))

        bond_hits = defaultdict(set)

        for donors_sel, acceptors_sel, direction in [
            ("protein", "resname UNK", "protein->ligand"),
            ("resname UNK", "protein", "ligand->protein"),
        ]:
            hba = HydrogenBondAnalysis(
                u,
                donors_sel=donors_sel,
                acceptors_sel=acceptors_sel,
                hydrogens_sel="name H*",
                d_a_cutoff=3.5,
                d_h_a_angle_cutoff=120,
            )
            hba.run(step=stride, verbose=False)

            if hba.results.hbonds is not None and len(hba.results.hbonds) > 0:
                for row in hba.results.hbonds:
                    frame_idx = int(row[0])
                    donor_idx = int(row[1])
                    acceptor_idx = int(row[3])
                    label = _bond_label(u, donor_idx, acceptor_idx)
                    bond_hits[label].add(frame_idx)

        occupancies = []
        for label, frames in bond_hits.items():
            windowed = _window_occupancy(frames, analyzed_frame_ids)
            occupancies.append(
                {
                    "donor_acceptor": label,
                    "occupancy_pct": windowed["overall_pct"],
                    "occupancy_early_pct": windowed["early_pct"],
                    "occupancy_late_pct": windowed["late_pct"],
                    "window_delta_pct": windowed["late_pct"] - windowed["early_pct"],
                    "n_frames_present": len(frames),
                }
            )
        all_occupancies.append(occupancies)

    if not all_occupancies:
        return None

    combined_occ = defaultdict(list)
    combined_early = defaultdict(list)
    combined_late = defaultdict(list)
    combined_delta = defaultdict(list)
    combined_frames = defaultdict(list)

    for occ_list in all_occupancies:
        for row in occ_list:
            label = row["donor_acceptor"]
            combined_occ[label].append(row["occupancy_pct"])
            combined_early[label].append(row["occupancy_early_pct"])
            combined_late[label].append(row["occupancy_late_pct"])
            combined_delta[label].append(row["window_delta_pct"])
            combined_frames[label].append(row["n_frames_present"])

    num_reps = len(all_occupancies)
    for label in combined_occ:
        while len(combined_occ[label]) < num_reps:
            combined_occ[label].append(0.0)
            combined_early[label].append(0.0)
            combined_late[label].append(0.0)
            combined_delta[label].append(0.0)
            combined_frames[label].append(0)

    final_occupancies = []
    for label in combined_occ:
        avg_occ = sum(combined_occ[label]) / num_reps
        if avg_occ > 0:
            final_occupancies.append(
                {
                    "donor_acceptor": label,
                    "occupancy_pct": avg_occ,
                    "occupancy_early_pct": sum(combined_early[label]) / num_reps,
                    "occupancy_late_pct": sum(combined_late[label]) / num_reps,
                    "window_delta_pct": sum(combined_delta[label]) / num_reps,
                    "n_frames_present": int(sum(combined_frames[label]) / num_reps),
                }
            )

    final_occupancies.sort(key=lambda item: item["occupancy_pct"], reverse=True)

    csv_path = os.path.join(OUT_DIR, f"{case_id}_hbonds.csv")
    with open(csv_path, "w") as f:
        f.write("donor_acceptor,occupancy_pct,occupancy_early_pct,occupancy_late_pct,window_delta_pct,n_frames_present\n")
        for row in final_occupancies:
            f.write(
                f"{row['donor_acceptor']},{row['occupancy_pct']:.2f},{row['occupancy_early_pct']:.2f},"
                f"{row['occupancy_late_pct']:.2f},{row['window_delta_pct']:.2f},{row['n_frames_present']}\n"
            )

    configured = config.get("critical_bonds", {})
    case_targets = configured.get(case_id, configured.get("default", []))
    critical_rows = [row for row in final_occupancies if row["donor_acceptor"] in set(case_targets)]

    summary_payload = make_metric_summary(
        "hbonds",
        case_id,
        {
            "n_bonds": len(final_occupancies),
            "top_occupancy_pct": final_occupancies[0]["occupancy_pct"] if final_occupancies else None,
        },
        derived_metrics={"top_bonds": final_occupancies[:10]},
        interpretation=[],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_hbonds_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "occupancies": final_occupancies,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(description="H-bond persistence — 10ns_direct production")
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
    config = _load_hbond_config()

    results = {}
    skipped = []
    with ProcessPoolExecutor(max_workers=min(4, len(cases_to_run))) as pool:
        futures = {
            pool.submit(compute_hbonds, case_id, meta, stride, stage6_cases, config): case_id
            for case_id, meta in cases_to_run.items()
        }
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

    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    for idx, case_id in enumerate(CASE_META.keys()):
        ax = axes[idx // 2][idx % 2]
        if case_id not in results:
            ax.text(0.5, 0.5, f"{case_id}\n(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(CASE_META[case_id]["label"])
            continue

        top10 = results[case_id]["occupancies"][:10]
        if not top10:
            ax.text(0.5, 0.5, "No H-bonds", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(results[case_id]["label"])
            continue

        labels = [row["donor_acceptor"] for row in reversed(top10)]
        values = [row["occupancy_pct"] for row in reversed(top10)]
        y_pos = range(len(labels))
        bars = ax.barh(y_pos, values, color=CASE_META[case_id]["color"], edgecolor="black", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("Occupancy (%)")
        ax.set_title(results[case_id]["label"], fontweight="semibold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="x")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=6)

    fig.tight_layout()
    bar_stem = os.path.join(OUT_DIR, "hbond_top10_bars")
    save_pub_figure(fig, bar_stem)
    print(f"\nTop-10 bar chart saved: {bar_stem}.png / .pdf")

    all_bonds = sorted(
        {
            row["donor_acceptor"]
            for data in results.values()
            for row in data["occupancies"][:15]
        }
    )

    if all_bonds:
        case_ids = [c for c in CASE_META.keys() if c in results]
        matrix = np.zeros((len(all_bonds), len(case_ids)))
        for j, case_id in enumerate(case_ids):
            occ_lookup = {row["donor_acceptor"]: row["occupancy_pct"] for row in results[case_id]["occupancies"]}
            for i, bond in enumerate(all_bonds):
                matrix[i, j] = occ_lookup.get(bond, 0.0)

        fig, ax = plt.subplots(figsize=(7, max(3, len(all_bonds) * 0.25)))
        im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(case_ids)))
        ax.set_xticklabels([results[c]["label"] for c in case_ids], rotation=20, ha="right")
        ax.set_yticks(range(len(all_bonds)))
        ax.set_yticklabels(all_bonds)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Occupancy (%)")
        fig.tight_layout()
        hm_stem = os.path.join(OUT_DIR, "hbond_occupancy_heatmap")
        save_pub_figure(fig, hm_stem)
        print(f"Heatmap saved: {hm_stem}.png / .pdf")

    print(f"\n{'=' * 80}")
    print(f"{'Case':<18} {'# Unique':>10} {'Top bond':>35} {'Occ':>8}")
    print(f"{'-' * 80}")
    for case_id, data in results.items():
        top = data["occupancies"][0] if data["occupancies"] else None
        if top:
            print(f"{case_id:<18} {len(data['occupancies']):>10} {top['donor_acceptor']:>35} {top['occupancy_pct']:>7.1f}%")
        else:
            print(f"{case_id:<18} {0:>10} {'N/A':>35} {'N/A':>8}")
    print(f"{'=' * 80}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "hbonds_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "hbonds",
            "case_summaries": {case_id: results[case_id]["summary_path"] for case_id in results},
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