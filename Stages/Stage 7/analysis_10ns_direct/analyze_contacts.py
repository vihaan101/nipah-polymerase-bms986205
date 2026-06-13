"""\nStage 7 — Contact map: binding pocket fingerprint & allosteric context (10ns_direct)\nComputes residue-residue heavy-atom contact frequency for binding pocket and a\ndistal allosteric control region across all 4 cases.\n"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array

from stage7_eval_common import (robust_universe, 
    get_replicate_trajectory,
    CASE_META,
    apply_pub_style,
    get_replicate_trajectory,
    load_stage6_cases,
    require_file,
    resolve_stage6_path,
    save_pub_figure,
)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    criterion_result,
    make_metric_summary,
    matrix_window_similarity,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "contacts_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 2 * 3600  # 2 h per case; contact maps are CPU-bound but CIFS can stall
CONTACT_CUTOFF = 4.5


def _define_regions(topology, trajectory):
    u = robust_universe(topology, trajectory)
    u.trajectory[0]

    ligand = u.select_atoms("resname UNK and not name H*")
    protein = u.select_atoms("protein")
    if len(ligand) == 0:
        return [], []

    pocket_atoms = u.select_atoms("protein and (around 5.0 (resname UNK and not name H*))")
    pocket_resids = sorted(set(pocket_atoms.residues.resids))

    lig_pos = ligand.positions
    allosteric_resids = []
    for res in protein.residues:
        heavy = res.atoms.select_atoms("not name H*")
        if len(heavy) == 0:
            continue
        dists = distance_array(heavy.positions, lig_pos)
        if dists.min() > 20.0:
            allosteric_resids.append(res.resid)

    rng = np.random.RandomState(42)
    if len(allosteric_resids) > 30:
        allosteric_resids = sorted(rng.choice(allosteric_resids, 30, replace=False))

    return pocket_resids, allosteric_resids


def _compute_contact_frequency(u, region_resids, frame_indices):
    n_res = len(region_resids)
    contact_count = np.zeros((n_res, n_res), dtype=float)
    protein = u.select_atoms("protein")

    all_indices = []
    boundaries = [0]
    valid_orig = []
    for i, rid in enumerate(region_resids):
        heavy = protein.select_atoms(f"resid {rid} and not name H*")
        if len(heavy) > 0:
            all_indices.extend(heavy.indices.tolist())
            boundaries.append(boundaries[-1] + len(heavy))
            valid_orig.append(i)

    if not valid_orig or not frame_indices:
        return contact_count

    merged = u.atoms[all_indices]
    orig_indices = np.array(valid_orig)
    bounds = np.array(boundaries[:-1])

    for frame_idx in frame_indices:
        u.trajectory[frame_idx]
        dist_matrix = distance_array(merged.positions, merged.positions)
        contact_bool = (dist_matrix <= CONTACT_CUTOFF).astype(np.int32)
        row_reduced = np.add.reduceat(contact_bool, bounds, axis=0)
        pair_reduced = np.add.reduceat(row_reduced, bounds, axis=1)
        in_contact = np.triu(pair_reduced > 0, k=1)
        vi, vj = np.where(in_contact)
        oi = orig_indices[vi]
        oj = orig_indices[vj]
        np.add.at(contact_count, (oi, oj), 1)
        np.add.at(contact_count, (oj, oi), 1)

    return contact_count / len(frame_indices)


def _save_contact_csv(path, freq_matrix, resids):
    with open(path, "w") as f:
        f.write("," + ",".join(str(r) for r in resids) + "\n")
        for i, rid in enumerate(resids):
            row = ",".join(f"{freq_matrix[i, j]:.4f}" for j in range(len(resids)))
            f.write(f"{rid},{row}\n")


def _pair_rows(freq_matrix, early_matrix, late_matrix, resids):
    rows = []
    for i in range(len(resids)):
        for j in range(i + 1, len(resids)):
            rows.append(
                {
                    "resid_i": int(resids[i]),
                    "resid_j": int(resids[j]),
                    "frequency": float(freq_matrix[i, j]),
                    "early_frequency": float(early_matrix[i, j]),
                    "late_frequency": float(late_matrix[i, j]),
                    "window_delta": float(late_matrix[i, j] - early_matrix[i, j]),
                }
            )
    rows.sort(key=lambda row: row["frequency"], reverse=True)
    return rows


def _residue_persistence(pair_rows):
    residue_stats = {}
    for row in pair_rows:
        for resid_key in ("resid_i", "resid_j"):
            resid = row[resid_key]
            residue_stats.setdefault(resid, []).append(row["frequency"])
    out = []
    for resid, vals in residue_stats.items():
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "resid": int(resid),
                "mean_pair_frequency": float(arr.mean()),
                "max_pair_frequency": float(arr.max()),
                "n_pairs": int(arr.size),
            }
        )
    out.sort(key=lambda row: row["mean_pair_frequency"], reverse=True)
    return out


def compute_contacts(case_id, meta, stride, pocket_resids, allosteric_resids, stage6_cases):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))

    print(f"\n{'=' * 60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'=' * 60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_pocket_freq = []
    all_pocket_early = []
    all_pocket_late = []
    all_allo_freq = []

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        all_frame_indices = list(range(0, len(u.trajectory), stride))
        split = max(1, len(all_frame_indices) // 2)
        early_frames = all_frame_indices[:split]
        late_frames = all_frame_indices[split:]
        if not late_frames:
            late_frames = all_frame_indices

        pocket_freq = _compute_contact_frequency(u, pocket_resids, all_frame_indices)
        pocket_early = _compute_contact_frequency(u, pocket_resids, early_frames)
        pocket_late = _compute_contact_frequency(u, pocket_resids, late_frames)
        allosteric_freq = _compute_contact_frequency(u, allosteric_resids, all_frame_indices)

        all_pocket_freq.append(pocket_freq)
        all_pocket_early.append(pocket_early)
        all_pocket_late.append(pocket_late)
        all_allo_freq.append(allosteric_freq)

    if not all_pocket_freq:
        return None

    pocket_freq = np.mean(all_pocket_freq, axis=0)
    pocket_early = np.mean(all_pocket_early, axis=0)
    pocket_late = np.mean(all_pocket_late, axis=0)
    allosteric_freq = np.mean(all_allo_freq, axis=0)

    pocket_csv = os.path.join(OUT_DIR, f"{case_id}_pocket_contacts.csv")
    _save_contact_csv(pocket_csv, pocket_freq, pocket_resids)
    print(f"  Saved: {pocket_csv}")

    allo_csv = os.path.join(OUT_DIR, f"{case_id}_allosteric_contacts.csv")
    _save_contact_csv(allo_csv, allosteric_freq, allosteric_resids)
    print(f"  Saved: {allo_csv}")

    pair_rows = _pair_rows(pocket_freq, pocket_early, pocket_late, pocket_resids)
    pair_csv = os.path.join(OUT_DIR, f"{case_id}_pocket_contact_pairs.csv")
    with open(pair_csv, "w") as f:
        f.write("resid_i,resid_j,frequency,early_frequency,late_frequency,window_delta\n")
        for row in pair_rows:
            f.write(
                f"{row['resid_i']},{row['resid_j']},{row['frequency']:.4f},{row['early_frequency']:.4f},"
                f"{row['late_frequency']:.4f},{row['window_delta']:.4f}\n"
            )
    print(f"  Saved: {pair_csv}")

    window_similarity = matrix_window_similarity(pocket_early, pocket_late)
    residue_persistence = _residue_persistence(pair_rows)
    summary_payload = make_metric_summary(
        "contacts",
        case_id,
        {
            "n_pocket_residues": len(pocket_resids),
            "n_allosteric_residues": len(allosteric_resids),
            "top_pair_frequency": pair_rows[0]["frequency"] if pair_rows else None,
        },
        derived_metrics={
            "top_pairs": pair_rows[:15],
            "residue_persistence": residue_persistence[:15],
            "window_similarity": window_similarity,
        },
        interpretation=[
            criterion_result(
                "contact_persistence",
                STATUS_DESCRIPTIVE,
                "Residue-specific contact persistence is reported descriptively. Early/late window similarity is included for reproducibility-friendly comparison without universal contact-count thresholds.",
                extra={
                    "window_similarity": window_similarity,
                    "top_pairs": pair_rows[:5],
                },
            )
        ],
        limitations=[
            "No experimental residue-priority list or replica ensemble is available for stronger contact interpretation.",
        ],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_contacts_summary.json")
    write_json(summary_path, summary_payload)
    print(f"  Saved: {summary_path}")

    return {
        "label": meta["label"],
        "pocket_freq": pocket_freq,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(description="Contact maps — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None, help="Run only this case ID (e.g. 'C_BMS_WT')")
    parser.add_argument("--stride", type=int, default=STRIDE, help=f"Frame stride (default: {STRIDE})")
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

    ref_case_id = "A_ERDRP_WT" if "A_ERDRP_WT" in stage6_cases else next(
        (c for c in stage6_cases if "WT" in c), next(iter(stage6_cases))
    )
    ref_topo = str(resolve_stage6_path(stage6_cases[ref_case_id]["topology_pdb"]))
    ref_traj = str(get_replicate_trajectory(ref_case_id, 1))
    require_file(ref_topo, f"{ref_case_id} topology.pdb for region definition")
    print("Defining binding pocket and allosteric regions from A_ERDRP_WT frame 0...")
    pocket_resids, allosteric_resids = _define_regions(ref_topo, ref_traj)
    print(f"  Binding pocket: {len(pocket_resids)} residues")
    print(f"  Allosteric control: {len(allosteric_resids)} residues")
    if not pocket_resids:
        print("FATAL: No binding pocket residues found.")
        sys.exit(1)

    results = {}
    skipped = []
    max_workers = int(os.environ.get("STAGE7_CONTACTS_MAX_WORKERS", min(4, len(cases_to_run))))
    with ProcessPoolExecutor(max_workers=min(max_workers, len(cases_to_run))) as pool:
        futures = {
            pool.submit(compute_contacts, case_id, meta, stride, pocket_resids, allosteric_resids, stage6_cases): case_id
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
    ref_matrix = results.get("A_ERDRP_WT", {}).get("pocket_freq")
    for idx, case_id in enumerate(CASE_META.keys()):
        ax = axes[idx // 2][idx % 2]
        if case_id not in results:
            ax.text(0.5, 0.5, f"{case_id}\n(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(CASE_META[case_id]["label"])
            continue

        freq = results[case_id]["pocket_freq"]
        if case_id == "A_ERDRP_WT" or ref_matrix is None:
            im = ax.imshow(freq, cmap="Blues", aspect="auto", vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Frequency")
        else:
            diff = freq - ref_matrix
            vmax = max(abs(diff.min()), abs(diff.max()), 0.1)
            im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Delta frequency vs A_ERDRP_WT")

        n_res = len(pocket_resids)
        tick_step = max(1, n_res // 8)
        tick_pos = list(range(0, n_res, tick_step))
        tick_labels = [str(pocket_resids[i]) for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("Residue ID")
        ax.set_title(results[case_id]["label"], fontweight="semibold")

    fig.tight_layout()
    plot_stem = os.path.join(OUT_DIR, "contact_map_grid")
    save_pub_figure(fig, plot_stem)
    print(f"\nContact map grid saved: {plot_stem}.png / .pdf")

    print(f"\n{'=' * 80}")
    print(f"{'Case':<18} {'Top pair freq':>14} {'Pairs reported':>16}")
    print(f"{'-' * 80}")
    for case_id in CASE_META:
        if case_id in results:
            matrix = results[case_id]["pocket_freq"]
            tri = matrix[np.triu_indices(matrix.shape[0], k=1)]
            top_pair = float(np.max(tri)) if tri.size else 0.0
            print(f"{case_id:<18} {top_pair:>13.3f} {int(tri.size):>16}")
    print(f"{'=' * 80}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "contacts_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "contacts",
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