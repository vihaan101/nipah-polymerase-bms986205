"""\nStage 7 — Per-residue RMSF flexibility profile (10ns_direct production)\nCα RMSF after backbone alignment, with W730 mutation-site and\nbinding-pocket annotations.\n"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

from stage7_eval_common import (robust_universe, get_replicate_trajectory, CASE_META, load_stage6_cases, resolve_stage6_path, find_trajectory, find_trajectory_chain, require_file, apply_pub_style, save_pub_figure)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    criterion_result,
    make_metric_summary,
    series_summary,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "rmsf_10ns_direct")

STRIDE = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h; in_memory alignment loads full 10ns_direct trajectory


def _find_mutation_site(wt_topology, mut_topology):
    """Auto-detect the W730A mutation site by comparing WT and MUT topologies.\n\n    Tries resname TRP + resid 730 first (canonical numbering).  If absent,\n    finds the TRP residue in WT that becomes ALA in the mutant.\n    """
    u_wt = robust_universe(wt_topology)
    direct = u_wt.select_atoms("resname TRP and resid 730 and name CA")
    if len(direct) > 0:
        return int(direct[0].resid)

    u_mut = robust_universe(mut_topology)
    wt_ca = u_wt.select_atoms("protein and name CA")
    mut_ca = u_mut.select_atoms("protein and name CA")

    wt_map = {a.resid: a.resname for a in wt_ca}
    mut_map = {a.resid: a.resname for a in mut_ca}

    for resid in wt_map:
        if wt_map[resid] == "TRP" and mut_map.get(resid) == "ALA":
            return resid

    return None


def _find_binding_pocket_resids(topology, trajectory):
    """Return set of protein resids within 5 A of ligand in frame 0."""
    u = robust_universe(topology, trajectory)
    u.trajectory[0]
    ligand = u.select_atoms("resname UNK")
    if len(ligand) == 0:
        return set()
    pocket = u.select_atoms(f"protein and (around 5.0 resname UNK) and not resname UNK")
    return set(pocket.residues.resids)


def _find_core_resids(topology, trajectory):
    """Return a simple buried-core proxy from frame-0 CA distances to protein COM."""
    u = robust_universe(topology, trajectory)
    u.trajectory[0]
    ca = u.select_atoms("protein and name CA")
    if len(ca) == 0:
        return set()
    com = ca.center_of_mass()
    dists = np.linalg.norm(ca.positions - com, axis=1)
    cutoff = np.quantile(dists, 0.4)
    return set(ca.resids[dists <= cutoff])


def compute_rmsf(case_id, meta, stride, stage6_cases):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))

    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    require_file(topology, f"topology.pdb for {case_id}")

    all_rmsf_values = []
    resids = None
    resnames = None

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        align.AlignTraj(u, u, select="backbone", in_memory=True).run(step=stride, verbose=False)

        ca = u.select_atoms("protein and name CA")
        rmsf_calc = rms.RMSF(ca)
        rmsf_calc.run(step=stride, verbose=False)
        
        all_rmsf_values.append(rmsf_calc.results.rmsf)

        if resids is None:
            resids = ca.resids
            resnames = ca.resnames

    if not all_rmsf_values:
        return None

    rmsf_arr = np.array(all_rmsf_values)
    rmsf_mean = rmsf_arr.mean(axis=0)
    rmsf_std = rmsf_arr.std(axis=0)

    print(f"  RMSF — mean: {rmsf_mean.mean():.2f} A  max: {rmsf_mean.max():.2f} A")

    csv_path = os.path.join(OUT_DIR, f"{case_id}_rmsf.csv")
    with open(csv_path, "w") as f:
        f.write("resid,resname,rmsf_A_mean,rmsf_A_std\n")
        for rid, rn, rm, rs in zip(resids, resnames, rmsf_mean, rmsf_std):
            f.write(f"{rid},{rn},{rm:.4f},{rs:.4f}\n")

    return {
        "label": meta["label"],
        "resids": resids,
        "resnames": resnames,
        "rmsf": rmsf_mean,
        "rmsf_std": rmsf_std,
    }



def main():
    parser = argparse.ArgumentParser(description="Per-residue RMSF — 10ns_direct production")
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

    results = {}
    skipped = []
    max_workers = int(os.environ.get("STAGE7_RMSF_MAX_WORKERS", min(4, len(cases_to_run))))
    with ProcessPoolExecutor(max_workers=min(max_workers, len(cases_to_run))) as pool:
        futures = {pool.submit(compute_rmsf, case_id, meta, stride, stage6_cases): case_id
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

    # ── Auto-detect W730 mutation site ────────────────────────────────────────
    mutation_resid = None
    try:
        wt_topo = str(resolve_stage6_path(stage6_cases["A_ERDRP_WT"]["topology_pdb"]))
        mut_topo = str(resolve_stage6_path(stage6_cases["B_ERDRP_MUT"]["topology_pdb"]))
        require_file(wt_topo, "A_ERDRP_WT topology.pdb for mutation-site detection")
        require_file(mut_topo, "B_ERDRP_MUT topology.pdb for mutation-site detection")
        mutation_resid = _find_mutation_site(wt_topo, mut_topo)
        if mutation_resid is not None:
            print(f"\n    W730A mutation site detected at PDB resid {mutation_resid}")
        else:
            print("\n    WARNING: Could not auto-detect W730A mutation site.")
    except KeyError as e:
        print(f"\n    NOTE: Mutation-site detection skipped (missing case {e} in manifest)")
        wt_topo = None

    # ── Identify binding pocket from A_ERDRP_WT (or first available WT case) ──
    pocket_resids = set()
    core_resids = set()
    ref_case_id = "A_ERDRP_WT" if "A_ERDRP_WT" in stage6_cases else next(
        (c for c in stage6_cases if "WT" in c), next(iter(stage6_cases))
    )
    try:
        ref_topo = str(resolve_stage6_path(stage6_cases[ref_case_id]["topology_pdb"]))
        ref_traj = find_trajectory_chain(ref_case_id)
        pocket_resids = _find_binding_pocket_resids(ref_topo, ref_traj)
        core_resids = _find_core_resids(ref_topo, ref_traj)
        print(f"  Binding pocket: {len(pocket_resids)} residues")
        print(f"  Core proxy: {len(core_resids)} residues")
    except Exception as e:
        print(f"  WARNING: Could not detect binding pocket from {ref_case_id}: {e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))

    if pocket_resids:
        for i, resid in enumerate(sorted(pocket_resids)):
            ax.axvline(x=resid, color="orange", alpha=0.3, linewidth=0.8,
                       label="Binding pocket residues" if i == 0 else None)

    if mutation_resid is not None:
        ax.axvline(x=mutation_resid, color="black", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"W730 (resid {mutation_resid})")

    for case_id, data in results.items():
        meta = CASE_META[case_id]
        ax.plot(data["resids"], data["rmsf"],
                color=meta["color"], linestyle=meta["linestyle"],
                lw=0.8, label=data["label"])
        ax.fill_between(data["resids"], data["rmsf"] - data["rmsf_std"], data["rmsf"] + data["rmsf_std"], color=meta["color"], alpha=0.2)

    ax.set_xlabel("Residue ID")
    ax.set_ylabel("RMSF (\u00c5)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=min(r["resids"].min() for r in results.values()),
                right=max(r["resids"].max() for r in results.values()))
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plot_stem = os.path.join(OUT_DIR, "rmsf_overlay")
    save_pub_figure(fig, plot_stem)
    print(f"\nPlot saved: {plot_stem}.png / .pdf")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Case':<18} {'Mean RMSF':>10} {'Max RMSF':>10} {'Max Resid':>10}")
    print(f"{'-'*60}")
    for case_id, data in results.items():
        r = data["rmsf"]
        max_idx = np.argmax(r)
        print(f"{case_id:<18} {r.mean():>9.2f}Å {r.max():>9.2f}Å {data['resids'][max_idx]:>10}")
    print(f"{'='*60}")
    print(f"\nAll outputs in: {OUT_DIR}")

    for case_id, data in results.items():
        rmsf_values = data["rmsf"]
        resids = np.asarray(data["resids"])
        pocket_mask = np.isin(resids, sorted(pocket_resids)) if pocket_resids else np.zeros_like(resids, dtype=bool)
        core_mask = np.isin(resids, sorted(core_resids)) if core_resids else np.zeros_like(resids, dtype=bool)
        mutation_mask = resids == mutation_resid if mutation_resid is not None else np.zeros_like(resids, dtype=bool)

        pocket_mean = float(rmsf_values[pocket_mask].mean()) if pocket_mask.any() else None
        core_mean = float(rmsf_values[core_mask].mean()) if core_mask.any() else None
        mutation_rmsf = float(rmsf_values[mutation_mask][0]) if mutation_mask.any() else None
        top_indices = np.argsort(rmsf_values)[-5:][::-1]
        top_residues = [
            {
                "resid": int(resids[idx]),
                "resname": str(data["resnames"][idx]),
                "rmsf_A": float(rmsf_values[idx]),
                "in_pocket": bool(resids[idx] in pocket_resids),
                "in_core_proxy": bool(resids[idx] in core_resids),
            }
            for idx in top_indices
        ]

        summary_payload = make_metric_summary(
            "rmsf",
            case_id,
            series_summary(rmsf_values),
            derived_metrics={
                "pocket_vs_core": {
                    "pocket_mean_rmsf_A": pocket_mean,
                    "core_proxy_mean_rmsf_A": core_mean,
                    "difference_A": (pocket_mean - core_mean) if pocket_mean is not None and core_mean is not None else None,
                },
                "mutation_site": {
                    "resid": int(mutation_resid) if mutation_resid is not None else None,
                    "rmsf_A": mutation_rmsf,
                },
                "top_flexible_residues": top_residues,
                "replica_hooks": {
                    "replica_comparison_supported": True,
                    "replica_inputs_present": False,
                },
            },
            interpretation=[
                criterion_result(
                    "rmsf_profile",
                    STATUS_DESCRIPTIVE,
                    "RMSF is reported descriptively. Pocket/core comparisons are provided to aid interpretation without imposing a universal threshold.",
                    extra={
                        "pocket_residue_count": len(pocket_resids),
                        "core_proxy_residue_count": len(core_resids),
                    },
                )
            ],
            notes=[
                "Pocket residues come from the frame-0 WT reference structure.",
                "Core proxy uses the most buried 40% of frame-0 CA atoms by distance to protein COM.",
            ],
            limitations=[
                "No experimental flexibility proxy or replica ensemble is available in this pipeline.",
            ],
        )
        summary_path = os.path.join(OUT_DIR, f"{case_id}_rmsf_summary.json")
        write_json(summary_path, summary_payload)
        print(f"Summary saved: {summary_path}")

    overview_path = os.path.join(OUT_DIR, "rmsf_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "rmsf",
            "case_summaries": {
                case_id: os.path.join(OUT_DIR, f"{case_id}_rmsf_summary.json")
                for case_id in results
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