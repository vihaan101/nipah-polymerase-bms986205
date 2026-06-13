#!/usr/bin/env python3
"""\nStage 7 — Principal Component Analysis (10ns_direct production)\nCovariance-based PCA on C-alpha atom positions after backbone alignment.\nTwo analyses are written per case: global protein PCA and pocket-focused PCA.\n"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import align

from stage7_eval_common import (
    get_replicate_trajectory,
    CASE_META,
    apply_pub_style,
    load_stage6_cases,
    require_file,
    resolve_stage6_path,
    save_pub_figure,
    sampled_cumulative_time_ns,
    robust_universe,
)
from stage7_success_criteria import (
    STATUS_CONCERN,
    STATUS_DESCRIPTIVE,
    STATUS_NOT_ASSESSED,
    STATUS_SUPPORTS,
    criterion_result,
    make_metric_summary,
    rmsip,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "pca_10ns_direct")

STRIDE = 10
N_PCS = 10
WORKER_TIMEOUT_S = 3 * 3600  # 3 h per case; in-memory covariance on full 10ns_direct trajectory


def _detect_pocket_resids(u):
    u.trajectory[0]
    pocket_atoms = u.select_atoms("protein and (around 5.0 (resname UNK and not name H*))")
    return sorted(set(pocket_atoms.residues.resids))


def _collect_coords(u, atom_group, ref, stride):
    coords = []
    times = sampled_cumulative_time_ns(u, stride)
    for ts in u.trajectory[::stride]:
        align.alignto(u, ref, select="backbone")
        coords.append(atom_group.positions.flatten())
    return np.array(coords), np.array(times)


def _pca_from_coords(coords, times):
    coords_mean = coords.mean(axis=0)
    coords_centered = coords - coords_mean
    cov = np.cov(coords_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    total_var = eigenvalues.sum()
    explained_pct = 100.0 * eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative_pct = np.cumsum(explained_pct)
    n_pcs = min(N_PCS, len(eigenvalues))
    projections = coords_centered @ eigenvectors[:, :n_pcs]
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "explained_pct": explained_pct,
        "cumulative_pct": cumulative_pct,
        "projections": projections,
        "times": times,
        "n_pcs": n_pcs,
    }


def _windowed_convergence(coords, times):
    if len(coords) < 8:
        return {
            "status": STATUS_NOT_ASSESSED,
            "detail": "Too few frames for windowed PCA comparison.",
            "metrics": {"rmsip_top3": None, "pc1_explained_delta_pct": None},
        }
    split = len(coords) // 2
    first = _pca_from_coords(coords[:split], times[:split])
    second = _pca_from_coords(coords[split:], times[split:])
    overlap = rmsip(first["eigenvectors"], second["eigenvectors"], n_components=3)
    pc1_delta = abs(float(first["explained_pct"][0] - second["explained_pct"][0]))
    supports = overlap is not None and overlap >= 0.6 and pc1_delta <= 10.0
    return {
        "status": STATUS_SUPPORTS if supports else STATUS_CONCERN,
        "detail": "Windowed PCA overlap is a heuristic convergence proxy; PCA remains descriptive unless dominant subspaces stabilize.",
        "metrics": {
            "rmsip_top3": overlap,
            "pc1_explained_delta_pct": pc1_delta,
            "first_half_pc1_pct": float(first["explained_pct"][0]),
            "second_half_pc1_pct": float(second["explained_pct"][0]),
        },
    }


def compute_pca(case_id, meta, stride, stage6_cases):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))

    print(f"\n{'=' * 60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'=' * 60}")

    require_file(topology, f"topology.pdb for {case_id}")
    
    all_outputs = []
    pocket_resids = []

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)
        ref = robust_universe(topology, traj_path)
        ref.trajectory[0]

        if not pocket_resids:
            pocket_resids = _detect_pocket_resids(u)

        outputs = {}
        for mode, atom_group in [
            ("global", u.select_atoms("protein and name CA")),
            (
                "pocket",
                u.select_atoms(
                    "protein and name CA and (" + " or ".join(f"resid {rid}" for rid in pocket_resids) + ")"
                ) if pocket_resids else None,
            ),
        ]:
            if atom_group is None or len(atom_group) == 0:
                outputs[mode] = None
                continue
            coords, times = _collect_coords(u, atom_group, ref, stride)
            pca_data = _pca_from_coords(coords, times)
            pca_data["convergence"] = _windowed_convergence(coords, times)
            outputs[mode] = pca_data
        
        all_outputs.append(outputs)

    if not all_outputs:
        return None

    final_outputs = {}
    for mode in ["global", "pocket"]:
        valid_outputs = [out[mode] for out in all_outputs if out[mode] is not None]
        if not valid_outputs:
            final_outputs[mode] = None
            continue

        all_exp = [out["explained_pct"] for out in valid_outputs]
        all_cum = [out["cumulative_pct"] for out in valid_outputs]
        all_eigenvals = [out["eigenvalues"] for out in valid_outputs]
        
        min_len = min(len(x) for x in all_exp)
        
        exp_arr = np.array([x[:min_len] for x in all_exp])
        cum_arr = np.array([x[:min_len] for x in all_cum])
        eig_arr = np.array([x[:min_len] for x in all_eigenvals])

        mean_exp = np.mean(exp_arr, axis=0)
        std_exp = np.std(exp_arr, axis=0)
        mean_cum = np.mean(cum_arr, axis=0)
        std_cum = np.std(cum_arr, axis=0)
        mean_eig = np.mean(eig_arr, axis=0)

        final_data = {
            "eigenvalues": mean_eig,
            "explained_pct": mean_exp,
            "explained_pct_std": std_exp,
            "cumulative_pct": mean_cum,
            "cumulative_pct_std": std_cum,
            "projections": valid_outputs[0]["projections"],
            "times": valid_outputs[0]["times"],
            "n_pcs": valid_outputs[0]["n_pcs"],
            "convergence": valid_outputs[0]["convergence"]
        }
        final_outputs[mode] = final_data

        eigen_csv = os.path.join(OUT_DIR, f"{case_id}_eigenvalues_{mode}.csv")
        with open(eigen_csv, "w") as f:
            f.write("pc_index,eigenvalue,explained_variance_pct_mean,explained_variance_pct_std,cumulative_pct_mean,cumulative_pct_std\n")
            for i in range(min(N_PCS, len(mean_eig))):
                f.write(
                    f"{i+1},{mean_eig[i]:.6f},{mean_exp[i]:.4f},{std_exp[i]:.4f},{mean_cum[i]:.4f},{std_cum[i]:.4f}\n"
                )

    summary_payload = make_metric_summary(
        "pca",
        case_id,
        {
            "pocket_residue_count": len(pocket_resids),
            "global_pc1_pct": float(final_outputs["global"]["explained_pct"][0]) if final_outputs["global"] is not None else None,
            "pocket_pc1_pct": float(final_outputs["pocket"]["explained_pct"][0]) if final_outputs["pocket"] is not None else None,
        },
        derived_metrics={
            mode: (
                {
                    "pc1_pct": float(data["explained_pct"][0]),
                    "pc1_3_pct": float(data["cumulative_pct"][min(2, len(data["cumulative_pct"]) - 1)]),
                }
                if data is not None
                else None
            )
            for mode, data in final_outputs.items()
        },
        interpretation=[],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_pca_summary.json")
    write_json(summary_path, summary_payload)

    return {"label": meta["label"], "outputs": final_outputs, "summary_path": summary_path}



def main():
    parser = argparse.ArgumentParser(description="PCA analysis — 10ns_direct production")
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
            pool.submit(compute_pca, case_id, meta, stride, stage6_cases): case_id
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

    for mode in ["pocket", "global"]:
        fig, axes = plt.subplots(2, 2, figsize=(7, 6))
        axes_flat = axes.flatten()
        for idx, case_id in enumerate(CASE_META):
            ax = axes_flat[idx]
            if case_id not in results or results[case_id]["outputs"][mode] is None:
                ax.text(0.5, 0.5, f"{case_id}\n(no data)", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(CASE_META[case_id]["label"], fontsize=7)
                continue
            proj = results[case_id]["outputs"][mode]["projections"]
            if proj.shape[1] >= 2:
                ax.hexbin(proj[:, 0], proj[:, 1], gridsize=25, cmap="viridis", mincnt=1)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
            else:
                ax.text(0.5, 0.5, "< 2 PCs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(CASE_META[case_id]["label"], fontsize=7)
        fig.suptitle(f"{mode.capitalize()} PCA — PC1 vs PC2", fontsize=8, fontweight="bold")
        fig.tight_layout()
        stem = os.path.join(OUT_DIR, f"pca_{mode}_landscape")
        save_pub_figure(fig, stem)
        print(f"\n{mode.capitalize()} landscape saved: {stem}.png / .pdf")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax_idx, mode in enumerate(["global", "pocket"]):
        ax = axes[ax_idx]
        for case_id in CASE_META:
            if case_id not in results or results[case_id]["outputs"][mode] is None:
                continue
            meta = CASE_META[case_id]
            pca_data = results[case_id]["outputs"][mode]
            n_show = min(N_PCS, len(pca_data["cumulative_pct"]))
            x_vals = np.arange(1, n_show + 1)
            y_vals = pca_data["cumulative_pct"][:n_show]
            y_err = pca_data["cumulative_pct_std"][:n_show]
            
            ax.plot(
                x_vals,
                y_vals,
                color=meta["color"],
                linestyle=meta["linestyle"],
                marker=meta["marker"],
                markersize=3,
                lw=1.2,
                label=meta["label"],
            )
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, color=meta["color"], alpha=0.2)
        ax.set_xlabel("PC index")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title(f"{mode.capitalize()} PCA", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=5, loc="lower right")
    fig.tight_layout()
    var_stem = os.path.join(OUT_DIR, "pca_explained_variance")
    save_pub_figure(fig, var_stem)
    print(f"Explained variance saved: {var_stem}.png / .pdf")

    print(f"\n{'=' * 90}")
    print(f"{'Case':<18} {'Mode':<8} {'PC1%':>8} {'PC1-3%':>8} {'RMSIP':>8} {'dPC1%':>8}")
    print(f"{'-' * 90}")
    for case_id in CASE_META:
        if case_id not in results:
            continue
        for mode in ["global", "pocket"]:
            data = results[case_id]["outputs"][mode]
            if data is None:
                continue
            conv = data["convergence"]["metrics"]
            pc1 = data["explained_pct"][0]
            pc3 = data["cumulative_pct"][min(2, len(data["cumulative_pct"]) - 1)]
            print(
                f"{case_id:<18} {mode:<8} {pc1:>7.1f}% {pc3:>7.1f}% "
                f"{(conv['rmsip_top3'] if conv['rmsip_top3'] is not None else float('nan')):>8.3f} "
                f"{(conv['pc1_explained_delta_pct'] if conv['pc1_explained_delta_pct'] is not None else float('nan')):>8.2f}"
            )
    print(f"{'=' * 90}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "pca_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "pca",
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