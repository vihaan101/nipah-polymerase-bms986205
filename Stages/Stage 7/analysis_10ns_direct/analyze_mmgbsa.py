"""\nStage 7 — Single-trajectory MM-GBSA binding energy (10ns_direct production)\nStrips explicit water, rebuilds with implicit solvent (OBC2), computes\nE_complex - E_receptor - E_ligand per frame.\n"""

import argparse
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

import openmm as mm
from openmm import app, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule

from stage7_eval_common import (robust_universe, get_replicate_trajectory, CASE_META, load_stage6_cases, resolve_stage6_path, find_trajectory, require_file, PROJECT_ROOT, apply_pub_style, save_pub_figure, MARKER_EVERY, sampled_cumulative_time_ns, resolve_stage6_case_path)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    bootstrap_mean_ci,
    criterion_result,
    late_window_metrics,
    block_average,
    make_metric_summary,
    series_summary,
    split_window_drift,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "mmgbsa_10ns_direct")

STRIDE = 25
KJ_TO_KCAL = 1.0 / 4.184
WORKER_TIMEOUT_S = 5 * 3600  # 5 h per drug group; GPU energy eval is slow


def _block_size(n_points):
    if n_points <= 0:
        return 1
    return max(5, n_points // 4)


def _write_stripped_pdb(u, selection_str, out_path):
    """Write a PDB containing only the selected atoms."""
    sel = u.select_atoms(selection_str)
    sel.write(out_path)
    return out_path


def _build_context(pdb_path, forcefield):
    """Build an OpenMM Context from a stripped PDB and ForceField, preferring GPU."""
    pdb = app.PDBFile(pdb_path)
    system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    for platform_name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
            context = mm.Context(system, integrator, platform)
            return context, pdb
        except Exception:
            continue
    raise RuntimeError("No suitable OpenMM platform found")



def compute_mmgbsa(case_id, meta, stride, stage6_cases, ff=None):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))
    ligand_sdf = str(resolve_stage6_case_path(case_id, "ligand_repaired.sdf"))

    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    require_file(topology, f"topology.pdb for {case_id}")
    require_file(ligand_sdf, f"ligand_repaired.sdf for {case_id}")

    all_dg_values = []
    all_times = []
    all_means = []

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue
            
        u = robust_universe(topology, traj_path)
        n_frames = len(u.trajectory)

        STRIP_SEL = "not (resname HOH or resname WAT or resname NA or resname CL or resname ZN)"
        complex_sel = u.select_atoms(STRIP_SEL)
        receptor_sel = u.select_atoms("protein")
        ligand_sel = u.select_atoms("resname UNK")

        if len(ligand_sel) == 0:
            print(f"  WARNING: no ligand found, skipping rep {rep_id}.")
            continue

        with tempfile.TemporaryDirectory(prefix=f"mmgbsa_{case_id}_rep{rep_id}_") as tmpdir:
            u.trajectory[0]
            complex_pdb = _write_stripped_pdb(u, STRIP_SEL, os.path.join(tmpdir, "complex.pdb"))
            receptor_pdb = _write_stripped_pdb(u, "protein", os.path.join(tmpdir, "receptor.pdb"))
            ligand_pdb = _write_stripped_pdb(u, "resname UNK", os.path.join(tmpdir, "ligand.pdb"))

            if ff is None:
                lig_mol = Molecule.from_file(ligand_sdf)
                gaff = GAFFTemplateGenerator(molecules=[lig_mol])
                ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
                ff.registerTemplateGenerator(gaff.generator)

            ctx_complex, _ = _build_context(complex_pdb, ff)
            ctx_receptor, _ = _build_context(receptor_pdb, ff)
            ctx_ligand, _ = _build_context(ligand_pdb, ff)

            complex_idx = complex_sel.indices
            receptor_idx = receptor_sel.indices
            ligand_idx = ligand_sel.indices

            dg_values = []
            times_ns = []

            sampled_times_ns = sampled_cumulative_time_ns(u, stride)
            for frame_i, (time_ns, ts) in enumerate(zip(sampled_times_ns, u.trajectory[::stride])):
                all_pos = u.atoms.positions

                complex_pos = all_pos[complex_idx] * 0.1
                receptor_pos = all_pos[receptor_idx] * 0.1
                ligand_pos = all_pos[ligand_idx] * 0.1

                ctx_complex.setPositions(complex_pos)
                ctx_receptor.setPositions(receptor_pos)
                ctx_ligand.setPositions(ligand_pos)

                e_complex = ctx_complex.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                e_receptor = ctx_receptor.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
                e_ligand = ctx_ligand.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

                dg = (e_complex - e_receptor - e_ligand) * KJ_TO_KCAL
                dg_values.append(dg)
                times_ns.append(time_ns)

            dg_values = np.array(dg_values)
            times_ns = np.array(times_ns)
            
            all_dg_values.append(dg_values)
            all_times.append(times_ns)
            all_means.append(dg_values.mean())
            
            csv_path = os.path.join(OUT_DIR, f"{case_id}_mmgbsa_rep{rep_id}.csv")
            with open(csv_path, "w") as f:
                f.write("time_ns,dG_bind_kcal_mol\n")
                for t, dg in zip(times_ns, dg_values):
                    f.write(f"{t:.4f},{dg:.4f}\n")

    if not all_dg_values:
        return None

    min_len = min(len(t) for t in all_times)
    times_ns = np.array(all_times[0][:min_len])
    dg_arr = np.array([dg[:min_len] for dg in all_dg_values])
    
    mean_dg_series = dg_arr.mean(axis=0)
    std_dg_series = dg_arr.std(axis=0)
    
    overall_mean = np.mean(all_means)
    overall_sem = np.std(all_means) / np.sqrt(len(all_means))

    print(f"  ΔG_bind — overall mean: {overall_mean:.2f} kcal/mol  SEM: {overall_sem:.2f}")

    csv_path = os.path.join(OUT_DIR, f"{case_id}_mmgbsa.csv")
    with open(csv_path, "w") as f:
        f.write("time_ns,dG_bind_kcal_mol_mean,dG_bind_kcal_mol_std\n")
        for t, m, s in zip(times_ns, mean_dg_series, std_dg_series):
            f.write(f"{t:.4f},{m:.4f},{s:.4f}\n")

    summary_payload = make_metric_summary(
        "mmgbsa",
        case_id,
        {
            "overall_mean": overall_mean,
            "overall_sem": overall_sem,
            "replicates": len(all_means)
        },
        derived_metrics={},
        interpretation=[],
        limitations=[]
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_mmgbsa_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "times_ns": times_ns,
        "dg": mean_dg_series,
        "dg_std": std_dg_series,
        "mean": overall_mean,
        "std": overall_sem,
        "summary_path": summary_path,
    }



def _process_drug_group(case_items, stride, stage6_cases):
    """Process cases sharing the same ligand, building ForceField once (6b optimization)."""
    group_results = {}
    group_errors = []
    ff = None
    for case_id, meta in case_items:
        try:
            # Build FF from first case's ligand (shared within drug group)
            if ff is None:
                ligand_sdf = str(resolve_stage6_case_path(case_id, "ligand_repaired.sdf"))
                require_file(ligand_sdf, f"ligand_repaired.sdf for {case_id}")
                print(f"  Building ForceField for {meta['drug']}...")
                lig_mol = Molecule.from_file(ligand_sdf)
                gaff = GAFFTemplateGenerator(molecules=[lig_mol])
                ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
                ff.registerTemplateGenerator(gaff.generator)
            data = compute_mmgbsa(case_id, meta, stride, stage6_cases, ff=ff)
            if data is not None:
                group_results[case_id] = data
        except Exception as e:
            print(f"  ERROR: {case_id} skipped: {e}")
            group_errors.append((case_id, str(e)))
    return group_results, group_errors


def main():
    parser = argparse.ArgumentParser(description="MM-GBSA binding energy — 10ns_direct production")
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

    # Group cases by drug for ForceField reuse (6b) and parallelize groups (7a)
    drug_groups = defaultdict(list)
    for case_id, meta in cases_to_run.items():
        drug_groups[meta["drug"]].append((case_id, meta))

    results = {}
    skipped = []
    with ProcessPoolExecutor(max_workers=len(drug_groups)) as pool:
        futures = {pool.submit(_process_drug_group, group_cases, stride, stage6_cases): drug
                   for drug, group_cases in drug_groups.items()}
        try:
            for future in as_completed(futures, timeout=WORKER_TIMEOUT_S):
                drug = futures[future]
                try:
                    group_results, group_errors = future.result()
                    results.update(group_results)
                    skipped.extend(group_errors)
                except Exception as e:
                    for case_id, _ in drug_groups[drug]:
                        print(f"  ERROR: {case_id} skipped: {e}", flush=True)
                        skipped.append((case_id, str(e)))
        except FutureTimeoutError:
            for future, drug in futures.items():
                if not future.done():
                    for case_id, _ in drug_groups[drug]:
                        print(f"  TIMEOUT: {case_id} worker exceeded {WORKER_TIMEOUT_S}s — cancelling", flush=True)
                        skipped.append((case_id, f"worker timeout after {WORKER_TIMEOUT_S}s"))
                    future.cancel()

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Time-series plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 3.5))

    for case_id, data in results.items():
        meta = CASE_META[case_id]
        ax.plot(data["times_ns"], data["dg"],
                color=meta["color"], linestyle=meta["linestyle"],
                marker=meta["marker"], markevery=MARKER_EVERY, markersize=3,
                lw=1.2, label=data["label"])
        ax.fill_between(data["times_ns"], data["dg"] - data["dg_std"], data["dg"] + data["dg_std"], color=meta["color"], alpha=0.2)

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"$\Delta G_{\mathrm{bind}}$ (kcal/mol)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(left=0)

    fig.tight_layout()
    ts_stem = os.path.join(OUT_DIR, "mmgbsa_timeseries")
    save_pub_figure(fig, ts_stem)
    print(f"\nTime-series plot saved: {ts_stem}.png / .pdf")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.5, 3))

    case_ids = list(results.keys())
    means = [results[c]["mean"] for c in case_ids]
    stds = [results[c]["std"] for c in case_ids]
    colors = [CASE_META[c]["color"] for c in case_ids]
    labels = [results[c]["label"] for c in case_ids]

    bars = ax.bar(range(len(case_ids)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(case_ids)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta G_{\mathrm{bind}}$ (kcal/mol)")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m - s - 2, f"{m:.1f}", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    bar_stem = os.path.join(OUT_DIR, "mmgbsa_bar")
    save_pub_figure(fig, bar_stem)
    print(f"Bar chart saved: {bar_stem}.png / .pdf")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Case':<18} {'Mean':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    for case_id, data in results.items():
        dg = data["dg"]
        print(f"{case_id:<18} {dg.mean():>10.2f}  {dg.std():>9.2f}  {dg.min():>9.2f}  {dg.max():>9.2f}  kcal/mol")
    print(f"{'='*70}")
    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "mmgbsa_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "mmgbsa",
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