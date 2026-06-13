#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 — Per-residue energy decomposition (10ns_direct production)
Decomposes the total MM-GBSA binding energy into pairwise per-residue
contributions. For each pocket residue R near the ligand:
  dG_pair(R) = E(R + ligand, implicit) - E(R alone, implicit) - E(ligand alone, implicit)

This reveals that TRP591 (W730) contributes significantly in WT but ~0 in MUT,
while other residues compensate, explaining why total MM-GBSA doesn't change.

D_BMS_MUT anomaly filtering: YES — ligand positions are invalid during
PBC wrapping artifact frames.

Validation: Sum of per-residue dG is reported vs total MM-GBSA as INFO
(WARNING-only, no FAIL threshold — pairwise implicit-solvent decomposition
has 30-100+ kcal/mol expected discrepancy from many-body/solvation effects).
"""

import argparse
import json
import math
import multiprocessing
import os
import resource
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MDAnalysis as mda

import openmm as mm
from openmm import app, unit
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Molecule

from stage7_eval_common import (robust_universe, 
    get_replicate_trajectory,
    CASE_META, load_stage6_cases, resolve_stage6_path,
    get_replicate_trajectory, require_file, PROJECT_ROOT,
    apply_pub_style, save_pub_figure,
    RMSD_50NS_DIR, MMGBSA_50NS_DIR, resolve_stage6_case_path, sampled_cumulative_time_ns,
)
from stage7_success_criteria import (
    STATUS_CONCERN,
    STATUS_DESCRIPTIVE,
    STATUS_NOT_ASSESSED,
    STATUS_SUPPORTS,
    criterion_result,
    make_metric_summary,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "decomp_10ns_direct")
CHECKPOINT_DIR = os.path.join(OUT_DIR, "_checkpoints")

STRIDE = 25
KJ_TO_KCAL = 1.0 / 4.184
DECOMP_CUTOFF = 6.0  # Angstrom — detect residues within this distance of ligand
MUTATION_RESID = 591  # Always include in decomposition
WORKER_TIMEOUT_S = 7 * 3600  # 7 h per drug group; builds 2N+1 contexts then loops frames
CHECKPOINT_EVERY_FRAMES = max(
    1, int(os.environ.get("STAGE7_DECOMP_CHECKPOINT_EVERY_FRAMES", "10")))
LIVE_SYNC_ROOT = os.environ.get("STAGE7_AZURE_SYNC_DIR", "").strip()

# D_BMS_MUT anomaly filtering (Azure-aware via stage7_eval_common)
RMSD_DIR = str(RMSD_50NS_DIR)
MMGBSA_DIR = str(MMGBSA_50NS_DIR)


def _case_output_paths(case_id):
    return {
        "perresidue_csv": os.path.join(OUT_DIR, f"{case_id}_perresidue_decomp.csv"),
        "timeseries_csv": os.path.join(OUT_DIR, f"{case_id}_decomp_timeseries.csv"),
        "summary_json": os.path.join(OUT_DIR, f"{case_id}_decomp_summary.json"),
    }


def _case_checkpoint_paths(case_id):
    case_dir = os.path.join(CHECKPOINT_DIR, case_id)
    return {
        "case_dir": case_dir,
        "state_json": os.path.join(case_dir, "state.json"),
        "partial_npz": os.path.join(case_dir, "partial_timeseries.npz"),
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
        print(f"  [CHECKPOINT SYNC] {reason} -> {dest}", flush=True)
    except Exception as exc:
        print(f"  [CHECKPOINT SYNC WARNING] {reason}: {exc}", flush=True)


def _load_completed_case_result(case_id):
    paths = _case_output_paths(case_id)
    if not all(os.path.exists(path) for path in paths.values()):
        return None

    perresidue = pd.read_csv(paths["perresidue_csv"])
    with open(paths["summary_json"]) as fh:
        summary_payload = json.load(fh)

    primary = summary_payload.get("primary_output", {})
    return {
        "label": CASE_META[case_id]["label"],
        "decomp_results": [
            {
                "resid": int(row["resid"]),
                "resname": row["resname"],
                "mean_dg": float(row["mean_dG_kcal_mol"]),
                "std_dg": float(row["std_dG_kcal_mol"]),
            }
            for _, row in perresidue.iterrows()
        ],
        "decomp_resids": [int(row["resid"]) for _, row in perresidue.iterrows()],
        "total_sum": float(primary.get("total_sum_dG_kcal_mol", float("nan"))),
        "mmgbsa_total": float(primary.get("mmgbsa_total_kcal_mol", float("nan"))),
        "n_analyzed": int(primary.get("n_frames", 0)),
        "summary_path": paths["summary_json"],
    }


def _write_stripped_pdb(u, selection_str, out_path):
    """Write a PDB containing only the selected atoms."""
    sel = u.select_atoms(selection_str)
    sel.write(out_path)
    return out_path


def _build_context(pdb_path, forcefield):
    """Build an OpenMM Context from a stripped PDB and ForceField, preferring GPU.

    Uses ignoreExternalBonds=True so AMBER14 templates match fragment residues
    that lack inter-residue peptide bonds (e.g. isolated single-residue PDBs
    from per-residue decomposition).
    """
    pdb = app.PDBFile(pdb_path)
    system = forcefield.createSystem(
        pdb.topology, nonbondedMethod=app.NoCutoff, ignoreExternalBonds=True)
    for platform_name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = mm.Platform.getPlatformByName(platform_name)
            integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
            context = mm.Context(system, integrator, platform)
            print(f"      Context on {platform_name} ({pdb_path.split('/')[-1]})", flush=True)
            return context, pdb
        except Exception:
            continue
    raise RuntimeError("No suitable OpenMM platform found")


def _get_artifact_frame_set(case_id):
    """Get set of strided frame indices to skip for D_BMS_MUT.

    Uses row-index mapping: RMSD stride=10, decomp stride=25.
    RMSD row i → trajectory frame i*10. Decomp frame j → trajectory frame j*25.
    Map: decomp_frame = floor(rmsd_row * 10 / 25).
    """
    if case_id != "D_BMS_MUT":
        return set()

    rmsd_csv = os.path.join(RMSD_DIR, "D_BMS_MUT_ligand_rmsd.csv")
    if not os.path.exists(rmsd_csv):
        print("    WARNING: RMSD CSV not found, no artifact filtering")
        return set()

    df = pd.read_csv(rmsd_csv)
    artifact_rows = df.index[df["ligand_rmsd_A"] > 10.0].tolist()

    artifact_frames = set()
    for rmsd_row in artifact_rows:
        decomp_frame = int(rmsd_row * 10 / STRIDE)
        artifact_frames.add(decomp_frame)

    if artifact_frames:
        print(f"    D_BMS_MUT: will skip {len(artifact_frames)} artifact frames")
    return artifact_frames


def compute_decomp(case_id, meta, stride, stage6_cases, ff=None):
    topology = str(resolve_stage6_path(stage6_cases[case_id]["topology_pdb"]))
    ligand_sdf = str(resolve_stage6_case_path(case_id, "ligand_repaired.sdf"))

    print(f"\n{'='*60}")
    print(f"  {case_id}: {meta['label']}")
    print(f"{'='*60}")

    require_file(topology, f"topology.pdb for {case_id}")
    require_file(ligand_sdf, f"ligand_repaired.sdf for {case_id}")

    ckpt_paths = _case_checkpoint_paths(case_id)
    os.makedirs(ckpt_paths["case_dir"], exist_ok=True)

    all_times = []
    all_values_matrices = []
    completed_reps = set()

    if os.path.exists(ckpt_paths["partial_npz"]):
        try:
            saved = np.load(ckpt_paths["partial_npz"], allow_pickle=True)
            n_done = int(saved["n_reps"])
            all_times = [saved[f"times_{i}"] for i in range(n_done)]
            all_values_matrices = [saved[f"values_{i}"] for i in range(n_done)]
            completed_reps = set(range(1, n_done + 1))
            print(f"  Resuming decomp: {n_done}/5 reps already done")
        except Exception as exc:
            print(f"  WARNING: checkpoint load failed ({exc}), starting over")
            all_times = []
            all_values_matrices = []
            completed_reps = set()

    decomp_resids = []
    n_skipped = 0
    n_nan = 0
    artifact_frames = _get_artifact_frame_set(case_id)
    contexts_built = False
    ctx_lig = None
    ctx_res = {}
    ctx_pair = {}
    lig_idx = None
    resid_to_resname = {}
    res_indices = {}
    pair_indices = {}

    for rep_id in range(1, 6):
        try:
            traj_path = str(get_replicate_trajectory(case_id, rep_id))
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            continue

        u = robust_universe(topology, traj_path)

        if not contexts_built:
            u.trajectory[0]
            nearby = u.select_atoms(f"protein and (around {DECOMP_CUTOFF} (resname UNK and not name H*))")
            decomp_resids = sorted(set(nearby.residues.resids))
            if MUTATION_RESID not in decomp_resids:
                decomp_resids.append(MUTATION_RESID)
                decomp_resids.sort()

            if not decomp_resids:
                print(f"  ERROR: no decomposition residues found")
                return None

            for resid in decomp_resids:
                atoms = u.select_atoms(f"resid {resid} and protein and name CA")
                resid_to_resname[resid] = atoms[0].resname if len(atoms) > 0 else "UNK"

            ligand_sel = u.select_atoms("resname UNK")

            with tempfile.TemporaryDirectory(prefix=f"decomp_{case_id}_setup_") as tmpdir:
                lig_pdb = _write_stripped_pdb(u, "resname UNK", os.path.join(tmpdir, "ligand.pdb"))
                lig_idx = ligand_sel.indices

                if ff is None:
                    lig_mol = Molecule.from_file(ligand_sdf)
                    gaff = GAFFTemplateGenerator(molecules=[lig_mol])
                    ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
                    ff.registerTemplateGenerator(gaff.generator)

                ctx_lig, _ = _build_context(lig_pdb, ff)
                for resid in decomp_resids:
                    res_sel_str = f"resid {resid} and protein"
                    pair_sel_str = f"(resid {resid} and protein) or resname UNK"
                    res_pdb = _write_stripped_pdb(u, res_sel_str, os.path.join(tmpdir, f"res_{resid}.pdb"))
                    pair_pdb = _write_stripped_pdb(u, pair_sel_str, os.path.join(tmpdir, f"pair_{resid}.pdb"))
                    res_indices[resid] = u.select_atoms(res_sel_str).indices
                    pair_indices[resid] = u.select_atoms(pair_sel_str).indices
                    ctx_res[resid], _ = _build_context(res_pdb, ff)
                    ctx_pair[resid], _ = _build_context(pair_pdb, ff)

            contexts_built = True

        if rep_id in completed_reps:
            print(f"  Skipping rep {rep_id} (checkpointed)")
            continue

        times_ns = []
        values_matrix = []

        sampled_times_ns = sampled_cumulative_time_ns(u, stride)
        for frame_i, (time_ns, ts) in enumerate(zip(sampled_times_ns, u.trajectory[::stride])):
            if frame_i in artifact_frames:
                n_skipped += 1
                continue

            all_pos = u.atoms.positions

            ctx_lig.setPositions(all_pos[lig_idx] * 0.1)
            e_lig = ctx_lig.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

            if math.isnan(e_lig):
                n_nan += 1
                continue

            frame_has_nan = False
            frame_values = []

            for resid in decomp_resids:
                ctx_res[resid].setPositions(all_pos[res_indices[resid]] * 0.1)
                e_res = ctx_res[resid].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

                ctx_pair[resid].setPositions(all_pos[pair_indices[resid]] * 0.1)
                e_pair = ctx_pair[resid].getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

                if math.isnan(e_res) or math.isnan(e_pair):
                    frame_values.append(float("nan"))
                    if not frame_has_nan:
                        n_nan += 1
                        frame_has_nan = True
                    continue

                dg_pair = (e_pair - e_res - e_lig) * KJ_TO_KCAL
                frame_values.append(dg_pair)

            times_ns.append(time_ns)
            values_matrix.append(frame_values)

        all_times.append(np.array(times_ns))
        all_values_matrices.append(np.array(values_matrix))

        n_done = len(all_times)
        save_dict = {"n_reps": n_done}
        for i in range(n_done):
            save_dict[f"times_{i}"] = all_times[i]
            save_dict[f"values_{i}"] = all_values_matrices[i]
        np.savez(ckpt_paths["partial_npz"], **save_dict)
        _sync_live_outputs(f"{case_id} decomp rep {rep_id}/5")

    if not all_values_matrices:
        return None

    min_len = min(len(t) for t in all_times)
    times_ns = all_times[0][:min_len]
    val_arr = np.array([v[:min_len] for v in all_values_matrices])
    values_matrix = np.nanmean(val_arr, axis=0)
    std_matrix = np.nanstd(val_arr, axis=0)
    
    n_analyzed = len(times_ns)

    per_residue_timeseries = {
        resid: values_matrix[:, idx].tolist() if values_matrix.size else []
        for idx, resid in enumerate(decomp_resids)
    }

    decomp_results = []
    for resid in decomp_resids:
        values = np.array(per_residue_timeseries[resid])
        valid = values[~np.isnan(values)]
        if len(valid) > 0:
            mean_dg = valid.mean()
            std_dg = valid.std()
        else:
            mean_dg = float("nan")
            std_dg = float("nan")

        decomp_results.append({
            "resid": resid,
            "resname": resid_to_resname.get(resid, "UNK"),
            "mean_dg": mean_dg,
            "std_dg": std_dg,
        })

    total_sum = sum(r["mean_dg"] for r in decomp_results if not math.isnan(r["mean_dg"]))

    mmgbsa_csv = os.path.join(MMGBSA_DIR, f"{case_id}_mmgbsa.csv")
    mmgbsa_total = float("nan")
    if os.path.exists(mmgbsa_csv):
        df_mm = pd.read_csv(mmgbsa_csv)
        if "dG_bind_kcal_mol_mean" in df_mm.columns:
            mmgbsa_total = df_mm["dG_bind_kcal_mol_mean"].mean()
        elif "dG_bind_kcal_mol" in df_mm.columns:
            mmgbsa_total = df_mm["dG_bind_kcal_mol"].mean()

    discrepancy = abs(total_sum - mmgbsa_total) if not math.isnan(mmgbsa_total) else float("nan")
    relative_discrepancy = (
        discrepancy / abs(mmgbsa_total)
        if not math.isnan(discrepancy) and not math.isnan(mmgbsa_total) and mmgbsa_total != 0
        else float("nan")
    )

    r591 = [r for r in decomp_results if r["resid"] == MUTATION_RESID]
    paper_specific_hook = None
    if r591:
        r591_dg = r591[0]["mean_dg"]
        r591_resname = r591[0]["resname"]
        ranked = sorted(
            [r for r in decomp_results if not math.isnan(r["mean_dg"])],
            key=lambda row: row["mean_dg"],
        )
        rank_lookup = {row["resid"]: idx + 1 for idx, row in enumerate(ranked)}
        paper_specific_hook = {
            "resid": MUTATION_RESID,
            "resname": r591_resname,
            "mean_dG_kcal_mol": r591_dg,
            "rank_by_favorability": rank_lookup.get(MUTATION_RESID),
        }

    csv_path = os.path.join(OUT_DIR, f"{case_id}_perresidue_decomp.csv")
    with open(csv_path, "w") as f:
        f.write("resid,resname,mean_dG_kcal_mol,std_dG_kcal_mol,pct_of_total\n")
        for r in decomp_results:
            pct = (r["mean_dg"] / total_sum * 100.0
                   if total_sum != 0 and not math.isnan(r["mean_dg"]) else 0)
            f.write(f"{r['resid']},{r['resname']},{r['mean_dg']:.4f},{r['std_dg']:.4f},{pct:.2f}\n")

    ts_csv = os.path.join(OUT_DIR, f"{case_id}_decomp_timeseries.csv")
    with open(ts_csv, "w") as f:
        header = "time_ns," + ",".join(f"resid_{r}" for r in decomp_resids)
        f.write(header + "\n")
        for i, t in enumerate(times_ns):
            vals = []
            for resid in decomp_resids:
                v = per_residue_timeseries[resid][i] if i < len(per_residue_timeseries[resid]) else float("nan")
                vals.append(f"{v:.4f}")
            f.write(f"{t:.4f},{','.join(vals)}\n")

    recovery_status = STATUS_NOT_ASSESSED
    if not math.isnan(discrepancy):
        recovery_status = (
            STATUS_SUPPORTS
            if discrepancy <= 25.0 or (not math.isnan(relative_discrepancy) and relative_discrepancy <= 0.5)
            else STATUS_CONCERN
        )
    summary_payload = make_metric_summary(
        "decomp",
        case_id,
        {
            "n_residues": len(decomp_resids),
            "n_frames": n_analyzed,
            "total_sum_dG_kcal_mol": total_sum,
            "mmgbsa_total_kcal_mol": mmgbsa_total,
        },
        derived_metrics={
            "decomposition_recovery": {
                "absolute_discrepancy_kcal_mol": discrepancy,
                "relative_discrepancy": relative_discrepancy,
            },
            "mutation_site_hook": paper_specific_hook,
            "top_residues": decomp_results[:10],
        },
        interpretation=[
            criterion_result(
                "decomp_recovery",
                recovery_status,
                "Decomposition-total recovery is reported as an approximate internal-consistency check.",
                extra={
                    "absolute_discrepancy_kcal_mol": discrepancy,
                    "relative_discrepancy": relative_discrepancy,
                },
                threshold={"absolute_discrepancy_kcal_mol": 25.0, "relative_discrepancy": 0.5},
                threshold_note="Heuristic tolerance band for approximate pairwise decomposition recovery.",
            )
        ],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_decomp_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "label": meta["label"],
        "decomp_results": decomp_results,
        "decomp_resids": decomp_resids,
        "total_sum": total_sum,
        "mmgbsa_total": mmgbsa_total,
        "n_analyzed": n_analyzed,
        "summary_path": summary_path,
    }



def _process_drug_group(case_items, stride, stage6_cases):
    """Process cases sharing the same ligand, building ForceField once."""
    group_results = {}
    group_errors = []
    ff = None
    for case_id, meta in case_items:
        try:
            if ff is None:
                ligand_sdf = str(resolve_stage6_case_path(case_id, "ligand_repaired.sdf"))
                require_file(ligand_sdf, f"ligand_repaired.sdf for {case_id}")
                print(f"  Building ForceField for {meta['drug']}...")
                lig_mol = Molecule.from_file(ligand_sdf)
                gaff = GAFFTemplateGenerator(molecules=[lig_mol])
                ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
                ff.registerTemplateGenerator(gaff.generator)
            data = compute_decomp(case_id, meta, stride, stage6_cases, ff=ff)
            if data is not None:
                group_results[case_id] = data
        except Exception as e:
            print(f"  ERROR: {case_id} skipped: {e}")
            group_errors.append((case_id, str(e)))
    return group_results, group_errors


def main():
    parser = argparse.ArgumentParser(
        description="Per-residue energy decomposition — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None,
                        help="Run only this case ID (e.g. 'C_BMS_WT')")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Frame stride (default: {STRIDE})")
    args = parser.parse_args()
    if args.case is None:
        args.case = os.environ.get("STAGE7_TARGET_CASE") or None

    # Raise file descriptor limit before spawning workers — each worker opens
    # 2N+1 OpenMM contexts (B_ERDRP_MUT has ~135) which can exceed the default
    # soft limit of 1024. ulimit in the parent shell does not propagate to
    # Python-spawned subprocesses so we set it here programmatically.
    try:
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        _target = min(65536, _hard if _hard != resource.RLIM_INFINITY else 65536)
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
        print(f"  File descriptor limit raised: {_soft} → {_target} (hard: {_hard})")
    except Exception as e:
        print(f"  WARNING: could not raise fd limit: {e}")

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

    # Group by drug for ForceField reuse
    drug_groups = defaultdict(list)
    for case_id, meta in cases_to_run.items():
        drug_groups[meta["drug"]].append((case_id, meta))

    results = {}
    skipped = []
    pending_groups = defaultdict(list)
    for case_id, group_cases in drug_groups.items():
        for case_name, meta in group_cases:
            completed = _load_completed_case_result(case_name)
            if completed is not None:
                print(f"  Reusing completed decomp outputs for {case_name}")
                results[case_name] = completed
            else:
                pending_groups[case_id].append((case_name, meta))

    if pending_groups:
        # Use 'spawn' to avoid CUDA fork-safety issues on Linux (H100 GPU)
        mp_ctx = multiprocessing.get_context("spawn")
        max_workers = int(os.environ.get("STAGE7_DECOMP_MAX_WORKERS", str(len(pending_groups))))
        max_workers = max(1, min(max_workers, len(pending_groups)))
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_ctx) as pool:
            futures = {
                pool.submit(_process_drug_group, group_cases, stride, stage6_cases): drug
                for drug, group_cases in pending_groups.items()
            }
            try:
                for future in as_completed(futures, timeout=WORKER_TIMEOUT_S):
                    drug = futures[future]
                    try:
                        group_results, group_errors = future.result()
                        results.update(group_results)
                        skipped.extend(group_errors)
                    except Exception as e:
                        for case_id, _ in pending_groups[drug]:
                            print(f"  ERROR: {case_id} skipped: {e}", flush=True)
                            skipped.append((case_id, str(e)))
            except FutureTimeoutError:
                for future, drug in futures.items():
                    if not future.done():
                        for case_id, _ in pending_groups[drug]:
                            print(f"  TIMEOUT: {case_id} worker exceeded {WORKER_TIMEOUT_S}s — cancelling", flush=True)
                            skipped.append((case_id, f"worker timeout after {WORKER_TIMEOUT_S}s"))
                        future.cancel()

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Bar chart: per-residue dG comparison, 4 cases side-by-side ───────────
    # Find union of all decomposition residues across cases
    all_resids = sorted(set(
        r for data in results.values() for r in data["decomp_resids"]))

    fig, ax = plt.subplots(figsize=(10, 4))

    case_ids = [c for c in CASE_META if c in results]
    n_cases = len(case_ids)
    n_resids = len(all_resids)
    x = np.arange(n_resids)
    width = 0.8 / max(n_cases, 1)

    for i, case_id in enumerate(case_ids):
        data = results[case_id]
        # Build lookup
        dg_lookup = {r["resid"]: r["mean_dg"] for r in data["decomp_results"]}
        values = [dg_lookup.get(resid, 0) for resid in all_resids]

        ax.bar(x + i * width, values, width,
               color=CASE_META[case_id]["color"],
               label=CASE_META[case_id]["label"],
               edgecolor="black", linewidth=0.3)

    # Highlight mutation site
    if MUTATION_RESID in all_resids:
        mut_idx = all_resids.index(MUTATION_RESID)
        ax.axvspan(mut_idx - 0.5, mut_idx + n_cases * width + 0.5,
                   color="red", alpha=0.1, label=f"Resid {MUTATION_RESID}")

    ax.set_xticks(x + width * (n_cases - 1) / 2)
    ax.set_xticklabels([str(r) for r in all_resids], rotation=90, fontsize=5)
    ax.set_xlabel("Residue ID")
    ax.set_ylabel(r"$\Delta G_{\mathrm{pair}}$ (kcal/mol)")
    ax.set_title("Per-Residue Energy Decomposition", fontsize=8, fontweight="bold")
    ax.legend(fontsize=5, loc="lower left")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    fig.tight_layout()
    bar_stem = os.path.join(OUT_DIR, "decomp_bar_comparison")
    save_pub_figure(fig, bar_stem)
    print(f"\nBar chart saved: {bar_stem}.png / .pdf")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Case':<18} {'Sum dG':>10} {'MM-GBSA':>10} {'Discr.':>10} {'N_res':>6} {'N_frm':>6}")
    print(f"{'-'*80}")
    for case_id in CASE_META:
        if case_id in results:
            d = results[case_id]
            discr = abs(d["total_sum"] - d["mmgbsa_total"]) if not math.isnan(d["mmgbsa_total"]) else float("nan")
            print(f"{case_id:<18} {d['total_sum']:>9.2f}  {d['mmgbsa_total']:>9.2f}  "
                  f"{discr:>9.2f}  {len(d['decomp_resids']):>5}  {d['n_analyzed']:>5}")
    print(f"{'='*80}")

    # TRP591 / ALA591 comparison
    print(f"\nResid {MUTATION_RESID} contributions:")
    print(f"{'Case':<18} {'Resname':<8} {'mean dG':>10}")
    print(f"{'-'*40}")
    for case_id in CASE_META:
        if case_id in results:
            r591 = [r for r in results[case_id]["decomp_results"] if r["resid"] == MUTATION_RESID]
            if r591:
                print(f"{case_id:<18} {r591[0]['resname']:<8} {r591[0]['mean_dg']:>9.2f}")
    print(f"{'='*40}")

    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "decomp_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "decomp",
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