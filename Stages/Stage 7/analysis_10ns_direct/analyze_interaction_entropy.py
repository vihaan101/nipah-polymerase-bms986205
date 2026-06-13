#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""\nStage 7 — Interaction Entropy (IE) correction for MM-GBSA (10ns_direct production)\nEstimates the conformational entropy penalty (-TdS) missing from single-trajectory\nMM-GBSA using the Duan et al. (JACS, 2016) method.\n\nReads existing MM-GBSA per-frame CSVs, no trajectory loading needed.\n\nD_BMS_MUT anomaly filtering: YES — excludes PBC artifact frames via row-index\nmapping (not time matching, due to non-monotonic time resets at DCD tier boundaries).\n"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.special import logsumexp

from stage7_eval_common import (
    CASE_META, apply_pub_style, save_pub_figure, PROJECT_ROOT,
    RMSD_50NS_DIR, MMGBSA_50NS_DIR,
)
from stage7_success_criteria import (
    STATUS_DESCRIPTIVE,
    bootstrap_mean_ci,
    criterion_result,
    make_metric_summary,
    split_window_drift,
    write_json,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "entropy_10ns_direct")

# Existing MM-GBSA and RMSD CSV locations (Azure-aware via stage7_eval_common)
MMGBSA_DIR = str(MMGBSA_50NS_DIR)
RMSD_DIR = str(RMSD_50NS_DIR)

# Physical constants
T = 300.0  # K (simulation temperature)
KB = 0.001987204  # kcal/(mol*K) Boltzmann constant
KT = KB * T  # 0.5961612 kcal/mol
BETA = 1.0 / KT  # 1.6774 mol/kcal

# Stride ratio for row-index mapping: RMSD stride=10, MMGBSA stride=25
RMSD_STRIDE = 10
MMGBSA_STRIDE = 25


def _filter_dbmsmut_artifacts(dg_values):
    """Filter D_BMS_MUT MM-GBSA frames that correspond to PBC wrapping artifacts.\n\n    Uses ROW-INDEX MAPPING instead of time matching because CSV time columns have\n    non-monotonic resets at DCD tier boundaries (Codex finding from eng review).\n\n    RMSD CSV has 1051 rows (stride=10), MMGBSA CSV has 421 rows (stride=25).\n    RMSD row i corresponds to trajectory frame i*10.\n    MMGBSA row j corresponds to trajectory frame j*25.\n    Mapping: mmgbsa_row = floor(rmsd_row * RMSD_STRIDE / MMGBSA_STRIDE)\n    """
    rmsd_csv = os.path.join(RMSD_DIR, "D_BMS_MUT_ligand_rmsd.csv")
    if not os.path.exists(rmsd_csv):
        print("  WARNING: D_BMS_MUT ligand RMSD CSV not found, skipping artifact filter")
        return dg_values, len(dg_values)

    df_rmsd = pd.read_csv(rmsd_csv)
    artifact_rmsd_rows = df_rmsd.index[df_rmsd["ligand_rmsd_A"] > 10.0].tolist()

    if not artifact_rmsd_rows:
        print("  No PBC artifact frames detected in D_BMS_MUT RMSD")
        return dg_values, len(dg_values)

    # Map RMSD row indices to MMGBSA row indices
    artifact_mmgbsa_rows = set()
    for rmsd_row in artifact_rmsd_rows:
        mmgbsa_row = int(rmsd_row * RMSD_STRIDE / MMGBSA_STRIDE)
        if 0 <= mmgbsa_row < len(dg_values):
            artifact_mmgbsa_rows.add(mmgbsa_row)

    n_original = len(dg_values)
    mask = np.ones(n_original, dtype=bool)
    for row in artifact_mmgbsa_rows:
        mask[row] = False

    filtered = dg_values[mask]
    n_filtered = len(filtered)
    print(f"  D_BMS_MUT: filtered {n_original - n_filtered} artifact frames "
          f"({n_original} → {n_filtered})")

    return filtered, n_filtered


def compute_ie(case_id, meta):
    all_neg_tds = []
    all_dg_corrected = []
    
    n_frames = 0
    mean_dg_overall = 0
    std_dg_overall = 0

    for rep_id in range(1, 6):
        mmgbsa_csv = os.path.join(MMGBSA_DIR, f"{case_id}_mmgbsa_rep{rep_id}.csv")
        if not os.path.exists(mmgbsa_csv):
            continue

        df = pd.read_csv(mmgbsa_csv)
        if "dG_bind_kcal_mol" not in df.columns:
            continue
        dg_values = df["dG_bind_kcal_mol"].values

        n_frames = len(dg_values)
        if n_frames == 0:
            continue

        mean_dg = dg_values.mean()
        std_dg = dg_values.std()
        delta = dg_values - mean_dg
        beta_delta = BETA * delta
        neg_tds = KT * (logsumexp(beta_delta) - np.log(n_frames))
        dg_corrected = mean_dg + neg_tds
        
        all_neg_tds.append(neg_tds)
        all_dg_corrected.append(dg_corrected)
        mean_dg_overall += mean_dg
        std_dg_overall += std_dg
        
    if not all_neg_tds:
        return None
        
    neg_tds = np.mean(all_neg_tds)
    dg_corrected = np.mean(all_dg_corrected)
    mean_dg = mean_dg_overall / len(all_neg_tds)
    std_dg = std_dg_overall / len(all_neg_tds)

    summary_payload = make_metric_summary(
        "interaction_entropy",
        case_id,
        {
            "mean_dG_mmgbsa": mean_dg,
            "std_dG": std_dg,
            "neg_TdS_IE": neg_tds,
            "dG_corrected": dg_corrected,
            "n_frames": n_frames,
        },
        derived_metrics={},
        interpretation=[],
        limitations=[],
    )
    summary_path = os.path.join(OUT_DIR, f"{case_id}_interaction_entropy_summary.json")
    write_json(summary_path, summary_payload)

    return {
        "case_id": case_id,
        "label": meta["label"],
        "drug": meta["drug"],
        "receptor": meta["receptor"],
        "mean_dg": mean_dg,
        "std_dg": std_dg,
        "neg_tds": neg_tds,
        "dg_corrected": dg_corrected,
        "n_frames": n_frames,
        "summary_path": summary_path,
    }



def main():
    parser = argparse.ArgumentParser(
        description="Interaction Entropy correction — 10ns_direct production")
    parser.add_argument("--case", type=str, default=None,
                        help="Run only this case ID (e.g. 'C_BMS_WT')")
    args = parser.parse_args()

    cases_to_run = CASE_META
    if args.case:
        if args.case not in CASE_META:
            print(f"FATAL: Unknown case '{args.case}'. Valid: {list(CASE_META.keys())}")
            sys.exit(1)
        cases_to_run = {args.case: CASE_META[args.case]}

    os.makedirs(OUT_DIR, exist_ok=True)
    apply_pub_style()

    # ── Compute IE for each case ─────────────────────────────────────────────
    results = {}
    for case_id, meta in cases_to_run.items():
        try:
            data = compute_ie(case_id, meta)
            if data is not None:
                results[case_id] = data
        except Exception as e:
            print(f"  ERROR: {case_id} failed: {e}")

    if not results:
        print("\nFATAL: No cases processed.")
        sys.exit(1)

    # ── Write summary CSV ────────────────────────────────────────────────────
    summary_csv = os.path.join(OUT_DIR, "interaction_entropy_summary.csv")
    with open(summary_csv, "w") as f:
        f.write("case_id,mean_dG_mmgbsa,std_dG,neg_TdS_IE,dG_corrected,n_frames\n")
        for case_id in CASE_META:
            if case_id in results:
                r = results[case_id]
                f.write(f"{case_id},{r['mean_dg']:.4f},{r['std_dg']:.4f},"
                        f"{r['neg_tds']:.4f},{r['dg_corrected']:.4f},{r['n_frames']}\n")
    print(f"\nSummary CSV saved: {summary_csv}")

    # ── Write ddG corrected CSV ──────────────────────────────────────────────
    ddg_csv = os.path.join(OUT_DIR, "ddG_corrected.csv")
    with open(ddg_csv, "w") as f:
        f.write("drug,dG_WT_corrected,dG_MUT_corrected,ddG_corrected\n")
        for drug in ["ERDRP-0519", "BMS-986205"]:
            wt_case = [c for c, r in results.items() if r["drug"] == drug and r["receptor"] == "WT"]
            mut_case = [c for c, r in results.items() if r["drug"] == drug and r["receptor"] != "WT"]
            if wt_case and mut_case:
                dg_wt = results[wt_case[0]]["dg_corrected"]
                dg_mut = results[mut_case[0]]["dg_corrected"]
                ddg = dg_mut - dg_wt
                f.write(f"{drug},{dg_wt:.4f},{dg_mut:.4f},{ddg:.4f}\n")
    print(f"ddG CSV saved: {ddg_csv}")

    # ── Grouped bar chart: raw vs corrected dG ───────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3.5))

    case_ids = [c for c in CASE_META if c in results]
    n = len(case_ids)
    x = np.arange(n)
    width = 0.35

    raw_means = [results[c]["mean_dg"] for c in case_ids]
    corrected = [results[c]["dg_corrected"] for c in case_ids]
    raw_stds = [results[c]["std_dg"] for c in case_ids]
    colors = [CASE_META[c]["color"] for c in case_ids]
    labels = [results[c]["label"] for c in case_ids]

    bars_raw = ax.bar(x - width / 2, raw_means, width, yerr=raw_stds, capsize=3,
                      color=colors, edgecolor="black", linewidth=0.5,
                      alpha=0.5, label=r"$\langle\Delta G\rangle_{\mathrm{MM-GBSA}}$")
    bars_corr = ax.bar(x + width / 2, corrected, width,
                       color=colors, edgecolor="black", linewidth=0.5,
                       label=r"$\Delta G_{\mathrm{corrected}}$")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(r"$\Delta G$ (kcal/mol)")
    ax.legend(loc="lower left", fontsize=6)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Annotate corrected values
    for i, val in enumerate(corrected):
        ax.text(x[i] + width / 2, val - 2, f"{val:.1f}", ha="center",
                fontsize=6, fontweight="bold")

    fig.tight_layout()
    bar_stem = os.path.join(OUT_DIR, "entropy_comparison_bar")
    save_pub_figure(fig, bar_stem)
    print(f"Bar chart saved: {bar_stem}.png / .pdf")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Case':<18} {'<dG>':>10} {'std':>8} {'-TdS':>10} {'dG_corr':>10} {'N':>6}")
    print(f"{'-'*80}")
    for case_id in CASE_META:
        if case_id in results:
            r = results[case_id]
            print(f"{case_id:<18} {r['mean_dg']:>9.2f}  {r['std_dg']:>7.2f}  "
                  f"{r['neg_tds']:>9.2f}  {r['dg_corrected']:>9.2f}  {r['n_frames']:>5}")
    print(f"{'='*80}")

    # ddG summary
    print(f"\n{'Drug':<15} {'dG_WT_corr':>12} {'dG_MUT_corr':>12} {'ddG_corr':>10}")
    print(f"{'-'*55}")
    for drug in ["ERDRP-0519", "BMS-986205"]:
        wt_case = [c for c, r in results.items() if r["drug"] == drug and r["receptor"] == "WT"]
        mut_case = [c for c, r in results.items() if r["drug"] == drug and r["receptor"] != "WT"]
        if wt_case and mut_case:
            dg_wt = results[wt_case[0]]["dg_corrected"]
            dg_mut = results[mut_case[0]]["dg_corrected"]
            ddg = dg_mut - dg_wt
            print(f"{drug:<15} {dg_wt:>11.2f}  {dg_mut:>11.2f}  {ddg:>9.2f}")
    print(f"{'='*55}")

    print(f"\nAll outputs in: {OUT_DIR}")

    overview_path = os.path.join(OUT_DIR, "interaction_entropy_metric_index.json")
    write_json(
        overview_path,
        {
            "metric_name": "interaction_entropy",
            "case_summaries": {
                case_id: results[case_id]["summary_path"] for case_id in results
            },
        },
    )
    print(f"Metric index saved: {overview_path}")


if __name__ == "__main__":
    main()