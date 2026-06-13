# -*- coding: utf-8 -*-
"""
make_comparison_figures.py — 4-case comparison figures from downloaded eval results.

Reads CSVs/JSONs from eval_results_10ns_direct/<CASE>/<metric_10ns_direct>/.
No MDAnalysis needed — pandas + matplotlib only.

Usage:
    python make_comparison_figures.py [--results-dir PATH]

Output:  eval_results_10ns_direct/comparison_figures/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Style constants (mirrors stage7_eval_common.py) ─────────────────────────
CASE_META = {
    "A_ERDRP_WT":  {"label": "A: ERDRP-0519 / WT (Control)",     "color": "#0072B2", "linestyle": "-",  "marker": "o"},
    "B_ERDRP_MUT": {"label": "B: ERDRP-0519 / W730A (Failure)",   "color": "#E69F00", "linestyle": "--", "marker": "s"},
    "C_BMS_WT":    {"label": "C: BMS-986205 / WT (Success)",      "color": "#009E73", "linestyle": "-.", "marker": "^"},
    "D_BMS_MUT":   {"label": "D: BMS-986205 / W730A (Resistance)","color": "#CC79A7", "linestyle": ":",  "marker": "D"},
}
CASES = list(CASE_META.keys())
PUB_DPI = 600
LINE_W = 1.5   # pt — visible at 600 DPI without looking heavy
FILL_A = 0.12  # std-band alpha — low enough not to clog overlapping bands


def apply_pub_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "lines.antialiased": True,
        "figure.dpi": 150,    # screen preview DPI (keep low — savefig overrides)
        "savefig.dpi": PUB_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_pub_figure(fig, path_stem):
    for ext in (".png", ".pdf"):
        fig.savefig(path_stem + ext, dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def _load(results_dir, case, subdir, filename, index_col=None, comment=None):
    path = results_dir / case / subdir / filename
    if not path.exists():
        return None
    try:
        kwargs = {}
        if index_col is not None:
            kwargs["index_col"] = index_col
        if comment is not None:
            kwargs["comment"] = comment
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f"  WARNING: {path.name}: {e}")
        return None


def _short_label(case):
    return CASE_META[case]["label"].split(" (")[0]


# ── 1. Backbone RMSD time series ─────────────────────────────────────────────
def plot_backbone_rmsd(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for case in CASES:
        df = _load(results_dir, case, "rmsd_10ns_direct", f"{case}_backbone_rmsd.csv")
        if df is None:
            continue
        m = CASE_META[case]
        t = df["time_ns"]
        mean, std = df["backbone_rmsd_A_mean"], df["backbone_rmsd_A_std"]
        ax.plot(t, mean, color=m["color"], linestyle=m["linestyle"], lw=LINE_W, label=m["label"])
        ax.fill_between(t, mean - std, mean + std, color=m["color"], alpha=FILL_A)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Backbone RMSD (Å)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "backbone_rmsd_overlay"))
    print("  backbone_rmsd_overlay.png")


# ── 2. Ligand RMSD time series ───────────────────────────────────────────────
def plot_ligand_rmsd(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for case in CASES:
        df = _load(results_dir, case, "rmsd_10ns_direct", f"{case}_ligand_rmsd.csv")
        if df is None:
            continue
        m = CASE_META[case]
        t = df["time_ns"]
        mean, std = df["ligand_rmsd_A_mean"], df["ligand_rmsd_A_std"]
        ax.plot(t, mean, color=m["color"], linestyle=m["linestyle"], lw=LINE_W, label=m["label"])
        ax.fill_between(t, mean - std, mean + std, color=m["color"], alpha=FILL_A)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Ligand RMSD (Å)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "ligand_rmsd_overlay"))
    print("  ligand_rmsd_overlay.png")


# ── 3. RMSF per-residue ──────────────────────────────────────────────────────
def plot_rmsf(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for case in CASES:
        df = _load(results_dir, case, "rmsf_10ns_direct", f"{case}_rmsf.csv")
        if df is None:
            continue
        m = CASE_META[case]
        resids = df["resid"]
        mean, std = df["rmsf_A_mean"], df["rmsf_A_std"]
        ax.plot(resids, mean, color=m["color"], linestyle=m["linestyle"], lw=LINE_W, label=m["label"])
        ax.fill_between(resids, mean - std, mean + std, color=m["color"], alpha=FILL_A)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Residue ID")
    ax.set_ylabel("RMSF (Å)")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "rmsf_overlay"))
    print("  rmsf_overlay.png")


# ── 4. MM-GBSA time series ────────────────────────────────────────────────────
def plot_mmgbsa_timeseries(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for case in CASES:
        df = _load(results_dir, case, "mmgbsa_10ns_direct", f"{case}_mmgbsa.csv")
        if df is None:
            continue
        m = CASE_META[case]
        # downsample if very dense
        stride = max(1, len(df) // 500)
        df = df.iloc[::stride]
        t = df["time_ns"]
        mean, std = df["dG_bind_kcal_mol_mean"], df["dG_bind_kcal_mol_std"]
        ax.plot(t, mean, color=m["color"], linestyle=m["linestyle"], lw=LINE_W, label=m["label"])
        ax.fill_between(t, mean - std, mean + std, color=m["color"], alpha=FILL_A)
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("ΔG bind (kcal/mol)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "mmgbsa_timeseries"))
    print("  mmgbsa_timeseries.png")


# ── 5. MM-GBSA bar ────────────────────────────────────────────────────────────
def plot_mmgbsa_bar(results_dir, out_dir):
    items = []
    for case in CASES:
        df = _load(results_dir, case, "mmgbsa_10ns_direct", f"{case}_mmgbsa.csv")
        if df is None:
            continue
        m = CASE_META[case]
        n = len(df)
        items.append(dict(
            case=case, label=m["label"], color=m["color"],
            mean=df["dG_bind_kcal_mol_mean"].mean(),
            sem=df["dG_bind_kcal_mol_std"].mean() / np.sqrt(max(n, 1)),
        ))
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(items))
    bars = ax.bar(x, [it["mean"] for it in items],
                  yerr=[it["sem"] for it in items],
                  color=[it["color"] for it in items],
                  capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([it["label"].split(":")[0] for it in items])
    ax.set_ylabel("ΔG bind (kcal/mol)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=it["color"]) for it in items]
    ax.legend(handles, [it["label"] for it in items], fontsize=6, loc="lower right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "mmgbsa_bar"))
    print("  mmgbsa_bar.png")


# ── 6. Pocket volume time series ──────────────────────────────────────────────
def plot_pocket_volume_timeseries(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(10, 4))
    any_data = False
    for case in CASES:
        df = _load(results_dir, case, "pocket_volume_10ns_direct", f"{case}_pocket_volume.csv")
        if df is None:
            continue
        m = CASE_META[case]
        stride = max(1, len(df) // 500)
        df = df.iloc[::stride]
        # Support both old (volume_A3) and new (volume_A3_mean) CSV formats
        if "volume_A3_mean" in df.columns:
            mean = df["volume_A3_mean"]
            std = df["volume_A3_std"]
            ax.plot(df["time_ns"], mean, color=m["color"], linestyle=m["linestyle"],
                    lw=LINE_W, label=m["label"])
            ax.fill_between(df["time_ns"], mean - std, mean + std, color=m["color"], alpha=FILL_A)
        else:
            ax.plot(df["time_ns"], df["volume_A3"],
                    color=m["color"], linestyle=m["linestyle"], lw=LINE_W, label=m["label"])
        any_data = True
    if not any_data:
        plt.close(fig); return
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Pocket volume (Å³)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "pocket_volume_timeseries"))
    print("  pocket_volume_timeseries.png")


# ── 7. Pocket volume bar ──────────────────────────────────────────────────────
def plot_pocket_volume_bar(results_dir, out_dir):
    items = []
    for case in CASES:
        df = _load(results_dir, case, "pocket_volume_10ns_direct", f"{case}_pocket_volume.csv")
        if df is None:
            continue
        m = CASE_META[case]
        if "volume_A3_mean" in df.columns:
            mean_val = float(df["volume_A3_mean"].mean())
            std_val = float(df["volume_A3_std"].mean())
        else:
            mean_val = float(df["volume_A3"].mean())
            std_val = float(df["volume_A3"].std())
        items.append(dict(label=m["label"], color=m["color"], mean=mean_val, std=std_val))
    if not items:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(items))
    ax.bar(x, [it["mean"] for it in items],
           yerr=[it["std"] for it in items],
           color=[it["color"] for it in items],
           capsize=4, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([it["label"].split(":")[0] for it in items])
    ax.set_ylabel("Pocket volume (Å³)")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    handles = [plt.Rectangle((0, 0), 1, 1, color=it["color"]) for it in items]
    ax.legend(handles, [it["label"] for it in items], fontsize=6, loc="upper right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "pocket_volume_bar"))
    print("  pocket_volume_bar.png")


# ── 8. H-bond union heatmap ───────────────────────────────────────────────────
def plot_hbond_comparison(results_dir, out_dir):
    all_dfs = {}
    for case in CASES:
        df = _load(results_dir, case, "hbond_10ns_direct", f"{case}_hbonds.csv")
        if df is not None:
            all_dfs[case] = df

    if not all_dfs:
        return

    # Union of bonds with max occupancy across any case
    bond_max = {}
    for df in all_dfs.values():
        for _, row in df.iterrows():
            bond = row["donor_acceptor"]
            bond_max[bond] = max(bond_max.get(bond, 0.0), float(row["occupancy_pct"]))

    top_bonds = sorted(bond_max, key=lambda b: bond_max[b], reverse=True)[:15]

    matrix = np.zeros((len(top_bonds), len(CASES)))
    for ci, case in enumerate(CASES):
        if case not in all_dfs:
            continue
        df = all_dfs[case].set_index("donor_acceptor")
        for bi, bond in enumerate(top_bonds):
            if bond in df.index:
                matrix[bi, ci] = float(df.loc[bond, "occupancy_pct"])

    fig_h = max(4.0, len(top_bonds) * 0.38)
    fig, ax = plt.subplots(figsize=(5, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([_short_label(c) for c in CASES], rotation=25, ha="right", fontsize=6)
    ax.set_yticks(range(len(top_bonds)))
    ax.set_yticklabels(top_bonds, fontsize=6)
    for ci in range(len(CASES)):
        for bi in range(len(top_bonds)):
            val = matrix[bi, ci]
            if val >= 5:
                ax.text(ci, bi, f"{val:.0f}%", ha="center", va="center",
                        fontsize=5, color="black" if val < 65 else "white")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Occupancy (%)", fontsize=7)
    ax.set_title("H-bond occupancy — top 15 across all cases", fontsize=8)
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "hbond_union_heatmap"))
    print("  hbond_union_heatmap.png")


# ── 9. PLIF comparison heatmap ────────────────────────────────────────────────
def plot_plif_comparison(results_dir, out_dir):
    all_dfs = {}
    for case in CASES:
        df = _load(results_dir, case, "plif_10ns_direct", f"{case}_plif_frequencies.csv", comment="#")
        if df is not None:
            all_dfs[case] = df

    if not all_dfs:
        return

    combined = pd.concat(all_dfs.values(), keys=all_dfs.keys(), names=["case"]).reset_index(level=0)
    combined["key"] = combined["residue"] + " | " + combined["interaction_type"]

    key_max = combined.groupby("key")["frequency_pct"].max()
    top_keys = key_max.nlargest(25).index.tolist()

    matrix = np.zeros((len(top_keys), len(CASES)))
    for ci, case in enumerate(CASES):
        if case not in all_dfs:
            continue
        df = all_dfs[case].copy()
        df["key"] = df["residue"] + " | " + df["interaction_type"]
        df_idx = df.set_index("key")
        for ki, key in enumerate(top_keys):
            if key in df_idx.index:
                matrix[ki, ci] = float(df_idx.loc[key, "frequency_pct"])

    fig_h = max(5.0, len(top_keys) * 0.32)
    fig, ax = plt.subplots(figsize=(5, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([_short_label(c) for c in CASES], rotation=25, ha="right", fontsize=6)
    ax.set_yticks(range(len(top_keys)))
    ax.set_yticklabels(top_keys, fontsize=5)
    for ci in range(len(CASES)):
        for ki in range(len(top_keys)):
            val = matrix[ki, ci]
            if val >= 5:
                ax.text(ci, ki, f"{val:.0f}", ha="center", va="center",
                        fontsize=5, color="black" if val < 60 else "white")
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Frequency (%)", fontsize=7)
    ax.set_title("PLIF interaction frequencies — top 25 across all cases", fontsize=8)
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "plif_comparison"))
    print("  plif_comparison.png")


# ── 10. Contact maps (4-panel side by side) ───────────────────────────────────
def plot_contacts_grid(results_dir, out_dir):
    dfs = {}
    for case in CASES:
        df = _load(results_dir, case, "contacts_10ns_direct", f"{case}_pocket_contacts.csv", index_col=0)
        if df is not None:
            df.index = df.index.astype(str)
            df.columns = df.columns.astype(str)
            dfs[case] = df

    if not dfs:
        return

    # Union residue set
    all_resids = sorted(
        set().union(*[set(df.index.tolist()) for df in dfs.values()]),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    n = len(all_resids)
    idx_map = {r: i for i, r in enumerate(all_resids)}

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=True)
    im = None
    for ax, case in zip(axes, CASES):
        mat = np.zeros((n, n))
        if case in dfs:
            df = dfs[case]
            for ri in df.index:
                if ri not in idx_map:
                    continue
                for ci in df.columns:
                    if ci not in idx_map:
                        continue
                    mat[idx_map[ri], idx_map[ci]] = float(df.loc[ri, ci])
        im = ax.imshow(mat, cmap="hot_r", vmin=0, vmax=1, aspect="auto")
        ax.set_title(_short_label(case), fontsize=6)
        tick_step = max(1, n // 6)
        ax.set_xticks(range(0, n, tick_step))
        ax.set_xticklabels(all_resids[::tick_step], rotation=90, fontsize=5)
        if ax is axes[0]:
            ax.set_yticks(range(0, n, tick_step))
            ax.set_yticklabels(all_resids[::tick_step], fontsize=5)
            ax.set_ylabel("Residue")
    if im is not None:
        cbar = plt.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.04)
        cbar.set_label("Contact frequency", fontsize=7)
    fig.suptitle("Pocket contact maps — all 4 cases", fontsize=8)
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "contact_maps_4case"))
    print("  contact_maps_4case.png")


# ── 11. Per-residue decomposition ─────────────────────────────────────────────
def plot_decomp_comparison(results_dir, out_dir):
    dfs = {}
    resname_map = {}
    for case in CASES:
        df = _load(results_dir, case, "decomp_10ns_direct", f"{case}_perresidue_decomp.csv")
        if df is None:
            continue
        df = df.set_index("resid")
        dfs[case] = df
        if "resname" in df.columns:
            for rid, row in df.iterrows():
                resname_map[rid] = row["resname"]

    if not dfs:
        return

    all_resids = sorted(set().union(*[set(df.index.tolist()) for df in dfs.values()]))
    x = np.arange(len(all_resids))
    bar_width = 0.2

    fig_w = max(8, len(all_resids) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, 4.0))

    for i, case in enumerate(CASES):
        m = CASE_META[case]
        vals, errs = [], []
        for rid in all_resids:
            if case in dfs and rid in dfs[case].index:
                vals.append(float(dfs[case].loc[rid, "mean_dG_kcal_mol"]))
                errs.append(float(dfs[case].loc[rid, "std_dG_kcal_mol"]))
            else:
                vals.append(0.0); errs.append(0.0)
        offset = (i - 1.5) * bar_width
        ax.bar(x + offset, vals, bar_width, yerr=errs, label=m["label"],
               color=m["color"], capsize=2, edgecolor="black", linewidth=0.3,
               error_kw={"linewidth": 0.5})

    ax.set_xticks(x)
    xlabels = [f"{rid}\n{resname_map.get(rid, '')}" for rid in all_resids]
    ax.set_xticklabels(xlabels, fontsize=5)
    ax.set_ylabel("ΔG decomp (kcal/mol)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=6, loc="lower right")
    ax.set_title("Per-residue decomposition — all 4 cases", fontsize=8)
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "decomp_comparison"))
    print("  decomp_comparison.png")


# ── 12. Interaction entropy bar ───────────────────────────────────────────────
def plot_entropy_bar(results_dir, out_dir):
    rows = []
    for case in CASES:
        path = results_dir / case / "entropy_10ns_direct" / "interaction_entropy_summary.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            match = df[df["case_id"] == case] if "case_id" in df.columns else df.iloc[:1]
            if len(match):
                rows.append((case, match.iloc[0]))
        except Exception as e:
            print(f"  WARNING: entropy {case}: {e}")

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(rows))
    bar_width = 0.25
    specs = [
        ("mean_dG_mmgbsa", "MM-GBSA ΔG",  "#4393C3"),
        ("neg_TdS_IE",     "−TΔS (IE)",    "#D6604D"),
        ("dG_corrected",   "ΔG corrected", "#74C476"),
    ]
    for i, (col, lbl, clr) in enumerate(specs):
        vals = [float(row[col]) if col in row.index else 0.0 for _, row in rows]
        ax.bar(x + (i - 1) * bar_width, vals, bar_width,
               label=lbl, color=clr, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([CASE_META[c]["label"].split(":")[0] for c, _ in rows])
    ax.set_ylabel("Energy (kcal/mol)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(fontsize=6)
    ax.set_title("Binding free energy components — all 4 cases", fontsize=8)
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "entropy_bar"))
    print("  entropy_bar.png")


# ── 13. PCA cumulative explained variance ────────────────────────────────────
def plot_pca_variance(results_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    any_data = False
    for scope, ax in zip(["global", "pocket"], axes):
        for case in CASES:
            df = _load(results_dir, case, "pca_10ns_direct", f"{case}_eigenvalues_{scope}.csv")
            if df is None:
                continue
            m = CASE_META[case]
            ax.plot(df["pc_index"], df["cumulative_pct_mean"],
                    color=m["color"], linestyle=m["linestyle"],
                    lw=LINE_W, marker=m["marker"], markersize=3, label=m["label"])
            ax.fill_between(
                df["pc_index"],
                df["cumulative_pct_mean"] - df["cumulative_pct_std"],
                df["cumulative_pct_mean"] + df["cumulative_pct_std"],
                color=m["color"], alpha=FILL_A,
            )
            any_data = True
        ax.axhline(80, color="gray", linestyle=":", linewidth=0.7, alpha=0.7, label="80%")
        ax.set_xlabel("PC index")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title(f"PCA — {scope}", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 105)
    if not any_data:
        plt.close(fig); return
    axes[0].legend(fontsize=6, loc="lower right")
    fig.tight_layout()
    save_pub_figure(fig, str(out_dir / "pca_explained_variance"))
    print("  pca_explained_variance.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate 4-case comparison figures from eval results CSVs")
    parser.add_argument("--results-dir", default=None,
                        help="Path to eval_results_10ns_direct/ (auto-detected if omitted)")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir).expanduser().resolve()
    else:
        # Auto-detect: sibling of analysis_10ns_direct/
        script_dir = Path(__file__).resolve().parent
        results_dir = script_dir.parent / "eval_results_10ns_direct"

    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        sys.exit(1)

    out_dir = results_dir / "comparison_figures"
    out_dir.mkdir(exist_ok=True)

    apply_pub_style()
    print(f"Reading from: {results_dir}")
    print(f"Writing to:   {out_dir}")
    print()

    plot_backbone_rmsd(results_dir, out_dir)
    plot_ligand_rmsd(results_dir, out_dir)
    plot_rmsf(results_dir, out_dir)
    plot_mmgbsa_timeseries(results_dir, out_dir)
    plot_mmgbsa_bar(results_dir, out_dir)
    plot_pocket_volume_timeseries(results_dir, out_dir)
    plot_pocket_volume_bar(results_dir, out_dir)
    plot_hbond_comparison(results_dir, out_dir)
    plot_plif_comparison(results_dir, out_dir)
    plot_contacts_grid(results_dir, out_dir)
    plot_decomp_comparison(results_dir, out_dir)
    plot_entropy_bar(results_dir, out_dir)
    plot_pca_variance(results_dir, out_dir)

    print()
    print(f"Done. {len(list(out_dir.glob('*.png')))} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
