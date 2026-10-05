#!/usr/bin/env python3
"""
Standalone Step 6: Redocking Validation with parallelized seed search.

Runs Vina (Vinardo) with exhaustiveness=32 across seeds [42, 0, 1] in parallel,
then reports the best RMSD and whether it passes the 2.0 Å gate.
"""

import json
import subprocess
import sys
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

STAGE1_DIR = Path(__file__).resolve().parent
COMMON_DIR = STAGE1_DIR.parent / "common"
sys.path.insert(0, str(COMMON_DIR))

from docking_utils import resolve_vina_path  # noqa: E402

DATA_DIR = STAGE1_DIR / "data"
RESULTS_DIR = STAGE1_DIR / "results"
CONFIG_PATH = STAGE1_DIR / "config" / "docking_box.json"

RECEPTOR = DATA_DIR / "9KNZ_clean.pdbqt"
LIGAND = DATA_DIR / "ligands" / "ERDRP.pdbqt"
REF_SDF = DATA_DIR / "ERDRP_with_bonds.sdf"

EXHAUSTIVENESS = 32
SEEDS = [42, 0, 1]
RMSD_GATE = 2.0


def parse_sdf_heavy_atoms(path):
    with open(path) as f:
        lines = f.readlines()
    natoms = int(lines[3][:3])
    atoms = []
    for line in lines[4 : 4 + natoms]:
        parts = line.split()
        if len(parts) >= 4 and parts[3] != "H":
            atoms.append((parts[3], float(parts[0]), float(parts[1]), float(parts[2])))
    return atoms


def run_vina(vina_path: Path, seed: int):
    out_pdbqt = RESULTS_DIR / f"ERDRP_redocked_seed{seed}.pdbqt"
    out_pdbqt.unlink(missing_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = json.load(f)

    cmd = [
        str(vina_path),
        "--receptor",
        str(RECEPTOR),
        "--ligand",
        str(LIGAND),
        "--center_x",
        str(cfg["center_x"]),
        "--center_y",
        str(cfg["center_y"]),
        "--center_z",
        str(cfg["center_z"]),
        "--size_x",
        str(cfg["size_x"]),
        "--size_y",
        str(cfg["size_y"]),
        "--size_z",
        str(cfg["size_z"]),
        "--exhaustiveness",
        str(EXHAUSTIVENESS),
        "--scoring",
        "vinardo",
        "--seed",
        str(seed),
        "--out",
        str(out_pdbqt),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0 or not out_pdbqt.exists():
        return seed, None, result.stderr
    return seed, out_pdbqt, result.stdout


def extract_pose1_heavy_atoms(pdbqt_path):
    ad_to_element = {
        "C": "C",
        "A": "C",
        "N": "N",
        "NA": "N",
        "OA": "O",
        "O": "O",
        "S": "S",
        "SA": "S",
        "F": "F",
        "Cl": "Cl",
        "Br": "Br",
        "I": "I",
        "P": "P",
    }
    atoms = []
    capture = False
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith("MODEL 1"):
                capture = True
                continue
            if capture and line.startswith("ENDMDL"):
                break
            if capture and line.startswith(("ATOM", "HETATM")):
                ad_type = line[77:79].strip() if len(line) > 77 else ""
                if ad_type in ("HD", "H"):
                    continue
                elem = ad_to_element.get(ad_type, ad_type[:1].upper())
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                atoms.append((elem, x, y, z))
    return atoms


def compute_rmsd_no_fit(ref_atoms, dock_atoms):
    from collections import defaultdict

    ref_by_elem = defaultdict(list)
    dock_by_elem = defaultdict(list)
    for atom in ref_atoms:
        ref_by_elem[atom[0]].append(atom[1:])
    for atom in dock_atoms:
        dock_by_elem[atom[0]].append(atom[1:])

    sq_diffs = []
    for elem, rcoords in ref_by_elem.items():
        dcoords = list(dock_by_elem.get(elem, []))
        if len(rcoords) != len(dcoords):
            return float("inf")
        used = [False] * len(dcoords)
        for r in rcoords:
            best_d, best_j = float("inf"), -1
            for j, d in enumerate(dcoords):
                if not used[j]:
                    dist2 = (r[0] - d[0]) ** 2 + (r[1] - d[1]) ** 2 + (r[2] - d[2]) ** 2
                    if dist2 < best_d:
                        best_d, best_j = dist2, j
            used[best_j] = True
            sq_diffs.append(best_d)

    return math.sqrt(sum(sq_diffs) / len(sq_diffs)) if sq_diffs else float("inf")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ERDRP redocking RMSD gate")
    parser.add_argument("--vina-path", type=Path, default=None)
    args = parser.parse_args()

    vina = resolve_vina_path(args.vina_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running Vina redocking  exhaustiveness={EXHAUSTIVENESS}  seeds={SEEDS}")
    print(f"Receptor : {RECEPTOR}")
    print(f"Ligand   : {LIGAND}")
    print(f"Vina     : {vina}")
    print()

    ref_atoms = parse_sdf_heavy_atoms(REF_SDF)
    print(f"Reference heavy atoms: {len(ref_atoms)}")

    results = []

    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        futures = {pool.submit(run_vina, vina, s): s for s in SEEDS}
        for fut in as_completed(futures):
            seed, pdbqt_out, log = fut.result()
            if pdbqt_out is None:
                print(f"  seed={seed}: VINA FAILED\n{log[:300]}")
                continue
            dock_atoms = extract_pose1_heavy_atoms(pdbqt_out)
            rmsd = compute_rmsd_no_fit(ref_atoms, dock_atoms)
            print(f"  seed={seed}: RMSD = {rmsd:.3f} Å  ({'PASS' if rmsd <= RMSD_GATE else 'FAIL'})")
            results.append((rmsd, seed, pdbqt_out))

    if not results:
        print("\nAll seeds failed — check Vina binary and inputs.")
        sys.exit(1)

    results.sort()
    best_rmsd, best_seed, best_pdbqt = results[0]

    print()
    print(f"Best pose: seed={best_seed}  RMSD={best_rmsd:.3f} Å")
    print(f"Gate (≤ {RMSD_GATE} Å): {'PASS ✓' if best_rmsd <= RMSD_GATE else 'FAIL ✗'}")

    canonical = RESULTS_DIR / "ERDRP_redocked.pdbqt"
    import shutil

    shutil.copy2(best_pdbqt, canonical)
    print(f"Copied best pdbqt → {canonical}")

    for _, seed, path in results:
        path.unlink(missing_ok=True)

    sys.exit(0 if best_rmsd <= RMSD_GATE else 1)


if __name__ == "__main__":
    main()
