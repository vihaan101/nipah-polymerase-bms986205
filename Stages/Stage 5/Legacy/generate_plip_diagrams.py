"""
Generate PLIP 2D protein-ligand interaction diagrams for BMS-986205 seed-42
Vinardo poses in IDO1 WT and W730A mutant receptors.

Inputs:
  - Stages/Stage 1/data/9KNZ_clean.pdb       (WT receptor)
  - Stages/Stage 1/data/9KNZ_W730A.pdb       (MUT receptor)
  - Stages/Stage 5/results/matrix/BMS_WT_seed_42.pdbqt
  - Stages/Stage 5/results/matrix/BMS_MUT_seed_42.pdbqt

Outputs go to the directory passed as --outdir (default: Stages/Stage 5/results/plip/).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DATA  = PROJECT_ROOT / "Stages" / "Stage 1" / "data"
STAGE5_MAT   = PROJECT_ROOT / "Stages" / "Stage 5" / "results" / "matrix"

REDUCE_BIN = Path("/Users/vihaanagrawal/ADFRsuite-1.0/bin/reduce")
PLIP_BIN   = Path("/Users/vihaanagrawal/miniforge/envs/md_mac/bin/plip")

PLIP_ENV = {
    **os.environ,
    "BABEL_LIBDIR":  "/Users/vihaanagrawal/miniforge/envs/md_mac/lib/openbabel/3.1.0",
    "BABEL_DATADIR": "/Users/vihaanagrawal/miniforge/envs/md_mac/share/openbabel/3.1.0",
    "PATH": f"/Users/vihaanagrawal/miniforge/envs/md_mac/bin:{os.environ.get('PATH', '')}",
}

CASES = [
    {
        "name":         "BMS_WT_seed42",
        "receptor_pdb": STAGE1_DATA / "9KNZ_clean.pdb",
        "ligand_pdbqt": STAGE5_MAT  / "BMS_WT_seed_42.pdbqt",
        "label":        "BMS-986205 / IDO1 WT (seed-42 Vinardo)",
    },
    {
        "name":         "BMS_MUT_seed42",
        "receptor_pdb": STAGE1_DATA / "9KNZ_W730A.pdb",
        "ligand_pdbqt": STAGE5_MAT  / "BMS_MUT_seed_42.pdbqt",
        "label":        "BMS-986205 / IDO1 W730A (seed-42 Vinardo)",
    },
]


def pdbqt_atom_to_pdb(line: str) -> str:
    """Convert one PDBQT ATOM/HETATM line to standard PDB format.

    PDBQT adds charge + atom-type columns after col 54.  Strip them and
    reconstruct the element symbol column (1-indexed 77-78, Python [76:78]).
    """
    base = line[:54].rstrip()
    # PDBQT atom type is the last whitespace-delimited token (e.g. "A", "HD")
    # Element is typically the first 1-2 chars of that token, uppercased
    tokens = line[54:].split()
    element = tokens[-1][0] if tokens else ""
    return f"{base:<54}{'':20}{element:>2}\n"


def extract_ligand_pdb(pdbqt_path: Path, resname: str = "LIG",
                       chain: str = "X", resnum: int = 1) -> list[str]:
    """Parse MODEL 1 ATOM/HETATM lines from a Vina PDBQT, return PDB lines."""
    lines = []
    capture = False
    with open(pdbqt_path) as fh:
        for raw in fh:
            if raw.startswith("MODEL 1"):
                capture = True
                continue
            if capture:
                if raw.startswith(("ENDMDL", "MODEL")):
                    break
                if raw.startswith(("ATOM", "HETATM")):
                    pdb_line = pdbqt_atom_to_pdb(raw)
                    # Overwrite residue name (cols 17-20), chain (col 21),
                    # residue number (cols 22-25) to ensure PLIP detects ligand
                    pdb_line = (
                        "HETATM"
                        + pdb_line[6:17]
                        + f"{resname:<3} "
                        + chain
                        + f"{resnum:>4}    "
                        + pdb_line[30:54]
                        + pdb_line[54:]
                    )
                    lines.append(pdb_line)
    if not lines:
        raise RuntimeError(f"No ATOM/HETATM lines found in MODEL 1 of {pdbqt_path}")
    return lines


def add_hydrogens(receptor_pdb: Path, out_path: Path) -> Path:
    """Run reduce to add H atoms to receptor; write to out_path."""
    print(f"  Adding H atoms via reduce: {receptor_pdb.name} → {out_path.name}")
    result = subprocess.run(
        [str(REDUCE_BIN), "-NOFLIP", str(receptor_pdb)],
        capture_output=True, text=True,
    )
    # reduce exits non-zero even on success; check output instead
    protonated = result.stdout
    h_count = sum(1 for l in protonated.splitlines()
                  if l.startswith(("ATOM", "HETATM")) and l[12:16].strip().startswith("H"))
    if h_count == 0:
        print(f"  WARNING: reduce produced no H atoms for {receptor_pdb.name}")
    else:
        print(f"  reduce added {h_count} H atoms")
    out_path.write_text(protonated)
    return out_path


def build_complex_pdb(receptor_pdb: Path, ligand_lines: list[str],
                      out_path: Path) -> Path:
    """Merge protonated receptor + ligand into a single PDB complex file."""
    with open(receptor_pdb) as fh:
        receptor_lines = [l for l in fh
                          if l.startswith(("ATOM", "HETATM", "TER"))]
    with open(out_path, "w") as fh:
        fh.writelines(receptor_lines)
        fh.write("TER\n")
        fh.writelines(ligand_lines)
        fh.write("END\n")
    print(f"  Complex written: {out_path.name} "
          f"({len(receptor_lines)} receptor + {len(ligand_lines)} ligand lines)")
    return out_path


def run_plip(complex_pdb: Path, out_dir: Path) -> list[Path]:
    """Run PLIP on complex_pdb; return list of generated PNG paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PLIP_BIN),
        "-f", str(complex_pdb),
        "-p",          # PNG output
        "-x",          # XML report
        "-t",          # text report
        "-o", str(out_dir),
    ]
    print(f"  Running PLIP: {' '.join(cmd[-6:])}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, env=PLIP_ENV,
    )
    stdout = result.stdout
    stderr = "\n".join(
        l for l in result.stderr.splitlines()
        if "Open Babel Error" not in l and "did not load" not in l
        and "dlopen" not in l and "===" not in l
    )
    if stdout.strip():
        print(f"  PLIP stdout: {stdout.strip()[:400]}")
    if stderr.strip():
        print(f"  PLIP stderr: {stderr.strip()[:400]}")

    pngs = list(out_dir.glob("*.png"))
    if not pngs:
        print(f"  WARNING: PLIP produced no PNG in {out_dir}. "
              f"Files present: {[f.name for f in out_dir.iterdir()]}")
    else:
        print(f"  PNG(s) generated: {[p.name for p in pngs]}")
    return pngs


def copy_to_output(src: Path, dest_dir: Path, dest_name: str) -> None:
    import shutil
    dest = dest_dir / dest_name
    shutil.copy2(src, dest)
    print(f"  Copied → {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PLIP interaction diagrams")
    parser.add_argument(
        "--outdir", type=Path,
        default=Path(__file__).parent / "results" / "plip",
        help="Directory for intermediate PLIP outputs",
    )
    parser.add_argument(
        "--figdir", type=Path,
        default=None,
        help="Directory to copy final PNGs into (e.g. comparison_figures/)",
    )
    args = parser.parse_args()

    # Pre-flight: PLIP available
    test = subprocess.run(
        [str(PLIP_BIN), "-h"], capture_output=True, env=PLIP_ENV,
    )
    if test.returncode not in (0, 2):  # -h exits 2 on some versions
        raise RuntimeError(f"PLIP not functional at {PLIP_BIN}")
    print("Pre-flight: PLIP OK")

    figdir = args.figdir
    if figdir:
        figdir.mkdir(parents=True, exist_ok=True)

    for case in CASES:
        print(f"\n{'='*60}")
        print(f"Case: {case['name']}  ({case['label']})")
        print(f"{'='*60}")

        case_dir = args.outdir / case["name"]
        case_dir.mkdir(parents=True, exist_ok=True)

        # 1. Add H to receptor
        receptor_h = case_dir / f"{case['receptor_pdb'].stem}_H.pdb"
        add_hydrogens(case["receptor_pdb"], receptor_h)

        # 2. H-atom count check
        h_count = sum(
            1 for l in receptor_h.read_text().splitlines()
            if l.startswith(("ATOM", "HETATM")) and l[12:16].strip().startswith("H")
        )
        print(f"  H atoms in protonated receptor: {h_count}")

        # 3. Extract ligand from PDBQT
        lig_lines = extract_ligand_pdb(case["ligand_pdbqt"])
        print(f"  Ligand atoms extracted: {len(lig_lines)}")

        # 4. Build complex
        complex_pdb = case_dir / f"{case['name']}_complex.pdb"
        build_complex_pdb(receptor_h, lig_lines, complex_pdb)

        # 5. Run PLIP
        plip_out = case_dir / "plip_output"
        pngs = run_plip(complex_pdb, plip_out)

        # 6. Copy PNGs to figdir
        if figdir and pngs:
            for png in pngs:
                copy_to_output(png, figdir, f"{case['name']}_{png.name}")

    print(f"\nDone. Figures in: {figdir or args.outdir}")


if __name__ == "__main__":
    main()
