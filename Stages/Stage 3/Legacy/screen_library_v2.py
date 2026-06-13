
import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from Bio.PDB import PDBParser
from scipy.spatial import distance
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem

# Paths
STAGE3_DIR = Path(__file__).parent  # Stages/Stage 3/
STAGE3_DATA = STAGE3_DIR / "data"
STAGE3_RESULTS = STAGE3_DIR / "results"
STAGE3_DATA.mkdir(parents=True, exist_ok=True)
STAGE3_RESULTS.mkdir(parents=True, exist_ok=True)
VINA_PATH = STAGE3_DIR.parent.parent / "scripts" / "vina"
LIGANDS_DIR = STAGE3_DATA / "ligands"
LIGANDS_DIR.mkdir(exist_ok=True)

# Fixed Configs
TARGET_RES_ID = 730
CHAIN_ID = "A"

with open(STAGE3_DIR.parent / "Stage 1" / "config" / "docking_box.json") as f:
    BOX_CONFIG = json.load(f)

# Receptors — rigid-rigid symmetry per Stage 2 mandate
WT_RIGID = STAGE3_DIR.parent / "Stage 1" / "data" / "9KNZ_clean.pdbqt"
MUT_RIGID = STAGE3_DIR.parent / "Stage 1" / "data" / "9KNZ_W730A.pdbqt"

# Canonical WT PDB for ghost sidechain reference (used for BOTH wt and mut distances)
WT_PDB = STAGE3_DIR.parent / "Stage 1" / "data" / "9KNZ_clean.pdb"
GHOST_BACKBONE_AND_CB = {"N", "CA", "C", "O", "CB"}

# Ghost clash threshold (non-covalent VdW overlap)
GHOST_CLASH_THRESHOLD = 2.5  # Angstrom

# Allosteric Independence minimum distance
ALLOSTERIC_MIN_DISTANCE = 5.0  # Angstrom from nearest ghost sidechain atom


def get_ghost_sidechain_coords(wt_pdb, chain_id, resid):
    """Extract the 9 TRP indole sidechain atom coordinates (the 'ghost' atoms
    deleted by W730A mutation). Always uses the WT PDB as reference."""
    if not wt_pdb.exists():
        raise FileNotFoundError(f"WT PDB missing: {wt_pdb}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("WT", str(wt_pdb))
    try:
        residue = structure[0][chain_id][resid]
    except KeyError:
        raise RuntimeError(f"Residue {chain_id}:{resid} not found in {wt_pdb}")
    if residue.resname != "TRP":
        raise RuntimeError(f"Expected TRP at {chain_id}:{resid}, found {residue.resname}")
    coords = []
    for atom in residue:
        if atom.name not in GHOST_BACKBONE_AND_CB:
            coords.append(atom.get_coord())
    if len(coords) != 9:
        raise RuntimeError(f"Expected 9 ghost sidechain atoms, found {len(coords)}")
    return np.array(coords)


def calculate_min_distance(ligand_pdbqt, ghost_coords):
    """Minimum distance from docked ligand MODEL 1 to ghost sidechain atoms."""
    if not os.path.exists(ligand_pdbqt):
        raise FileNotFoundError(f"Ligand PDBQT missing: {ligand_pdbqt}")
    ligand_coords = []
    with open(ligand_pdbqt, 'r') as f:
        in_model1 = False
        for line in f:
            if line.startswith("MODEL 1"):
                in_model1 = True
            elif line.startswith("ENDMDL"):
                break
            elif in_model1 and line.startswith(("ATOM", "HETATM")):
                try:
                    ligand_coords.append([
                        float(line[30:38]), float(line[38:46]), float(line[46:54])
                    ])
                except ValueError:
                    pass
    if not ligand_coords:
        raise RuntimeError(f"No ligand coordinates found in {ligand_pdbqt}")
    return float(np.min(distance.cdist(np.array(ligand_coords), ghost_coords)))


def run_vina(receptor_rigid, ligand_pdbqt, output_file):
    """Rigid-receptor-only docking per Stage 2 mandate."""
    cmd = [
        str(VINA_PATH),
        "--receptor", str(receptor_rigid),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(BOX_CONFIG["center_x"]),
        "--center_y", str(BOX_CONFIG["center_y"]),
        "--center_z", str(BOX_CONFIG["center_z"]),
        "--size_x", str(BOX_CONFIG["size_x"]),
        "--size_y", str(BOX_CONFIG["size_y"]),
        "--size_z", str(BOX_CONFIG["size_z"]),
        "--exhaustiveness", str(BOX_CONFIG.get("exhaustiveness", 16)),
        "--num_modes", "1",
        "--scoring", "vinardo",
        "--seed", str(BOX_CONFIG.get("seed", 42)),
        "--out", str(output_file),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        return False


def parse_affinity(pdbqt_file):
    if not os.path.exists(pdbqt_file): return None
    with open(pdbqt_file, 'r') as f:
        for line in f:
            if "REMARK VINA RESULT" in line:
                return float(line.split()[3])
    return None


def process_compound(row):
    name = row['name']
    smiles = row['smiles']
    print(f"Docking {name}...")

    # 0. pH-aware protonation (Dimorphite-DL preferred)
    try:
        from dimorphite_dl import DimorphiteDL
        dimorphite = DimorphiteDL(min_ph=7.4, max_ph=7.4, max_variants=1, label_states=False)
        protonated = dimorphite.protonate(smiles)
        if protonated:
            smiles = protonated[0]
    except ImportError:
        pass  # Accept neutral SMILES — do NOT use rdMolStandardize.Uncharger

    # 1. Prepare Ligand PDBQT
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"name": name, "error": "Invalid SMILES"}
        mol = Chem.AddHs(mol)
        embed_code = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if embed_code == -1:
            return {"name": name, "error": "3D embedding failed"}

        # Force-field minimization: MMFF94 preferred, UFF fallback
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]
        ligand_pdbqt = LIGANDS_DIR / f"{name}.pdbqt"
        with open(ligand_pdbqt, "w") as f:
            f.write(pdbqt_string)
    except Exception as e:
        return {"name": name, "error": f"Ligand prep: {str(e)}"}

    # 2. Dock WT (Rigid)
    wt_out = STAGE3_RESULTS / f"{name}_wt.pdbqt"
    if not run_vina(WT_RIGID, ligand_pdbqt, wt_out):
        return {"name": name, "error": "WT Docking failed"}

    # 3. Dock MUT (Rigid)
    mut_out = STAGE3_RESULTS / f"{name}_mut.pdbqt"
    if not run_vina(MUT_RIGID, ligand_pdbqt, mut_out):
        return {"name": name, "error": "MUT Docking failed"}

    # 4. Analyze Results
    ghost_coords = get_ghost_sidechain_coords(WT_PDB, CHAIN_ID, TARGET_RES_ID)

    res = {
        "name": name,
        "wt_affinity": parse_affinity(wt_out),
        "mut_affinity": parse_affinity(mut_out),
        "wt_dist": calculate_min_distance(wt_out, ghost_coords),
        "mut_dist": calculate_min_distance(mut_out, ghost_coords),
    }

    if res["wt_affinity"] is not None and res["mut_affinity"] is not None:
        res["delta_affinity"] = res["mut_affinity"] - res["wt_affinity"]
        res["delta_dist"] = res["mut_dist"] - res["wt_dist"]

    # Ghost clash check — reject compounds occupying the deleted TRP sidechain volume
    mut_ghost_dist = res["mut_dist"]
    res["ghost_clash_dist"] = mut_ghost_dist
    res["ghost_clash"] = mut_ghost_dist < GHOST_CLASH_THRESHOLD

    return res


def main():
    library_path = STAGE3_DATA / "filtered_library.csv"
    if not library_path.exists():
        print(f"FATAL: {library_path} not found. Run select_library_v2.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(library_path)
    if df.empty:
        print(f"FATAL: {library_path} is empty.", file=sys.stderr)
        sys.exit(1)

    # Run in parallel
    num_procs = min(cpu_count(), len(df))
    print(f"Starting screening on {num_procs} cores...")
    with Pool(num_procs) as p:
        results = p.map(process_compound, [row for _, row in df.iterrows()])

    # Save results
    final_df = pd.DataFrame([r for r in results if "error" not in r])
    errors = [r for r in results if "error" in r]

    final_df.to_csv(STAGE3_RESULTS / "resistance_screening_results.csv", index=False)
    print(f"Screening complete. {len(final_df)} compounds successful, {len(errors)} failed.")

    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e['name']}: {e['error']}")

    # Apply Filters (Allosteric Independence + Ghost Clash)
    potential_hits = final_df[final_df['wt_affinity'] <= -7.0]
    # Filter 1: must be spatially separated from mutation site in MUT context
    allosteric_hits = potential_hits[potential_hits['mut_dist'] >= ALLOSTERIC_MIN_DISTANCE]
    # Filter 2: pose must not shift significantly between WT and MUT
    stable_hits = allosteric_hits[allosteric_hits['delta_dist'].abs() < 2.0]
    # Filter 3: must not exploit vacuum hole
    clean_hits = stable_hits[stable_hits['ghost_clash'] == False]

    print("\n=== TOP RESILIENT CANDIDATES ===")
    if not clean_hits.empty:
        print(clean_hits.sort_values('wt_affinity')[
            ['name', 'wt_affinity', 'delta_affinity', 'delta_dist', 'ghost_clash_dist']
        ])
    else:
        print("No resilient candidates met all criteria (Potency + Allosteric Distance + Stability + Ghost Clash).")

if __name__ == "__main__":
    main()
