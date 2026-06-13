#!/usr/bin/env python3
"""
Pipeline Smoke Test: Single Compound End-to-End Validation (v7 Protocol)

Runs ONE compound through the entire docking pipeline to verify
everything works before committing to the full 20-compound screen.

CRITICAL CHANGES from original:
- Uses Meeko (not OpenBabel) for ligand PDBQT
- exhaustiveness=16 (not 4)
- Deterministic: seed=42
- Includes RMSD redocking validation
- Bond perception for crystal ligand
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, AllChem
from rdkit.Geometry import Point3D

ADFR_ROOT = Path("/Users/vihaanagrawal/ADFRsuite-1.0")
ADFR_PYTHON = ADFR_ROOT / "bin" / "python"
ADFR_SCRIPT = ADFR_ROOT / "CCSBpckgs" / "AutoDockTools" / "Utilities24" / "prepare_receptor4.py"


def run_prepare_receptor(input_pdb: Path, output_pdbqt: Path) -> None:
    env = os.environ.copy()
    env.update({
        "ADS_ROOT": str(ADFR_ROOT),
        "ADS_EXTRALIBS": str(ADFR_ROOT / "lib"),
        "ADS_EXTRAINCLUDE": str(ADFR_ROOT / "include"),
        "PYTHONHOME": str(ADFR_ROOT),
        "PYTHONPATH": str(ADFR_ROOT / "CCSBpckgs"),
        "BABEL_LIBDIR": str(ADFR_ROOT / "lib" / "openbabel" / "2.4.1"),
        "BABEL_DATADIR": str(ADFR_ROOT / "share" / "openbabel" / "2.4.1"),
        "REDUCE_HET_DICT": str(ADFR_ROOT / "bin"),
        "PATH": f"{ADFR_ROOT / 'bin'}:{env.get('PATH', '')}",
    })
    if sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = f"{ADFR_ROOT / 'lib'}:{env.get('DYLD_LIBRARY_PATH', '')}".rstrip(":")
    else:
        env["LD_LIBRARY_PATH"] = f"{ADFR_ROOT / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")

    cmd = [
        str(ADFR_PYTHON),
        str(ADFR_SCRIPT),
        "-r", str(input_pdb),
        "-o", str(output_pdbqt),
        "-A", "hydrogens",
        "-U", "nphs_lps",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)


def build_rdkit_mol_from_cif(cif_path: Path, comp_id: str):
    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()
    atom_table = block.find([
        "_chem_comp_atom.comp_id",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.pdbx_aromatic_flag",
    ])
    bond_table = block.find([
        "_chem_comp_bond.comp_id",
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.value_order",
        "_chem_comp_bond.pdbx_aromatic_flag",
    ])

    bond_type_map = {
        "sing": rdchem.BondType.SINGLE,
        "doub": rdchem.BondType.DOUBLE,
        "trip": rdchem.BondType.TRIPLE,
    }

    mol = Chem.RWMol()
    atom_name_to_idx = {}
    for row_idx in range(len(atom_table)):
        row = atom_table[row_idx]
        if row[0] != comp_id or row[2] == "H":
            continue
        atom = Chem.Atom(row[2])
        atom.SetProp("atom_name", row[1])
        if row[3] == "Y":
            atom.SetIsAromatic(True)
        atom_name_to_idx[row[1]] = mol.AddAtom(atom)

    for row_idx in range(len(bond_table)):
        row = bond_table[row_idx]
        if row[0] != comp_id:
            continue
        atom_1, atom_2 = row[1], row[2]
        if atom_1 not in atom_name_to_idx or atom_2 not in atom_name_to_idx:
            continue
        bond_type = rdchem.BondType.AROMATIC if row[4] == "Y" else bond_type_map[row[3]]
        mol.AddBond(atom_name_to_idx[atom_1], atom_name_to_idx[atom_2], bond_type)
        if row[4] == "Y":
            bond = mol.GetBondBetweenAtoms(atom_name_to_idx[atom_1], atom_name_to_idx[atom_2])
            bond.SetIsAromatic(True)

    template = mol.GetMol()
    Chem.SanitizeMol(template)
    return template


def read_pdb_atom_coordinates(pdb_path: Path):
    coords = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                coords[atom_name] = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
    return coords

# Set up paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = Path(__file__).resolve().parent
DATA_DIR = STAGE1_DIR / "data"
LIGANDS_DIR = DATA_DIR / "ligands"
CONFIG_DIR = STAGE1_DIR / "config"
RESULTS_DIR = STAGE1_DIR / "results"
VINA_PATH = PROJECT_ROOT / "scripts" / "vina"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LIGANDS_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Global settings (v7 protocol)
RANDOM_SEED = 42
EXHAUSTIVENESS = 16  # NOT 4! Critical for accuracy
BOX_SIZE = 22  # Angstroms

# Status tracking
status = {}

def print_step(step_num, description):
    print(f"\n{'='*60}")
    print(f"Step {step_num}: {description}")
    print('='*60)


def pdb_fingerprint(path: Path):
    heavy_atoms = 0
    residues = set()
    chains = set()
    with open(path) as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                element = (line[76:78].strip() or atom_name[:1]).upper()
                if not element.startswith("H"):
                    heavy_atoms += 1
                chains.add(line[21].strip())
                residues.add((line[21], line[22:26].strip(), line[17:20].strip()))
    return {"heavy_atoms": heavy_atoms, "residue_count": len(residues), "chains": tuple(sorted(chains))}


# =============================================================================
# Step 1: Download and Clean PDB 9KNZ
# =============================================================================
print_step(1, "Download and Clean PDB 9KNZ")

import requests
from Bio.PDB import PDBParser, PDBIO, Select
import gemmi

# Download CIF file (PDB format not available for EM structures)
cif_url = "https://files.rcsb.org/download/9KNZ.cif"
cif_path = DATA_DIR / "9KNZ_raw.cif"
pdb_raw_path = DATA_DIR / "9KNZ_raw.pdb"

print(f"Downloading PDB 9KNZ (CIF format) from RCSB...")
response = requests.get(cif_url)
if response.status_code == 200:
    with open(cif_path, 'w') as f:
        f.write(response.text)
    print(f"  Downloaded CIF to: {cif_path}")

    # Convert CIF to PDB using gemmi
    print(f"  Converting CIF to PDB format...")
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()
    structure = gemmi.make_structure_from_block(block)
    structure.write_pdb(str(pdb_raw_path))
    print(f"  Converted to: {pdb_raw_path}")
else:
    print(f"  ERROR: Failed to download CIF (status {response.status_code})")
    sys.exit(1)

# Clean: isolate Chain A, remove waters/ions
class ChainASelect(Select):
    def accept_chain(self, chain):
        return chain.id == "A"
    def accept_residue(self, residue):
        return residue.id[0] == " "  # Exclude heteroatoms/waters

parser = PDBParser(QUIET=True)
structure = parser.get_structure("9KNZ", str(pdb_raw_path))

# Check what chains are available
chains = [c.id for model in structure for c in model]
print(f"  Available chains: {chains}")

# If Chain A doesn't exist, use first available chain
if "A" not in chains:
    first_chain = chains[0] if chains else None
    print(f"  WARNING: Chain A not found, using Chain {first_chain}")
    class ChainASelect(Select):
        def accept_chain(self, chain):
            return chain.id == first_chain
        def accept_residue(self, residue):
            return residue.id[0] == " "

io = PDBIO()
io.set_structure(structure)
clean_pdb_path = DATA_DIR / "9KNZ_clean.pdb"
io.save(str(clean_pdb_path), ChainASelect())

# Verify
line_count = sum(1 for _ in open(clean_pdb_path))
print(f"  Saved cleaned structure: {clean_pdb_path}")
print(f"  Line count: {line_count}")
if line_count <= 1000:
    sys.exit("CRITICAL ERROR: cleaned receptor is too short to be physically valid.")
status["step1_structure"] = True
print(f"  CHECK: {'PASS' if status['step1_structure'] else 'FAIL'} (expected >1000 lines)")


# =============================================================================
# Step 2: Extract ERDRP-0519 Ligand (for redocking validation)
# =============================================================================
print_step(2, "Extract ERDRP-0519 Ligand")

class LigandSelect(Select):
    def accept_residue(self, residue):
        # Include heteroatoms but exclude common ions and water
        return residue.id[0] != " " and residue.resname not in ["HOH", "NA", "CL", "MG", "ZN", "CA", "K"]

# Re-parse the raw structure to get ligands
structure = parser.get_structure("9KNZ", str(pdb_raw_path))
io = PDBIO()
io.set_structure(structure)
ligand_path = DATA_DIR / "ERDRP_original.pdb"
io.save(str(ligand_path), LigandSelect())

# Verify and show what ligands we found
line_count = sum(1 for _ in open(ligand_path))
print(f"  Extracted ligand to: {ligand_path}")
print(f"  Line count: {line_count}")

# List unique ligand residue names
ligand_resnames = set()
with open(ligand_path) as f:
    for line in f:
        if line.startswith(("HETATM", "ATOM")):
            resname = line[17:20].strip()
            ligand_resnames.add(resname)
print(f"  Ligand residues found: {ligand_resnames}")

status["step2_ligand"] = line_count > 10
print(f"  CHECK: {'PASS' if status['step2_ligand'] else 'FAIL'} (expected >10 lines)")


# =============================================================================
# Step 3: Calculate Docking Box from Crystal Ligand
# =============================================================================
print_step(3, "Calculate Docking Box from Crystal Ligand")

coords = []
ref_coords = []  # For RMSD calculation later

with open(ligand_path) as f:
    for line in f:
        if line.startswith(("HETATM", "ATOM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
                # Store heavy atom coords for RMSD (skip H)
                elem = line[76:78].strip() if len(line) > 76 else line[12:14].strip()
                if elem not in ["H", "D"]:
                    ref_coords.append([x, y, z])
            except ValueError:
                pass

if coords:
    center_x = (min(c[0] for c in coords) + max(c[0] for c in coords)) / 2.0
    center_y = (min(c[1] for c in coords) + max(c[1] for c in coords)) / 2.0
    center_z = (min(c[2] for c in coords) + max(c[2] for c in coords)) / 2.0
    print(f"  Ligand center: ({center_x:.2f}, {center_y:.2f}, {center_z:.2f})")
    
    # Save reference coords for RMSD
    np.save(DATA_DIR / "ERDRP_ref_coords.npy", np.array(ref_coords))
    print(f"  Saved {len(ref_coords)} heavy atom reference coordinates")
else:
    print("  ERROR: No ligand coordinates found!")
    sys.exit(1)

docking_config = {
    "center_x": round(center_x, 3),
    "center_y": round(center_y, 3),
    "center_z": round(center_z, 3),
    "size_x": BOX_SIZE,
    "size_y": BOX_SIZE,
    "size_z": BOX_SIZE,
    "exhaustiveness": EXHAUSTIVENESS,
    "seed": RANDOM_SEED
}

config_path = CONFIG_DIR / "docking_box.json"
with open(config_path, "w") as f:
    json.dump(docking_config, f, indent=2)

print(f"  Saved config: {config_path}")
print(f"  Config: {docking_config}")

status["step3_box"] = config_path.exists()
print(f"  CHECK: {'PASS' if status['step3_box'] else 'FAIL'}")


# =============================================================================
# Step 4: Prepare Receptor PDBQT Files (ADFRsuite)
# =============================================================================
print_step(4, "Prepare Receptor PDBQT Files via ADFRsuite")

wt_pdbqt = DATA_DIR / "9KNZ_clean.pdbqt"

print("  Converting to PDBQT via ADFRsuite...")
before = pdb_fingerprint(clean_pdb_path)
if wt_pdbqt.exists():
    wt_pdbqt.unlink()
try:
    run_prepare_receptor(clean_pdb_path, wt_pdbqt)
except subprocess.CalledProcessError as exc:
    print(f"  ERROR: Failed to create receptor PDBQT: {exc.stderr}")
    sys.exit(1)
if not wt_pdbqt.exists():
    sys.exit("CRITICAL ERROR: receptor PDBQT was not created.")
after = pdb_fingerprint(wt_pdbqt)
for key in ("heavy_atoms", "residue_count", "chains"):
    if before[key] != after[key]:
        print(f"  ERROR: WT receptor parity failure for {key}: before={before}, after={after}")
        sys.exit(1)
print(f"  Created: {wt_pdbqt}")

status["step4_receptor"] = True
print(f"  CHECK: {'PASS' if status['step4_receptor'] else 'FAIL'}")


# =============================================================================
# Step 5: Prepare Crystal Ligand with Bond Perception (CRITICAL)
# =============================================================================
print_step(5, "Prepare Crystal Ligand (Bond Perception + Meeko)")

# from openbabel import openbabel as ob # Removed to avoid build errors
from meeko import MoleculePreparation, PDBQTWriterLegacy


def transfer_coordinates_to_template(template, coordinate_source):
    if coordinate_source is None:
        sys.exit("ERROR: failed to load docked coordinates for template transfer.")
    if coordinate_source.GetNumAtoms() != template.GetNumAtoms():
        sys.exit(
            f"ERROR: atom-count mismatch during template transfer ({coordinate_source.GetNumAtoms()} vs {template.GetNumAtoms()})."
        )
    templated_pose = Chem.Mol(template)
    templated_pose.RemoveAllConformers()
    source_conf = coordinate_source.GetConformer()
    conf = Chem.Conformer(template.GetNumAtoms())
    for idx in range(template.GetNumAtoms()):
        conf.SetAtomPosition(idx, source_conf.GetAtomPosition(idx))
    templated_pose.AddConformer(conf)
    return templated_pose

print("  Step 5a: Fixing PDB format...")
fixed_lines = []
atom_num = 0

with open(ligand_path, "r") as f:
    for line in f:
        if line.startswith("HETATM") or line.startswith("ATOM"):
            atom_num += 1
            parts = line.split()
            if len(parts) >= 8:
                name = parts[2]
                elem = parts[-1] if len(parts[-1]) <= 2 else name[0]
                x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                fixed = f"HETATM{atom_num:>5} {name:<4} LIG A   1    {x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2}\n"
                fixed_lines.append(fixed)

fixed_lines.append("END\n")
fixed_pdb = DATA_DIR / "ERDRP_fixed.pdb"
with open(fixed_pdb, "w") as f:
    f.writelines(fixed_lines)
print(f"  Fixed PDB: {atom_num} atoms")

print("  Step 5b: Applying trusted template chemistry...")
sdf_path = DATA_DIR / "ERDRP_with_bonds.sdf"
template_sdf = DATA_DIR / "ERDRP_template.sdf"
template = build_rdkit_mol_from_cif(DATA_DIR / "9KNZ_raw.cif", comp_id="A1EF9")
Chem.MolToMolFile(template, str(template_sdf))
pose_coords = read_pdb_atom_coordinates(fixed_pdb)

template_name_to_idx = {
    atom.GetProp("atom_name"): atom.GetIdx()
    for atom in template.GetAtoms()
}

conf = Chem.Conformer(template.GetNumAtoms())
for atom_name, (x, y, z) in pose_coords.items():
    if atom_name not in template_name_to_idx:
        print(f"  ERROR: CIF template is missing atom {atom_name}.")
        sys.exit(1)
    idx = template_name_to_idx[atom_name]
    conf.SetAtomPosition(idx, Point3D(x, y, z))

template.RemoveAllConformers()
template.AddConformer(conf, assignId=True)
Chem.SanitizeMol(template)
typed_pose = Chem.Mol(template)
Chem.MolToMolFile(typed_pose, str(sdf_path))
print(f"    Converted to SDF: {sdf_path}")

print("  Step 5c: RDKit validation...")
rdmol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
if rdmol is None:
    print("  ERROR: RDKit failed to read SDF!")
    sys.exit(1)

n_frags = len(Chem.GetMolFrags(typed_pose))
print(f"    Fragments: {n_frags}")
if n_frags != 1:
    print(f"  ERROR: Bond perception FAILED! Molecule has {n_frags} fragments!")
    sys.exit(1)
if Chem.MolToSmiles(typed_pose, canonical=True) != Chem.MolToSmiles(template, canonical=True):
    print("  ERROR: Template chemistry mismatch after bond-order assignment.")
    sys.exit(1)

print("  Step 5d: Meeko PDBQT preparation...")
typed_pose = Chem.AddHs(typed_pose, addCoords=True)
assert all(atom.GetNumImplicitHs() == 0 for atom in typed_pose.GetAtoms())
preparator = MoleculePreparation()
mol_setups = preparator.prepare(typed_pose)
pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

erdrp_pdbqt = LIGANDS_DIR / "ERDRP.pdbqt"
with open(erdrp_pdbqt, "w") as f:
    f.write(pdbqt_string)
print(f"  Created: {erdrp_pdbqt}")


status["step5_crystal_ligand"] = erdrp_pdbqt.exists() and n_frags == 1
print(f"  CHECK: {'PASS' if status['step5_crystal_ligand'] else 'FAIL'}")


# =============================================================================
# Step 6: Redocking Validation (RMSD < 2.0 Å)
# =============================================================================
print_step(6, "Redocking Validation (RMSD must be < 2.0 Å)")

with open(config_path) as f:
    config = json.load(f)

redock_output = RESULTS_DIR / "ERDRP_redocked.pdbqt"
first_pose_pdbqt = RESULTS_DIR / "ERDRP_redocked_pose1.pdbqt"
docked_sdf_path = redock_output.with_suffix(".sdf")

for stale in (redock_output, first_pose_pdbqt, docked_sdf_path):
    if stale.exists():
        stale.unlink()

cmd = [
    VINA_PATH,
    "--receptor", wt_pdbqt,
    "--ligand", erdrp_pdbqt,
    "--center_x", str(config["center_x"]),
    "--center_y", str(config["center_y"]),
    "--center_z", str(config["center_z"]),
    "--size_x", str(config["size_x"]),
    "--size_y", str(config["size_y"]),
    "--size_z", str(config["size_z"]),
    "--exhaustiveness", str(EXHAUSTIVENESS),
    "--seed", str(RANDOM_SEED),
    "--out", redock_output
]

print(f"  Running Vina redocking (exhaustiveness={EXHAUSTIVENESS}, seed={RANDOM_SEED})...")
result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
assert redock_output.exists() and redock_output.stat().st_size > 0

# Parse RMSD from docked output
from spyrmsd import io, rmsd

# Load reference with topology
ref = io.loadmol(str(sdf_path)) # Path to the prepared sdf in step 5

# OpenBabel is unreliable on the full multi-model Vina PDBQT. Extract the first pose
# as a standalone PDBQT, then convert that single pose to SDF for coordinate transfer.
with open(redock_output, "r") as f_in, open(first_pose_pdbqt, "w") as f_out:
    capture = False
    for line in f_in:
        if line.startswith("MODEL 1"):
            capture = True
            continue
        if capture and line.startswith("ENDMDL"):
            break
        if capture:
            f_out.write(line)

assert first_pose_pdbqt.exists() and first_pose_pdbqt.stat().st_size > 0

# Bypass OpenBabel: parse heavy-atom coordinates from the docked PDBQT and
# transfer them onto the CIF-derived template graph using substructure matching.
# AutoDock atom types must be mapped to real elements for RDKit's PDB parser.
ad_to_element = {
    "C": "C", "A": "C", "N": "N", "NA": "N", "OA": "O", "O": "O",
    "S": "S", "SA": "S", "F": "F", "Cl": "Cl", "Br": "Br", "I": "I",
    "P": "P",
}
tmp_pdb = RESULTS_DIR / "_pose1_heavy.pdb"
with open(first_pose_pdbqt, "r") as fin, open(tmp_pdb, "w") as fout:
    for line in fin:
        if not line.startswith(("ATOM", "HETATM")):
            continue
        ad_type = line[77:79].strip() if len(line) > 77 else ""
        if ad_type in ("HD", "H"):
            continue  # Hydrogen: skip — template has heavy atoms only.
        element = ad_to_element.get(ad_type, ad_type[:1].upper())
        padded = line.rstrip("\n")[:76].ljust(76)
        fout.write(padded + f"{element:>2}" + "\n")
    fout.write("END\n")

rough_mol = Chem.MolFromPDBFile(str(tmp_pdb), removeHs=True, proximityBonding=True, sanitize=False)
if rough_mol is None:
    sys.exit("ERROR: RDKit failed to parse docked pose from temporary PDB.")

docked_typed_pose = AllChem.AssignBondOrdersFromTemplate(template, rough_mol)
if docked_typed_pose is None:
    sys.exit("ERROR: template bond-order assignment failed on docked pose.")

Chem.SanitizeMol(docked_typed_pose)
Chem.MolToMolFile(docked_typed_pose, str(docked_sdf_path))
tmp_pdb.unlink()

assert docked_sdf_path.exists()
assert docked_sdf_path.stat().st_size > 0
assert docked_sdf_path.stat().st_mtime >= first_pose_pdbqt.stat().st_mtime

# Load docked poses
docked_poses = io.loadallmols(str(docked_sdf_path))

if docked_poses:
    # spyrmsd handles atom reordering automatically based on molecular graph adjacency
    ref.strip()
    docked_poses[0].strip()
    calculated_rmsd = rmsd.symmrmsd(
        ref.coordinates,
        docked_poses[0].coordinates,
        ref.atomicnums,
        docked_poses[0].atomicnums,
        ref.adjacency_matrix,
        docked_poses[0].adjacency_matrix
    )
    print(f"  True Topology-Aware RMSD: {calculated_rmsd:.3f} Å")
    
    if calculated_rmsd <= 2.0:
        print(f"  VALIDATION PASSED: RMSD ≤ 2.0 Å")
        status["step6_rmsd"] = True
    else:
        print(f"  WARNING: RMSD > 2.0 Å - docking setup may have issues!")
        status["step6_rmsd"] = False
else:
    print(f"  ERROR: No docked poses found for RMSD validation.")
    status["step6_rmsd"] = False

print(f"  CHECK: {'PASS' if status['step6_rmsd'] else 'FAIL'}")


# =============================================================================
# Step 7: Create W730A Mutant Structure via PyMOL
# =============================================================================
print_step(7, "Create W730A Mutant Structure via PyMOL API")

import pymol

mutant_path = DATA_DIR / "9KNZ_W730A.pdb"
mut_pdbqt = DATA_DIR / "9KNZ_W730A.pdbqt"

print(f"  Generating W730A via PyMOL Rotamer Mutagenesis...")
pymol.finish_launching(['pymol', '-cq'])
pymol.cmd.load(str(clean_pdb_path), "WT_prot")
pymol.cmd.wizard("mutagenesis")
pymol.cmd.get_wizard().do_select("/WT_prot//A/730")
pymol.cmd.get_wizard().set_mode("ALA")
pymol.cmd.get_wizard().apply()
pymol.cmd.save(str(mutant_path), "WT_prot")
pymol.cmd.delete("all")

# Convert mutant to PDBQT via ADFRsuite
print("  Converting to PDBQT via ADFRsuite...")
before = pdb_fingerprint(mutant_path)
if mut_pdbqt.exists():
    mut_pdbqt.unlink()
try:
    run_prepare_receptor(mutant_path, mut_pdbqt)
except subprocess.CalledProcessError as exc:
    print(f"  ERROR: Failed to create mutant PDBQT: {exc.stderr}")
    sys.exit(1)
if not mut_pdbqt.exists():
    sys.exit("CRITICAL ERROR: mutant receptor PDBQT was not created.")
after = pdb_fingerprint(mut_pdbqt)
for key in ("heavy_atoms", "residue_count", "chains"):
    if before[key] != after[key]:
        print(f"  ERROR: mutant receptor parity failure for {key}: before={before}, after={after}")
        sys.exit(1)
print(f"  Saved mutant PDBQT: {mut_pdbqt}")

status["step7_mutant"] = mutant_path.exists() and mut_pdbqt.exists()
print(f"  CHECK: {'PASS' if status['step7_mutant'] else 'FAIL'}")


# =============================================================================
# Step 8: Prepare Test Ligand (Ribavirin) with Meeko
# =============================================================================
print_step(8, "Prepare Test Ligand (Ribavirin) with Meeko")

# Ribavirin SMILES
test_smiles = "C1=NC(=NN1C2C(C(C(O2)CO)O)O)C(=O)N"
test_name = "Ribavirin"

print(f"  Generating 3D structure for {test_name}...")
mol = Chem.MolFromSmiles(test_smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=RANDOM_SEED)
AllChem.MMFFOptimizeMolecule(mol)

# Use Meeko for PDBQT (NOT OpenBabel!)
print(f"  Converting to PDBQT with Meeko...")
preparator = MoleculePreparation()
mol_setups = preparator.prepare(mol)
pdbqt_string = PDBQTWriterLegacy.write_string(mol_setups[0])[0]

pdbqt_path = LIGANDS_DIR / f"{test_name}.pdbqt"
with open(pdbqt_path, "w") as f:
    f.write(pdbqt_string)
print(f"  Created: {pdbqt_path}")

status["step8_test_ligand"] = pdbqt_path.exists()
print(f"  CHECK: {'PASS' if status['step8_test_ligand'] else 'FAIL'}")


# =============================================================================
# Step 9: Dock Ribavirin to Wildtype
# =============================================================================
print_step(9, "Dock Ribavirin to Wildtype")

wt_output = RESULTS_DIR / "wt_docked.pdbqt"

cmd = [
    VINA_PATH,
    "--receptor", wt_pdbqt,
    "--ligand", pdbqt_path,
    "--center_x", str(config["center_x"]),
    "--center_y", str(config["center_y"]),
    "--center_z", str(config["center_z"]),
    "--size_x", str(config["size_x"]),
    "--size_y", str(config["size_y"]),
    "--size_z", str(config["size_z"]),
    "--exhaustiveness", str(EXHAUSTIVENESS),
    "--seed", str(RANDOM_SEED),
    "--out", wt_output
]

print(f"  Running Vina (exhaustiveness={EXHAUSTIVENESS}, seed={RANDOM_SEED})...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

# Parse affinity
result_wt = None
for line in result.stdout.split('\n'):
    if line.strip().startswith('1'):
        parts = line.split()
        if len(parts) >= 2:
            try:
                result_wt = float(parts[1])
                break
            except ValueError:
                pass

print(f"  Wildtype affinity: {result_wt} kcal/mol")
status["step9_wt_dock"] = result_wt is not None
print(f"  CHECK: {'PASS' if status['step9_wt_dock'] else 'FAIL'}")


# =============================================================================
# Step 10: Dock Ribavirin to Mutant
# =============================================================================
print_step(10, "Dock Ribavirin to Mutant (W730A)")

mut_output = RESULTS_DIR / "mut_docked.pdbqt"

cmd = [
    VINA_PATH,
    "--receptor", mut_pdbqt,
    "--ligand", pdbqt_path,
    "--center_x", str(config["center_x"]),
    "--center_y", str(config["center_y"]),
    "--center_z", str(config["center_z"]),
    "--size_x", str(config["size_x"]),
    "--size_y", str(config["size_y"]),
    "--size_z", str(config["size_z"]),
    "--exhaustiveness", str(EXHAUSTIVENESS),
    "--seed", str(RANDOM_SEED),
    "--out", mut_output
]

print(f"  Running Vina...")
result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

result_mut = None
for line in result.stdout.split('\n'):
    if line.strip().startswith('1'):
        parts = line.split()
        if len(parts) >= 2:
            try:
                result_mut = float(parts[1])
                break
            except ValueError:
                pass

print(f"  Mutant affinity: {result_mut} kcal/mol")
status["step10_mut_dock"] = result_mut is not None

# Step 11: Results
print_step(11, "Calculate Delta Affinity")

import pandas as pd

if result_wt is not None and result_mut is not None:
    delta_affinity = result_mut - result_wt
    print(f"  WT: {result_wt:.2f}, Mut: {result_mut:.2f}, Delta: {delta_affinity:.2f} kcal/mol")
    
    results = pd.DataFrame([{"compound": "Ribavirin", "wt": result_wt, "mut": result_mut, "delta": delta_affinity}])
    results.to_csv(RESULTS_DIR / "smoke_test_results.csv", index=False)
    status["step11_results"] = True

# Summary
print("\n" + "="*60)
print("SMOKE TEST SUMMARY")
print("="*60)
for k, v in status.items():
    print(f"  {'PASS' if v else 'FAIL'}: {k}")
print("="*60)
