#!/usr/bin/env python3
"""
Scientific Reconstruction Script for BMS-986205 Complex
Follows the "True Scratch" protocol:
1. Ligand Bond Repair (PDBQT + SMILES -> SDF)
2. Protein Cleanup (Strip H, Fix CYX)
3. System Build (Amber14SB + GAFF2 + TIP3P + Ions)
4. Minimization & Equilibration (Safe Protocol)
"""

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from openmmforcefields.generators import GAFFTemplateGenerator
from pdbfixer import PDBFixer
from openff.toolkit.topology import Molecule

# --- FAIL-1: Manifest-driven, absolutely-anchored path resolution ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE1_DIR = PROJECT_ROOT / "Stages" / "Stage 1"
STAGE4_DIR = PROJECT_ROOT / "Stages" / "Stage 4"
STAGE6_DIR = Path(__file__).resolve().parent
RESULTS_DIR = STAGE6_DIR / "results"

LEAD_MANIFEST = STAGE4_DIR / "results" / "verification" / "lead_selection.json"

STD_RESIDUES = frozenset([
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'CYX', 'HID', 'HIE', 'HIP', 'ASH', 'GLH', 'LYN',
])

def require_file(path: Path, label: str) -> Path:
    """Fail-loud file existence and non-empty check."""
    if not path.exists():
        raise FileNotFoundError(f"Stage 6 contract violation: {label} not found at {path}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"Stage 6 contract violation: {label} is empty at {path}")
    return path

def load_lead_pose() -> Path:
    """Resolve the BMS-986205 pose via the Stage 4 lead-selection manifest."""
    require_file(LEAD_MANIFEST, "Stage 4 lead-selection manifest")
    with open(LEAD_MANIFEST) as fh:
        manifest = json.load(fh)
    if manifest.get("lead_name") != "BMS-986205":
        raise RuntimeError(
            f"Stage 6 contract violation: expected lead BMS-986205, "
            f"got {manifest.get('lead_name')!r}"
        )
    pose_value = manifest.get("pose_path")
    if not pose_value:
        raise RuntimeError("Stage 6 contract violation: manifest missing 'pose_path'")
    pose_path = Path(pose_value)
    if not pose_path.is_absolute():
        pose_path = PROJECT_ROOT / pose_path
    return require_file(pose_path, "Stage 4 selected best pose")

# Configuration - all paths absolutely anchored
RAW_RECEPTOR_PDB = require_file(
    STAGE1_DIR / "data" / "9KNZ_raw.pdb",
    "Stage 1 canonical raw receptor"
)

# --- Matrix-of-4 case configuration ---
BMS_SMILES = "C[C@@H](C(=O)Nc1ccc(Cl)cc1)[C@H]1CC[C@@H](c2ccnc3ccc(F)cc32)CC1"

_erdrp_sdf_path = STAGE1_DIR / "data" / "ERDRP_template.sdf"
require_file(_erdrp_sdf_path, "Stage 1 ERDRP template SDF")
_erdrp_mol = Chem.MolFromMolFile(str(_erdrp_sdf_path))
if _erdrp_mol is None:
    raise RuntimeError(f"Failed to parse ERDRP template SDF at {_erdrp_sdf_path}")
ERDRP_SMILES = Chem.MolToSmiles(_erdrp_mol)

STAGE5_MATRIX = PROJECT_ROOT / "Stages" / "Stage 5" / "results" / "matrix"

CASES = [
    {
        "case_id": "A_ERDRP_WT",
        "label": "Control",
        "drug": "ERDRP-0519",
        "receptor": "WT",
        "smiles": ERDRP_SMILES,
        "pose_pdbqt": require_file(
            STAGE5_MATRIX / "ERDRP_WT_seed_42.pdbqt",
            "Stage 5 ERDRP WT seed-42 pose"
        ),
        "apply_mutation": False,
    },
    {
        "case_id": "B_ERDRP_MUT",
        "label": "Failure",
        "drug": "ERDRP-0519",
        "receptor": "W730A",
        "smiles": ERDRP_SMILES,
        "pose_pdbqt": require_file(
            STAGE5_MATRIX / "ERDRP_MUT_seed_42.pdbqt",
            "Stage 5 ERDRP MUT seed-42 pose"
        ),
        "apply_mutation": True,
    },
    {
        "case_id": "C_BMS_WT",
        "label": "Success",
        "drug": "BMS-986205",
        "receptor": "WT",
        "smiles": BMS_SMILES,
        "pose_pdbqt": load_lead_pose(),
        "apply_mutation": False,
    },
    {
        "case_id": "D_BMS_MUT",
        "label": "Resilience",
        "drug": "BMS-986205",
        "receptor": "W730A",
        "smiles": BMS_SMILES,
        "pose_pdbqt": require_file(
            STAGE5_MATRIX / "BMS_MUT_seed_42.pdbqt",
            "Stage 5 BMS MUT seed-42 pose"
        ),
        "apply_mutation": True,
    },
]

def setup_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def repair_ligand_bonds(pdbqt_path, smiles, output_sdf):
    print(f"[1/4] Repairing Ligand Bonds...")
    print(f"      Input: {pdbqt_path}")
    print(f"      SMILES: {smiles}")
    
    # 1. Load SMILES (Topology Source)
    ref_mol = Chem.MolFromSmiles(smiles)
    if ref_mol is None:
        raise ValueError("Invalid SMILES string")
    ref_mol = Chem.AddHs(ref_mol, addCoords=True) # Add 3D coords to reference for shape alignment if needed (optional)
    
    # 2. Extract Coordinates from PDBQT (Model 1)
    # We parse manually to ensure we get the first model and coordinates correctly
    lines = []
    with open(pdbqt_path, 'r') as f:
        in_model_1 = False
        for line in f:
            if line.startswith("MODEL 1"):
                in_model_1 = True
            elif line.startswith("ENDMDL"):
                if in_model_1: break
            elif in_model_1 and (line.startswith("ATOM") or line.startswith("HETATM")):
                # Filter for Ligand Only (UNL) to exclude flexible sidechains (e.g., TRP)
                # PDB residue name is cols 17-20 (0-indexed: 17,18,19)
                res_name = line[17:20]
                if "UNL" in res_name:
                    lines.append(line[:66])
    
    pdb_block = "\n".join(lines)
    # Load COORDINATES (Keep Hs for now, but we might strip them for matching)
    mol_coords = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    
    if not mol_coords:
        raise ValueError("Failed to parse coordinates from PDBQT")

    # 3. Assign Bond Orders
    print("      Assigning bond orders from SMILES...")
    try:
        # STRATEGY: Heavy Atom Match Only
        # Vina PDBQT Hs can be messy. SMILES Hs are implicit.
        # We strip Hs from both for the topological match.
        
        ref_mol_no_h = Chem.RemoveHs(ref_mol)
        mol_coords_no_h = Chem.RemoveHs(mol_coords)
        
        # Assign bonds to the heavy-atom coordinate mol using the heavy-atom SMILES ref
        new_mol_no_h = AllChem.AssignBondOrdersFromTemplate(ref_mol_no_h, mol_coords_no_h)
        
        # Now we have heavy atoms with correct bonds and 3D coords.
        # We need to add Protonation for pH 7.4.
        # Simple AddHs() adds to satisfy valence.
        new_mol = Chem.AddHs(new_mol_no_h, addCoords=True)
        
        # Optimization (Optional): Constrained minimization of just the Hs? 
        # For now, just generate coords for Hs. RDKit AddHs(addCoords=True) does reasonable geometry.
        
    except Exception as e:
        print(f"      Error assigning bond orders: {e}")
        # Fallback: Try looser matching or different SMILES reader if needed.
        raise ValueError(f"Bond Assignment Failed: {e}")
        
    # 4. Protonation Fix (pH 7.4)
    # We have heavy atoms with correct topology. Now we ensure Hs are correct.
    # The Vina output might have Hs but they might be geometric only.
    # Safe bet: AddHs based on valency.
    new_mol.UpdatePropertyCache(strict=False)
    
    # 5. Save SDF
    w = Chem.SDWriter(output_sdf)
    w.write(new_mol)
    w.close()
    print(f"      Saved Corrected Ligand: {output_sdf}")
    return output_sdf

def prepare_protein_and_build_system(raw_pdb_path, ligand_sdf_path,
                                      apply_mutation=False, output_dir=None):
    if output_dir is None:
        output_dir = str(RESULTS_DIR)
    print(f"[2/4] Preparing Protein & Building System...")
    
    # 1. Load Raw PDB & Clean with PDBFixer (Handles Missing Atoms/Caps)
    print(f"      Loading raw PDB into PDBFixer: {raw_pdb_path}")
    fixer = PDBFixer(filename=raw_pdb_path)
    
    chains_to_remove_indices = []
    for i, chain in enumerate(fixer.topology.chains()):
        keep = False
        if chain.id == 'A':
            keep = True
        else:
            for res in chain.residues():
                if res.name == 'ZN':
                    keep = True
                    break
        if not keep:
            chains_to_remove_indices.append(i)
            
    if chains_to_remove_indices:
        print(f"      Removing {len(chains_to_remove_indices)} unwanted chains...")
        fixer.removeChains(chains_to_remove_indices)

    # --- Apply W730A mutation if MUT case (must precede A1E removal) ---
    if apply_mutation:
        print("      Applying W730A mutation (TRP-730 -> ALA-730)...")
        fixer.applyMutations(["TRP-730-ALA"], "A")
        found_ala_730 = False
        for res in fixer.topology.residues():
            if res.chain.id == 'A' and res.id == '730' and res.name == 'ALA':
                found_ala_730 = True
                break
        if not found_ala_730:
            raise RuntimeError(
                "Stage 6 assertion failure: W730A mutation was not applied. "
                "Residue 730 in chain A is not ALA."
            )
        print("      W730A mutation applied and verified.")

    # --- FAIL-2: Remove co-crystallized ligands BEFORE fixer repair ---
    # This must happen on fixer state before findMissingAtoms/addMissingHydrogens
    modeller_clean = app.Modeller(fixer.topology, fixer.positions)
    atoms_to_remove = []
    removed_residues = []
    for res in modeller_clean.topology.residues():
        if res.name not in STD_RESIDUES and res.name not in ['ZN', 'HOH', 'WAT']:
            removed_residues.append(f"{res.name} {res.id}")
            for atom in res.atoms():
                atoms_to_remove.append(atom)
    if atoms_to_remove:
        print(f"      Removing {len(removed_residues)} non-standard residues: "
              f"{', '.join(removed_residues)}")
        modeller_clean.delete(atoms_to_remove)

    # Write cleaned protein to temp PDB and reload into fresh PDBFixer
    tmp_clean = os.path.join(output_dir, "_cleaned_for_fixer.pdb")
    with open(tmp_clean, 'w') as f:
        app.PDBFile.writeFile(modeller_clean.topology, modeller_clean.positions, f)
    fixer = PDBFixer(filename=tmp_clean)

    # --- Bug 23 mitigation: disable missing-residue loop building ---
    fixer.findMissingResidues()
    fixer.missingResidues = {}  # CRITICAL: prevents Stadium Bug (Bug 23)

    # --- Repair missing atoms and add hydrogens ---
    print("      Adding missing atoms (including terminal caps)...")
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    print("      Adding hydrogens (pH 7.4)...")
    fixer.addMissingHydrogens(7.4)

    # Single Modeller from this point forward - no state loss
    # Note: fixer.addMissingHydrogens(7.4) above already added hydrogens;
    # do NOT call modeller.addHydrogens() again to avoid duplicate protons.
    modeller = app.Modeller(fixer.topology, fixer.positions)

    # --- Post-cleanup assertion ---
    non_std_remaining = []
    for res in modeller.topology.residues():
        if res.name not in STD_RESIDUES and res.name not in ['ZN', 'HOH', 'WAT']:
            non_std_remaining.append(f"{res.name} {res.id}")
    if non_std_remaining:
        raise RuntimeError(
            f"Stage 6 assertion failure: {len(non_std_remaining)} non-standard "
            f"residues remain after cleanup: {non_std_remaining}. "
            f"A1E/ERDRP co-crystal ligand was not successfully removed."
        )
    print("      Post-cleanup assertion: 0 non-standard residues remain. PASS.")

    # 3. Handle CYS -> CYX (Zinc Coordination) - NOW AFTER HYDROGENS
    # Identify CYS residues near ZN (from saved logic above, need to re-find or move logic)
    
    # We need to re-scan because atom indices changed after adding protons!
    # But positions of heavy atoms (S, ZN) are stable.
    
    # Re-build topology/positions map
    # ... actually, we can just find them by residue index if we tracked it?
    # Modeller adds atoms but usually preserves residue order.
    # Let's simple re-run the distance check on the CURRENT modeller state.
    
    print("      Identifying Zn-coordinating Cysteines...")
    # Update positions
    positions = modeller.positions
    coords = positions.value_in_unit(unit.nanometers)
    
    zn_atoms = [atom for atom in modeller.topology.atoms() if atom.residue.name == 'ZN']
    cys_to_fix = []
    
    for zn in zn_atoms:
        zn_pos = coords[zn.index]
        # Search all atoms? Optimized: Search CYS SG only
        for atom in modeller.topology.atoms():
             if atom.residue.name == 'CYS' and atom.name == 'SG':
                 dist_sq = sum((zn_pos[i]-coords[atom.index][i])**2 for i in range(3))
                 if dist_sq < 0.09: # 3.0 A = 0.3 nm -> 0.09 nm^2
                     if atom.residue not in cys_to_fix:
                         cys_to_fix.append(atom.residue)
                         
    print(f"      Renaming {len(cys_to_fix)} CYS to CYX and removing HG...")
    atoms_to_delete = []
    for res in cys_to_fix:
        print(f"        - {res.name} {res.id} {res.chain.id} -> CYX")
        res.name = 'CYX'
        # Find HG and mark for deletion
        for atom in res.atoms():
            if atom.name == 'HG':
                atoms_to_delete.append(atom)
                
    if atoms_to_delete:
        print(f"      Removing {len(atoms_to_delete)} HG atoms from CYX residues.")
        modeller.delete(atoms_to_delete)
    
    # 5. Merge Ligand
    print("      Merging Ligand...")
    ligand_mol = Molecule.from_file(ligand_sdf_path)
    ligand_top = ligand_mol.to_topology()
    # We need to align ligand positions?
    # The SDF has positions from Vina.
    # Modeller needs OpenMM topology and positions.
    # OpenFF to OpenMM
    off_top = ligand_mol.to_topology()
    omm_top = off_top.to_openmm()

    # Debug: Print type/content
    raw_pos = ligand_mol.conformers[0]
    print(f"      Ligand Conformer Type: {type(raw_pos)}")
    
    # BULLETPROOF CONVERSION:
    # 1. Ensure we have a quantity (OpenFF usually gives Angstroms)
    if not unit.is_quantity(raw_pos):
        raw_pos = raw_pos * unit.angstroms
        
    # 2. Convert to Nanometers (standard OpenMM unit)
    raw_pos_nm = raw_pos.value_in_unit(unit.nanometers)
    
    # 3. Create list of Vec3
    # raw_pos_nm should be a numpy array of shape (N, 3)
    omm_pos_list = []
    for i in range(len(raw_pos_nm)):
        x, y, z = raw_pos_nm[i]
        omm_pos_list.append(mm.Vec3(x, y, z))
        
    # 4. Wrap matches Modeller requirement (List of Vec3, or Quantity wrapping it)
    # Ideally Modeller wants a list of coordinates. If we pass a list of Vec3, 
    # it might complain if the existing list allows units?
    # Modeller.positions is a Quantity. We should pass a Quantity.
    final_pos = unit.Quantity(omm_pos_list, unit.nanometers)
    
    modeller.add(omm_top, final_pos)
    
    # 6. Build System (Solvate & Ionize)
    print("      Solvating (TIP3P, Padding 1.0nm)...")
    ff = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    
    # Create Template for Ligand (GAFF2)
    print("      Generating Ligand Parameters (GAFF2 + AM1-BCC)...")
    gaff = GAFFTemplateGenerator(molecules=ligand_mol)
    ff.registerTemplateGenerator(gaff.generator)
    
    modeller.addSolvent(ff, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
    
    # 7. Create System
    print("      Creating OpenMM System...")
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, 
                             nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds)
                             
    # FAIL-4(a): Barostat removed from build phase; deferred to NPT phase in minimize_and_equilibrate()

    # Save System & State
    out_xml = os.path.join(output_dir, "system.xml")
    with open(out_xml, 'w') as f:
        f.write(mm.XmlSerializer.serialize(system))
    print(f"      Saved System XML: {out_xml}")
    
    out_pdb = os.path.join(output_dir, "topology.pdb")
    with open(out_pdb, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"      Saved System PDB: {out_pdb}")
    
    return system, modeller.topology, modeller.positions

def minimize_and_equilibrate(system, topology, positions, results_dir):
    """
    Full minimization + NVT + NPT equilibration protocol.
    Returns dict of output artifact paths.
    """
    print("[3/4] Minimization & Equilibration...")

    # --- Add positional restraints BEFORE creating Simulation ---
    force = mm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
    force.addGlobalParameter("k", 1000.0 * unit.kilojoules_per_mole / unit.nanometers**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    restrained_count = 0
    for atom in topology.atoms():
        is_backbone = (atom.name in ['CA', 'C', 'N'] and
                       atom.residue.name not in ['HOH', 'NA', 'CL', 'ZN', 'WAT'])
        is_ligand_heavy = (atom.residue.name not in
                           ['HOH', 'NA', 'CL', 'ZN', 'WAT']
                           and atom.residue.name not in STD_RESIDUES
                           and atom.element.symbol != 'H')
        if is_backbone or is_ligand_heavy:
            force.addParticle(atom.index, positions[atom.index])
            restrained_count += 1
    system.addForce(force)
    print(f"      Restrained {restrained_count} atoms (backbone + ligand heavy).")

    # --- Create Simulation ---
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # --- Minimize ---
    print("      Minimizing energy...")
    simulation.minimizeEnergy()
    min_state = simulation.context.getState(getEnergy=True)
    print(f"      Minimized. PE = {min_state.getPotentialEnergy()}")

    # --- NVT Equilibration (100 ps, no barostat) ---
    print("      Running NVT Equilibration (100 ps, Restrained)...")
    simulation.step(50000)  # 50000 * 2 fs = 100 ps, true NVT

    # --- Add barostat for NPT phase ---
    print("      Adding MonteCarloBarostat for NPT phase...")
    barostat = mm.MonteCarloBarostat(
        1.0 * unit.atmospheres, 300.0 * unit.kelvin, 250  # Bug 22 compliant
    )
    system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)

    # --- NPT Equilibration (200 ps) ---
    print("      Running NPT Equilibration (200 ps, Restrained)...")
    simulation.step(100000)  # 100000 * 2 fs = 200 ps, true NPT

    # --- Save outputs ---
    results_dir_path = Path(results_dir)
    chk_path = results_dir_path / "equilibrated_production_ready.chk"
    pdb_path = results_dir_path / "equilibrated_production_ready.pdb"

    simulation.saveCheckpoint(str(chk_path))
    state_xml_path = results_dir_path / "equilibrated_production_ready_state.xml"
    simulation.saveState(str(state_xml_path))
    print(f"      Saved state XML: {state_xml_path}")
    final_pos = simulation.context.getState(getPositions=True).getPositions()
    with open(str(pdb_path), 'w') as f:
        app.PDBFile.writeFile(topology, final_pos, f)

    print(f"      Saved checkpoint: {chk_path}")
    print(f"      Saved equilibrated PDB: {pdb_path}")
    print("[4/4] Reconstruction Complete.")

    return {
        "equilibrated_checkpoint": str(chk_path),
        "equilibrated_pdb": str(pdb_path),
        "equilibrated_state_xml": str(state_xml_path),
    }

if __name__ == "__main__":
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(
        description="Stage 6: Matrix of 4 System Reconstruction"
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Run only this case ID (e.g., 'C_BMS_WT'). Default: run all."
    )
    args = parser.parse_args()

    cases_to_run = CASES
    if args.case:
        cases_to_run = [c for c in CASES if c["case_id"] == args.case]
        if not cases_to_run:
            valid = [c["case_id"] for c in CASES]
            print(f"FATAL: Unknown case '{args.case}'. Valid: {valid}")
            sys.exit(1)

    setup_dirs()

    all_case_outputs = {}
    for case in cases_to_run:
        case_id = case["case_id"]
        case_dir = RESULTS_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_output_dir = str(case_dir)

        print(f"\n{'='*80}")
        print(f"  CASE {case_id}: {case['drug']} x {case['receptor']} ({case['label']})")
        print(f"{'='*80}")

        # Step 1: Repair ligand bonds
        repaired_sdf = os.path.join(case_output_dir, "ligand_repaired.sdf")
        try:
            repair_ligand_bonds(str(case["pose_pdbqt"]), case["smiles"], repaired_sdf)
        except Exception as e:
            print(f"FATAL [{case_id}]: Ligand repair failed: {e}")
            sys.exit(1)

        # Step 2: Prepare protein & build system
        try:
            system, topology, positions = prepare_protein_and_build_system(
                str(RAW_RECEPTOR_PDB), repaired_sdf,
                apply_mutation=case["apply_mutation"],
                output_dir=case_output_dir,
            )
        except Exception as e:
            print(f"FATAL [{case_id}]: System build failed: {e}")
            sys.exit(1)

        # Step 3: Minimize & equilibrate
        try:
            eq_outputs = minimize_and_equilibrate(
                system, topology, positions, case_output_dir
            )
        except Exception as e:
            print(f"FATAL [{case_id}]: Equilibration failed: {e}")
            sys.exit(1)

        all_case_outputs[case_id] = {
            "case_id": case_id,
            "drug": case["drug"],
            "receptor": case["receptor"],
            "label": case["label"],
            "pose_source": str(case["pose_pdbqt"]),
            "system_xml": str(case_dir / "system.xml"),
            "topology_pdb": str(case_dir / "topology.pdb"),
            "equilibrated_checkpoint": eq_outputs["equilibrated_checkpoint"],
            "equilibrated_pdb": eq_outputs["equilibrated_pdb"],
            "equilibrated_state_xml": eq_outputs["equilibrated_state_xml"],
        }

        # Write per-case manifest fragment (used by parallel merge in orchestrator)
        case_manifest = case_dir / "_case_manifest.json"
        with open(case_manifest, "w") as fh:
            json.dump(all_case_outputs[case_id], fh, indent=2)

    # Write unified Stage 6 manifest
    manifest = {
        "stage": 6,
        "raw_receptor_source": str(RAW_RECEPTOR_PDB),
        "cases": all_case_outputs,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
    }
    manifest_path = RESULTS_DIR / "stage6_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nStage 6 manifest written: {manifest_path}")
    print(f"Cases completed: {list(all_case_outputs.keys())}")

