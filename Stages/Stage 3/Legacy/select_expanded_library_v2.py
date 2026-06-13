
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import os
import sys

# Paths
STAGE3_DIR = Path(__file__).parent  # Stages/Stage 3/
STAGE3_DATA = STAGE3_DIR / "data"
STAGE3_DATA.mkdir(parents=True, exist_ok=True)
LAKE_DIR = STAGE3_DIR.parent / "data" / "lake"
DRUGS_FILE = LAKE_DIR / "Broad_Repurposing_Hub.csv"
SAMPLES_FILE = LAKE_DIR / "Broad_Repurposing_Samples.csv"
OUTPUT_CSV = STAGE3_DATA / "expanded_library.csv"

def calculate_props(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            return mw, logp
    except:
        pass
    return None, None

def main():
    print("=== Phase 2: Broad Library Expansion ===")
    
    # skip rows with "!" (metadata headers)
    try:
        df_drugs = pd.read_csv(DRUGS_FILE, sep='\t', comment='!')
        df_samples = pd.read_csv(SAMPLES_FILE, sep='\t', comment='!')
    except FileNotFoundError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(df_drugs)} compounds and {len(df_samples)} samples.")
    
    # Merge on 'pert_iname'
    merged = pd.merge(df_samples, df_drugs, on='pert_iname', how='inner')
    print(f"Merged library size: {len(merged)}")
    
    # Deduplicate by InChIKey or SMILES
    merged = merged.drop_duplicates(subset=['smiles'])
    print(f"Unique structures: {len(merged)}")
    
    # Filter for valid SMILES and properties
    valid_data = []
    
    print("Calculating properties... (this may take a moment)")
    # Optimizing: Apply pre-filter on string length to avoid tiny molecules first? 
    # No, RDKit is fast enough for <10k.
    
    count = 0
    for idx, row in merged.iterrows():
        smiles = row['smiles']
        if pd.isna(smiles): continue
        
        mw, logp = calculate_props(smiles)
        if mw is None: continue
        
        # Council Optimized Filters: LogP > 1.5, MW > 250
        if logp > 1.5 and mw > 250:
            valid_data.append({
                "name": row['pert_iname'],
                "smiles": smiles,
                "mw": mw,
                "logp": logp,
                "moa": row.get('moa', 'unknown'),
                "indication": row.get('indication', 'unknown')
            })
        
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count}...")

    final_df = pd.DataFrame(valid_data)
    print(f"Filtered Library (LogP > 1.5, MW > 250): {len(final_df)} compounds.")
    
    # Cap at 100 top candidates for 'Fast' Screening if list is huge
    # Or just save all. Let's save all for now, maybe subsample for the demo run.
    # Protocol says: "Downgrade to Rigid Docking for throughput".
    # Let's limit to 50 diverse candidates if > 50 to keep runtime < 10 mins.
    
    if len(final_df) > 50:
        print("Selecting 50 diverse candidates via MaxMin Tanimoto picking...")
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

        fps = []
        valid_indices = []
        for idx, row in final_df.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                valid_indices.append(idx)

        picker = MaxMinPicker()
        pick_indices = list(picker.LazyBitVectorPick(fps, len(fps), 50, seed=42))
        selected_orig = [valid_indices[i] for i in pick_indices]
        final_df = final_df.loc[selected_orig]
        
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved expanded library to {OUTPUT_CSV}")
    print(final_df[['name', 'mw', 'logp']].head())

if __name__ == "__main__":
    main()
