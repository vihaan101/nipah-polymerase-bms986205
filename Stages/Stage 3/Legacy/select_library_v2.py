
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
import sys

# Paths
STAGE3_DIR = Path(__file__).parent  # Stages/Stage 3/
STAGE3_DATA = STAGE3_DIR / "data"
STAGE3_DATA.mkdir(parents=True, exist_ok=True)
INPUT_CSV = STAGE3_DATA / "expanded_library.csv"
OUTPUT_CSV = STAGE3_DATA / "filtered_library.csv"

def calculate_logp(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Descriptors.MolLogP(mol)
    except:
        pass
    return None

def main():
    print("=== Phase 2: Smart Library Selection ===")
    if not INPUT_CSV.exists():
        print(f"FATAL: {INPUT_CSV} not found. Run select_expanded_library_v2.py first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} compounds from fallback library.")

    # Calculate LogP if missing
    if 'logp' not in df.columns:
        print("Calculating LogP values...")
        df['logp'] = df['smiles'].apply(calculate_logp)

    # Physically defensible ceilings (bRo5 extended drug-likeness)
    MAX_MW = 600
    MAX_LOGP = 5.0

    mw_limit = 500
    logp_limit = 3.0

    while True:
        filtered_df = df[(df['mw'] <= mw_limit) & (df['logp'] < logp_limit)]
        count = len(filtered_df)
        print(f"Filter (MW <= {mw_limit}, LogP < {logp_limit}): {count} compounds found.")

        if count >= 5:
            break

        # Independent relaxation with smaller steps
        new_mw = min(mw_limit + 25, MAX_MW)
        new_logp = min(logp_limit + 0.5, MAX_LOGP)

        if new_mw == mw_limit and new_logp == logp_limit:
            print(f"WARNING: Reached drug-likeness ceiling (MW={mw_limit}, LogP={logp_limit}). "
                  f"Only {count} compounds available.", file=sys.stderr)
            break

        mw_limit = new_mw
        logp_limit = new_logp

    # Save results
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(filtered_df)} compounds to {OUTPUT_CSV}")
    
    # Preview
    print("\nSelected Library Preview:")
    print(filtered_df[['name', 'mw', 'logp']].head())

if __name__ == "__main__":
    main()
