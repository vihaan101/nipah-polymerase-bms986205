"""Backward-compatible wrapper; canonical implementation in md_eval."""
from md_eval.analyze_ligand_rmsd import *  # noqa: F403

if __name__ == "__main__":
    from md_eval.analyze_ligand_rmsd import main
    main()
