"""Backward-compatible wrapper; canonical implementation in md_eval."""
from md_eval.analyze_pocket_volume import *  # noqa: F403

if __name__ == "__main__":
    from md_eval.analyze_pocket_volume import main
    main()
