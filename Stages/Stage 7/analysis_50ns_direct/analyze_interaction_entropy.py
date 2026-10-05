"""Backward-compatible wrapper; canonical implementation in md_eval."""
from md_eval.analyze_interaction_entropy import *  # noqa: F403

if __name__ == "__main__":
    from md_eval.analyze_interaction_entropy import main
    main()
