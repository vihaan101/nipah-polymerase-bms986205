"""Backward-compatible wrapper; canonical implementation in md_eval."""
from md_eval.verify_eval_ready_trajectories import *  # noqa: F403

if __name__ == "__main__":
    from md_eval.verify_eval_ready_trajectories import main
    main()
