"""Shim: canonical implementation lives in ``md_eval.stage7_eval_common``."""

import sys
from pathlib import Path

_COMMON = Path(__file__).resolve().parents[2] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from md_eval.stage7_eval_common import *  # noqa: F403
