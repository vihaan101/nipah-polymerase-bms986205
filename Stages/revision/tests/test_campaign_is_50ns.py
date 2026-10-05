"""Campaign scripts are the 50 ns direct protocol only."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

_OLD_PRODUCTION_LENGTH = re.compile(r"(?<![0-9_])5_000_000")


STAGES = Path(__file__).resolve().parents[2]
PRODUCTION = STAGES / "Stage 7" / "run_production_50ns_direct.py"
OLD_PRODUCTION = STAGES / "Stage 7" / "run_production_10ns_direct_v2.py"


def campaign_files() -> list[Path]:
    files = [
        PRODUCTION,
        STAGES / "common" / "md_eval" / "stage7_eval_common.py",
        STAGES / "common" / "md_eval" / "verify_eval_ready_trajectories.py",
    ]
    md_eval = STAGES / "common" / "md_eval"
    files.extend(sorted(md_eval.glob("analyze_*.py")))
    analysis = STAGES / "Stage 7" / "analysis_50ns_direct"
    if analysis.is_dir():
        files.extend(p for p in analysis.rglob("*.py") if p.is_file())
    return files


class CampaignIs50nsTests(unittest.TestCase):
    def test_production_entry_point_is_50ns_only(self) -> None:
        self.assertTrue(PRODUCTION.is_file(), "run_production_50ns_direct.py is missing")
        self.assertFalse(OLD_PRODUCTION.exists(), "10 ns production entry point still exists")
        text = PRODUCTION.read_text(encoding="utf-8")
        self.assertIn("TOTAL_STEPS   = 25_000_000", text)
        self.assertIn('TIER_LABEL    = "50ns_direct"', text)
        self.assertNotIn("10ns", text.lower())
        self.assertIsNone(_OLD_PRODUCTION_LENGTH.search(text))

    def test_eval_scripts_have_no_10ns_campaign_identifier(self) -> None:
        offenders = []
        for path in campaign_files():
            if not path.is_file():
                offenders.append(f"missing {path}")
                continue
            text = path.read_text(encoding="utf-8")
            if "10ns" in text.lower() or _OLD_PRODUCTION_LENGTH.search(text):
                offenders.append(str(path.relative_to(STAGES)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
