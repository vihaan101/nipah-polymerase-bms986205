import unittest
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the script directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "analysis_10ns_direct"))

from stage7_eval_common import robust_universe

class TestRobustUniverse(unittest.TestCase):
    def test_robust_universe_success(self):
        # Mock mda.Universe to succeed on first try
        with patch("MDAnalysis.Universe") as mock_universe:
            mock_u = MagicMock()
            mock_universe.return_status = mock_u
            
            u = robust_universe("topo.pdb", "traj.dcd")
            self.assertEqual(mock_universe.call_count, 1)

    def test_robust_universe_retry_success(self):
        # Mock mda.Universe to fail twice and then succeed
        with patch("MDAnalysis.Universe") as mock_universe:
            mock_u = MagicMock()
            mock_universe.side_effect = [
                Exception("Reading DCD header failed"),
                Exception("StopIteration"),
                mock_u
            ]
            
            with patch("time.sleep") as mock_sleep:
                u = robust_universe("topo.pdb", "traj.dcd", attempts=5, delay=0)
                self.assertEqual(mock_universe.call_count, 3)
                self.assertEqual(mock_sleep.call_count, 2)

    def test_robust_universe_fail_all(self):
        # Mock mda.Universe to fail all attempts
        with patch("MDAnalysis.Universe") as mock_universe:
            mock_universe.side_effect = Exception("Reading DCD header failed")
            
            with patch("time.sleep") as mock_sleep:
                with self.assertRaises(Exception):
                    robust_universe("topo.pdb", "traj.dcd", attempts=3, delay=0)
                self.assertEqual(mock_universe.call_count, 3)

if __name__ == "__main__":
    unittest.main()
