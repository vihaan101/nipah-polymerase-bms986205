import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from md_eval.stage7_eval_common import robust_universe, set_universe_factory


class TestRobustUniverse(unittest.TestCase):
    def tearDown(self) -> None:
        set_universe_factory(None)

    def test_robust_universe_success(self):
        mock_universe = MagicMock()
        set_universe_factory(mock_universe)
        robust_universe("topo.pdb", "traj.dcd")
        self.assertEqual(mock_universe.call_count, 1)

    def test_robust_universe_retry_success(self):
        mock_universe = MagicMock()
        mock_u = MagicMock()
        mock_universe.side_effect = [
            Exception("Reading DCD header failed"),
            Exception("StopIteration"),
            mock_u,
        ]
        set_universe_factory(mock_universe)

        with unittest.mock.patch("time.sleep"):
            u = robust_universe("topo.pdb", "traj.dcd", attempts=5, delay=0)
            self.assertEqual(mock_universe.call_count, 3)
            self.assertEqual(u, mock_u)

    def test_robust_universe_fail_all(self):
        mock_universe = MagicMock(side_effect=Exception("Reading DCD header failed"))
        set_universe_factory(mock_universe)

        with unittest.mock.patch("time.sleep"):
            with self.assertRaises(Exception):
                robust_universe("topo.pdb", "traj.dcd", attempts=3, delay=0)
            self.assertEqual(mock_universe.call_count, 3)


if __name__ == "__main__":
    unittest.main()
