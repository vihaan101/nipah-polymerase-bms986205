from __future__ import annotations

import unittest

import numpy as np

from pathlib import Path
import sys


TESTS_DIR = Path(__file__).resolve().parent
STAGE7_DIR = TESTS_DIR.parent
sys.path.insert(0, str(STAGE7_DIR))

from stage7_eval_common import minimum_image_displacements, minimum_image_distance


class Stage7PBCMetricTests(unittest.TestCase):
    def test_minimum_image_displacements_collapses_box_wrap_jump(self) -> None:
        box = np.array([117.0, 117.0, 117.0])
        wrapped = np.array([[116.7, 0.2, -0.1]])
        corrected = minimum_image_displacements(wrapped, box)
        self.assertAlmostEqual(float(corrected[0, 0]), -0.3, places=6)
        self.assertAlmostEqual(float(corrected[0, 1]), 0.2, places=6)
        self.assertAlmostEqual(float(corrected[0, 2]), -0.1, places=6)

    def test_minimum_image_distance_avoids_false_large_com_separation(self) -> None:
        box = np.array([117.0, 117.0, 117.0])
        point_a = np.array([116.6, 20.0, 20.0])
        point_b = np.array([0.4, 20.0, 20.0])
        raw = float(np.linalg.norm(point_a - point_b))
        corrected = minimum_image_distance(point_a, point_b, box)
        self.assertGreater(raw, 100.0)
        self.assertLess(corrected, 1.0)

    def test_invalid_box_lengths_fall_back_to_raw_vectors(self) -> None:
        vectors = np.array([[116.7, 0.0, 0.0]])
        corrected = minimum_image_displacements(vectors, [0.0, 117.0, 117.0])
        self.assertTrue(np.array_equal(corrected, vectors))


if __name__ == "__main__":
    unittest.main()
