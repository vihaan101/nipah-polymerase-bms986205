"""Highlighted-residue occupancy fails closed when the topology is wrong."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_revision  # noqa: E402


class ContactOccupancyTests(unittest.TestCase):
    def test_missing_residue_id_is_rejected(self) -> None:
        residues = [{"resid": 875, "resname": "HIS"}, {"resid": 806, "resname": "TRP"}]
        with self.assertRaises(ValueError):
            run_revision.contact_occupancy(residues, [])

    def test_unexpected_resname_is_rejected(self) -> None:
        residues = [
            {"resid": 875, "resname": "GLY"},
            {"resid": 806, "resname": "TRP"},
            {"resid": 730, "resname": "ALA"},
        ]
        with self.assertRaises(ValueError):
            run_revision.contact_occupancy(residues, [])

    def test_occupancy_series_follows_frames(self) -> None:
        residues = [
            {"resid": 875, "resname": "HIS"},
            {"resid": 806, "resname": "TRP"},
            {"resid": 730, "resname": "ALA"},
        ]
        frames = [
            {875: 1, 806: 0, 730: 1},
            {875: 1, 806: 1, 730: 0},
        ]
        series = run_revision.contact_occupancy(residues, frames)
        self.assertEqual(series[875], [1, 1])
        self.assertEqual(series[806], [0, 1])
        self.assertEqual(series[730], [1, 0])


if __name__ == "__main__":
    unittest.main()
