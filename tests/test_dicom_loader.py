"""
Unit tests for DICOM loading utilities.
"""

import unittest

import numpy as np
import pandas as pd

from src.data.metadata import expand_coordinates, parse_coordinates


class TestDICOMMetadata(unittest.TestCase):
    """Test DICOM metadata parsing."""

    def test_parse_coordinates_with_xyz(self):
        """Test parsing coordinates with all dimensions."""
        coord_str = "{'x': 100.5, 'y': 200.3, 'z': 50.7}"

        x, y, z = parse_coordinates(coord_str)

        self.assertAlmostEqual(x, 100.5)
        self.assertAlmostEqual(y, 200.3)
        self.assertAlmostEqual(z, 50.7)

    def test_parse_coordinates_without_z(self):
        """Test parsing coordinates without z dimension."""
        coord_str = "{'x': 100.5, 'y': 200.3}"

        x, y, z = parse_coordinates(coord_str)

        self.assertAlmostEqual(x, 100.5)
        self.assertAlmostEqual(y, 200.3)
        self.assertEqual(z, 0.0)  # Default value

    def test_parse_coordinates_invalid(self):
        """Test parsing invalid coordinates."""
        coord_str = "invalid string"

        x, y, z = parse_coordinates(coord_str)

        # Should return defaults
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0.0)
        self.assertEqual(z, 0.0)

    def test_expand_coordinates(self):
        """Test expanding coordinates column in DataFrame."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "SeriesInstanceUID": ["123", "456"],
                "coordinates": [
                    "{'x': 100.5, 'y': 200.3, 'z': 50.7}",
                    "{'x': 150.2, 'y': 250.8}",
                ],
            }
        )

        df_expanded = expand_coordinates(df)

        # Check new columns exist
        self.assertIn("x", df_expanded.columns)
        self.assertIn("y", df_expanded.columns)
        self.assertIn("z", df_expanded.columns)

        # Check values
        self.assertAlmostEqual(df_expanded.iloc[0]["x"], 100.5)
        self.assertAlmostEqual(df_expanded.iloc[0]["y"], 200.3)
        self.assertAlmostEqual(df_expanded.iloc[0]["z"], 50.7)

        # Second row should have z=0
        self.assertEqual(df_expanded.iloc[1]["z"], 0.0)


if __name__ == "__main__":
    unittest.main()
