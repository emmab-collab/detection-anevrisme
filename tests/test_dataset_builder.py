"""
Unit tests for DatasetBuilder.
"""

import unittest
import numpy as np
from src.bricks.dataset import DatasetBuilder
from src.bricks.preprocessing import Preprocessor


class TestDatasetBuilder(unittest.TestCase):
    """Test DatasetBuilder functionality."""

    def setUp(self):
        """Initialize test components."""
        self.preprocessor = Preprocessor()
        self.builder = DatasetBuilder(preprocessor=self.preprocessor, cube_size=48)

    def test_extract_cube_correct_size(self):
        """Test that extracted cubes have correct size."""
        # Create test volume
        volume = np.random.rand(200, 200, 100).astype(np.float32)
        center = np.array([100, 100, 50])

        cube = self.builder.extract_cube(volume, center)

        # Check cube shape
        self.assertEqual(cube.shape, (48, 48, 48))

    def test_extract_cube_with_padding(self):
        """Test cube extraction at volume edges with padding."""
        volume = np.random.rand(200, 200, 100).astype(np.float32)

        # Test near edge
        center = np.array([10, 10, 10])
        cube = self.builder.extract_cube(volume, center)

        # Should still have correct shape due to padding
        self.assertEqual(cube.shape, (48, 48, 48))

    def test_pad_cube(self):
        """Test cube padding functionality."""
        # Create small cube
        small_cube = np.ones((30, 30, 30))

        # Pad to target size
        padded = self.builder._pad_cube(small_cube, target_size=48)

        # Check final shape
        self.assertEqual(padded.shape, (48, 48, 48))

    def test_create_position_vector(self):
        """Test one-hot position vector creation."""
        position_name = "Right Middle Cerebral Artery"

        vector = self.builder.create_position_vector(position_name)

        # Check shape
        self.assertEqual(vector.shape, (13,))

        # Check it's one-hot (only one element is 1.0)
        self.assertEqual(vector.sum(), 1.0)

    def test_extract_non_overlapping_cubes(self):
        """Test non-overlapping cube extraction."""
        volume = np.random.rand(144, 144, 144).astype(np.float32)

        cubes = self.builder.extract_non_overlapping_cubes(volume)

        # Check all cubes have correct size
        for cube in cubes:
            self.assertEqual(cube.shape, (48, 48, 48))

        # With stride=48, we should get 3x3x3 = 27 cubes
        expected_count = 27
        self.assertEqual(len(cubes), expected_count)


if __name__ == "__main__":
    unittest.main()
