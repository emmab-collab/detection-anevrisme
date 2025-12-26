"""
Unit tests for preprocessing module.
"""
import unittest
import numpy as np
from src.preprocessing.transforms import resample, crop, normalization


class TestPreprocessingTransforms(unittest.TestCase):
    """Test preprocessing transformations."""

    def setUp(self):
        """Create test data."""
        # Create a simple 3D volume
        self.test_volume = np.random.rand(100, 100, 50).astype(np.float32)
        self.test_spacing = (1.0, 1.0, 2.0)  # mm

    def test_normalization(self):
        """Test volume normalization to [0, 1] range."""
        normalized = normalization(self.test_volume)

        # Check output range
        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)

        # Check shape preservation
        self.assertEqual(normalized.shape, self.test_volume.shape)

    def test_crop_removes_background(self):
        """Test that cropping removes zero-padding."""
        # Create volume with zero padding
        padded_volume = np.zeros((150, 150, 80))
        padded_volume[25:125, 25:125, 15:65] = self.test_volume

        cropped, indices = crop(padded_volume)

        # Cropped volume should be smaller
        self.assertLess(cropped.shape[0], padded_volume.shape[0])
        self.assertLess(cropped.shape[1], padded_volume.shape[1])

    def test_resample_changes_spacing(self):
        """Test that resampling changes volume dimensions."""
        target_spacing = (0.5, 0.5, 0.5)  # Higher resolution

        resampled = resample(
            self.test_volume,
            self.test_spacing,
            target_spacing=target_spacing
        )

        # With higher resolution, volume should be larger
        self.assertGreater(resampled.shape[0], self.test_volume.shape[0])
        self.assertGreater(resampled.shape[1], self.test_volume.shape[1])


if __name__ == '__main__':
    unittest.main()
