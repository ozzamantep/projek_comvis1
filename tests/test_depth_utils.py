import unittest

import numpy as np

from main import (
    DepthEMASmoother,
    DEFAULT_CAMERA_DEPTH_SCALE,
    calibrate_depth_map,
    get_depth_at_point,
    get_depth_in_bbox,
    get_depth_patch_stats,
    meters_to_cm,
    normalize_depth_for_display,
)


class DepthUtilsTests(unittest.TestCase):
    def test_webcam_depth_calibration_uses_empirical_scale(self):
        calibrated = calibrate_depth_map(
            np.array([[1.6, 1.7, 1.8]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            calibrated,
            np.array([[0.7948718, 0.8445513, 0.8942308]], dtype=np.float32),
            rtol=1e-6,
        )
        self.assertAlmostEqual(
            DEFAULT_CAMERA_DEPTH_SCALE, 0.4967948718, places=8
        )

    def test_depth_calibration_rejects_non_positive_or_non_finite_scale(self):
        for scale in (0.0, -1.0, np.nan, np.inf):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                calibrate_depth_map(np.ones((2, 2), dtype=np.float32), scale)

    def test_bbox_uses_median_patch_around_center(self):
        depth = np.full((7, 7), 2.0, dtype=np.float32)
        depth[3, 3] = 8.0
        depth[2, 2] = np.nan
        self.assertEqual(get_depth_in_bbox(depth, (1, 1, 5, 5)), 2.0)

    def test_point_ignores_invalid_values(self):
        depth = np.array([[np.nan, 3.0], [0.0, 5.0]], dtype=np.float32)
        self.assertEqual(get_depth_at_point(depth, 0, 0, patch_size=3), 4.0)

    def test_patch_stats_return_center_patch_shape_and_raw_variance(self):
        depth = np.array([[0.2, 0.4], [0.6, np.nan]], dtype=np.float32)
        stats = get_depth_patch_stats(depth, 0, 0, patch_size=3)
        self.assertAlmostEqual(stats.center_value, 0.2)
        self.assertAlmostEqual(stats.median, 0.4)
        self.assertAlmostEqual(stats.variance, np.var([0.2, 0.4, 0.6]))
        self.assertEqual(stats.valid_count, 3)
        self.assertEqual(stats.patch_shape, (2, 2))

    def test_out_of_bounds_point_is_rejected(self):
        with self.assertRaises(ValueError):
            get_depth_at_point(np.ones((2, 2), dtype=np.float32), 2, 0)

    def test_ema_smooths_frames_and_resets_on_shape_change(self):
        smoother = DepthEMASmoother(alpha=0.25)
        np.testing.assert_allclose(smoother.update(np.full((2, 2), 2.0)), 2.0)
        np.testing.assert_allclose(smoother.update(np.full((2, 2), 6.0)), 3.0)
        np.testing.assert_allclose(smoother.update(np.full((1, 2), 7.0)), 7.0)

    def test_display_normalization_does_not_change_metric_input(self):
        metric_depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        original = metric_depth.copy()
        display = normalize_depth_for_display(metric_depth)
        self.assertEqual(display.dtype, np.uint8)
        np.testing.assert_array_equal(metric_depth, original)


if __name__ == "__main__":
    unittest.main()
