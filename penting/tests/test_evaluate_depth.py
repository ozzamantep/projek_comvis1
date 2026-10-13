import unittest

import numpy as np

from main import DepthMetricAccumulator, compute_depth_metrics


class EvaluateDepthTests(unittest.TestCase):
    def test_perfect_prediction_metrics(self):
        depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        metrics = compute_depth_metrics(depth, depth)
        self.assertEqual(metrics["delta1"], 1.0)
        self.assertEqual(metrics["abs_rel"], 0.0)
        self.assertEqual(metrics["rmse_m"], 0.0)
        self.assertEqual(metrics["silog"], 0.0)
        self.assertEqual(metrics["valid_pixels"], 4)

    def test_invalid_pixels_are_masked(self):
        prediction = np.array([[2.0, np.nan], [8.0, 0.0]])
        target = np.array([[2.0, 4.0], [4.0, 1.0]])
        mask = np.array([[True, True], [True, False]])
        metrics = compute_depth_metrics(prediction, target, mask)
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertAlmostEqual(metrics["rmse_m"], np.sqrt(8.0))

    def test_accumulator_aggregates_pixels_across_images(self):
        accumulator = DepthMetricAccumulator()
        accumulator.update(np.array([1.0]), np.array([1.0]))
        accumulator.update(np.array([2.0]), np.array([1.0]))
        metrics = accumulator.compute()
        self.assertEqual(metrics["valid_pixels"], 2)
        self.assertAlmostEqual(metrics["rmse_m"], np.sqrt(0.5))

    def test_all_invalid_pixels_is_an_error(self):
        with self.assertRaises(ValueError):
            compute_depth_metrics(
                np.array([0.0, np.nan]),
                np.array([1.0, 2.0]),
            )


if __name__ == "__main__":
    unittest.main()
