import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..calibrate_depth import fit_calibration, load_validation_pairs


class CalibrationTests(unittest.TestCase):
    def test_ratio_fit_uses_median_measured_to_predicted_ratio(self):
        fit = fit_calibration(
            np.array([2.0, 4.0, 10.0]),
            np.array([1.0, 2.0, 6.0]),
            "ratio",
        )
        self.assertEqual(fit.slope, 0.5)
        self.assertEqual(fit.intercept_m, 0.0)
        self.assertAlmostEqual(fit.apply(8.0), 4.0)

    def test_linear_fit_recovers_scale_and_offset(self):
        fit = fit_calibration(
            np.array([1.0, 2.0, 4.0]),
            np.array([1.5, 2.5, 4.5]),
            "linear",
        )
        self.assertAlmostEqual(fit.slope, 1.0)
        self.assertAlmostEqual(fit.intercept_m, 0.5)

    def test_linear_fit_requires_distinct_predictions(self):
        with self.assertRaises(ValueError):
            fit_calibration(np.array([1.0, 1.0]), np.array([1.0, 2.0]), "linear")

    def test_csv_loader_skips_empty_template_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "measurements.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(
                    csv_file, fieldnames=["measured_m", "predicted_m"]
                )
                writer.writeheader()
                writer.writerow({"measured_m": "0.6", "predicted_m": "2.29"})
                writer.writerow({"measured_m": "", "predicted_m": ""})

            predicted, measured, skipped = load_validation_pairs(str(csv_path))

        np.testing.assert_allclose(predicted, [2.29])
        np.testing.assert_allclose(measured, [0.6])
        self.assertEqual(skipped, 1)


if __name__ == "__main__":
    unittest.main()
