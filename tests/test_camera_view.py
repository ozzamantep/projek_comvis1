import unittest
from unittest.mock import patch

import cv2
import numpy as np

from main import _draw_split_camera_view


class CameraViewTests(unittest.TestCase):
    def test_split_view_has_two_labeled_panels_and_center_depth(self):
        detection_view = np.zeros((100, 120, 3), dtype=np.uint8)
        depth_view = np.zeros_like(detection_view)
        depth_m = np.full((100, 120), 2.5, dtype=np.float32)
        drawn_text = []
        put_text = cv2.putText

        def record_text(*args, **kwargs):
            drawn_text.append(args[1])
            return put_text(*args, **kwargs)

        with patch("main.cv2.putText", side_effect=record_text):
            display = _draw_split_camera_view(
                detection_view, depth_view, depth_m, patch_size=5
            )

        self.assertEqual(display.shape, (100, 240, 3))
        self.assertGreater(np.count_nonzero(display[:, :120]), 0)
        self.assertGreater(np.count_nonzero(display[:, 120:]), 0)
        self.assertEqual(
            drawn_text,
            [
                "DETEKSI + JARAK",
                "Jarak tiap objek ditulis pada kotak",
                "PETA DEPTH + JARAK",
                "Depth tengah: 2.50 m",
            ],
        )

    def test_split_view_rejects_different_panel_sizes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            _draw_split_camera_view(
                np.zeros((20, 30, 3), dtype=np.uint8),
                np.zeros((20, 31, 3), dtype=np.uint8),
                np.ones((20, 30), dtype=np.float32),
                patch_size=3,
            )


if __name__ == "__main__":
    unittest.main()
