import threading
import unittest
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from main import create_web_handler


class FakeModel:
    class_names = ("chair",)
    input_width = 640
    input_height = 480

    def predict(self, frame):
        height, width = frame.shape[:2]
        detection = np.array(
            [[width * 0.1, height * 0.1, width * 0.9, height * 0.9, 0.9, 0]],
            dtype=np.float32,
        )
        depth = np.full((height, width), 2.0, dtype=np.float32)
        return detection, depth


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            create_web_handler(FakeModel(), patch_size=3),
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def test_home_page_contains_web_camera_ui(self):
        with urlopen(self.base_url) as response:
            body = response.read().decode("utf-8")
        self.assertIn("Nyalakan kamera", body)
        self.assertIn("Pilih foto", body)

    def test_health_reports_model_contract(self):
        with urlopen(f"{self.base_url}/api/health") as response:
            payload = response.read().decode("utf-8")
        self.assertIn('"ready":true', payload)
        self.assertIn('"classes":1', payload)

    def test_inference_returns_detection_distance_and_depth_preview(self):
        frame = np.zeros((80, 120, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(ok)
        request = Request(
            f"{self.base_url}/api/infer?confidence=0.25",
            data=encoded.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )

        with urlopen(request) as response:
            payload = json.loads(response.read())

        self.assertEqual((payload["width"], payload["height"]), (120, 80))
        self.assertEqual(len(payload["detections"]), 1)
        self.assertEqual(payload["detections"][0]["class_name"], "chair")
        self.assertAlmostEqual(payload["detections"][0]["distance_m"], 2.0)
        self.assertGreater(len(payload["depth_preview"]), 0)

    def test_invalid_image_is_rejected_with_bad_request(self):
        request = Request(
            f"{self.base_url}/api/infer",
            data=b"not an image",
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
