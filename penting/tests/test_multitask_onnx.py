import unittest

import numpy as np

from ..multitask_onnx import MultitaskOnnxModel


class FakeSession:
    def __init__(self):
        self.input = None
        self.detections = np.zeros((1, 2, 6), dtype=np.float32)
        self.depth = np.full((1, 1, 480, 640), 2.5, dtype=np.float32)
        self.detections[0, 0] = [32, 100, 600, 400, 0.9, 0]

    def run(self, output_names, inputs):
        del output_names
        self.input = inputs["images"]
        return [self.detections, self.depth]


class MultitaskOnnxTests(unittest.TestCase):
    def test_predict_runs_shared_heads_and_unletterboxes_outputs(self):
        session = FakeSession()
        model = MultitaskOnnxModel(
            session=session,
            input_name="images",
            detection_output_name="det",
            depth_output_name="depth",
            input_width=640,
            input_height=480,
            max_detections=2,
            class_names=("chair",),
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        detections, depth = model.predict(frame)

        self.assertEqual(session.input.shape, (1, 3, 480, 640))
        self.assertEqual(session.input.dtype, np.float32)
        self.assertEqual(detections.shape, (2, 6))
        np.testing.assert_allclose(detections[0, :4], [10, 6.25, 187.5, 100])
        self.assertEqual(depth.shape, (100, 200))
        np.testing.assert_allclose(depth, 2.5)

    def test_rejects_wrong_detection_output_shape(self):
        session = FakeSession()
        session.detections = np.zeros((1, 1, 6), dtype=np.float32)
        model = MultitaskOnnxModel(
            session=session,
            input_name="images",
            detection_output_name="det",
            depth_output_name="depth",
            input_width=640,
            input_height=480,
            max_detections=2,
            class_names=("chair",),
        )

        with self.assertRaisesRegex(ValueError, "Detection output has shape"):
            model.predict(np.zeros((100, 200, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
