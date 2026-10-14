import unittest

import numpy as np
import torch
from torch import nn

from multitask_pt import MultitaskTorchModel


class FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(nn.Identity() for _ in range(23))
        self.detections = torch.zeros((1, 300, 6), dtype=torch.float32)
        self.detections[0, 0] = torch.tensor([32, 100, 600, 400, 0.9, 0])

    def forward(self, image):
        for layer in self.model:
            image = layer(image)
        return self.detections, {}


class FakeDepthHead(nn.Module):
    def forward(self, features):
        del features
        return torch.full((1, 1, 480, 640), 2.5)


class MultitaskTorchTests(unittest.TestCase):
    def test_predict_runs_both_heads_and_unletterboxes_outputs(self):
        model = MultitaskTorchModel(
            detector=FakeDetector(),
            depth_head=FakeDepthHead(),
            class_names=("chair",),
            input_height=480,
            input_width=640,
            device=torch.device("cpu"),
        )
        frame = np.zeros((100, 200, 3), dtype=np.uint8)

        detections, depth = model.predict(frame)

        self.assertEqual(detections.shape, (300, 6))
        np.testing.assert_allclose(detections[0, :4], [10, 6.25, 187.5, 100])
        self.assertEqual(depth.shape, (100, 200))
        np.testing.assert_allclose(depth, 2.5)

    def test_predict_rejects_wrong_detection_output_shape(self):
        detector = FakeDetector()
        detector.detections = torch.zeros((1, 1, 6), dtype=torch.float32)
        model = MultitaskTorchModel(
            detector=detector,
            depth_head=FakeDepthHead(),
            class_names=("chair",),
            input_height=480,
            input_width=640,
            device=torch.device("cpu"),
        )

        with self.assertRaisesRegex(ValueError, "Detection output has shape"):
            model.predict(np.zeros((100, 200, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
