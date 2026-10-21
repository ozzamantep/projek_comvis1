import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from architecture_pt import ArchitectureTorchModel, decode_detections


class FakeArchitecture(nn.Module):
    def forward(self, image: Tensor) -> dict[str, Tensor]:
        batch, _, height, width = image.shape
        outputs = {}
        for stride, level in ((8, "p3"), (16, "p4"), (32, "p5")):
            level_height, level_width = height // stride, width // stride
            outputs[f"{level}_bbox"] = torch.zeros(
                batch, 4, level_height, level_width
            )
            outputs[f"{level}_cls"] = torch.full(
                (batch, 1, level_height, level_width), -10.0
            )
        outputs["p3_cls"][0, 0, 10, 20] = 10.0
        outputs["depth"] = torch.full((batch, 1, height, width), 0.5)
        return outputs


class ArchitectureInferenceTests(unittest.TestCase):
    def test_decode_detections_converts_cell_offsets_to_xyxy(self):
        outputs = {
            "p3_bbox": torch.zeros(1, 4, 60, 80),
            "p3_cls": torch.full((1, 1, 60, 80), -10.0),
            "p4_bbox": torch.zeros(1, 4, 30, 40),
            "p4_cls": torch.full((1, 1, 30, 40), -10.0),
            "p5_bbox": torch.zeros(1, 4, 15, 20),
            "p5_cls": torch.full((1, 1, 15, 20), -10.0),
        }
        outputs["p3_cls"][0, 0, 10, 20] = 10.0

        detections = decode_detections(outputs, 480, 640, confidence=0.5)

        self.assertEqual(detections.shape, (1, 6))
        np.testing.assert_allclose(detections[0, :4], [160, 80, 168, 88])
        self.assertGreater(float(detections[0, 4]), 0.99)
        self.assertEqual(float(detections[0, 5]), 0.0)

    def test_predict_restores_detection_and_depth_to_source_frame(self):
        model = ArchitectureTorchModel(
            FakeArchitecture(),
            class_names=("person",),
            input_height=480,
            input_width=640,
            depth_max_m=10.0,
            device=torch.device("cpu"),
        )

        detections, depth = model.predict(np.zeros((240, 320, 3), dtype=np.uint8))

        self.assertEqual(detections.shape, (1, 6))
        np.testing.assert_allclose(detections[0, :4], [80, 40, 84, 44])
        self.assertEqual(depth.shape, (240, 320))
        np.testing.assert_allclose(depth, 5.0)

    def test_loader_accepts_report_blueprint_checkpoint(self):
        from architecture_pt import load_architecture_torch_model
        from arsitektur import build_model

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "report.pt"
            model = build_model(num_classes=1)
            torch.save(
                {
                    "kind": "arsitektur_multitask",
                    "classes": ["person"],
                    "imgsz": [64, 96],
                    "depth_max_m": 10.0,
                    "width_mult": 0.25,
                    "depth_mult": 0.5,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            loaded = load_architecture_torch_model(checkpoint_path)

        detections, depth = loaded.predict(
            np.zeros((48, 72, 3), dtype=np.uint8)
        )
        self.assertEqual(detections.shape[1], 6)
        self.assertEqual(depth.shape, (48, 72))
        self.assertTrue(np.isfinite(depth).all())


if __name__ == "__main__":
    unittest.main()
