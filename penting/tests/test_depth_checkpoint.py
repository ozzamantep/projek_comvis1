import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ..arsitektur_yolo_depth import load_depth_checkpoint


class DepthCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = nn.Linear(2, 1)

    def test_loads_compatible_state_dict(self):
        expected = {
            "weight": torch.tensor([[1.0, 2.0]]),
            "bias": torch.tensor([3.0]),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "depth.pt"
            torch.save(expected, checkpoint)

            load_depth_checkpoint(self.model, checkpoint, self.device)

        torch.testing.assert_close(self.model.weight, expected["weight"])
        torch.testing.assert_close(self.model.bias, expected["bias"])

    def test_rejects_incompatible_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "depthnet_best.pt"
            torch.save({"encoder.weight": torch.ones(1)}, checkpoint)

            with self.assertRaisesRegex(ValueError, "incompatible with YOLO26nDepth"):
                load_depth_checkpoint(self.model, checkpoint, self.device)

    def test_missing_checkpoint_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            load_depth_checkpoint(
                self.model,
                "missing_depth_checkpoint.pt",
                self.device,
            )


if __name__ == "__main__":
    unittest.main()
