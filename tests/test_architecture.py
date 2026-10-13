import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Install requirements-architecture.txt to test the model.")
class ArchitectureTests(unittest.TestCase):
    def test_forward_outputs_reported_detection_scales_and_relative_depth(self):
        from arsitektur import build_model

        model = build_model(num_classes=4).eval()
        images = torch.zeros(1, 3, 64, 96)

        with torch.inference_mode():
            outputs = model(images)

        self.assertEqual(outputs["p3_bbox"].shape, (1, 4, 8, 12))
        self.assertEqual(outputs["p3_cls"].shape, (1, 4, 8, 12))
        self.assertEqual(outputs["p4_bbox"].shape, (1, 4, 4, 6))
        self.assertEqual(outputs["p4_cls"].shape, (1, 4, 4, 6))
        self.assertEqual(outputs["p5_bbox"].shape, (1, 4, 2, 3))
        self.assertEqual(outputs["p5_cls"].shape, (1, 4, 2, 3))
        self.assertEqual(outputs["depth"].shape, (1, 1, 64, 96))
        self.assertTrue(torch.all((outputs["depth"] >= 0) & (outputs["depth"] <= 1)))

    def test_forward_rejects_dimensions_not_divisible_by_32(self):
        from arsitektur import build_model

        model = build_model().eval()
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            model(torch.zeros(1, 3, 65, 96))


if __name__ == "__main__":
    unittest.main()
