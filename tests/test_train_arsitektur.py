import contextlib
import io
import json
import random
import tempfile
import time
import unittest
from pathlib import Path
import zipfile

import cv2
import numpy as np
import torch

from train_arsitektur import (
    ActivityIndicator,
    COCO_ANNOTATIONS_URL,
    COCO_IMAGES_URL,
    _download_file,
    build_detection_targets,
    compute_losses,
    discover_coco_records,
    discover_sun_records,
    ensure_coco_annotations,
    ensure_sun_dataset,
    _focal_classification_loss,
    prepare_sample,
)


class ArchitectureTrainingTests(unittest.TestCase):
    def test_focal_classification_loss_keeps_sparse_positive_labels_significant(self):
        logits = torch.zeros(1, 1, 80, 80, requires_grad=True)
        targets = torch.zeros_like(logits)
        targets[0, 0, 40, 40] = 1.0

        loss = _focal_classification_loss(logits, targets)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.1)
        self.assertLess(float(logits.grad[0, 0, 40, 40].detach()), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0].detach()), 0.0)

    def test_coco_assets_use_secure_s3_path_style_urls(self):
        expected_base = "https://s3.amazonaws.com/images.cocodataset.org/"
        self.assertTrue(COCO_ANNOTATIONS_URL.startswith(expected_base))
        self.assertTrue(COCO_IMAGES_URL.startswith(expected_base))

    def test_activity_indicator_animates_and_reports_final_message(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with ActivityIndicator("loading batch", interval=0.005) as indicator:
                time.sleep(0.02)
                indicator.update("training batch 1/3")

        rendered = output.getvalue()
        self.assertTrue(any(f"[{frame}]" in rendered for frame in "|/-\\"))
        self.assertIn("[done] training batch 1/3", rendered)

    def test_download_file_reports_progress_and_writes_complete_file(self):
        from unittest.mock import patch

        class FakeResponse(io.BytesIO):
            status = 200
            headers = {"Content-Length": "10"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "sample.zip"
            output = io.StringIO()
            with patch(
                "train_arsitektur.urllib.request.urlopen",
                return_value=FakeResponse(b"0123456789"),
            ), contextlib.redirect_stdout(output):
                _download_file("https://example.invalid/sample.zip", destination)

            self.assertEqual(destination.read_bytes(), b"0123456789")
            self.assertIn("100.0%", output.getvalue())
            self.assertIn("ETA", output.getvalue())

    def test_letterbox_keeps_image_depth_and_boxes_aligned(self):
        image = np.zeros((32, 64, 3), dtype=np.uint8)
        image[4:28, 16:48] = 255
        depth = np.zeros((32, 64), dtype=np.float32)
        depth[4:28, 16:48] = 2.0
        boxes = ((16.0, 4.0, 48.0, 28.0, 1),)

        image_tensor, depth_tensor, boxes_tensor = prepare_sample(
            image,
            depth,
            boxes,
            (64, 64),
            train=False,
            rng=random.Random(0),
            depth_max_m=10.0,
        )

        self.assertEqual(tuple(image_tensor.shape), (3, 64, 64))
        self.assertEqual(tuple(depth_tensor.shape), (64, 64))
        torch.testing.assert_close(
            boxes_tensor,
            torch.tensor([[16.0, 20.0, 48.0, 44.0, 1.0]]),
        )
        self.assertTrue(torch.all(depth_tensor[20:44, 16:48] == 0.2))
        self.assertTrue(torch.all(depth_tensor[:16] == 0))

    def test_detection_targets_assign_each_box_to_one_scale(self):
        outputs = {
            "p3_bbox": torch.zeros(1, 4, 8, 12),
            "p3_cls": torch.zeros(1, 4, 8, 12),
            "p4_bbox": torch.zeros(1, 4, 4, 6),
            "p4_cls": torch.zeros(1, 4, 4, 6),
            "p5_bbox": torch.zeros(1, 4, 2, 3),
            "p5_cls": torch.zeros(1, 4, 2, 3),
        }
        classes, boxes, masks = build_detection_targets(
            outputs,
            [torch.tensor([[8.0, 8.0, 40.0, 40.0, 2.0]])],
            torch.device("cpu"),
        )

        self.assertEqual(sum(int(mask.sum()) for mask in masks), 1)
        self.assertEqual(float(classes[0][0, 2].sum()), 1.0)
        self.assertEqual(tuple(boxes[0].shape), (1, 4, 8, 12))

    def test_coco_batch_has_zero_depth_loss_and_finite_detection_loss(self):
        outputs = {
            "p3_bbox": torch.zeros(1, 4, 8, 12, requires_grad=True),
            "p3_cls": torch.zeros(1, 4, 8, 12, requires_grad=True),
            "p4_bbox": torch.zeros(1, 4, 4, 6, requires_grad=True),
            "p4_cls": torch.zeros(1, 4, 4, 6, requires_grad=True),
            "p5_bbox": torch.zeros(1, 4, 2, 3, requires_grad=True),
            "p5_cls": torch.zeros(1, 4, 2, 3, requires_grad=True),
            "depth": torch.rand(1, 1, 64, 96, requires_grad=True),
        }
        total, parts = compute_losses(
            outputs,
            torch.zeros(1, 64, 96),
            [torch.tensor([[8.0, 8.0, 40.0, 40.0, 2.0]])],
            torch.device("cpu"),
        )

        self.assertEqual(float(parts["depth"].detach()), 0.0)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertIsNotNone(outputs["p3_cls"].grad)

    def test_coco_instances_annotations_become_detection_only_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_dir = root / "train2017"
            annotation_dir = root / "annotations"
            image_dir.mkdir()
            annotation_dir.mkdir()
            image_path = image_dir / "000000000001.jpg"
            self.assertTrue(
                cv2.imwrite(str(image_path), np.zeros((32, 32, 3), dtype=np.uint8))
            )
            annotation_path = annotation_dir / "instances_train2017.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": 1,
                                "width": 32,
                                "height": 32,
                                "file_name": image_path.name,
                            }
                        ],
                        "categories": [{"id": 3, "name": "person"}],
                        "annotations": [
                            {
                                "image_id": 1,
                                "category_id": 3,
                                "bbox": [4, 5, 12, 15],
                                "iscrowd": 0,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = discover_coco_records(root, None, ("person",))

        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].depth_path)
        self.assertEqual(records[0].boxes, ((4.0, 5.0, 16.0, 20.0, 0),))

    def test_coco_subset_limits_images_before_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotations_dir = root / "annotations"
            annotations_dir.mkdir()
            images = [
                {
                    "id": image_id,
                    "width": 32,
                    "height": 32,
                    "file_name": f"{image_id:012d}.jpg",
                }
                for image_id in (1, 2, 3)
            ]
            data = {
                "images": images,
                "categories": [{"id": 3, "name": "person"}],
                "annotations": [
                    {
                        "image_id": image["id"],
                        "category_id": 3,
                        "bbox": [4, 5, 12, 15],
                        "iscrowd": 0,
                    }
                    for image in images
                ],
            }
            (annotations_dir / "instances_train2017.json").write_text(
                json.dumps(data), encoding="utf-8"
            )

            records = discover_coco_records(
                root,
                None,
                ("person",),
                max_records=2,
                seed=7,
                allow_missing_images=True,
            )

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.image_path.name.endswith(".jpg") for record in records))
        self.assertTrue(
            all(record.image_path.parent.name == "train2017" for record in records)
        )

    def test_sun_discovery_skips_invalid_annotation_and_keeps_valid_scenes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad_scene = root / "SUNRGBD" / "kv1" / "b3dodata" / "img_bad"
            good_scene = root / "SUNRGBD" / "kv1" / "b3dodata" / "img_good"
            for scene in (bad_scene, good_scene):
                (scene / "annotation2Dfinal").mkdir(parents=True)
                (scene / "image").mkdir()
                (scene / "depth").mkdir()
                cv2.imwrite(
                    str(scene / "image" / "image.png"),
                    np.zeros((32, 32, 3), dtype=np.uint8),
                )
                cv2.imwrite(
                    str(scene / "depth" / "depth.png"),
                    np.full((32, 32), 1000, dtype=np.uint16),
                )

            (bad_scene / "annotation2Dfinal" / "index.json").write_text(
                '{"objects": }', encoding="utf-8"
            )
            (good_scene / "annotation2Dfinal" / "index.json").write_text(
                json.dumps(
                    {
                        "objects": ["person"],
                        "frames": [
                            {
                                "polygon": [
                                    {
                                        "object": 0,
                                        "x": [4, 24, 24, 4],
                                        "y": [4, 4, 24, 24],
                                    }
                                ]
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            records = discover_sun_records(
                root / "SUNRGBD", ("person",), 10.0, "off"
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_path.parent.parent, good_scene)

    def test_sun_dataset_extracts_archive_into_requested_folder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            download_dir = root / "downloads"
            download_dir.mkdir()
            archive_path = download_dir / "SUNRGBD.zip"
            annotation_name = "SUNRGBD/scene/annotation2Dfinal/index.json"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(annotation_name, "{}")

            dataset_root = root / "datasets" / "SUNRGBD"
            ensure_sun_dataset(dataset_root, download_dir)

            self.assertTrue((dataset_root / annotation_name).is_file())

    def test_coco_annotations_extract_only_train_instances_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            download_dir = root / "downloads"
            download_dir.mkdir()
            archive_path = download_dir / "coco_annotations_trainval2017.zip"
            member = "annotations/instances_train2017.json"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(member, "{}")
                archive.writestr(
                    "annotations/instances_val2017.json",
                    '{"should_not_extract": true}',
                )

            annotation_path = ensure_coco_annotations(root / "coco", download_dir)

            self.assertEqual(annotation_path.read_text(encoding="utf-8"), "{}")
            self.assertFalse(
                (root / "coco" / "annotations" / "instances_val2017.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
