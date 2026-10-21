"""Inference adapter for checkpoints produced by train_arsitektur.py."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from arsitektur import ARCHITECTURE_ID, build_model


DETECTION_LEVELS = ((8, "p3"), (16, "p4"), (32, "p5"))
DEFAULT_CONFIDENCE = 0.05
NMS_IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 300


def _class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray
) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        class_indices = np.flatnonzero(class_ids == class_id)
        order = class_indices[np.argsort(scores[class_indices])[::-1]]
        while order.size:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
            intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
                0.0, boxes[current, 3] - boxes[current, 1]
            )
            areas_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
                0.0, boxes[rest, 3] - boxes[rest, 1]
            )
            union = area_current + areas_rest - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = rest[iou <= NMS_IOU_THRESHOLD]
    if not kept:
        return np.zeros((0,), dtype=np.int64)
    kept_array = np.asarray(kept, dtype=np.int64)
    return kept_array[np.argsort(scores[kept_array])[::-1][:MAX_DETECTIONS]]


def decode_detections(
    outputs: dict[str, Tensor],
    image_height: int,
    image_width: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> NDArray[np.float32]:
    """Decode the trainer's cell-offset/log-size box representation."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in the interval [0, 1].")
    decoded_boxes: list[np.ndarray] = []
    decoded_scores: list[np.ndarray] = []
    decoded_classes: list[np.ndarray] = []

    for stride, prefix in DETECTION_LEVELS:
        regression = outputs[f"{prefix}_bbox"][0].detach().float().cpu().numpy()
        class_logits = outputs[f"{prefix}_cls"][0].detach().float().cpu().numpy()
        if regression.shape[0] != 4 or class_logits.ndim != 3:
            raise ValueError(f"Unexpected {prefix} detection output shape.")
        probability = 1.0 / (1.0 + np.exp(-np.clip(class_logits, -80, 80)))
        class_ids, row_ids, col_ids = np.where(probability >= confidence)
        if not len(class_ids):
            continue
        tx = regression[0, row_ids, col_ids]
        ty = regression[1, row_ids, col_ids]
        tw = np.clip(regression[2, row_ids, col_ids], -10.0, 10.0)
        th = np.clip(regression[3, row_ids, col_ids], -10.0, 10.0)
        center_x = (col_ids + 0.5 + tx) * stride
        center_y = (row_ids + 0.5 + ty) * stride
        width = np.exp(tw) * stride
        height = np.exp(th) * stride
        boxes = np.stack(
            (
                center_x - width / 2.0,
                center_y - height / 2.0,
                center_x + width / 2.0,
                center_y + height / 2.0,
            ),
            axis=1,
        )
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, image_width)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, image_height)
        valid = (
            np.isfinite(boxes).all(axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
        )
        decoded_boxes.append(boxes[valid].astype(np.float32))
        decoded_scores.append(probability[class_ids, row_ids, col_ids][valid])
        decoded_classes.append(class_ids[valid].astype(np.int64))

    if not decoded_boxes:
        return np.zeros((0, 6), dtype=np.float32)
    boxes = np.concatenate(decoded_boxes)
    scores = np.concatenate(decoded_scores).astype(np.float32)
    class_ids = np.concatenate(decoded_classes)
    keep = _class_aware_nms(boxes, scores, class_ids)
    return np.column_stack(
        (boxes[keep], scores[keep], class_ids[keep].astype(np.float32))
    ).astype(np.float32)


class ArchitectureTorchModel:
    """Run the trained blueprint and adapt its outputs for the camera UI."""

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: tuple[str, ...],
        input_height: int,
        input_width: int,
        depth_max_m: float,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device).eval()
        self.class_names = class_names
        self.input_height = input_height
        self.input_width = input_width
        self.depth_max_m = depth_max_m

    def predict(
        self, frame_bgr: NDArray[np.uint8]
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a three-channel BGR image.")
        source_height, source_width = frame_bgr.shape[:2]
        if source_height <= 0 or source_width <= 0:
            raise ValueError("frame_bgr must have positive dimensions.")

        scale = min(
            self.input_width / source_width,
            self.input_height / source_height,
        )
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = cv2.resize(
            frame_bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_left = (self.input_width - resized_width) // 2
        pad_top = (self.input_height - resized_height) // 2
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            self.input_height - resized_height - pad_top,
            pad_left,
            self.input_width - resized_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        image = image_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(image).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_tensor)
        detections = decode_detections(
            outputs, self.input_height, self.input_width
        )
        if detections.size:
            detections[:, (0, 2)] = (
                detections[:, (0, 2)] - pad_left
            ) / scale
            detections[:, (1, 3)] = (
                detections[:, (1, 3)] - pad_top
            ) / scale
            detections[:, (0, 2)] = detections[:, (0, 2)].clip(
                0, source_width
            )
            detections[:, (1, 3)] = detections[:, (1, 3)].clip(
                0, source_height
            )

        depth = outputs["depth"][0, 0].detach().float().cpu().numpy()
        depth = np.clip(depth, 0.0, 1.0) * np.float32(self.depth_max_m)
        depth_content = depth[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ]
        depth_source = cv2.resize(
            depth_content,
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return detections, np.asarray(depth_source, dtype=np.float32)


def load_architecture_torch_model(
    model_path: str | Path,
    device: torch.device | None = None,
) -> ArchitectureTorchModel:
    """Safely load and validate a checkpoint from the architecture trainer."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "arsitektur_multitask":
        raise ValueError("Checkpoint is not a supported arsitektur training checkpoint.")
    checkpoint_architecture = checkpoint.get("architecture_id")
    if checkpoint_architecture not in (None, ARCHITECTURE_ID):
        raise ValueError(
            f"Unsupported architecture checkpoint: {checkpoint_architecture!r}."
        )

    class_names_value = checkpoint.get("classes")
    if not isinstance(class_names_value, list) or not class_names_value:
        raise ValueError("Checkpoint must contain a non-empty class list.")
    class_names = tuple(str(name) for name in class_names_value)

    image_size = checkpoint.get("imgsz")
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(not isinstance(value, int) or value < 32 or value % 32 for value in image_size)
    ):
        raise ValueError("Checkpoint must contain a valid stride-32 (height, width) input size.")
    input_height, input_width = image_size

    depth_max_m = checkpoint.get("depth_max_m")
    if not isinstance(depth_max_m, (int, float)) or not math.isfinite(depth_max_m) or depth_max_m <= 0:
        raise ValueError("Checkpoint must contain a positive finite depth_max_m.")
    width_mult = checkpoint.get("width_mult")
    depth_mult = checkpoint.get("depth_mult")
    if not isinstance(width_mult, (int, float)) or not isinstance(depth_mult, (int, float)):
        raise ValueError("Checkpoint must contain valid model scaling parameters.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    model = build_model(
        num_classes=len(class_names),
        width_mult=float(width_mult),
        depth_mult=float(depth_mult),
    )
    model.load_state_dict(state, strict=True)
    return ArchitectureTorchModel(
        model,
        class_names,
        input_height,
        input_width,
        float(depth_max_m),
        device,
    )
