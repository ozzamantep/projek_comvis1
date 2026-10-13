"""Inference helpers for the jointly trained detection-and-depth ONNX model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MultitaskOnnxModel:
    session: Any
    input_name: str
    detection_output_name: str
    depth_output_name: str
    input_width: int
    input_height: int
    max_detections: int
    class_names: tuple[str, ...]

    def predict(
        self,
        frame_bgr: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a three-channel BGR image.")
        source_height, source_width = frame_bgr.shape[:2]
        if source_width <= 0 or source_height <= 0:
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
        image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255.0)

        detections, depth_output = self.session.run(
            [self.detection_output_name, self.depth_output_name],
            {self.input_name: image},
        )
        detections = np.asarray(detections, dtype=np.float32)
        expected_detection_shape = (1, self.max_detections, 6)
        if detections.shape != expected_detection_shape:
            raise ValueError(
                f"Detection output has shape {detections.shape}; "
                f"expected {expected_detection_shape}."
            )
        depth = np.asarray(depth_output, dtype=np.float32)
        expected_depth_shape = (1, 1, self.input_height, self.input_width)
        if depth.shape != expected_depth_shape:
            raise ValueError(
                f"Depth output has shape {depth.shape}; expected {expected_depth_shape}."
            )

        boxes = detections[0].copy()
        boxes[:, (0, 2)] = (boxes[:, (0, 2)] - pad_left) / scale
        boxes[:, (1, 3)] = (boxes[:, (1, 3)] - pad_top) / scale
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, source_width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, source_height)

        depth_content = depth[0, 0][
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ]
        depth_source = cv2.resize(
            depth_content,
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return boxes, np.asarray(depth_source, dtype=np.float32)


def load_multitask_onnx_model(
    model_path: str | Path,
    metadata_path: str | Path,
) -> MultitaskOnnxModel:
    """Load one ONNX graph and validate it against its training metadata."""
    import onnxruntime as ort

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    input_config = metadata["input"]
    output_config = metadata["outputs"]
    input_name = str(input_config["name"])
    detection_name = "det"
    depth_name = "depth"
    input_shape = tuple(int(value) for value in input_config["shape"])
    if len(input_shape) != 4 or input_shape[:2] != (1, 3):
        raise ValueError(f"Unsupported multitask input shape in metadata: {input_shape}.")
    _, _, input_height, input_width = input_shape
    detection_shape = tuple(int(value) for value in output_config["det"]["shape"])
    depth_shape = tuple(int(value) for value in output_config["depth"]["shape"])
    if len(detection_shape) != 3 or detection_shape[0] != 1 or detection_shape[2] != 6:
        raise ValueError(f"Unsupported detection output shape in metadata: {detection_shape}.")
    if depth_shape != (1, 1, input_height, input_width):
        raise ValueError(
            f"Depth output shape {depth_shape} must match model input {(1, 3, input_height, input_width)}."
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    actual_input = {value.name: value for value in session.get_inputs()}
    actual_outputs = {value.name: value for value in session.get_outputs()}
    if input_name not in actual_input:
        raise ValueError(f"ONNX input {input_name!r} not found; got {tuple(actual_input)}.")
    if detection_name not in actual_outputs or depth_name not in actual_outputs:
        raise ValueError(
            f"ONNX must expose {detection_name!r} and {depth_name!r}; "
            f"got {tuple(actual_outputs)}."
        )
    expected_shapes = {
        input_name: input_shape,
        detection_name: detection_shape,
        depth_name: depth_shape,
    }
    actual_tensors = {**actual_input, **actual_outputs}
    for name, expected in expected_shapes.items():
        actual = tuple(int(value) for value in actual_tensors[name].shape)
        if actual != expected:
            raise ValueError(
                f"ONNX tensor {name!r} has shape {actual}; metadata expects {expected}."
            )
        if actual_tensors[name].type != "tensor(float)":
            raise ValueError(
                f"ONNX tensor {name!r} must be float32, got {actual_tensors[name].type}."
            )

    class_names = tuple(str(name) for name in metadata["class_names"])
    return MultitaskOnnxModel(
        session=session,
        input_name=input_name,
        detection_output_name=detection_name,
        depth_output_name=depth_name,
        input_width=input_width,
        input_height=input_height,
        max_detections=detection_shape[1],
        class_names=class_names,
    )
