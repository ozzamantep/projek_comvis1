"""Load and run the calibrated YOLO26n-Depth checkpoint in metric units."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn


def load_calibrated_depth_model(
    weights_path: str | Path,
    device: torch.device,
) -> nn.Module:
    """Load the trained Ultralytics depth model, including its metric calibration."""
    from ultralytics import YOLO

    yolo = YOLO(str(weights_path), task="depth")
    if yolo.task != "depth":
        raise ValueError(f"Expected a depth checkpoint, got task {yolo.task!r}.")

    model = yolo.model.to(device).eval()
    depth_head = model.model[-1]
    if not hasattr(depth_head, "cal_a") or not hasattr(depth_head, "cal_b"):
        raise ValueError(
            "The depth checkpoint is missing its learned metric calibration buffers."
        )
    return model


def predict_calibrated_depth_meters(
    model: nn.Module,
    frame_bgr: np.ndarray,
    device: torch.device,
    input_size: int = 640,
) -> np.ndarray:
    """Predict calibrated depth in meters and restore the source image geometry."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a three-channel BGR image.")
    if input_size < 32 or input_size % 32:
        raise ValueError("input_size must be a positive multiple of 32.")

    source_height, source_width = frame_bgr.shape[:2]
    scale = min(input_size / source_width, input_size / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        frame_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_left = (input_size - resized_width) // 2
    pad_top = (input_size - resized_height) // 2
    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        input_size - resized_height - pad_top,
        pad_left,
        input_size - resized_width - pad_left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
    image = image.unsqueeze(0).to(device=device, dtype=torch.float32).div_(255.0)

    with torch.inference_mode():
        prediction: Any = model(image)
    if not isinstance(prediction, torch.Tensor) or prediction.ndim != 4:
        raise ValueError("Depth model must return a BCHW tensor during inference.")
    if prediction.shape[0] != 1 or prediction.shape[1] != 1:
        raise ValueError(
            f"Expected one depth map per image; got output {tuple(prediction.shape)}."
        )

    prediction = functional.interpolate(
        prediction,
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=True,
    )
    depth_square = prediction[0, 0].detach().cpu().numpy()
    depth_content = depth_square[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ]
    return cv2.resize(
        depth_content.astype(np.float32, copy=False),
        (source_width, source_height),
        interpolation=cv2.INTER_LINEAR,
    )
