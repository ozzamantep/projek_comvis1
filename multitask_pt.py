"""PyTorch inference for the trained multitask YOLO26n checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F
from ultralytics.nn.tasks import DetectionModel


DETECTION_FEATURE_LAYERS = (16, 19, 22)
DEPTH_BATCH_NORM_EPS = 1e-5


class ConvBNAct(nn.Module):
    """Checkpoint-compatible convolution, batch normalization, and SiLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=DEPTH_BATCH_NORM_EPS)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class TrainedDepthHead(nn.Module):
    """Depth branch matching the uploaded multitask checkpoint."""

    def __init__(self, calibration_a: Tensor, calibration_b: Tensor) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            (
                ConvBNAct(64, 256, 1),
                ConvBNAct(128, 256, 1),
                ConvBNAct(256, 256, 1),
            )
        )
        self.refine = nn.ModuleList(
            nn.Sequential(ConvBNAct(256, 256, 3), ConvBNAct(256, 256, 3))
            for _ in range(2)
        )
        self.head = nn.Sequential(
            ConvBNAct(256, 128, 3),
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            ConvBNAct(128, 64, 3),
            nn.Conv2d(64, 1, kernel_size=1),
        )
        self.register_buffer("cal_a", calibration_a.reshape(1).clone())
        self.register_buffer("cal_b", calibration_b.reshape(1).clone())

    @staticmethod
    def _upsample(x: Tensor, scale_factor: int) -> Tensor:
        return F.interpolate(
            x,
            scale_factor=scale_factor,
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, features: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        p3, p4, p5 = (projection(feature) for projection, feature in zip(self.proj, features))
        x = self.refine[1](self._upsample(p5, 2) + p4)
        x = self.refine[0](self._upsample(x, 2) + p3)
        log_depth = self.head(x).clamp(-4.0, 5.0)
        depth_m = log_depth.exp().pow(self.cal_a) * self.cal_b.exp()
        return self._upsample(depth_m, 4)


class MultitaskTorchModel:
    """Run both trained heads in PyTorch and restore outputs to frame space."""

    def __init__(
        self,
        detector: nn.Module,
        depth_head: nn.Module,
        class_names: tuple[str, ...],
        input_height: int,
        input_width: int,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.detector = detector.to(self.device).eval()
        self.depth_head = depth_head.to(self.device).eval()
        self.class_names = class_names
        self.input_height = input_height
        self.input_width = input_width
        self.max_detections = 300
        self._features: dict[int, Tensor] = {}

        for layer_index in DETECTION_FEATURE_LAYERS:
            self.detector.model[layer_index].register_forward_hook(
                self._capture_feature(layer_index)
            )

    def _capture_feature(self, layer_index: int) -> Any:
        def capture(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            del _module, _inputs
            self._features[layer_index] = output

        return capture

    def predict(
        self,
        frame_bgr: NDArray[np.uint8],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Letterbox a BGR frame, run the trained model, and undo letterboxing."""
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
        input_tensor = torch.from_numpy(image).to(self.device)

        self._features.clear()
        with torch.inference_mode():
            detector_output = self.detector(input_tensor)
            detections = (
                detector_output[0]
                if isinstance(detector_output, tuple)
                else detector_output
            )
            missing_features = set(DETECTION_FEATURE_LAYERS) - self._features.keys()
            if missing_features:
                raise RuntimeError(
                    f"Detector did not produce depth features at layers "
                    f"{sorted(missing_features)}."
                )
            depth = self.depth_head(
                tuple(self._features[index] for index in DETECTION_FEATURE_LAYERS)
            )

        detections_np = detections[0].detach().cpu().numpy().astype(np.float32)
        expected_detection_shape = (self.max_detections, 6)
        if detections_np.shape != expected_detection_shape:
            raise ValueError(
                f"Detection output has shape {detections_np.shape}; "
                f"expected {expected_detection_shape}."
            )
        expected_depth_shape = (1, 1, self.input_height, self.input_width)
        if tuple(depth.shape) != expected_depth_shape:
            raise ValueError(
                f"Depth output has shape {tuple(depth.shape)}; "
                f"expected {expected_depth_shape}."
            )

        detections_np[:, (0, 2)] = (
            detections_np[:, (0, 2)] - pad_left
        ) / scale
        detections_np[:, (1, 3)] = (
            detections_np[:, (1, 3)] - pad_top
        ) / scale
        detections_np[:, (0, 2)] = np.clip(
            detections_np[:, (0, 2)], 0, source_width
        )
        detections_np[:, (1, 3)] = np.clip(
            detections_np[:, (1, 3)], 0, source_height
        )

        depth_np = depth[0, 0].detach().cpu().numpy().astype(np.float32)
        depth_content = depth_np[
            pad_top : pad_top + resized_height,
            pad_left : pad_left + resized_width,
        ]
        depth_source = cv2.resize(
            depth_content,
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        return detections_np, np.asarray(depth_source, dtype=np.float32)


def _build_detection_model(num_classes: int) -> DetectionModel:
    model_config = (
        Path(ultralytics.__file__).parent / "cfg" / "models" / "26" / "yolo26.yaml"
    )
    model_definition = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    model_definition["scale"] = "n"
    detector = DetectionModel(cfg=model_definition, ch=3, verbose=False)
    head = detector.model[-1]

    for branch_name in ("cv3", "one2one_cv3"):
        for branch in getattr(head, branch_name):
            classifier = branch[-1]
            branch[-1] = nn.Conv2d(
                classifier.in_channels,
                num_classes,
                classifier.kernel_size,
                classifier.stride,
                classifier.padding,
                bias=classifier.bias is not None,
            )

    head.nc = num_classes
    head.no = num_classes + 4 * head.reg_max
    head.end2end = True
    return detector


def load_multitask_torch_model(model_path: str | Path) -> MultitaskTorchModel:
    """Safely load the inference checkpoint and strictly validate both branches."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "inference_multitask":
        raise ValueError("Checkpoint is not a supported multitask inference .pt file.")

    class_names_value = checkpoint.get("names")
    if not isinstance(class_names_value, list) or not class_names_value:
        raise ValueError("Checkpoint must contain a non-empty class-name list.")
    class_names = tuple(str(name) for name in class_names_value)

    image_size = checkpoint.get("imgsz")
    if not isinstance(image_size, (tuple, list)) or len(image_size) != 2:
        raise ValueError("Checkpoint must contain its (height, width) input size.")
    input_height, input_width = (int(value) for value in image_size)
    if (input_height, input_width) != (480, 640):
        raise ValueError(
            f"Unsupported checkpoint input size {(input_height, input_width)}."
        )

    state = checkpoint.get("ema")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain an EMA inference state dict.")
    detection_state = {
        key.removeprefix("det."): value
        for key, value in state.items()
        if key.startswith("det.")
    }
    depth_state = {
        key.removeprefix("depth."): value
        for key, value in state.items()
        if key.startswith("depth.")
    }
    if not detection_state or not depth_state:
        raise ValueError("Checkpoint must contain both detection and depth weights.")

    detector = _build_detection_model(len(class_names))
    detector.load_state_dict(detection_state, strict=True)
    depth_head = TrainedDepthHead(state["depth.cal_a"], state["depth.cal_b"])
    depth_head.load_state_dict(depth_state, strict=True)
    detector.names = dict(enumerate(class_names))

    return MultitaskTorchModel(
        detector,
        depth_head,
        class_names,
        input_height,
        input_width,
    )
