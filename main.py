"""Main desktop app for multitask detection and metric depth.

Commands: camera, evaluate, validate, and calibrate.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_MODEL_PATH = (
    "yolo26n_multitask_coco4_v1_20261013-163528-539afa/"
    "inference/multitask.onnx"
)
DEFAULT_METADATA_PATH = (
    "yolo26n_multitask_coco4_v1_20261013-163528-539afa/"
    "inference/model_metadata.json"
)

# Depth application
def meters_to_cm(meters: ArrayLike) -> NDArray[np.float32]:
    """Convert meters to centimeters."""
    return np.asarray(meters, dtype=np.float32) * np.float32(100.0)


class DepthPatchStats(NamedTuple):
    center_value: float
    median: float
    variance: float
    valid_count: int
    patch_shape: tuple[int, int]


def get_depth_patch_stats(
    depth_map: ArrayLike, x: float, y: float, patch_size: int = 1
) -> DepthPatchStats:
    """Return center value and valid-pixel statistics for a square patch."""
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_map must be a 2D array.")
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer.")
    if not np.isfinite(x) or not np.isfinite(y):
        raise ValueError("Point coordinates must be finite.")

    center_x, center_y = int(round(x)), int(round(y))
    height, width = depth.shape
    if center_x < 0 or center_x >= width or center_y < 0 or center_y >= height:
        raise ValueError("Point is outside the depth map.")

    radius = patch_size // 2
    patch = depth[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        raise ValueError("No valid positive depth values at the requested point.")
    return DepthPatchStats(
        center_value=float(depth[center_y, center_x]),
        median=float(np.median(valid)),
        variance=float(np.var(valid)),
        valid_count=int(valid.size),
        patch_shape=(int(patch.shape[0]), int(patch.shape[1])),
    )


def get_depth_at_point(
    depth_m: ArrayLike, x: float, y: float, patch_size: int = 1
) -> float:
    """Return the median valid depth around an (x, y) image coordinate."""
    return get_depth_patch_stats(depth_m, x, y, patch_size).median


def get_depth_in_bbox(
    depth_m: ArrayLike, bbox: ArrayLike, patch_size: int = 5
) -> float:
    """Sample median depth in a patch centered on an xyxy bounding box."""
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        raise ValueError("bbox must contain four finite xyxy coordinates.")
    x1, y1, x2, y2 = box
    if x2 < x1 or y2 < y1:
        raise ValueError("bbox coordinates must satisfy x2 >= x1 and y2 >= y1.")
    return get_depth_at_point(
        depth_m,
        x=(x1 + x2) / 2.0,
        y=(y1 + y2) / 2.0,
        patch_size=patch_size,
    )


@dataclass
class DepthEMASmoother:
    """Exponential smoothing for raw metric depth maps across video frames."""

    alpha: float = 0.35

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be finite and in the interval (0, 1].")
        self._value: NDArray[np.float32] | None = None

    def reset(self) -> None:
        """Discard the previous frame state."""
        self._value = None

    def update(self, depth_m: ArrayLike) -> NDArray[np.float32]:
        """Smooth a metric depth map; a changed shape starts a fresh sequence."""
        current = np.asarray(depth_m, dtype=np.float32)
        if current.ndim != 2:
            raise ValueError("depth_m must be a 2D array.")

        if self._value is None or self._value.shape != current.shape:
            self._value = current.copy()
        else:
            self._value = (
                self.alpha * current + (1.0 - self.alpha) * self._value
            ).astype(np.float32)
        return self._value.copy()


def normalize_depth_for_display(depth_m: ArrayLike) -> NDArray[np.uint8]:
    """Normalize one frame for visualization only, never for metric sampling."""
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_m must be a 2D array.")

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)

    low, high = np.percentile(depth[valid], (2.0, 98.0))
    if high - low < 1e-6:
        return np.zeros(depth.shape, dtype=np.uint8)

    normalized = np.zeros(depth.shape, dtype=np.float32)
    normalized[valid] = np.clip(
        (depth[valid] - low) / (high - low), 0.0, 1.0
    )
    normalized[valid] = 1.0 - normalized[valid]
    return (normalized * 255.0).astype(np.uint8)

# Onnx application
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

# Calibrate application
@dataclass(frozen=True)
class Calibration:
    method: str
    slope: float
    intercept_m: float
    sample_count: int

    def apply(self, predicted_m: float) -> float:
        return self.slope * predicted_m + self.intercept_m


def load_validation_pairs(csv_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load non-empty positive measured/predicted meter pairs from validation CSV."""
    predicted: list[float] = []
    measured: list[float] = []
    skipped = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"measured_m", "predicted_m"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV must contain measured_m and predicted_m columns."
            )
        for row_number, row in enumerate(reader, start=2):
            measured_text = (row.get("measured_m") or "").strip()
            predicted_text = (row.get("predicted_m") or "").strip()
            if not measured_text or not predicted_text:
                skipped += 1
                continue
            try:
                actual = float(measured_text)
                estimate = float(predicted_text)
            except ValueError as error:
                raise ValueError(
                    f"CSV row {row_number} contains a non-numeric distance."
                ) from error
            if (
                not math.isfinite(actual)
                or not math.isfinite(estimate)
                or actual <= 0
                or estimate <= 0
            ):
                raise ValueError(
                    f"CSV row {row_number} distances must be finite and positive."
                )
            measured.append(actual)
            predicted.append(estimate)

    if not predicted:
        raise ValueError("CSV has no complete positive calibration rows.")
    return np.asarray(predicted), np.asarray(measured), skipped


def fit_calibration(
    predicted_m: np.ndarray, measured_m: np.ndarray, method: str
) -> Calibration:
    """Fit median ratio scaling or a least-squares affine calibration."""
    predicted = np.asarray(predicted_m, dtype=np.float64)
    measured = np.asarray(measured_m, dtype=np.float64)
    if predicted.ndim != 1 or measured.ndim != 1 or predicted.shape != measured.shape:
        raise ValueError("predicted_m and measured_m must be equal-length 1D arrays.")
    if predicted.size == 0:
        raise ValueError("At least one calibration pair is required.")
    if (
        not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(measured))
        or np.any(predicted <= 0)
        or np.any(measured <= 0)
    ):
        raise ValueError("Calibration distances must be finite and positive.")

    if method == "ratio":
        slope = float(np.median(measured / predicted))
        intercept = 0.0
    elif method == "linear":
        if predicted.size < 2 or np.ptp(predicted) == 0:
            raise ValueError(
                "Linear calibration needs at least two distinct predicted distances."
            )
        slope, intercept = (float(value) for value in np.polyfit(predicted, measured, 1))
    else:
        raise ValueError("method must be 'ratio' or 'linear'.")
    return Calibration(
        method=method,
        slope=slope,
        intercept_m=intercept,
        sample_count=int(predicted.size),
    )


def calibrate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate model depth from measured_m/predicted_m CSV columns."
    )
    parser.add_argument("--csv", default="data/metric_validation.csv")
    parser.add_argument(
        "--method",
        choices=("ratio", "linear"),
        default="ratio",
        help="ratio: median(measured/predicted); linear: measured=a*predicted+b.",
    )
    args = parser.parse_args()
    try:
        predicted, measured, skipped = load_validation_pairs(args.csv)
        calibration = fit_calibration(predicted, measured, args.method)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    corrected = calibration.slope * predicted + calibration.intercept_m
    rmse = float(np.sqrt(np.mean(np.square(corrected - measured))))
    print("PERINGATAN: kalibrasi ini hanya berlaku untuk kamera yang dipakai saat validasi.")
    print("Ubah kamera, resolusi, zoom, atau pipeline input? Validasi dan kalibrasi ulang.")
    print(f"Pasangan valid: {calibration.sample_count}; baris kosong dilewati: {skipped}")
    print(
        f"Rumus koreksi: depth_kalibrasi_m = "
        f"{calibration.slope:.8f} * depth_model_m "
        f"+ ({calibration.intercept_m:.8f})"
    )
    if calibration.method == "ratio":
        print(f"Faktor rasio median: {calibration.slope:.8f}")
    print(f"RMSE pada data kalibrasi (bukan validasi terpisah): {rmse:.6f} m")

# Evaluate application
@dataclass
class DepthMetricAccumulator:
    """Accumulate pixel-weighted NYU depth metrics over multiple images."""

    silog_lambda: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.silog_lambda) or not 0.0 <= self.silog_lambda <= 1.0:
            raise ValueError("silog_lambda must be finite and in [0, 1].")
        self.valid_pixels = 0
        self.delta1_pixels = 0
        self.abs_rel_sum = 0.0
        self.squared_error_sum = 0.0
        self.log_error_sum = 0.0
        self.squared_log_error_sum = 0.0

    def update(
        self,
        prediction_m: ArrayLike,
        target_m: ArrayLike,
        valid_mask: ArrayLike | None = None,
    ) -> None:
        prediction = np.asarray(prediction_m, dtype=np.float64)
        target = np.asarray(target_m, dtype=np.float64)
        if prediction.shape != target.shape:
            raise ValueError("prediction_m and target_m must have the same shape.")

        valid = (
            np.isfinite(prediction)
            & np.isfinite(target)
            & (prediction > 0)
            & (target > 0)
        )
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape != target.shape:
                raise ValueError("valid_mask must have the same shape as the depth maps.")
            valid &= mask
        if not np.any(valid):
            return

        pred = prediction[valid]
        gt = target[valid]
        ratio = np.maximum(pred / gt, gt / pred)
        log_error = np.log(pred) - np.log(gt)

        self.valid_pixels += int(gt.size)
        self.delta1_pixels += int(np.count_nonzero(ratio < 1.25))
        self.abs_rel_sum += float(np.sum(np.abs(pred - gt) / gt))
        self.squared_error_sum += float(np.sum(np.square(pred - gt)))
        self.log_error_sum += float(np.sum(log_error))
        self.squared_log_error_sum += float(np.sum(np.square(log_error)))

    def compute(self) -> dict[str, float | int]:
        if self.valid_pixels == 0:
            raise ValueError("No valid depth pixels were accumulated.")
        count = self.valid_pixels
        mean_log_error = self.log_error_sum / count
        silog_variance = (
            self.squared_log_error_sum / count
            - self.silog_lambda * mean_log_error**2
        )
        return {
            "delta1": self.delta1_pixels / count,
            "abs_rel": self.abs_rel_sum / count,
            "rmse_m": float(np.sqrt(self.squared_error_sum / count)),
            "silog": float(np.sqrt(max(silog_variance, 0.0)) * 100.0),
            "valid_pixels": count,
        }


def compute_depth_metrics(
    prediction_m: ArrayLike,
    target_m: ArrayLike,
    valid_mask: ArrayLike | None = None,
    silog_lambda: float = 1.0,
) -> dict[str, float | int]:
    """Compute delta1, AbsRel, RMSE (meters), and SILog (percent)."""
    accumulator = DepthMetricAccumulator(silog_lambda=silog_lambda)
    accumulator.update(prediction_m, target_m, valid_mask)
    return accumulator.compute()


def evaluate_nyu_validation(args: argparse.Namespace) -> dict[str, Any]:
    import h5py


    model = load_multitask_onnx_model(args.model, args.metadata)

    with h5py.File(args.nyu_path, "r") as dataset_file:
        total_samples = int(dataset_file["images"].shape[0])
        sample_count = (
            total_samples
            if args.max_samples <= 0
            else min(args.max_samples, total_samples)
        )
        indices = np.arange(sample_count)
        rng = np.random.default_rng(args.split_seed)
        rng.shuffle(indices)
        split = max(1, int(sample_count * 0.9))
        validation_indices = indices[split:]
        if validation_indices.size == 0:
            raise ValueError(
                "Validation split is empty; use a larger dataset or --max-samples."
            )

        metrics = DepthMetricAccumulator(silog_lambda=args.silog_lambda)
        valid_images = 0
        excluded_out_of_range = 0

        for index in validation_indices:
            image_rgb = np.asarray(dataset_file["images"][int(index)]).transpose(
                2, 1, 0
            )
            target_m = np.asarray(dataset_file["depths"][int(index)]).transpose(
                1, 0
            ).astype(np.float32)

            image_bgr = np.ascontiguousarray(image_rgb[:, :, ::-1]).astype(
                np.uint8,
                copy=False,
            )
            _, prediction_m = model.predict(image_bgr)
            if prediction_m.shape != target_m.shape:
                raise ValueError(
                    f"Prediction shape {prediction_m.shape} does not match "
                    f"NYU target shape {target_m.shape}."
                )

            valid = (
                np.isfinite(target_m)
                & (target_m > 0)
                & (target_m <= args.max_depth_m)
            )
            excluded_out_of_range += int(
                np.count_nonzero(
                    np.isfinite(target_m)
                    & (target_m > args.max_depth_m)
                )
            )
            if np.any(valid):
                valid_images += 1
            metrics.update(prediction_m, target_m, valid)

    result = metrics.compute()
    result.update(
        {
            "runtime": "onnxruntime-cpu",
            "total_samples": total_samples,
            "samples_used_for_split": sample_count,
            "validation_images": int(validation_indices.size),
            "images_with_valid_depth": valid_images,
            "excluded_gt_pixels_above_max_depth": excluded_out_of_range,
            "max_depth_m": args.max_depth_m,
            "input_size": [model.input_height, model.input_width],
            "silog_lambda": args.silog_lambda,
        }
    )
    return result


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce depth metrics on the deterministic NYU validation split."
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--silog-lambda",
        type=float,
        default=1.0,
        help="SILog uses 100*sqrt(E[g^2]-lambda*E[g]^2); default 1 is scale invariant.",
    )
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    for path in (args.nyu_path, args.model, args.metadata):
        if not os.path.isfile(path):
            parser.error(f"File tidak ditemukan: {path}")
    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error("--max-depth-m harus bernilai positif.")
    if args.max_samples == 1:
        parser.error("--max-samples harus 0 atau minimal 2 agar split validasi ada.")
    if args.output_csv and os.path.exists(args.output_csv):
        raise FileExistsError(
            f"Refusing to overwrite existing file: {args.output_csv}"
        )

    result = evaluate_nyu_validation(args)
    print("NYU Depth V2 validation metrics (pixel-weighted, metric units)")
    print(f"Runtime: {result['runtime']}")
    print(
        f"Validation images: {result['validation_images']} "
        f"({result['images_with_valid_depth']} contain valid depth)"
    )
    print(f"Valid pixels: {result['valid_pixels']}")
    print(f"GT pixels excluded above {result['max_depth_m']:.2f} m: "
          f"{result['excluded_gt_pixels_above_max_depth']}")
    print(f"delta1 (< 1.25): {result['delta1']:.6f}")
    print(f"AbsRel:           {result['abs_rel']:.6f}")
    print(f"RMSE (m):         {result['rmse_m']:.6f}")
    print(
        f"SILog (x100, lambda={result['silog_lambda']:.2f}): "
        f"{result['silog']:.6f}"
    )

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(result.keys()))
            writer.writeheader()
            writer.writerow(result)
        print(f"Metrics CSV saved: {args.output_csv}")

# Validate application
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CSV_FIELDS = [
    "image",
    "x_px",
    "y_px",
    "measured_m",
    "predicted_m",
    "absolute_error_m",
]


def find_images(image_dir: str) -> list[Path]:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Folder foto tidak ditemukan: {image_dir}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _on_photo_mouse(event: int, x: int, y: int, flags: int, state: dict[str, Any]) -> None:
    del flags
    if event == state["cv2"].EVENT_LBUTTONDOWN:
        frame_width = state["frame_width"]
        if x >= frame_width * 2:
            return
        if x >= frame_width:
            x -= frame_width
        state["point"] = (x, y)


def validate_photos(args: argparse.Namespace) -> None:
    import cv2


    model = load_multitask_onnx_model(args.model, args.metadata)
    rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {"cv2": cv2, "point": None, "frame_width": 0}
    title = "Photo metric-depth validation"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, _on_photo_mouse, state)

    try:
        for image_path in find_images(args.image_dir):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Gagal membaca foto: {image_path}")
            _, depth_m = model.predict(frame)
            depth_color = cv2.applyColorMap(
                normalize_depth_for_display(depth_m), cv2.COLORMAP_INFERNO
            )
            state["point"] = None
            state["frame_width"] = frame.shape[1]

            while True:
                annotated = frame.copy()
                point = state["point"]
                if point is None:
                    instruction = "Klik pusat target; Enter: simpan; n: lewati; q: keluar"
                else:
                    try:
                        distance_m = get_depth_at_point(
                            depth_m, point[0], point[1], patch_size=args.patch_size
                        )
                        instruction = (
                            f"Prediksi {distance_m:.2f} m / "
                            f"{distance_m * 100:.0f} cm; Enter simpan; n lewati; q keluar"
                        )
                        cv2.circle(annotated, point, 7, (0, 255, 255), 2)
                    except ValueError:
                        instruction = "Depth tidak valid di titik ini; pilih titik lain"

                cv2.putText(
                    annotated,
                    instruction,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                display = np.hstack((annotated, depth_color))
                cv2.imshow(title, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("n"):
                    rows.append(
                        {
                            "image": image_path.name,
                            "x_px": "",
                            "y_px": "",
                            "measured_m": "",
                            "predicted_m": "",
                            "absolute_error_m": "",
                        }
                    )
                    break
                if key in (10, 13) and point is not None:
                    try:
                        distance_m = get_depth_at_point(
                            depth_m, point[0], point[1], patch_size=args.patch_size
                        )
                    except ValueError:
                        continue
                    rows.append(
                        {
                            "image": image_path.name,
                            "x_px": point[0],
                            "y_px": point[1],
                            "measured_m": "",
                            "predicted_m": f"{distance_m:.6f}",
                            "absolute_error_m": "",
                        }
                    )
                    break

            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()

    write_csv(args.output, rows)
    print(f"Saved {len(rows)} photo rows to {args.output}")
    print("Fill measured_m from the tape measure; compute absolute_error_m afterward.")


def validate_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a blank template or sample real-distance photos."
    )
    parser.add_argument("--image-dir", default="data/metric_photos")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output", default="data/metric_validation.csv")
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Create an empty measured/predicted table without loading the model.",
    )
    args = parser.parse_args()

    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error("--max-depth-m harus bernilai positif.")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        parser.error("--patch-size harus bilangan ganjil positif.")
    if os.path.exists(args.output):
        raise FileExistsError(f"Refusing to overwrite existing file: {args.output}")
    images = find_images(args.image_dir)
    if not images:
        raise FileNotFoundError(f"Tidak ada foto yang didukung di {args.image_dir}")

    if args.template_only:
        rows = [
            {
                "image": image.name,
                "x_px": "",
                "y_px": "",
                "measured_m": "",
                "predicted_m": "",
                "absolute_error_m": "",
            }
            for image in images
        ]
        write_csv(args.output, rows)
        print(f"Blank validation template saved: {args.output}")
        return

    for path in (args.model, args.metadata):
        if not os.path.isfile(path):
            parser.error(f"File model/metadata tidak ditemukan: {path}")
    validate_photos(args)

# Camera application
def resolve_class_ids(names: Any, requested_names: list[str] | None) -> list[int] | None:
    """Resolve requested class names against the multitask model's labels."""
    if not requested_names:
        return None
    if isinstance(names, dict):
        class_items = [(int(class_id), str(name)) for class_id, name in names.items()]
    elif isinstance(names, (list, tuple)):
        class_items = list(enumerate(str(name) for name in names))
    else:
        raise TypeError("Model class names must be a mapping or a sequence.")

    normalized_to_id = {name.strip().casefold(): class_id for class_id, name in class_items}
    unknown = [
        requested
        for requested in requested_names
        if requested.strip().casefold() not in normalized_to_id
    ]
    if unknown:
        available = ", ".join(name for _, name in class_items)
        raise ValueError(
            f"Unknown class name(s): {', '.join(unknown)}. "
            f"Available names: {available}"
        )
    return list(
        dict.fromkeys(normalized_to_id[name.strip().casefold()] for name in requested_names)
    )


def _draw_detection_overlay(
    frame: np.ndarray,
    detections: np.ndarray,
    names: tuple[str, ...],
    depth_m: np.ndarray,
    patch_size: int,
    max_depth_m: float,
    confidence: float,
    class_ids: list[int] | None,
    debug: bool,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = frame.shape[:2]

    for row in detections:
        x1, y1, x2, y2, score, raw_class_id = row
        if not np.isfinite(row).all() or score < confidence:
            continue
        class_id = int(round(float(raw_class_id)))
        if class_id not in range(len(names)):
            continue
        if class_ids is not None and class_id not in class_ids:
            continue
        x1, y1, x2, y2 = np.rint((x1, y1, x2, y2)).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width - 1)
        y1, y2 = np.clip((y1, y2), 0, height - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        try:
            distance_m = get_depth_in_bbox(depth_m, (x1, y1, x2, y2), patch_size)
            distance_cm = float(meters_to_cm(distance_m))
            distance_text = f"{distance_m:.2f} m / {distance_cm:.0f} cm"
        except ValueError:
            distance_m = None
            distance_text = "depth tidak valid"

        label = (
            f"{names[class_id]} {float(score):.2f} - {distance_text}"
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.circle(annotated, (center_x, center_y), 3, (0, 0, 255), -1)
        text_y = max(20, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (50, 220, 50),
            2,
            cv2.LINE_AA,
        )
        if debug:
            try:
                raw_stats = get_depth_patch_stats(
                    depth_m / np.float32(max_depth_m),
                    center_x,
                    center_y,
                    patch_size=patch_size,
                )
                height_patch, width_patch = raw_stats.patch_shape
                debug_lines = [
                    f"raw center={raw_stats.center_value:.4f} "
                    f"median={raw_stats.median:.4f}",
                    f"patch={width_patch}x{height_patch} "
                    f"var={raw_stats.variance:.6f} n={raw_stats.valid_count}",
                ]
                for line_index, debug_line in enumerate(debug_lines, start=1):
                    cv2.putText(
                        annotated,
                        debug_line,
                        (x1, min(height - 5, text_y + line_index * 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            except ValueError:
                cv2.putText(
                    annotated,
                    "depth patch invalid",
                    (x1, min(height - 5, text_y + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return annotated


def _on_camera_mouse(event: int, x: int, y: int, flags: int, state: dict[str, Any]) -> None:
    del flags
    if event != cv2.EVENT_LBUTTONDOWN or state.get("depth_m") is None:
        return

    frame_width = state["frame_width"]
    if state["side_by_side"]:
        if x >= frame_width * 2:
            return
        if x >= frame_width:
            x -= frame_width
    elif x >= frame_width:
        return
    state["tap_point"] = (x, y)


def _draw_split_camera_view(
    detection_view: NDArray[np.uint8],
    depth_view: NDArray[np.uint8],
    depth_m: NDArray[np.float32],
    patch_size: int,
) -> NDArray[np.uint8]:
    """Join detection and depth views, with metric readings visible on both."""
    if detection_view.shape != depth_view.shape:
        raise ValueError("Detection and depth views must have the same shape.")

    height, width = detection_view.shape[:2]
    center_x, center_y = width // 2, height // 2
    try:
        center_distance = get_depth_at_point(
            depth_m, center_x, center_y, patch_size=patch_size
        )
        depth_reading = f"Depth tengah: {center_distance:.2f} m"
    except ValueError:
        depth_reading = "Depth tengah: tidak valid"

    for panel, title, detail in (
        (detection_view, "DETEKSI + JARAK", "Jarak tiap objek ditulis pada kotak"),
        (depth_view, "PETA DEPTH + JARAK", depth_reading),
    ):
        cv2.rectangle(panel, (0, 0), (width, 58), (20, 27, 40), thickness=-1)
        cv2.putText(
            panel,
            title,
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            detail,
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 230, 190),
            1,
            cv2.LINE_AA,
        )
    return np.hstack((detection_view, depth_view))


def run_camera(
    model: MultitaskOnnxModel,
    camera_index: int,
    confidence: float,
    class_ids: list[int] | None,
    patch_size: int,
    max_depth_m: float,
    debug: bool,
) -> None:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Kamera index {camera_index} tidak dapat dibuka.")

    title = "YOLO26n multitask detection + metric depth"
    state: dict[str, Any] = {
        "depth_m": None,
        "frame_width": 0,
        "side_by_side": True,
        "tap_point": None,
    }
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, _on_camera_mouse, state)
    smoother = DepthEMASmoother()
    show_depth_overlay = True

    print("Satu model YOLO26n multitask: detection head + depth head")
    print(
        "Tampilan awal split: deteksi + jarak di kiri, depth + jarak di kanan. "
        "Klik untuk mengukur titik; s: ganti tampilan; d: overlay on/off; q: keluar."
    )
    if debug:
        print(
            "Debug aktif: nilai raw di center bbox, ukuran patch efektif, "
            "variansi patch raw-depth."
        )

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Gagal membaca frame dari kamera.")

            detections, depth_m = model.predict(frame)
            depth_m = smoother.update(depth_m)
            state["depth_m"] = depth_m
            state["frame_width"] = frame.shape[1]

            annotated = _draw_detection_overlay(
                frame,
                detections,
                model.class_names,
                depth_m,
                patch_size,
                max_depth_m,
                confidence,
                class_ids,
                debug,
            )
            depth_color = cv2.applyColorMap(
                normalize_depth_for_display(depth_m), cv2.COLORMAP_INFERNO
            )
            depth_annotated = _draw_detection_overlay(
                depth_color,
                detections,
                model.class_names,
                depth_m,
                patch_size,
                max_depth_m,
                confidence,
                class_ids,
                False,
            )

            if state["tap_point"] is not None:
                point_x, point_y = state["tap_point"]
                try:
                    point_depth = get_depth_at_point(
                        depth_m, point_x, point_y, patch_size=patch_size
                    )
                    point_cm = float(meters_to_cm(point_depth))
                    tap_text = (
                        f"Tap: {point_depth:.2f} m / {point_cm:.0f} cm"
                    )
                    cv2.circle(annotated, (point_x, point_y), 6, (0, 255, 255), 2)
                except ValueError:
                    tap_text = "Tap: depth tidak valid"
                cv2.putText(
                    annotated,
                    tap_text,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    depth_annotated,
                    (point_x, point_y),
                    6,
                    (0, 255, 255),
                    2,
                )

            if state["side_by_side"]:
                display = _draw_split_camera_view(
                    annotated, depth_annotated, depth_m, patch_size
                )
            elif show_depth_overlay:
                display = cv2.addWeighted(annotated, 0.68, depth_color, 0.32, 0)
            else:
                display = annotated
            cv2.imshow(title, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_depth_overlay = not show_depth_overlay
            elif key == ord("s"):
                state["side_by_side"] = not state["side_by_side"]
    finally:
        camera.release()
        cv2.destroyAllWindows()


def camera_main() -> None:
    parser = argparse.ArgumentParser(
        description="Single YOLO26n multitask model for detections and metric depth"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Single ONNX checkpoint with detection and depth outputs.",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help="Training metadata for shapes and class labels.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--conf",
        "--confidence",
        dest="confidence",
        type=float,
        default=0.25,
        help="Ambang confidence deteksi (default: 0.25).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS_NAME",
        help='Hanya tampilkan kelas dari model, misalnya --classes chair "dining table".',
    )
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tampilkan raw depth di center bbox, ukuran patch, dan variansi patch.",
    )
    args = parser.parse_args()

    for path in (args.model, args.metadata):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File model/metadata tidak ditemukan: {path}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence harus berada pada rentang [0, 1].")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        parser.error("--patch-size harus bilangan ganjil positif.")
    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error("--max-depth-m harus bernilai positif.")

    model = load_multitask_onnx_model(args.model, args.metadata)
    try:
        class_ids = resolve_class_ids(model.class_names, args.classes)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    run_camera(
        model,
        args.camera,
        args.confidence,
        class_ids,
        args.patch_size,
        args.max_depth_m,
        args.debug,
    )


WEB_PAGE = r"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO26n Depth — Localhost</title>
  <style>
    :root{color-scheme:dark;--bg:#0b1020;--panel:#121a2d;--line:#26334c;--muted:#9aabc4;--accent:#7ce0bd;--text:#eef4ff}
    *{box-sizing:border-box}body{margin:0;background:radial-gradient(ellipse at top,#172644,var(--bg) 60%);color:var(--text);font:15px/1.5 system-ui,sans-serif}
    main{width:min(1180px,94vw);margin:34px auto 56px}.top{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:22px}
    h1{font-size:clamp(26px,4vw,40px);line-height:1.1;margin:0 0 8px;letter-spacing:-.04em}.sub,.hint{color:var(--muted)}
    .badge{border:1px solid #38556a;border-radius:99px;color:var(--accent);padding:7px 12px;white-space:nowrap}
    .panel{background:color-mix(in srgb,var(--panel) 94%,transparent);border:1px solid var(--line);border-radius:18px;padding:18px;box-shadow:0 18px 50px #0003}
    .controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:15px}
    button,.upload{border:1px solid #3b506c;border-radius:10px;background:#1b2941;color:var(--text);padding:10px 15px;font:inherit;cursor:pointer}
    button.primary{background:var(--accent);border-color:var(--accent);color:#092018;font-weight:700}
    button:disabled{opacity:.45;cursor:not-allowed}.upload input{display:none}.range{display:flex;align-items:center;gap:8px;color:var(--muted);margin-left:auto}
    input[type=range]{accent-color:var(--accent)}.status{color:var(--muted);min-height:24px;margin:3px 0 12px}
    .status.error{color:#ff9b9b}.view{position:relative;min-height:140px;background:#080c15;border-radius:12px;overflow:hidden;display:flex;align-items:center;justify-content:center}
    .view video,.view img{display:block;width:100%;height:auto;max-height:70vh;object-fit:contain}.view canvas{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
    #video{display:none}#photo{display:none}.empty{padding:52px 20px;text-align:center;color:var(--muted)}
    .results{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,320px);gap:18px;margin-top:18px}
    .results h2{font-size:16px;margin:0 0 10px}.depth{width:100%;aspect-ratio:4/3;object-fit:cover;border-radius:10px;background:#080c15}
    .detections{display:grid;gap:8px}.item{display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:7px 2px}
    .item span:last-child{color:var(--accent);white-space:nowrap}.empty-list{color:var(--muted)}
    footer{color:var(--muted);font-size:13px;margin-top:16px}
    @media(max-width:700px){main{margin-top:20px}.top{align-items:start;flex-direction:column}.results{grid-template-columns:1fr}.range{margin-left:0;width:100%}.view video,.view img{max-height:55vh}}
  </style>
</head>
<body>
<main>
  <header class="top">
    <div><h1>YOLO26n <span style="color:var(--accent)">Depth</span></h1><div class="sub">Deteksi objek dan estimasi jarak dari satu model multitask.</div></div>
    <div class="badge">Berjalan lokal · ONNX</div>
  </header>
  <section class="panel">
    <div class="controls">
      <button id="start" class="primary">Nyalakan kamera</button>
      <button id="stop" disabled>Matikan</button>
      <label class="upload">Pilih foto<input id="file" type="file" accept="image/*"></label>
      <label class="range">Confidence <input id="confidence" type="range" min="5" max="80" value="25"><span id="confValue">0.25</span></label>
    </div>
    <div id="status" class="status">Memeriksa model lokal…</div>
    <div id="view" class="view">
      <div id="empty" class="empty">Nyalakan kamera atau pilih foto untuk mulai.</div>
      <video id="video" autoplay playsinline muted></video>
      <img id="photo" alt="Foto yang dipilih">
      <canvas id="overlay"></canvas>
    </div>
    <div class="results">
      <div><h2>Objek terdeteksi</h2><div id="detections" class="detections"><div class="empty-list">Belum ada hasil.</div></div></div>
      <div><h2>Peta depth</h2><img id="depth" class="depth" alt="Peta kedalaman model"></div>
    </div>
    <footer>Video/foto diproses di komputer ini melalui localhost; tidak dikirim ke layanan luar.</footer>
  </section>
</main>
<script>
const video=document.querySelector("#video"),photo=document.querySelector("#photo"),overlay=document.querySelector("#overlay");
const ctx=overlay.getContext("2d"),statusEl=document.querySelector("#status"),empty=document.querySelector("#empty");
const start=document.querySelector("#start"),stop=document.querySelector("#stop"),file=document.querySelector("#file");
const confidence=document.querySelector("#confidence"),confValue=document.querySelector("#confValue"),depthEl=document.querySelector("#depth");
const list=document.querySelector("#detections"),capture=document.createElement("canvas");
let stream=null,running=false,inFlight=false,mode="camera",lastSourceSize=[0,0];
function status(text,error=false){statusEl.textContent=text;statusEl.classList.toggle("error",error)}
confidence.addEventListener("input",()=>{confValue.textContent=(Number(confidence.value)/100).toFixed(2)});
function showSource(){empty.style.display="none";video.style.display=mode==="camera"?"block":"none";photo.style.display=mode==="photo"?"block":"none"}
function clearOverlay(){ctx.clearRect(0,0,overlay.width,overlay.height)}
function captureFrame(){
  const w=mode==="camera"?video.videoWidth:photo.naturalWidth,h=mode==="camera"?video.videoHeight:photo.naturalHeight;
  if(!w||!h)return null;
  const scale=Math.min(1,960/w,960/h),cw=Math.max(1,Math.round(w*scale)),ch=Math.max(1,Math.round(h*scale));
  capture.width=cw;capture.height=ch;capture.getContext("2d").drawImage(mode==="camera"?video:photo,0,0,cw,ch);
  overlay.width=w;overlay.height=h;lastSourceSize=[cw,ch];
  return new Promise(resolve=>capture.toBlob(resolve,"image/jpeg",0.82));
}
function draw(items,sourceWidth,sourceHeight){
  clearOverlay();const scaleX=overlay.width/sourceWidth,scaleY=overlay.height/sourceHeight;
  ctx.lineWidth=Math.max(2,overlay.width/500);ctx.font=`600 ${Math.max(13,overlay.width/65)}px system-ui`;
  for(const d of items){
    const [x1,y1,x2,y2]=d.box,x=x1*scaleX,y=y1*scaleY,w=(x2-x1)*scaleX,h=(y2-y1)*scaleY;
    ctx.strokeStyle="#7ce0bd";ctx.fillStyle="#7ce0bd";ctx.strokeRect(x,y,w,h);
    const text=`${d.class_name} ${d.distance_m.toFixed(2)} m · ${(d.score*100).toFixed(0)}%`;
    const th=Math.max(21,overlay.width/30),tw=ctx.measureText(text).width+12,ty=Math.max(th,y);
    ctx.fillRect(x,ty-th,tw,th);ctx.fillStyle="#092018";ctx.fillText(text,x+6,ty-th/4);
  }
  list.replaceChildren();
  if(!items.length){list.innerHTML='<div class="empty-list">Tidak ada objek pada ambang confidence ini.</div>';return}
  for(const d of items){const row=document.createElement("div");row.className="item";
    const name=document.createElement("span");name.textContent=`${d.class_name} · ${(d.score*100).toFixed(0)}%`;
    const dist=document.createElement("span");dist.textContent=`${d.distance_m.toFixed(2)} m (${(d.distance_m*100).toFixed(0)} cm)`;
    row.append(name,dist);list.append(row)}
}
async function infer(){
  if(inFlight)return;inFlight=true;
  try{
    const blob=await captureFrame();if(!blob)return;
    const started=performance.now();
    const response=await fetch(`/api/infer?confidence=${Number(confidence.value)/100}`,{method:"POST",headers:{"Content-Type":"image/jpeg"},body:blob});
    const data=await response.json();if(!response.ok)throw new Error(data.error||"Inferensi gagal.");
    draw(data.detections,data.width,data.height);
    depthEl.src=`data:image/jpeg;base64,${data.depth_preview}`;
    status(`Inferensi ${data.inference_ms.toFixed(0)} ms · ${data.detections.length} objek`);
  }catch(error){status(error.message||String(error),true);running=false;start.disabled=false;stop.disabled=true}
  finally{inFlight=false}
  if(running)setTimeout(infer,350);
}
start.addEventListener("click",async()=>{
  try{if(!navigator.mediaDevices?.getUserMedia)throw new Error("Browser tidak mendukung akses kamera.");
    stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"},audio:false});
    video.srcObject=stream;await video.play();mode="camera";showSource();running=true;start.disabled=true;stop.disabled=false;status("Kamera aktif; memproses frame…");infer();
  }catch(error){status(error.message||String(error),true)}
});
stop.addEventListener("click",()=>{running=false;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}video.srcObject=null;start.disabled=false;stop.disabled=true;status("Kamera dimatikan.")});
file.addEventListener("change",()=>{
  const chosen=file.files?.[0];if(!chosen)return;
  running=false;if(stream){stream.getTracks().forEach(t=>t.stop());stream=null}video.srcObject=null;
  mode="photo";photo.src=URL.createObjectURL(chosen);photo.onload=()=>{showSource();status("Foto siap; menjalankan inferensi…");infer()};
});
(async()=>{try{const r=await fetch("/api/health");const d=await r.json();if(!r.ok)throw new Error(d.error||"Model tidak siap.");status(`Model siap · ${d.classes} kelas · input ${d.input_width}×${d.input_height}`)}catch(e){status(e.message||String(e),true)}})();
</script>
</body>
</html>
"""


def create_web_handler(model: MultitaskOnnxModel, patch_size: int = 15):
    inference_lock = threading.Lock()

    class WebHandler(BaseHTTPRequestHandler):
        server_version = "YOLO26nDepthLocal/1.0"

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/":
                body = WEB_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif route == "/api/health":
                self._send_json(
                    {
                        "ready": True,
                        "classes": len(model.class_names),
                        "input_width": model.input_width,
                        "input_height": model.input_height,
                    }
                )
            else:
                self._send_json({"error": "Halaman tidak ditemukan."}, 404)

        def do_POST(self) -> None:
            route = urlsplit(self.path)
            if route.path != "/api/infer":
                self._send_json({"error": "Endpoint tidak ditemukan."}, 404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json({"error": "Content-Length tidak valid."}, 400)
                return
            if content_length <= 0 or content_length > 8 * 1024 * 1024:
                self._send_json({"error": "Ukuran foto harus antara 1 byte dan 8 MB."}, 413)
                return

            try:
                confidence_values = parse_qs(route.query).get("confidence", ["0.25"])
                confidence = float(confidence_values[0])
                if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError("Confidence harus berada pada rentang 0 sampai 1.")
                encoded_image = self.rfile.read(content_length)
                frame = cv2.imdecode(
                    np.frombuffer(encoded_image, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if frame is None:
                    raise ValueError("Data bukan foto JPEG/PNG yang dapat dibaca.")

                started = time.perf_counter()
                with inference_lock:
                    detections, depth_m = model.predict(frame)
                inference_ms = (time.perf_counter() - started) * 1000.0

                results: list[dict[str, Any]] = []
                for row in detections:
                    if not np.isfinite(row).all() or row[4] < confidence:
                        continue
                    class_id = int(round(float(row[5])))
                    if class_id not in range(len(model.class_names)):
                        continue
                    x1, y1, x2, y2 = (float(value) for value in row[:4])
                    if x2 <= x1 or y2 <= y1:
                        continue
                    try:
                        distance_m = get_depth_in_bbox(
                            depth_m,
                            (x1, y1, x2, y2),
                            patch_size=patch_size,
                        )
                    except ValueError:
                        continue
                    results.append(
                        {
                            "box": [x1, y1, x2, y2],
                            "score": float(row[4]),
                            "class_id": class_id,
                            "class_name": model.class_names[class_id],
                            "distance_m": distance_m,
                        }
                    )

                depth_color = cv2.applyColorMap(
                    normalize_depth_for_display(depth_m),
                    cv2.COLORMAP_INFERNO,
                )
                ok, preview = cv2.imencode(
                    ".jpg",
                    depth_color,
                    [cv2.IMWRITE_JPEG_QUALITY, 78],
                )
                if not ok:
                    raise RuntimeError("Gagal membuat pratinjau peta depth.")
                self._send_json(
                    {
                        "width": int(frame.shape[1]),
                        "height": int(frame.shape[0]),
                        "detections": results,
                        "depth_preview": base64.b64encode(preview).decode("ascii"),
                        "inference_ms": inference_ms,
                    }
                )
            except ValueError as error:
                self._send_json({"error": str(error)}, 400)
            except Exception as error:
                self.log_error("Inferensi gagal: %s", error)
                self._send_json({"error": f"Inferensi gagal: {error}"}, 500)

        def log_message(self, format: str, *args: Any) -> None:
            super().log_message(format, *args)

    return WebHandler


def web_main() -> None:
    parser = argparse.ArgumentParser(
        description="Jalankan YOLO26n multitask sebagai web app lokal."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("Web lokal hanya boleh bind ke localhost (127.0.0.1).")
    if not 1 <= args.port <= 65535:
        parser.error("--port harus berada pada rentang 1 hingga 65535.")
    for path in (args.model, args.metadata):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File model/metadata tidak ditemukan: {path}")

    model = load_multitask_onnx_model(args.model, args.metadata)
    server = ThreadingHTTPServer(
        (args.host, args.port),
        create_web_handler(model),
    )
    print(f"Web app berjalan di http://{args.host}:{args.port}")
    print("Tekan Ctrl+C di terminal untuk menghentikan server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")
    finally:
        server.server_close()


def main() -> None:
    """Dispatch the desktop application commands from one entry point."""
    commands = {
        "web": web_main,
        "camera": camera_main,
        "evaluate": evaluate_main,
        "validate": validate_main,
        "calibrate": calibrate_main,
    }
    parser = argparse.ArgumentParser(
        description="YOLO26n multitask detection and metric-depth application"
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(commands),
        default="web",
        help="local web app (default), camera, evaluation, photo validation, or calibration",
    )
    arguments = sys.argv[1:]
    if arguments and arguments[0] in commands:
        command = arguments.pop(0)
    elif arguments and arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    else:
        command = "web"
    sys.argv = [sys.argv[0], *arguments]
    commands[command]()


if __name__ == "__main__":
    main()
