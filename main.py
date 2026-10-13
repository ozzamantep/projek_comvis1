"""Terminal camera app for multitask detection and metric depth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

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
DEFAULT_CAMERA_DEPTH_SCALE = 0.4967948718

# Depth application
def meters_to_cm(meters: ArrayLike) -> NDArray[np.float32]:
    """Convert meters to centimeters."""
    return np.asarray(meters, dtype=np.float32) * np.float32(100.0)


def calibrate_depth_map(
    depth_m: ArrayLike, scale: float = DEFAULT_CAMERA_DEPTH_SCALE
) -> NDArray[np.float32]:
    """Apply the empirical webcam scale to a metric depth map."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Depth calibration scale must be finite and positive.")
    return np.asarray(depth_m, dtype=np.float32) * np.float32(scale)


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
    """Load one ONNX graph and validate it against its model metadata."""
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
    depth_scale: float,
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
    print(f"Kalibrasi skala depth kamera: x{depth_scale:.4f}")
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
            depth_m = calibrate_depth_map(smoother.update(depth_m), depth_scale)
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
        help="Model metadata for input/output shapes and class labels.",
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
        "--depth-scale",
        type=float,
        default=DEFAULT_CAMERA_DEPTH_SCALE,
        help=(
            "Kalibrasi skala webcam (default x0.4968 dari sepuluh pasangan "
            "ukur; khusus setup kamera ini)."
        ),
    )
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
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("--depth-scale harus bernilai positif.")

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
        args.depth_scale,
    )


def main() -> None:
    """Run the camera directly, optionally accepting the `camera` prefix."""
    if len(sys.argv) > 1 and sys.argv[1] == "camera":
        del sys.argv[1]
    camera_main()


if __name__ == "__main__":
    main()
