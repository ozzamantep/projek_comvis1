"""Desktop demo for the jointly trained detection-and-depth model."""

from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
import numpy as np

if __package__:
    from .depth_utils import (
        DepthEMASmoother,
        get_depth_at_point,
        get_depth_in_bbox,
        get_depth_patch_stats,
        meters_to_cm,
        normalize_depth_for_display,
    )
    from .multitask_onnx import MultitaskOnnxModel, load_multitask_onnx_model
else:
    from depth_utils import (
        DepthEMASmoother,
        get_depth_at_point,
        get_depth_in_bbox,
        get_depth_patch_stats,
        meters_to_cm,
        normalize_depth_for_display,
    )
    from multitask_onnx import MultitaskOnnxModel, load_multitask_onnx_model


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


def _on_mouse(event: int, x: int, y: int, flags: int, state: dict[str, Any]) -> None:
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
        "side_by_side": False,
        "tap_point": None,
    }
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, _on_mouse, state)
    smoother = DepthEMASmoother()
    show_depth_overlay = True

    print("Satu model YOLO26n multitask: detection head + depth head")
    print(
        "Klik untuk mengukur depth titik; d: overlay on/off; "
        "s: side-by-side; q: keluar."
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

            if state["side_by_side"]:
                display = np.hstack((annotated, depth_color))
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single YOLO26n multitask model for detections and metric depth"
    )
    parser.add_argument(
        "--model",
        default="data/multitask.onnx",
        help="Single ONNX checkpoint with detection and depth outputs.",
    )
    parser.add_argument(
        "--metadata",
        default="data/model_metadata.json",
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


if __name__ == "__main__":
    main()
