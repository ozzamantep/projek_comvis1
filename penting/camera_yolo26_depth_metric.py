"""Desktop demo: COCO object detection plus the local metric-depth model."""

from __future__ import annotations

import argparse
import os
from typing import Any

import cv2
import numpy as np
import torch

if __package__:
    from .depth_utils import (
        DepthEMASmoother,
        get_depth_at_point,
        get_depth_in_bbox,
        get_depth_patch_stats,
        meters_to_cm,
        normalize_depth_for_display,
    )
    from .ultralytics_depth import (
        load_calibrated_depth_model,
        predict_calibrated_depth_meters,
    )
else:
    from depth_utils import (
        DepthEMASmoother,
        get_depth_at_point,
        get_depth_in_bbox,
        get_depth_patch_stats,
        meters_to_cm,
        normalize_depth_for_display,
    )
    from ultralytics_depth import (
        load_calibrated_depth_model,
        predict_calibrated_depth_meters,
    )


DEPTH_INPUT_SIZE = 640


def load_depth_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load the calibrated YOLO26n-Depth checkpoint used by the current pipeline."""
    return load_calibrated_depth_model(weights_path, device)


def predict_depth_meters(
    model: torch.nn.Module,
    device: torch.device,
    frame_bgr: np.ndarray,
    input_size: int = DEPTH_INPUT_SIZE,
) -> np.ndarray:
    """Return the model's calibrated meter predictions at camera resolution."""
    return predict_calibrated_depth_meters(
        model,
        frame_bgr,
        device,
        input_size=input_size,
    )


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def resolve_class_ids(names: Any, requested_names: list[str] | None) -> list[int] | None:
    """Resolve requested COCO class names against the detector's own labels."""
    if not requested_names:
        return None
    if isinstance(names, dict):
        class_items = [(int(class_id), str(name)) for class_id, name in names.items()]
    elif isinstance(names, (list, tuple)):
        class_items = list(enumerate(str(name) for name in names))
    else:
        raise TypeError("Detector class names must be a mapping or a sequence.")

    normalized_to_id = {name.strip().casefold(): class_id for class_id, name in class_items}
    unknown = [
        requested
        for requested in requested_names
        if requested.strip().casefold() not in normalized_to_id
    ]
    if unknown:
        available = ", ".join(name for _, name in class_items)
        raise ValueError(
            f"Unknown detector class name(s): {', '.join(unknown)}. "
            f"Available names: {available}"
        )
    return list(
        dict.fromkeys(normalized_to_id[name.strip().casefold()] for name in requested_names)
    )


def _draw_detection_overlay(
    frame: np.ndarray,
    result: Any,
    names: Any,
    depth_m: np.ndarray,
    patch_size: int,
    max_depth_m: float,
    debug: bool,
) -> np.ndarray:
    annotated = frame.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return annotated

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidences = boxes.conf.detach().cpu().numpy()
    height, width = frame.shape[:2]

    for box, class_id, confidence in zip(xyxy, classes, confidences):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip((x1, x2), 0, width - 1)
        y1, y2 = np.clip((y1, y2), 0, height - 1)
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        try:
            distance_m = get_depth_in_bbox(depth_m, (x1, y1, x2, y2), patch_size)
            distance_cm = float(meters_to_cm(distance_m))
            distance_text = f"{distance_m:.2f} m / {distance_cm:.0f} cm"
        except ValueError:
            distance_m = None
            distance_text = "depth tidak valid"

        label = (
            f"{_class_name(names, int(class_id))} "
            f"{float(confidence):.2f} - {distance_text}"
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
    detector: Any,
    depth_weights: str,
    camera_index: int,
    confidence: float,
    class_ids: list[int] | None,
    patch_size: int,
    max_depth_m: float,
    debug: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model = load_depth_model(depth_weights, device)
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Kamera index {camera_index} tidak dapat dibuka.")

    title = "YOLO26 COCO detection + metric depth"
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

    print(f"Detector COCO: {getattr(detector, 'ckpt_path', 'loaded detector')}")
    print(f"Model depth lokal: {depth_weights} | Device: {device}")
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

            depth_m = predict_depth_meters(depth_model, device, frame)
            depth_m = smoother.update(depth_m)
            state["depth_m"] = depth_m
            state["frame_width"] = frame.shape[1]

            results = detector(
                frame,
                verbose=False,
                conf=confidence,
                classes=class_ids,
            )
            result = results[0]
            annotated = _draw_detection_overlay(
                frame,
                result,
                detector.names,
                depth_m,
                patch_size,
                max_depth_m,
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
        description="COCO detector plus local metric-depth camera demo"
    )
    parser.add_argument("--detector", default="data/yolo26n.pt")
    parser.add_argument(
        "--depth-weights",
        default="data/yolo26n-depth-nyu-calibrated.pt",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--conf",
        "--confidence",
        dest="confidence",
        type=float,
        default=0.5,
        help="Ambang confidence detector (default: 0.5).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        metavar="COCO_NAME",
        help='Hanya tampilkan kelas detector ini, misalnya --classes person "cell phone".',
    )
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tampilkan raw depth di center bbox, ukuran patch, dan variansi patch.",
    )
    args = parser.parse_args()

    for path in (args.detector, args.depth_weights):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint tidak ditemukan: {path}")
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence harus berada pada rentang [0, 1].")
    if args.patch_size < 1 or args.patch_size % 2 == 0:
        parser.error("--patch-size harus bilangan ganjil positif.")
    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error("--max-depth-m harus bernilai positif.")

    from ultralytics import YOLO

    detector = YOLO(args.detector)
    try:
        class_ids = resolve_class_ids(detector.names, args.classes)
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    run_camera(
        detector,
        args.depth_weights,
        args.camera,
        args.confidence,
        class_ids,
        args.patch_size,
        args.max_depth_m,
        args.debug,
    )


if __name__ == "__main__":
    main()
