"""Interactive real-distance photo validation and CSV template generation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import numpy as np


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


def _on_mouse(event: int, x: int, y: int, flags: int, state: dict[str, Any]) -> None:
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
    import torch

    if __package__:
        from .camera_yolo26_depth_metric import (
            load_depth_model,
            predict_depth_meters,
        )
        from .depth_utils import get_depth_at_point, normalize_depth_for_display
    else:
        from camera_yolo26_depth_metric import (
            load_depth_model,
            predict_depth_meters,
        )
        from depth_utils import get_depth_at_point, normalize_depth_for_display

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_depth_model(args.weights, device)
    rows: list[dict[str, Any]] = []
    state: dict[str, Any] = {"cv2": cv2, "point": None, "frame_width": 0}
    title = "Photo metric-depth validation"
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, _on_mouse, state)

    try:
        for image_path in find_images(args.image_dir):
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Gagal membaca foto: {image_path}")
            depth_m = predict_depth_meters(
                model, device, frame, max_depth_m=args.max_depth_m
            )
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
            else:
                continue

            if key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()

    write_csv(args.output, rows)
    print(f"Saved {len(rows)} photo rows to {args.output}")
    print("Fill measured_m from the tape measure; compute absolute_error_m afterward.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a blank template or sample real-distance photos."
    )
    parser.add_argument("--image-dir", default="data/metric_photos")
    parser.add_argument("--weights", default="data/yolo26n_depth_nyu_full.pt")
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

    if not os.path.isfile(args.weights):
        parser.error(f"Checkpoint tidak ditemukan: {args.weights}")
    validate_photos(args)


if __name__ == "__main__":
    main()
