"""Diagnose preprocessing and spatial/input collapse in local depth checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

if __package__:
    from .arsitektur_yolo_depth import (
        YOLO26nDepth,
        load_depth_checkpoint,
    )
else:
    from arsitektur_yolo_depth import (
        YOLO26nDepth,
        load_depth_checkpoint,
    )


INPUT_WIDTH = 320
INPUT_HEIGHT = 240
DEFAULT_CHECKPOINTS = {
    "nyu_full": "data/yolo26n_depth_nyu_full.pt",
    "nyu": "data/yolo26n_depth_nyu.pt",
    "trained": "data/yolo26n_depth_trained.pt",
}
DEFAULT_ULTRALYTICS_WEIGHTS = "data/pendukung/yolo26n-depth.pt"


def preprocess_camera_runner(frame_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(
        frame_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0)


def preprocess_nyu_trainer_camera(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    trainer_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    trainer_bgr = cv2.resize(
        trainer_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    trainer_rgb = cv2.cvtColor(trainer_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(trainer_rgb.transpose(2, 0, 1).copy()).float().div_(255.0)


def preprocess_nyu_sample(
    image_data: np.ndarray,
    depth_data: np.ndarray,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    image_rgb = np.asarray(image_data).transpose(2, 1, 0).astype(np.uint8)
    depth_m = np.asarray(depth_data).transpose(1, 0).astype(np.float32)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    image_bgr = cv2.resize(
        image_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    depth_m = cv2.resize(
        depth_m,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_NEAREST,
    )
    target_raw = np.clip(depth_m / 10.0, 0.0, 1.0)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(image_rgb.transpose(2, 0, 1).copy())
        .float()
        .div_(255.0)
    )
    return tensor, image_rgb, target_raw.astype(np.float32)


def image_tensor_from_path(image_path: str) -> tuple[np.ndarray, torch.Tensor]:
    frame_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return frame_bgr, preprocess_camera_runner(frame_bgr)


def capture_camera(camera_index: int) -> np.ndarray:
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}.")
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not capture a frame from camera {camera_index}.")
        return frame
    finally:
        camera.release()


def load_local_model(weights_path: str, device: torch.device) -> YOLO26nDepth:
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    load_depth_checkpoint(model, weights_path, device)
    model.eval()
    return model


def predict_local_raw(
    model: YOLO26nDepth,
    image_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        output = model(image_tensor.unsqueeze(0).to(device))
    raw_depth = output["depth"][0, 0].detach().cpu().numpy()
    return np.asarray(raw_depth, dtype=np.float32)


def stats(values: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(array)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    selected = array[valid]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
    }


def sample_abs_rel(prediction_raw: np.ndarray, target_raw: np.ndarray) -> float:
    valid = (
        np.isfinite(prediction_raw)
        & np.isfinite(target_raw)
        & (target_raw > 0.0)
    )
    if not np.any(valid):
        raise ValueError("The selected NYU validation sample has no valid depth pixels.")
    return float(
        np.mean(
            np.abs(prediction_raw[valid] - target_raw[valid])
            / np.maximum(target_raw[valid], 1e-6)
        )
    )


def save_rgb_and_depth(
    output_dir: Path,
    name: str,
    image_rgb: np.ndarray,
    prediction_raw: np.ndarray,
) -> str:
    color_input = (np.clip(prediction_raw, 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(color_input, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(
        depth_color,
        (image_rgb.shape[1], image_rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    comparison = np.hstack(
        (cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), depth_color)
    )
    output_path = output_dir / f"{name}_rgb_depth.png"
    if not cv2.imwrite(str(output_path), comparison):
        raise OSError(f"Could not write diagnostic image: {output_path}")
    return output_path.name


def select_val_indices(total_samples: int, max_samples: int, count: int) -> np.ndarray:
    sample_count = total_samples if max_samples <= 0 else min(max_samples, total_samples)
    indices = np.arange(sample_count)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    split = max(1, int(sample_count * 0.9))
    val_indices = indices[split:]
    if val_indices.size < count:
        raise ValueError(
            f"Only {val_indices.size} validation samples available; "
            f"need at least {count}. Increase --max-samples or use the full NYU set."
        )
    positions = np.linspace(0, val_indices.size - 1, count, dtype=int)
    return val_indices[positions]


def checkpoint_collapse_report(
    weights_path: str,
    nyu_file: h5py.File,
    val_indices: np.ndarray,
    camera_tensor: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model = load_local_model(weights_path, device)
    records: list[dict[str, Any]] = []
    per_input_means: list[float] = []
    per_input_spatial_stds: list[float] = []

    for index in val_indices:
        image_tensor, _, _ = preprocess_nyu_sample(
            nyu_file["images"][int(index)],
            nyu_file["depths"][int(index)],
        )
        raw = predict_local_raw(model, image_tensor, device)
        per_input_means.append(float(np.mean(raw)))
        per_input_spatial_stds.append(float(np.std(raw)))
        records.append(
            {
                "input": f"nyu_val_{int(index)}",
                "mean_raw": float(np.mean(raw)),
                "spatial_std_raw": float(np.std(raw)),
            }
        )

    camera_raw = predict_local_raw(model, camera_tensor, device)
    per_input_means.append(float(np.mean(camera_raw)))
    per_input_spatial_stds.append(float(np.std(camera_raw)))
    records.append(
        {
            "input": "camera_or_image",
            "mean_raw": float(np.mean(camera_raw)),
            "spatial_std_raw": float(np.std(camera_raw)),
        }
    )
    del model

    return {
        "checkpoint": weights_path,
        "input_count": len(records),
        "spatial_std_raw_per_input": per_input_spatial_stds,
        "mean_spatial_std_raw": float(np.mean(per_input_spatial_stds)),
        "min_spatial_std_raw": float(np.min(per_input_spatial_stds)),
        "max_spatial_std_raw": float(np.max(per_input_spatial_stds)),
        "std_across_input_mean_raw": float(np.std(per_input_means)),
        "records": records,
    }


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic output: {output_dir}"
        )
    for path in (args.nyu_path, args.weights, args.nyu_checkpoint, args.trained_checkpoint):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    if args.ultralytics_weights and not Path(args.ultralytics_weights).is_file():
        raise FileNotFoundError(
            f"Ultralytics checkpoint not found: {args.ultralytics_weights}"
        )

    frame_bgr = (
        cv2.imread(args.image, cv2.IMREAD_COLOR)
        if args.image
        else capture_camera(args.camera_index)
    )
    if frame_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    primary = load_local_model(args.weights, device)

    runner_tensor = preprocess_camera_runner(frame_bgr)
    trainer_tensor = preprocess_nyu_trainer_camera(frame_bgr)
    runner_raw = predict_local_raw(primary, runner_tensor, device)
    trainer_raw = predict_local_raw(primary, trainer_tensor, device)
    camera_rgb = cv2.cvtColor(
        cv2.resize(
            frame_bgr,
            (INPUT_WIDTH, INPUT_HEIGHT),
            interpolation=cv2.INTER_AREA,
        ),
        cv2.COLOR_BGR2RGB,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    runner_image_name = save_rgb_and_depth(
        output_dir, "camera_runner", camera_rgb, runner_raw
    )
    trainer_image_name = save_rgb_and_depth(
        output_dir, "camera_trainer", camera_rgb, trainer_raw
    )
    np.save(output_dir / "camera_runner_raw.npy", runner_raw, allow_pickle=False)
    np.save(output_dir / "camera_trainer_raw.npy", trainer_raw, allow_pickle=False)

    report: dict[str, Any] = {
        "device": str(device),
        "primary_checkpoint": args.weights,
        "camera_or_image": args.image or f"camera_index_{args.camera_index}",
        "source_frame_shape_bgr": list(frame_bgr.shape),
        "model_input_shape_chw": list(runner_tensor.shape),
        "camera_preprocessing": {
            "runner_raw": stats(runner_raw),
            "trainer_raw": stats(trainer_raw),
            "tensor_max_abs_difference": float(
                (runner_tensor - trainer_tensor).abs().max()
            ),
            "tensor_mean_abs_difference": float(
                (runner_tensor - trainer_tensor).abs().mean()
            ),
            "prediction_max_abs_difference": float(
                np.max(np.abs(runner_raw - trainer_raw))
            ),
            "prediction_mean_abs_difference": float(
                np.mean(np.abs(runner_raw - trainer_raw))
            ),
            "runner_comparison_image": runner_image_name,
            "trainer_comparison_image": trainer_image_name,
        },
    }

    with h5py.File(args.nyu_path, "r") as nyu_file:
        val_indices = select_val_indices(
            int(nyu_file["images"].shape[0]),
            args.max_samples,
            args.collapse_samples,
        )
        sample_index = int(val_indices[0])
        nyu_tensor, nyu_rgb, target_raw = preprocess_nyu_sample(
            nyu_file["images"][sample_index],
            nyu_file["depths"][sample_index],
        )
        nyu_prediction = predict_local_raw(primary, nyu_tensor, device)
        valid_gt = np.isfinite(target_raw) & (target_raw > 0.0)
        nyu_image_name = save_rgb_and_depth(
            output_dir, f"nyu_val_{sample_index}", nyu_rgb, nyu_prediction
        )
        report["nyu_validation_sample"] = {
            "source_index": sample_index,
            "prediction_raw_0_1": stats(nyu_prediction),
            "ground_truth_depth_div_10": stats(target_raw, valid_gt),
            "abs_rel": sample_abs_rel(nyu_prediction, target_raw),
            "comparison_image": nyu_image_name,
        }

        checkpoint_paths = {
            "nyu_full": args.weights,
            "nyu": args.nyu_checkpoint,
            "trained": args.trained_checkpoint,
        }
        collapse_results: dict[str, Any] = {}
        for checkpoint_name, checkpoint_path in checkpoint_paths.items():
            try:
                collapse_results[checkpoint_name] = checkpoint_collapse_report(
                    checkpoint_path,
                    nyu_file,
                    val_indices,
                    runner_tensor,
                    device,
                )
            except (RuntimeError, ValueError, KeyError, OSError) as error:
                collapse_results[checkpoint_name] = {
                    "checkpoint": checkpoint_path,
                    "error": f"{type(error).__name__}: {error}",
                }
        report["local_checkpoint_collapse_checks"] = collapse_results

    ultralytics_result: dict[str, Any]
    if args.ultralytics_weights:
        try:
            from ultralytics import YOLO

            depth_model = YOLO(args.ultralytics_weights)
            result = depth_model(frame_bgr, verbose=False)[0]
            depth_result = getattr(result, "depth", None)
            if depth_result is None:
                raise RuntimeError(
                    "Ultralytics result has no .depth output for this checkpoint/task."
                )
            depth_map = depth_result.data.detach().float().cpu().numpy()
            if depth_map.ndim == 3:
                depth_map = depth_map[0]
            depth_map = np.asarray(depth_map, dtype=np.float32)
            ultralytics_result = {
                "checkpoint": args.ultralytics_weights,
                "raw_depth": stats(depth_map),
                "status": "inference_succeeded",
            }
        except Exception as error:
            ultralytics_result = {
                "checkpoint": args.ultralytics_weights,
                "status": "inference_failed",
                "error": f"{type(error).__name__}: {error}",
            }
    else:
        ultralytics_result = {"status": "skipped"}
    report["ultralytics_checkpoint_check"] = ultralytics_result

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    csv_path = output_dir / "collapse_per_input.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["checkpoint", "input", "mean_raw", "spatial_std_raw"])
        for checkpoint_name, result in report[
            "local_checkpoint_collapse_checks"
        ].items():
            if "records" not in result:
                continue
            for record in result["records"]:
                writer.writerow(
                    [
                        checkpoint_name,
                        record["input"],
                        record["mean_raw"],
                        record["spatial_std_raw"],
                    ]
                )

    print(json.dumps(report, indent=2))
    print(f"\nSaved report and comparison images to: {output_dir}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare camera/NYU preprocessing and test depth-output collapse."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Saved camera frame or other BGR image.")
    input_group.add_argument(
        "--camera-index",
        type=int,
        help="Capture one frame from a camera index.",
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument(
        "--weights", default=DEFAULT_CHECKPOINTS["nyu_full"]
    )
    parser.add_argument(
        "--nyu-checkpoint", default=DEFAULT_CHECKPOINTS["nyu"]
    )
    parser.add_argument(
        "--trained-checkpoint", default=DEFAULT_CHECKPOINTS["trained"]
    )
    parser.add_argument(
        "--ultralytics-weights",
        default=DEFAULT_ULTRALYTICS_WEIGHTS,
        help="Use an empty string to skip the optional Ultralytics checkpoint.",
    )
    parser.add_argument("--output-dir", default="diagnostik/depth_input")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Use 0 for the full NYU dataset, matching the default training split.",
    )
    parser.add_argument(
        "--collapse-samples",
        type=int,
        default=20,
        help="Number of distinct validation images; minimum is 20.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.collapse_samples < 20:
        parser.error("--collapse-samples must be at least 20.")
    if args.max_samples == 1:
        parser.error("--max-samples must be 0 or at least 2.")
    run_diagnostic(args)


if __name__ == "__main__":
    main()
