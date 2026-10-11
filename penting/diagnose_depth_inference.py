"""Compare NYU and camera preprocessing, checkpoint outputs, and depth collapse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch

if __package__:
    from .arsitektur_yolo_depth import YOLO26nDepth, load_depth_checkpoint
else:
    from arsitektur_yolo_depth import YOLO26nDepth, load_depth_checkpoint


INPUT_WIDTH = 320
INPUT_HEIGHT = 240


def preprocess_camera_runner(frame_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    resized_bgr = cv2.resize(
        frame_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().div_(255.0)
    return tensor, rgb


def preprocess_camera_trainer(frame_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    camera_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    trainer_bgr = cv2.cvtColor(camera_rgb, cv2.COLOR_RGB2BGR)
    resized_bgr = cv2.resize(
        trainer_bgr,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    trainer_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = (
        torch.from_numpy(trainer_rgb.transpose(2, 0, 1).copy())
        .float()
        .div_(255.0)
    )
    return tensor, trainer_rgb


def preprocess_nyu_sample(
    image_rgb_hdf5: np.ndarray,
    depth_hdf5: np.ndarray,
    max_depth_m: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    image_rgb = np.asarray(image_rgb_hdf5).transpose(2, 1, 0).astype(np.uint8)
    depth_m = np.asarray(depth_hdf5).transpose(1, 0).astype(np.float32)

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
    ground_truth_m = depth_m.copy()
    model_target = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = (
        torch.from_numpy(image_rgb.transpose(2, 0, 1).copy())
        .float()
        .div_(255.0)
    )
    return (
        image_tensor,
        image_rgb,
        model_target.astype(np.float32),
        ground_truth_m,
    )


def prediction_raw(model: YOLO26nDepth, image: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.inference_mode():
        output = model(image.unsqueeze(0).to(device))
    return output["depth"][0, 0].detach().cpu().numpy().astype(np.float32)


def depth_stats(depth: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float | int]:
    values = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    selected = values[valid]
    if selected.size == 0:
        return {"count": 0}
    return {
        "count": int(selected.size),
        "min": float(np.min(selected)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
        "spatial_std": float(np.std(selected)),
    }


def _raw_depth_color(raw_depth: np.ndarray) -> np.ndarray:
    gray = (np.clip(raw_depth, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def _metric_depth_color(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    gray = (np.clip(depth_m / max_depth_m, 0.0, 1.0) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def save_rgb_depth_outputs(
    output_dir: Path,
    name: str,
    rgb: np.ndarray,
    raw_prediction: np.ndarray,
) -> None:
    rgb_path = output_dir / f"{name}_rgb.png"
    raw_path = output_dir / f"{name}_prediction_raw.npy"
    depth_path = output_dir / f"{name}_prediction_raw_inferno.png"
    comparison_path = output_dir / f"{name}_rgb_and_prediction.png"
    depth_color = _raw_depth_color(raw_prediction)
    comparison = np.hstack((rgb, cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]))))

    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Failed to write {rgb_path}")
    if not cv2.imwrite(str(depth_path), depth_color):
        raise OSError(f"Failed to write {depth_path}")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise OSError(f"Failed to write {comparison_path}")
    np.save(raw_path, raw_prediction, allow_pickle=False)


def get_frame(args: argparse.Namespace) -> np.ndarray:
    if args.camera_frame:
        frame = cv2.imread(args.camera_frame, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read camera frame: {args.camera_frame}")
        return frame

    camera = cv2.VideoCapture(args.capture_camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(args.capture_camera_index)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {args.capture_camera_index}."
            )
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("Could not capture a frame from the camera.")
        return frame
    finally:
        camera.release()


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic folder: {output_dir}"
        )
    if not Path(args.nyu_path).is_file():
        raise FileNotFoundError(f"NYU MAT file not found: {args.nyu_path}")
    if not Path(args.weights).is_file():
        raise FileNotFoundError(f"Depth checkpoint not found: {args.weights}")

    frame_bgr = get_frame(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    load_depth_checkpoint(model, args.weights, device)
    model.eval()

    summary: dict[str, Any] = {
        "checkpoint": str(Path(args.weights).resolve()),
        "device": str(device),
        "input_size": [INPUT_WIDTH, INPUT_HEIGHT],
        "max_depth_m": args.max_depth_m,
        "camera_frame_shape_bgr": list(frame_bgr.shape),
    }

    with h5py.File(args.nyu_path, "r") as dataset_file:
        total_samples = int(dataset_file["images"].shape[0])
        sample_count = (
            total_samples
            if args.max_samples <= 0
            else min(total_samples, args.max_samples)
        )
        rng = np.random.default_rng(args.split_seed)
        indices = np.arange(sample_count)
        rng.shuffle(indices)
        split = max(1, int(sample_count * 0.9))
        val_indices = indices[split:]
        if val_indices.size == 0:
            raise ValueError("NYU validation split is empty.")

        sample_source_index = int(val_indices[0])
        image_tensor, image_rgb, target_normalized, target_m = preprocess_nyu_sample(
            dataset_file["images"][sample_source_index],
            dataset_file["depths"][sample_source_index],
            args.max_depth_m,
        )
        sample_prediction = prediction_raw(model, image_tensor, device)
        target_valid = (
            np.isfinite(target_m)
            & (target_m > 0.0)
        )
        sample_name = f"nyu_val_source_{sample_source_index}"
        save_rgb_depth_outputs(
            output_dir, sample_name, image_rgb, sample_prediction
        )
        target_color = _metric_depth_color(target_m, args.max_depth_m)
        target_file = output_dir / f"{sample_name}_ground_truth_m_inferno.png"
        target_npy = output_dir / f"{sample_name}_ground_truth_m.npy"
        if not cv2.imwrite(str(target_file), target_color):
            raise OSError(f"Failed to write {target_file}")
        np.save(target_npy, target_m, allow_pickle=False)

        selected_positions = np.linspace(
            0,
            val_indices.size - 1,
            num=min(args.collapse_samples, val_indices.size),
            dtype=int,
        )
        collapse_records: list[dict[str, Any]] = []
        input_means: list[float] = []
        input_spatial_stds: list[float] = []
        for ordinal, position in enumerate(selected_positions):
            source_index = int(val_indices[position])
            collapse_tensor, collapse_rgb, _, _ = preprocess_nyu_sample(
                dataset_file["images"][source_index],
                dataset_file["depths"][source_index],
                args.max_depth_m,
            )
            raw = prediction_raw(model, collapse_tensor, device)
            stats = depth_stats(raw)
            collapse_records.append(
                {"source_index": source_index, "raw_prediction": stats}
            )
            input_means.append(float(stats["mean"]))
            input_spatial_stds.append(float(stats["spatial_std"]))
            if args.save_collapse_maps:
                save_rgb_depth_outputs(
                    output_dir,
                    f"collapse_nyu_{ordinal:03d}_source_{source_index}",
                    collapse_rgb,
                    raw,
                )

    summary["nyu_validation_sample"] = {
        "source_index": sample_source_index,
        "raw_prediction": depth_stats(sample_prediction),
        "ground_truth_m": depth_stats(target_m, target_valid),
        "ground_truth_training_target_0_1": depth_stats(
            target_normalized, target_valid
        ),
        "rgb_file": f"{sample_name}_rgb.png",
        "prediction_raw_npy": f"{sample_name}_prediction_raw.npy",
        "prediction_color_file": f"{sample_name}_prediction_raw_inferno.png",
        "ground_truth_color_file": target_file.name,
    }
    summary["nyu_collapse_check"] = {
        "selected_val_samples": len(collapse_records),
        "records": collapse_records,
        "std_of_per_input_raw_means": float(np.std(input_means)),
        "mean_per_input_spatial_std": float(np.mean(input_spatial_stds)),
    }

    runner_tensor, runner_rgb = preprocess_camera_runner(frame_bgr)
    trainer_tensor, trainer_rgb = preprocess_camera_trainer(frame_bgr)
    runner_raw = prediction_raw(model, runner_tensor, device)
    trainer_raw = prediction_raw(model, trainer_tensor, device)
    save_rgb_depth_outputs(
        output_dir, "camera_runner_preprocessing", runner_rgb, runner_raw
    )
    save_rgb_depth_outputs(
        output_dir, "camera_trainer_preprocessing", trainer_rgb, trainer_raw
    )
    tensor_difference = (runner_tensor - trainer_tensor).abs()
    prediction_difference = (runner_raw - trainer_raw).astype(np.float64)
    summary["camera_preprocessing_comparison"] = {
        "runner_tensor_shape": list(runner_tensor.shape),
        "trainer_tensor_shape": list(trainer_tensor.shape),
        "runner_tensor_dtype": str(runner_tensor.dtype),
        "trainer_tensor_dtype": str(trainer_tensor.dtype),
        "max_absolute_tensor_difference": float(tensor_difference.max()),
        "mean_absolute_tensor_difference": float(tensor_difference.mean()),
        "runner_raw_prediction": depth_stats(runner_raw),
        "trainer_raw_prediction": depth_stats(trainer_raw),
        "max_absolute_prediction_difference": float(
            np.max(np.abs(prediction_difference))
        ),
        "mean_absolute_prediction_difference": float(
            np.mean(np.abs(prediction_difference))
        ),
        "runner_rgb_file": "camera_runner_preprocessing_rgb.png",
        "trainer_rgb_file": "camera_trainer_preprocessing_rgb.png",
    }
    combined_means = input_means + [
        float(summary["camera_preprocessing_comparison"]["runner_raw_prediction"]["mean"])
    ]
    combined_spatial_stds = input_spatial_stds + [
        float(summary["camera_preprocessing_comparison"]["runner_raw_prediction"]["spatial_std"])
    ]
    summary["combined_input_collapse_check"] = {
        "input_count": len(combined_means),
        "includes": "selected NYU validation images and the camera frame",
        "std_of_per_input_raw_means": float(np.std(combined_means)),
        "mean_per_input_spatial_std": float(np.mean(combined_spatial_stds)),
    }

    summary_file = output_dir / "diagnostic_summary.json"
    with summary_file.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print(f"Checkpoint: {summary['checkpoint']}")
    print(f"Device: {device}; NYU total={total_samples}; val={len(val_indices)}")
    print("\nOne NYU validation sample:")
    print(json.dumps(summary["nyu_validation_sample"], indent=2))
    print("\nCollapse check across NYU validation samples:")
    print(json.dumps(summary["nyu_collapse_check"], indent=2))
    print("\nSame camera frame, two preprocessing paths:")
    print(json.dumps(summary["camera_preprocessing_comparison"], indent=2))
    print("\nCombined collapse check (NYU inputs plus camera frame):")
    print(json.dumps(summary["combined_input_collapse_check"], indent=2))
    print(f"\nSaved diagnostic images, raw .npy maps, and summary: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare NYU/trainer and camera preprocessing using one depth checkpoint."
    )
    frame_source = parser.add_mutually_exclusive_group(required=True)
    frame_source.add_argument(
        "--camera-frame",
        help="Path to a saved BGR camera frame (same frame used for both pipelines).",
    )
    frame_source.add_argument(
        "--capture-camera-index",
        type=int,
        help="Capture one fresh frame from this camera index.",
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument(
        "--weights", default="data/yolo26n_depth_nyu_full.pt"
    )
    parser.add_argument(
        "--output-dir", default="diagnostics/depth_preprocessing_run"
    )
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--collapse-samples", type=int, default=16)
    parser.add_argument("--save-collapse-maps", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not np.isfinite(args.max_depth_m) or args.max_depth_m <= 0:
        parser.error("--max-depth-m must be finite and positive.")
    if args.max_samples == 1:
        parser.error("--max-samples must be 0 or at least 2.")
    if args.collapse_samples < 1:
        parser.error("--collapse-samples must be positive.")
    run_diagnostic(args)


if __name__ == "__main__":
    main()
