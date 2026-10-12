"""Evaluate the local metric-depth model on the NYU Depth V2 validation split."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


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
    import torch

    if __package__:
        from .ultralytics_depth import (
            load_calibrated_depth_model,
            predict_calibrated_depth_meters,
        )
    else:
        from ultralytics_depth import (
            load_calibrated_depth_model,
            predict_calibrated_depth_meters,
        )

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

        device = torch.device(
            args.device
            if args.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = load_calibrated_depth_model(args.weights, device)
        metrics = DepthMetricAccumulator(silog_lambda=args.silog_lambda)
        valid_images = 0
        excluded_out_of_range = 0

        with torch.inference_mode():
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
                prediction_m = predict_calibrated_depth_meters(
                    model,
                    image_bgr,
                    device,
                    input_size=args.input_size,
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
            "device": str(device),
            "total_samples": total_samples,
            "samples_used_for_split": sample_count,
            "validation_images": int(validation_indices.size),
            "images_with_valid_depth": valid_images,
            "excluded_gt_pixels_above_max_depth": excluded_out_of_range,
            "max_depth_m": args.max_depth_m,
            "input_size": args.input_size,
            "silog_lambda": args.silog_lambda,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce depth metrics on the deterministic NYU validation split."
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument(
        "--weights", default="data/yolo26n-depth-nyu-calibrated.pt"
    )
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--max-depth-m", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--silog-lambda",
        type=float,
        default=1.0,
        help="SILog uses 100*sqrt(E[g^2]-lambda*E[g]^2); default 1 is scale invariant.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or another torch device string.",
    )
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    for path in (args.nyu_path, args.weights):
        if not os.path.isfile(path):
            parser.error(f"File tidak ditemukan: {path}")
    if args.input_size < 32 or args.input_size % 32:
        parser.error("--input-size harus minimal 32 dan kelipatan 32.")
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
    print(f"Device: {result['device']}")
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


if __name__ == "__main__":
    main()
