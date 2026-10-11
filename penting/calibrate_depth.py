"""Fit a camera-specific depth correction from tape-measure validation CSV."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass

import numpy as np


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


def main() -> None:
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


if __name__ == "__main__":
    main()
