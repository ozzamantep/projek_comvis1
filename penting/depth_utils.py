"""Metric-depth helpers for the local YOLO26nDepth output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def raw_depth_to_meters(
    raw_depth: ArrayLike, max_depth_m: float = 10.0
) -> NDArray[np.float32]:
    """Convert the model's normalized raw depth to meters.

    The current NYU training pipeline divides metric depth by 10 and clips it
    to [0, 1]. This conversion is valid only for weights trained with that
    target scaling; do not pass per-frame display-normalized depth here.
    """
    if not np.isfinite(max_depth_m) or max_depth_m <= 0:
        raise ValueError("max_depth_m must be a finite positive number.")
    return np.asarray(raw_depth, dtype=np.float32) * np.float32(max_depth_m)


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
