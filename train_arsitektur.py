"""Train the standalone YOLO26n-Depth blueprint on SUN RGB-D and optional COCO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import ssl
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import ijson
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from arsitektur import ARCHITECTURE_ID, build_model


DEFAULT_CLASSES = ("person", "laptop", "cell phone", "bottle")
DEPTH_ROTATION_MASK = 0xFFFF
DETECTION_LEVELS = ((8, "p3"), (16, "p4"), (32, "p5"))
AP_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * index, 2) for index in range(10))
ASSIGNMENT_RADIUS_CELLS = 2.5
SUN_RGBD_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
COCO_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/"
    "annotations/annotations_trainval2017.zip"
)
COCO_IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2017"


def _class_key(name: str) -> str:
    return "".join(character for character in name.casefold() if character.isalnum())


def _human_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} TiB"


class ActivityIndicator:
    def __init__(self, message: str, interval: float = 0.15) -> None:
        self.message = message
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._animate, daemon=True)

    def _animate(self) -> None:
        frames = "|/-\\"
        frame_index = 0
        while not self._stop.wait(self.interval):
            with self._lock:
                message = self.message
            print(f"\r[{frames[frame_index % len(frames)]}] {message}", end="", flush=True)
            frame_index += 1

    def update(self, message: str) -> None:
        with self._lock:
            self.message = message

    def __enter__(self) -> ActivityIndicator:
        self._thread.start()
        return self

    def __exit__(self, exception_type: object, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        with self._lock:
            message = self.message
        status = "stopped" if exception_type is not None else "done"
        print(f"\r[{status}] {message}", flush=True)


@dataclass(frozen=True)
class Record:
    image_path: Path
    depth_path: Path | None
    boxes: tuple[tuple[float, float, float, float, int], ...]
    group: str
    depth_rotation: bool = False


def _first_image(folder: Path) -> Path | None:
    if not folder.is_dir():
        return None
    return next(
        (
            path
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ),
        None,
    )


def _depth_file(scene: Path) -> Path | None:
    for folder_name in ("depth", "depth_bfx"):
        folder = scene / folder_name
        if folder.is_dir():
            matches = sorted(folder.glob("*.png"))
            if matches:
                return matches[0]
    return None


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    offset = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "projek-comvis1-dataset-downloader/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    started_at = time.monotonic()
    progress = {"downloaded": offset, "total_bytes": None}
    progress_lock = threading.Lock()
    stop_progress = threading.Event()
    spinner_frames = "|/-\\"
    spinner_index = 0
    rendered_width = 0

    def render_progress() -> None:
        nonlocal spinner_index, rendered_width
        with progress_lock:
            downloaded = progress["downloaded"]
            total_bytes = progress["total_bytes"]
            elapsed = max(time.monotonic() - started_at, 0.001)
            rate = (downloaded - offset) / elapsed
            spinner = spinner_frames[spinner_index % len(spinner_frames)]
            spinner_index += 1
            if total_bytes:
                fraction = min(downloaded / total_bytes, 1.0)
                bar_width = 24
                filled = int(bar_width * fraction)
                if rate > 0:
                    eta_seconds = max(0, int((total_bytes - downloaded) / rate))
                    eta = (
                        f"{eta_seconds // 3600:02d}:"
                        f"{eta_seconds % 3600 // 60:02d}:"
                        f"{eta_seconds % 60:02d}"
                    )
                else:
                    eta = "--:--:--"
                message = (
                    f"  {destination.name} [{('#' * filled).ljust(bar_width, '-')}] "
                    f"{fraction * 100:5.1f}% {_human_bytes(downloaded)}/"
                    f"{_human_bytes(total_bytes)} {_human_bytes(int(rate))}/s "
                    f"ETA {eta} {spinner}"
                )
            else:
                message = (
                    f"  {destination.name}: {_human_bytes(downloaded)} downloaded "
                    f"{_human_bytes(int(rate))}/s {spinner}"
                )
            rendered_width = max(rendered_width, len(message))
            print(f"\r{message.ljust(rendered_width)}", end="", flush=True)

    def animate_progress() -> None:
        while not stop_progress.wait(0.15):
            render_progress()

    progress_thread = threading.Thread(target=animate_progress, daemon=True)
    progress_thread.start()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            append = offset > 0 and getattr(response, "status", None) == 206
            if offset and not append:
                offset = 0
            total_header = response.headers.get("Content-Length")
            total_bytes = offset + int(total_header) if total_header else None
            with progress_lock:
                progress["downloaded"] = offset
                progress["total_bytes"] = total_bytes
            mode = "ab" if append else "wb"
            downloaded = offset
            with partial_path.open(mode) as output:
                read_chunk = getattr(response, "read1", response.read)
                while chunk := read_chunk(256 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    with progress_lock:
                        progress["downloaded"] = downloaded
                    render_progress()
    except (OSError, urllib.error.URLError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                f"HTTPS certificate verification failed for {url}: {reason}. "
                "The network, proxy, or antivirus may be presenting a certificate "
                "for another hostname. Do not disable certificate verification. "
                "Try a trusted network. You may also download the official COCO "
                "annotation archive in a browser, extract instances_train2017.json, "
                "and pass its path with --coco-annotations; however, missing COCO "
                "images are downloaded from the same host and require working HTTPS."
            ) from exc
        raise RuntimeError(
            f"Download failed for {url}. Partial data is kept at {partial_path}; "
            "rerun the command to resume when the server supports HTTP Range."
        ) from exc
    finally:
        stop_progress.set()
        progress_thread.join()
        print(flush=True)
    os.replace(partial_path, destination)


def _safe_extract_zip(
    archive_path: Path,
    destination: Path,
    members: set[str] | None = None,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        selected = [
            info
            for info in archive.infolist()
            if members is None or info.filename in members
        ]
        required_bytes = sum(info.file_size for info in selected if not info.is_dir())
        free_bytes = shutil.disk_usage(destination).free
        if required_bytes > free_bytes:
            raise OSError(
                f"Not enough free disk space to extract {archive_path.name}: "
                f"needs {required_bytes / (1024**3):.1f} GiB, "
                f"has {free_bytes / (1024**3):.1f} GiB free."
            )
        extracted = 0
        for info in selected:
            target = (destination / info.filename).resolve()
            if os.path.commonpath((str(destination_resolved), str(target))) != str(
                destination_resolved
            ):
                raise ValueError(
                    f"Unsafe path in dataset archive: {info.filename!r}"
                )
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted += 1
            if extracted % 5000 == 0:
                print(f"  Extracted {extracted:,}/{len(selected):,} archive entries.")


def ensure_sun_dataset(
    root: Path, download_dir: Path, allow_download: bool = True
) -> None:
    root = root.expanduser().resolve()
    if root.is_dir() and any(root.rglob("annotation2Dfinal/index.json")):
        return
    if not allow_download:
        raise FileNotFoundError(
            f"SUN RGB-D was not found under {root}. Remove --no-download to fetch it."
        )
    archive_path = download_dir / "SUNRGBD.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        print(f"Downloading full SUN RGB-D dataset to {archive_path}.")
        _download_file(SUN_RGBD_URL, archive_path)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting SUN RGB-D into {root}; this can take a while.")
    _safe_extract_zip(archive_path, root)
    if not any(root.rglob("annotation2Dfinal/index.json")):
        raise ValueError(
            f"Archive extracted, but no SUN RGB-D annotations were found under {root}."
        )


def ensure_coco_annotations(
    root: Path, download_dir: Path, allow_download: bool = True
) -> Path:
    annotation_path = root / "annotations" / "instances_train2017.json"
    if annotation_path.is_file():
        return annotation_path
    if not allow_download:
        raise FileNotFoundError(
            f"COCO train annotations not found at {annotation_path}. "
            "Remove --no-download to fetch them."
        )
    archive_path = download_dir / "coco_annotations_trainval2017.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        print(f"Downloading COCO 2017 annotations to {archive_path}.")
        _download_file(COCO_ANNOTATIONS_URL, archive_path)
    member = "annotations/instances_train2017.json"
    with zipfile.ZipFile(archive_path) as archive:
        if member not in archive.namelist():
            raise ValueError(f"COCO annotation archive does not contain {member}.")
    print(f"Extracting COCO training annotations into {root / 'annotations'}.")
    _safe_extract_zip(archive_path, root, {member})
    if not annotation_path.is_file():
        raise OSError(f"Failed to extract COCO annotations to {annotation_path}.")
    return annotation_path


def download_coco_images(records: list[Record]) -> None:
    missing = [record for record in records if not record.image_path.is_file()]
    if not missing:
        print(f"All {len(records)} selected COCO images are already available.")
        return
    print(
        f"Downloading {len(missing)} selected COCO images "
        "(only images containing requested classes)."
    )
    for index, record in enumerate(missing, start=1):
        image_path = record.image_path
        url = f"{COCO_IMAGES_URL}/{image_path.name}"
        for attempt in range(3):
            partial_path = image_path.with_name(image_path.name + ".part")
            try:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                _download_file(url, image_path)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    image_path.unlink(missing_ok=True)
                    raise OSError(f"Downloaded file is not a readable image: {image_path}")
                break
            except (OSError, RuntimeError) as exc:
                partial_path.unlink(missing_ok=True)
                if attempt == 2:
                    raise RuntimeError(
                        f"Could not download COCO image {image_path.name}."
                    ) from exc
                time.sleep(attempt + 1)
        if index % 25 == 0 or index == len(missing):
            print(f"  Downloaded {index}/{len(missing)} COCO images.")


def _sun_group(scene: Path, root: Path, group_block: int = 100) -> str:
    relative = scene.relative_to(root).as_posix()
    parent, name = relative.rsplit("/", 1) if "/" in relative else ("", relative)
    match = re.match(r"^(.*?)_rgbf\d+", name, re.IGNORECASE)
    if match:
        return f"{parent}/{match.group(1)}"
    match = re.search(r"(\d+)$", name)
    if match and ("nyudata" in parent.lower() or "b3dodata" in parent.lower()):
        return f"{parent}/block{int(match.group(1)) // group_block}"
    if parent.count("/") >= 3:
        return parent
    return relative


def _annotation_objects(annotation: dict[str, Any]) -> list[str | None]:
    objects: list[str | None] = []
    for item in annotation.get("objects", []):
        value = item.get("name") if isinstance(item, dict) else item
        objects.append(
            value.split(":")[0].strip().lower() if isinstance(value, str) else None
        )
    return objects


def _annotation_polygons(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    frames = annotation.get("frames", [])
    if isinstance(frames, dict):
        frames = [frames]
    if not frames or not isinstance(frames[0], dict):
        return []
    polygons = frames[0].get("polygon", [])
    if isinstance(polygons, dict):
        return [polygons]
    return [item for item in polygons if isinstance(item, dict)]


def _detect_sun_object_base(annotation_paths: list[Path]) -> int:
    over_count = zero_count = 0
    for path in annotation_paths[:200]:
        try:
            with path.open(encoding="utf-8") as stream:
                annotation = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        object_count = len(_annotation_objects(annotation))
        for polygon in _annotation_polygons(annotation):
            object_id = polygon.get("object")
            if isinstance(object_id, int):
                over_count += int(object_id >= object_count)
                zero_count += int(object_id == 0)
    return 1 if over_count > 0 and zero_count == 0 else 0


def _parse_sun_boxes(
    annotation_path: Path,
    image_size: tuple[int, int],
    class_to_id: dict[str, int],
    object_base: int,
    min_box_px: float,
) -> tuple[tuple[float, float, float, float, int], ...]:
    try:
        with annotation_path.open(encoding="utf-8") as stream:
            annotation = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read SUN RGB-D annotation {annotation_path}") from exc

    image_width, image_height = image_size
    names = _annotation_objects(annotation)
    boxes: list[tuple[float, float, float, float, int]] = []
    for polygon in _annotation_polygons(annotation):
        object_id = polygon.get("object")
        if not isinstance(object_id, int):
            continue
        object_index = object_id - object_base
        if object_index < 0 or object_index >= len(names):
            continue
        class_id = class_to_id.get(names[object_index] or "")
        if class_id is None:
            continue
        try:
            xs = np.asarray(polygon.get("x", []), dtype=np.float32).reshape(-1)
            ys = np.asarray(polygon.get("y", []), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if len(xs) < 2 or len(xs) != len(ys):
            continue
        x1 = max(float(xs.min()), 0.0)
        y1 = max(float(ys.min()), 0.0)
        x2 = min(float(xs.max()), image_width - 1.0)
        y2 = min(float(ys.max()), image_height - 1.0)
        if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
            continue
        boxes.append((x1, y1, x2, y2, class_id))
    return tuple(boxes)


def _read_depth_m(path: Path, rotate: bool, max_depth_m: float) -> np.ndarray:
    depth_raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise OSError(f"Could not read depth map: {path}")
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]
    depth_u32 = depth_raw.astype(np.uint32)
    if rotate:
        depth_u32 = ((depth_u32 >> 3) | (depth_u32 << 13)) & DEPTH_ROTATION_MASK
    depth_m = depth_u32.astype(np.float32) / 1000.0
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0) | (depth_m > max_depth_m)] = 0
    return depth_m


def _detect_depth_rotation(scenes: list[Path]) -> bool:
    votes = used = 0
    for scene in scenes[:64]:
        path = _depth_file(scene)
        if path is None:
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise OSError(f"Could not read SUN RGB-D depth map: {path}")
        raw = raw.astype(np.uint16)
        rotated = ((raw.astype(np.uint32) >> 3) | (raw.astype(np.uint32) << 13))
        rotated = (rotated & DEPTH_ROTATION_MASK).astype(np.uint16)
        if np.count_nonzero(raw) < 100:
            continue
        rotated_median = float(np.median(rotated[rotated > 0])) / 1000.0
        raw_median = float(np.median(raw[raw > 0])) / 1000.0
        votes += int(0.3 < rotated_median < 8 and not 0.3 < raw_median < 8)
        used += 1
    return used > 0 and votes >= used / 2


def discover_sun_records(
    root: Path,
    classes: tuple[str, ...],
    max_depth_m: float,
    depth_rotation: str,
    min_box_px: float = 4.0,
) -> list[Record]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"SUN RGB-D root directory not found: {root}")
    annotations = sorted(root.rglob("annotation2Dfinal/index.json"))
    scenes = [path.parent.parent for path in annotations]
    if not scenes:
        raise FileNotFoundError(
            f"No annotation2Dfinal/index.json files found under {root}"
        )

    class_to_id = {_class_key(name): index for index, name in enumerate(classes)}
    if depth_rotation == "auto":
        rotate = _detect_depth_rotation(scenes)
    else:
        rotate = depth_rotation == "on"
    object_base = _detect_sun_object_base(annotations)
    print(f"SUN RGB-D depth rotation: {'on' if rotate else 'off'}; object index base: {object_base}")

    records: list[Record] = []
    for scene, annotation_path in zip(scenes, annotations):
        image_path = _first_image(scene / "image")
        depth_path = _depth_file(scene)
        if image_path is None or depth_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Could not read SUN RGB-D image: {image_path}")
        height, width = image.shape[:2]
        try:
            boxes = _parse_sun_boxes(
                annotation_path,
                (width, height),
                class_to_id,
                object_base,
                min_box_px,
            )
        except ValueError as exc:
            print(f"Skipping SUN RGB-D scene with invalid annotation: {exc}")
            continue
        records.append(
            Record(
                image_path=image_path,
                depth_path=depth_path,
                boxes=boxes,
                group=_sun_group(scene, root),
                depth_rotation=rotate,
            )
        )
    if not records:
        raise ValueError(f"No usable SUN RGB-D image/depth/annotation triples under {root}")
    print(f"Found {len(records)} SUN RGB-D records; depth rotation {'enabled' if rotate else 'disabled'}.")
    return records


def _split_records_by_group(
    records: list[Record],
    val_fraction: float,
    test_fraction: float,
    seed: int,
    dataset_name: str,
) -> tuple[list[Record], list[Record], list[Record]]:
    if (
        val_fraction <= 0
        or test_fraction <= 0
        or val_fraction + test_fraction >= 1
    ):
        raise ValueError(
            "Validation and test fractions must be positive and sum to less than 1."
        )
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 3:
        raise ValueError(
            f"Need at least three {dataset_name} groups for train/val/test splits."
        )
    target_val = max(1, round(len(records) * val_fraction))
    target_test = max(1, round(len(records) * test_fraction))
    val_groups: list[str] = []
    val_count = 0
    for key in keys:
        if val_count >= target_val or len(val_groups) == len(keys) - 2:
            break
        val_groups.append(key)
        val_count += len(groups[key])
    val_keys = set(val_groups)
    remaining_keys = [key for key in keys if key not in val_keys]
    test_groups: list[str] = []
    test_count = 0
    for key in remaining_keys:
        if test_count >= target_test or len(test_groups) == len(remaining_keys) - 1:
            break
        test_groups.append(key)
        test_count += len(groups[key])
    test_keys = set(test_groups)
    train = [
        record
        for record in records
        if record.group not in val_keys and record.group not in test_keys
    ]
    val = [record for record in records if record.group in val_keys]
    test = [record for record in records if record.group in test_keys]
    if not train or not val or not test:
        raise ValueError(
            f"{dataset_name} train/val/test split is empty; check group identifiers."
        )
    return train, val, test


def split_sun_records(
    records: list[Record],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Record], list[Record], list[Record]]:
    return _split_records_by_group(
        records, val_fraction, test_fraction, seed, "SUN RGB-D"
    )


def split_coco_records(
    records: list[Record],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[Record], list[Record], list[Record]]:
    return _split_records_by_group(records, val_fraction, test_fraction, seed, "COCO")


def discover_coco_records(
    root: Path,
    annotation_path: Path | None,
    classes: tuple[str, ...],
    min_box_px: float = 4.0,
    max_records: int | None = None,
    seed: int = 42,
    allow_missing_images: bool = False,
    local_only: bool = False,
) -> list[Record]:
    root = root.expanduser().resolve()
    annotations_path = (
        annotation_path.expanduser().resolve()
        if annotation_path is not None
        else root / "annotations" / "instances_train2017.json"
    )
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO instance annotations not found: {annotations_path}")
    images_dir = root / "train2017"
    if not images_dir.is_dir() and root.is_dir() and any(root.glob("*.jpg")):
        images_dir = root

    class_to_id = {_class_key(name): index for index, name in enumerate(classes)}
    try:
        with annotations_path.open("rb") as stream:
            category_to_class = {
                int(category["id"]): class_to_id[_class_key(category["name"])]
                for category in ijson.items(stream, "categories.item")
                if _class_key(category.get("name", "")) in class_to_id
            }
    except (OSError, ijson.JSONError) as exc:
        raise ValueError(f"Could not read COCO annotations {annotations_path}") from exc
    if not category_to_class:
        with annotations_path.open("rb") as stream:
            available = sorted(
                category.get("name", "")
                for category in ijson.items(stream, "categories.item")
            )
        raise ValueError(
            f"None of --classes match COCO categories. Requested {classes}; "
            f"available examples: {available[:20]}"
        )

    try:
        image_info: dict[int, dict[str, Any]] = {}
        with annotations_path.open("rb") as stream:
            for image in ijson.items(stream, "images.item"):
                image_id = int(image["id"])
                file_name = image.get("file_name") or f"{image_id:012d}.jpg"
                image_path = images_dir / file_name
                if local_only and not image_path.is_file():
                    continue
                image_info[image_id] = {
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                    "file_name": file_name,
                }
    except (OSError, ijson.JSONError) as exc:
        raise ValueError(f"Could not read COCO annotations {annotations_path}") from exc

    by_image: dict[int, list[tuple[float, float, float, float, int]]] = {}
    try:
        with annotations_path.open("rb") as stream:
            for annotation in ijson.items(stream, "annotations.item"):
                if annotation.get("iscrowd", 0):
                    continue
                class_id = category_to_class.get(int(annotation["category_id"]))
                image_id = int(annotation["image_id"])
                image = image_info.get(image_id)
                if class_id is None or image is None:
                    continue
                x, y, box_width, box_height = (
                    float(value) for value in annotation["bbox"]
                )
                x1 = max(x, 0.0)
                y1 = max(y, 0.0)
                x2 = min(x + box_width, float(image["width"]) - 1.0)
                y2 = min(y + box_height, float(image["height"]) - 1.0)
                if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
                    continue
                by_image.setdefault(image_id, []).append(
                    (x1, y1, x2, y2, class_id)
                )
    except (OSError, ijson.JSONError) as exc:
        raise ValueError(f"Could not read COCO annotations {annotations_path}") from exc

    candidates: list[Record] = []
    for image_id, boxes in by_image.items():
        image = image_info[image_id]
        file_name = image["file_name"]
        image_path = images_dir / file_name
        candidates.append(
            Record(
                image_path=image_path,
                depth_path=None,
                boxes=tuple(boxes),
                group=f"coco/{image_id}",
            )
        )
    if local_only:
        candidates = [record for record in candidates if record.image_path.is_file()]
        if not candidates:
            raise FileNotFoundError(
                "No local COCO images with the requested classes were found under "
                f"{images_dir}."
            )
    if max_records is not None and max_records < len(candidates):
        if max_records <= 0:
            candidates = []
        else:
            by_class: dict[int, list[Record]] = {}
            for record in candidates:
                for class_id in {box[4] for box in record.boxes}:
                    by_class.setdefault(class_id, []).append(record)
            rng = random.Random(seed)
            for class_records in by_class.values():
                rng.shuffle(class_records)
            class_ids = sorted(by_class)
            positions = {class_id: 0 for class_id in class_ids}
            chosen: list[Record] = []
            seen: set[Path] = set()
            while len(chosen) < max_records:
                progressed = False
                for class_id in class_ids:
                    class_records = by_class[class_id]
                    position = positions[class_id]
                    while position < len(class_records):
                        record = class_records[position]
                        position += 1
                        if record.image_path not in seen:
                            chosen.append(record)
                            seen.add(record.image_path)
                            break
                    positions[class_id] = position
                    if len(chosen) >= max_records:
                        break
                    progressed = progressed or position < len(class_records)
                if not progressed and len(chosen) < max_records:
                    remaining = [
                        record
                        for record in candidates
                        if record.image_path not in seen
                    ]
                    rng.shuffle(remaining)
                    chosen.extend(remaining[: max_records - len(chosen)])
                    break
            candidates = chosen
    if not allow_missing_images:
        missing = next(
            (record.image_path for record in candidates if not record.image_path.is_file()),
            None,
        )
        if missing is not None:
            raise FileNotFoundError(
                f"COCO image referenced by annotations is missing: {missing}"
            )
    records = candidates
    if not records:
        raise ValueError(f"No COCO images contain the requested classes in {annotations_path}")
    print(f"Selected {len(records)} COCO images for classes {classes}.")
    return records


def _augment_geometry(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    boxes_array: np.ndarray,
    rng: random.Random,
    scale_jitter: float,
    translate_jitter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply random zoom (crop or pad) plus translation to image, depth and boxes."""
    source_height, source_width = image_rgb.shape[:2]
    zoom = 1.0
    if scale_jitter > 0:
        zoom = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    zoom = max(0.25, zoom)
    shift_x = shift_y = 0.0
    if translate_jitter > 0:
        shift_x = rng.uniform(-translate_jitter, translate_jitter) * source_width
        shift_y = rng.uniform(-translate_jitter, translate_jitter) * source_height

    if zoom >= 1.0:
        crop_width = max(1, min(source_width, int(round(source_width / zoom))))
        crop_height = max(1, min(source_height, int(round(source_height / zoom))))
        left = rng.randint(0, source_width - crop_width)
        top = rng.randint(0, source_height - crop_height)
        image_rgb = cv2.resize(
            image_rgb[top : top + crop_height, left : left + crop_width],
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_m = cv2.resize(
            depth_m[top : top + crop_height, left : left + crop_width],
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        )
        if len(boxes_array):
            boxes_array[:, (0, 2)] = (
                boxes_array[:, (0, 2)] - left
            ) * (source_width / crop_width)
            boxes_array[:, (1, 3)] = (
                boxes_array[:, (1, 3)] - top
            ) * (source_height / crop_height)
    else:
        new_width = max(source_width, int(round(source_width / zoom)))
        new_height = max(source_height, int(round(source_height / zoom)))
        padded_image = np.full((new_height, new_width, 3), 114, dtype=np.uint8)
        padded_depth = np.zeros((new_height, new_width), dtype=np.float32)
        offset_x = (new_width - source_width) // 2
        offset_y = (new_height - source_height) // 2
        padded_image[
            offset_y : offset_y + source_height, offset_x : offset_x + source_width
        ] = image_rgb
        padded_depth[
            offset_y : offset_y + source_height, offset_x : offset_x + source_width
        ] = depth_m
        image_rgb, depth_m = padded_image, padded_depth
        if len(boxes_array):
            boxes_array[:, (0, 2)] = boxes_array[:, (0, 2)] + offset_x
            boxes_array[:, (1, 3)] = boxes_array[:, (1, 3)] + offset_y

    delta_x = int(round(shift_x))
    delta_y = int(round(shift_y))
    if delta_x or delta_y:
        canvas_height, canvas_width = image_rgb.shape[:2]
        shifted_image = np.full((canvas_height, canvas_width, 3), 114, dtype=np.uint8)
        shifted_depth = np.zeros((canvas_height, canvas_width), dtype=np.float32)
        source_x = max(0, -delta_x)
        source_y = max(0, -delta_y)
        target_x = max(0, delta_x)
        target_y = max(0, delta_y)
        copy_width = min(canvas_width - source_x, canvas_width - target_x)
        copy_height = min(canvas_height - source_y, canvas_height - target_y)
        if copy_width > 0 and copy_height > 0:
            shifted_image[
                target_y : target_y + copy_height, target_x : target_x + copy_width
            ] = image_rgb[source_y : source_y + copy_height, source_x : source_x + copy_width]
            shifted_depth[
                target_y : target_y + copy_height, target_x : target_x + copy_width
            ] = depth_m[source_y : source_y + copy_height, source_x : source_x + copy_width]
        image_rgb, depth_m = shifted_image, shifted_depth
        if len(boxes_array):
            boxes_array[:, (0, 2)] = boxes_array[:, (0, 2)] + delta_x
            boxes_array[:, (1, 3)] = boxes_array[:, (1, 3)] + delta_y

    if len(boxes_array):
        canvas_height, canvas_width = image_rgb.shape[:2]
        boxes_array[:, (0, 2)] = boxes_array[:, (0, 2)].clip(0, canvas_width)
        boxes_array[:, (1, 3)] = boxes_array[:, (1, 3)].clip(0, canvas_height)
        valid = (boxes_array[:, 2] - boxes_array[:, 0] > 1.0) & (
            boxes_array[:, 3] - boxes_array[:, 1] > 1.0
        )
        boxes_array = boxes_array[valid]
    return image_rgb, depth_m, boxes_array


def _augment_photometric(
    image_rgb: np.ndarray, rng: random.Random, strength: float
) -> np.ndarray:
    """Apply random hue/saturation/value jitter on an RGB uint8 image."""
    if strength <= 0:
        return image_rgb
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue_shift = rng.uniform(-9.0, 9.0) * strength
    saturation_scale = 1.0 + rng.uniform(-0.5, 0.5) * strength
    value_scale = 1.0 + rng.uniform(-0.4, 0.4) * strength
    hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * value_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def prepare_sample(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    boxes: tuple[tuple[float, float, float, float, int], ...],
    image_size: tuple[int, int],
    train: bool,
    rng: random.Random,
    depth_max_m: float,
    scale_jitter: float = 0.0,
    translate_jitter: float = 0.0,
    hsv_jitter: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    target_height, target_width = image_size
    source_height, source_width = image_rgb.shape[:2]
    boxes_array = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    if train and rng.random() < 0.5:
        image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
        depth_m = np.ascontiguousarray(depth_m[:, ::-1])
        if len(boxes_array):
            left = boxes_array[:, 0].copy()
            right = boxes_array[:, 2].copy()
            boxes_array[:, 0] = source_width - right
            boxes_array[:, 2] = source_width - left

    if train:
        image_rgb, depth_m, boxes_array = _augment_geometry(
            image_rgb,
            depth_m,
            boxes_array,
            rng,
            scale_jitter,
            translate_jitter,
        )
        image_rgb = _augment_photometric(image_rgb, rng, hsv_jitter)
        source_height, source_width = image_rgb.shape[:2]

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(round(source_width * scale), target_width))
    resized_height = max(1, min(round(source_height * scale), target_height))
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    resized_image = cv2.resize(
        image_rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    resized_depth = cv2.resize(
        depth_m, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    depth_canvas = np.zeros((target_height, target_width), dtype=np.float32)
    canvas[
        pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
    ] = resized_image
    depth_canvas[
        pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
    ] = resized_depth

    if len(boxes_array):
        boxes_array[:, (0, 2)] = boxes_array[:, (0, 2)] * scale + pad_left
        boxes_array[:, (1, 3)] = boxes_array[:, (1, 3)] * scale + pad_top
        boxes_array[:, (0, 2)] = boxes_array[:, (0, 2)].clip(
            pad_left, pad_left + resized_width
        )
        boxes_array[:, (1, 3)] = boxes_array[:, (1, 3)].clip(
            pad_top, pad_top + resized_height
        )
        valid = (boxes_array[:, 2] - boxes_array[:, 0] > 1.0) & (
            boxes_array[:, 3] - boxes_array[:, 1] > 1.0
        )
        boxes_array = boxes_array[valid]
    image_tensor = torch.from_numpy(canvas.copy()).permute(2, 0, 1).float() / 255.0
    depth_tensor = torch.from_numpy(depth_canvas / depth_max_m).clamp_(0.0, 1.0)
    boxes_tensor = torch.from_numpy(boxes_array)
    return image_tensor, depth_tensor, boxes_tensor


class MultitaskDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        records: list[Record],
        image_size: tuple[int, int],
        train: bool,
        depth_max_m: float,
        seed: int,
        scale_jitter: float = 0.0,
        translate_jitter: float = 0.0,
        hsv_jitter: float = 0.0,
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.train = train
        self.depth_max_m = depth_max_m
        self.seed = seed
        self.scale_jitter = scale_jitter
        self.translate_jitter = translate_jitter
        self.hsv_jitter = hsv_jitter
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        record = self.records[index]
        image_bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise OSError(f"Could not read image: {record.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        if record.depth_path is None:
            depth_m = np.zeros((height, width), dtype=np.float32)
        else:
            depth_m = _read_depth_m(
                record.depth_path, record.depth_rotation, self.depth_max_m
            )
            if depth_m.shape != (height, width):
                depth_m = cv2.resize(
                    depth_m, (width, height), interpolation=cv2.INTER_NEAREST
                )
        rng = random.Random(self.seed + self.epoch * len(self.records) + index)
        return prepare_sample(
            image_rgb,
            depth_m,
            record.boxes,
            self.image_size,
            self.train,
            rng,
            self.depth_max_m,
            self.scale_jitter if self.train else 0.0,
            self.translate_jitter if self.train else 0.0,
            self.hsv_jitter if self.train else 0.0,
        )


def collate_multitask(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, list[Tensor]]:
    images, depths, boxes = zip(*batch)
    return torch.stack(images), torch.stack(depths), list(boxes)


def _candidate_levels(box_width: float, box_height: float) -> list[int]:
    side = math.sqrt(max(box_width * box_height, 0.0))
    preferred = 0 if side < 64 else 1 if side < 160 else 2
    return sorted(range(3), key=lambda level: (abs(level - preferred), level))


def build_detection_targets(
    outputs: dict[str, Tensor],
    boxes_per_image: list[Tensor],
    device: torch.device,
    topk: int = 1,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Assign every GT box to up to ``topk`` free cells of the matching output scale.

    Multi-cell assignment (YOLO-style) gives each object several positive cells, so
    the regression branch receives a much stronger signal than a single cell per box.
    """
    if topk < 1:
        raise ValueError("topk must be at least 1.")
    class_targets: list[Tensor] = []
    box_targets: list[Tensor] = []
    positive_masks: list[Tensor] = []
    for _, prefix in DETECTION_LEVELS:
        logits = outputs[f"{prefix}_cls"]
        class_targets.append(torch.zeros_like(logits))
        box_targets.append(torch.zeros_like(outputs[f"{prefix}_bbox"]))
        positive_masks.append(
            torch.zeros(
                logits.shape[0],
                logits.shape[2],
                logits.shape[3],
                dtype=torch.bool,
                device=device,
            )
        )

    for batch_index, batch_boxes in enumerate(boxes_per_image):
        for row in batch_boxes.tolist():
            x1, y1, x2, y2, class_id = row
            width, height = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            center_x, center_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            assignments: list[tuple[int, int, int, int]] = []
            for level in _candidate_levels(width, height):
                stride, _ = DETECTION_LEVELS[level]
                grid_height, grid_width = outputs[
                    f"{DETECTION_LEVELS[level][1]}_cls"
                ].shape[-2:]
                center_col = min(int(center_x / stride), grid_width - 1)
                center_row = min(int(center_y / stride), grid_height - 1)
                radius = int(math.ceil(ASSIGNMENT_RADIUS_CELLS))
                candidates: list[tuple[float, int, int]] = []
                for row_index in range(
                    max(0, center_row - radius), min(grid_height, center_row + radius + 1)
                ):
                    for col_index in range(
                        max(0, center_col - radius),
                        min(grid_width, center_col + radius + 1),
                    ):
                        distance = math.hypot(
                            (col_index + 0.5) * stride - center_x,
                            (row_index + 0.5) * stride - center_y,
                        )
                        if distance > ASSIGNMENT_RADIUS_CELLS * stride:
                            continue
                        candidates.append((distance, row_index, col_index))
                candidates.sort(key=lambda item: item[0])
                for _, row_index, col_index in candidates:
                    if len(assignments) >= topk:
                        break
                    if positive_masks[level][batch_index, row_index, col_index]:
                        continue
                    assignments.append((level, row_index, col_index, stride))
                if assignments:
                    break
            if not assignments:
                raise RuntimeError(
                    "Could not assign a ground-truth box to an unused detector cell; "
                    "try a larger input size or inspect crowded annotations."
                )

            for level, row_index, col_index, stride in assignments:
                positive_masks[level][batch_index, row_index, col_index] = True
                class_targets[level][batch_index, int(class_id), row_index, col_index] = 1.0
                box_targets[level][batch_index, :, row_index, col_index] = torch.tensor(
                    (
                        center_x / stride - (col_index + 0.5),
                        center_y / stride - (row_index + 0.5),
                        math.log(max(width / stride, 1e-3)),
                        math.log(max(height / stride, 1e-3)),
                    ),
                    device=device,
                    dtype=box_targets[level].dtype,
                )
    return class_targets, box_targets, positive_masks


def _focal_classification_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Standard (RetinaNet/Ultralytics-style) focal loss averaged over all cells."""
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    focal_loss = alpha_t * (1.0 - p_t).pow(gamma) * cross_entropy
    return focal_loss.mean()


def compute_losses(
    outputs: dict[str, Tensor],
    target_depth: Tensor,
    boxes_per_image: list[Tensor],
    device: torch.device,
    topk: int = 1,
    cls_weight: float = 1.0,
    box_weight: float = 5.0,
    depth_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    class_targets, box_targets, positive_masks = build_detection_targets(
        outputs, boxes_per_image, device, topk
    )
    classification_loss = outputs["p3_cls"].new_zeros(())
    regression_parts: list[Tensor] = []
    for level, (_, prefix) in enumerate(DETECTION_LEVELS):
        logits = outputs[f"{prefix}_cls"]
        classification_loss = classification_loss + _focal_classification_loss(
            logits, class_targets[level], focal_alpha, focal_gamma
        ) / len(DETECTION_LEVELS)
        mask = positive_masks[level]
        if mask.any():
            predicted_boxes = outputs[f"{prefix}_bbox"].permute(0, 2, 3, 1)[mask]
            expected_boxes = box_targets[level].permute(0, 2, 3, 1)[mask]
            regression_parts.append(
                F.smooth_l1_loss(predicted_boxes, expected_boxes, beta=1.0)
            )
    box_loss = (
        torch.stack(regression_parts).mean()
        if regression_parts
        else outputs["p3_bbox"].sum() * 0.0
    )
    predicted_depth = outputs["depth"][:, 0]
    valid_depth = target_depth > 0
    if valid_depth.any():
        depth_loss = F.smooth_l1_loss(
            predicted_depth[valid_depth],
            target_depth[valid_depth],
            beta=0.05,
        )
    else:
        depth_loss = predicted_depth.sum() * 0.0
    total = (
        cls_weight * classification_loss
        + box_weight * box_loss
        + depth_weight * depth_loss
    )
    return total, {
        "classification": classification_loss,
        "box": box_loss,
        "depth": depth_loss,
    }


def _decode_level_detections(
    cls_logits: Tensor,
    bbox_predictions: Tensor,
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode one feature level into pixel-space boxes, scores and class ids."""
    probability = cls_logits.sigmoid()
    scores, class_ids = probability.max(dim=1)
    bbox = bbox_predictions.permute(0, 2, 3, 1)
    grid_height, grid_width = scores.shape[-2:]
    rows = torch.arange(grid_height, device=scores.device, dtype=bbox.dtype).view(-1, 1)
    cols = torch.arange(grid_width, device=scores.device, dtype=bbox.dtype).view(1, -1)
    center_x = (cols.unsqueeze(0) + 0.5 + bbox[..., 0]) * stride
    center_y = (rows.unsqueeze(0) + 0.5 + bbox[..., 1]) * stride
    width = torch.exp(bbox[..., 2].clamp(-6.0, 6.0)) * stride
    height = torch.exp(bbox[..., 3].clamp(-6.0, 6.0)) * stride
    boxes = torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )
    return boxes, scores, class_ids


def _nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Vectorised greedy non-maximum suppression for one class."""
    if boxes.shape[0] == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    ordered_boxes = boxes[order]
    iou = _box_iou(ordered_boxes, ordered_boxes).triu(1)
    keep = iou.max(dim=0).values <= iou_threshold
    return order[keep]


def _collect_detections(
    outputs: dict[str, Tensor],
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[tuple[Tensor, Tensor, Tensor]]:
    """Decode all three scales, drop low-confidence boxes and run class-wise NMS."""
    batch_size = int(outputs["p3_cls"].shape[0])
    num_classes = int(outputs["p3_cls"].shape[1])
    detections: list[tuple[Tensor, Tensor, Tensor]] = []
    for batch_index in range(batch_size):
        level_boxes: list[Tensor] = []
        level_scores: list[Tensor] = []
        level_classes: list[Tensor] = []
        for stride, prefix in DETECTION_LEVELS:
            boxes, scores, class_ids = _decode_level_detections(
                outputs[f"{prefix}_cls"], outputs[f"{prefix}_bbox"], stride
            )
            keep_mask = scores[batch_index] >= conf_threshold
            if not bool(keep_mask.any()):
                continue
            level_boxes.append(boxes[batch_index][keep_mask])
            level_scores.append(scores[batch_index][keep_mask])
            level_classes.append(class_ids[batch_index][keep_mask])
        if not level_boxes:
            detections.append(
                (
                    torch.zeros((0, 4)),
                    torch.zeros((0,)),
                    torch.zeros((0,), dtype=torch.long),
                )
            )
            continue
        candidate_boxes = torch.cat(level_boxes)
        candidate_scores = torch.cat(level_scores)
        candidate_classes = torch.cat(level_classes)
        if candidate_boxes.shape[0] > max_detections:
            top = candidate_scores.topk(max_detections).indices
            candidate_boxes = candidate_boxes[top]
            candidate_scores = candidate_scores[top]
            candidate_classes = candidate_classes[top]
        kept_boxes: list[Tensor] = []
        kept_scores: list[Tensor] = []
        kept_classes: list[Tensor] = []
        for class_id in range(num_classes):
            class_mask = candidate_classes == class_id
            if not bool(class_mask.any()):
                continue
            selected = _nms(
                candidate_boxes[class_mask],
                candidate_scores[class_mask],
                iou_threshold,
            )
            if selected.numel() == 0:
                continue
            kept_boxes.append(candidate_boxes[class_mask][selected])
            kept_scores.append(candidate_scores[class_mask][selected])
            kept_classes.append(candidate_classes[class_mask][selected])
        if not kept_boxes:
            detections.append(
                (
                    torch.zeros((0, 4)),
                    torch.zeros((0,)),
                    torch.zeros((0,), dtype=torch.long),
                )
            )
            continue
        detections.append(
            (
                torch.cat(kept_boxes),
                torch.cat(kept_scores),
                torch.cat(kept_classes),
            )
        )
    return detections


def _box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU between two sets of xyxy boxes."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device)
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=2)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (
        boxes_a[:, 3] - boxes_a[:, 1]
    ).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (
        boxes_b[:, 3] - boxes_b[:, 1]
    ).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=1e-7)


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """COCO-style all-point interpolated average precision."""
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def _evaluate_detection(
    predictions: list[tuple[Tensor, Tensor, Tensor]],
    targets: list[tuple[Tensor, Tensor]],
    num_classes: int,
    class_names: tuple[str, ...],
    iou_thresholds: tuple[float, ...] = AP_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Compute mAP@0.5, mAP@0.5:0.95, precision and recall for one split."""
    per_class_ap50: dict[str, float] = {}
    per_class_ap50_95: dict[str, float] = {}
    total_true_positives = 0
    total_false_positives = 0
    total_ground_truth = 0
    device = predictions[0][0].device if predictions else torch.device("cpu")
    for class_id in range(num_classes):
        class_name = (
            class_names[class_id] if class_id < len(class_names) else str(class_id)
        )
        detection_scores: list[Tensor] = []
        detection_boxes: list[Tensor] = []
        detection_images: list[Tensor] = []
        for image_index, (boxes, scores, class_ids) in enumerate(predictions):
            mask = class_ids == class_id
            if not bool(mask.any()):
                continue
            detection_scores.append(scores[mask])
            detection_boxes.append(boxes[mask])
            detection_images.append(
                torch.full(
                    (int(mask.sum().item()),),
                    image_index,
                    dtype=torch.long,
                    device=scores.device,
                )
            )
        class_ground_truth: list[Tensor] = []
        for image_index, (ground_truth_boxes, ground_truth_classes) in enumerate(targets):
            ground_truth_boxes = ground_truth_boxes.to(device)
            mask = ground_truth_classes.to(device) == class_id
            class_ground_truth.append(
                ground_truth_boxes[mask]
                if bool(mask.any())
                else torch.zeros((0, 4), device=device)
            )
        num_ground_truth = sum(int(item.shape[0]) for item in class_ground_truth)
        total_ground_truth += num_ground_truth
        if num_ground_truth == 0:
            continue
        if not detection_scores:
            per_class_ap50[class_name] = 0.0
            per_class_ap50_95[class_name] = 0.0
            continue
        scores = torch.cat(detection_scores)
        boxes = torch.cat(detection_boxes)
        image_indices = torch.cat(detection_images)
        order = scores.argsort(descending=True)
        ordered_boxes = boxes[order]
        ordered_images = image_indices[order]
        averages: list[float] = []
        true_positives_at_50 = 0
        false_positives_at_50 = 0
        for threshold in iou_thresholds:
            matched = [
                np.zeros(int(item.shape[0]), dtype=bool) for item in class_ground_truth
            ]
            hits = np.zeros(int(order.numel()), dtype=np.float64)
            for rank in range(int(order.numel())):
                image_index = int(ordered_images[rank].item())
                ground_truth = class_ground_truth[image_index]
                if ground_truth.shape[0] == 0:
                    continue
                ious = (
                    _box_iou(
                        ordered_boxes[rank : rank + 1], ground_truth
                    )[0]
                    .cpu()
                    .numpy()
                )
                for ground_truth_index in np.argsort(-ious):
                    if ious[ground_truth_index] < threshold:
                        break
                    if matched[image_index][ground_truth_index]:
                        continue
                    matched[image_index][ground_truth_index] = True
                    hits[rank] = 1.0
                    break
            if threshold == 0.5:
                true_positives_at_50 = float(hits.sum())
                false_positives_at_50 = float(len(hits) - hits.sum())
            cumulative_hits = np.cumsum(hits)
            false_positives = np.cumsum(1.0 - hits)
            recalls = cumulative_hits / num_ground_truth
            precisions = cumulative_hits / np.maximum(
                cumulative_hits + false_positives, 1e-9
            )
            averages.append(_interpolated_ap(recalls, precisions))
        total_true_positives += true_positives_at_50
        total_false_positives += false_positives_at_50
        per_class_ap50[class_name] = averages[0] if averages else 0.0
        per_class_ap50_95[class_name] = (
            float(np.mean(averages)) if averages else 0.0
        )
    evaluated_classes = len(per_class_ap50)
    map50 = (
        float(np.mean(list(per_class_ap50.values()))) if evaluated_classes else 0.0
    )
    map50_95 = (
        float(np.mean(list(per_class_ap50_95.values())))
        if evaluated_classes
        else 0.0
    )
    precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0.0
    )
    recall = (
        total_true_positives / total_ground_truth if total_ground_truth else 0.0
    )
    return {
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap50_95,
        "ground_truth_boxes": int(total_ground_truth),
        "evaluated_classes": evaluated_classes,
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, list[Tensor]]],
    device: torch.device,
    depth_max_m: float,
    description: str,
    classes: tuple[str, ...] = DEFAULT_CLASSES,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
    topk: int = 1,
    cls_weight: float = 1.0,
    box_weight: float = 5.0,
    depth_weight: float = 1.0,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    max_detections: int = 300,
) -> dict[str, Any]:
    model.eval()
    sums: dict[str, float] = {}
    sample_count = 0
    depth_abs_error = 0.0
    depth_valid_pixels = 0
    predictions: list[tuple[Tensor, Tensor, Tensor]] = []
    targets: list[tuple[Tensor, Tensor]] = []
    iterator = iter(loader)
    with torch.inference_mode():
        with ActivityIndicator(description) as activity:
            for batch_index in range(len(loader)):
                activity.update(
                    f"{description}: batch {batch_index + 1}/{len(loader)}"
                )
                images, depths, boxes = next(iterator)
                images = images.to(device, non_blocking=True)
                depths = depths.to(device, non_blocking=True)
                outputs = model(images, use_nms=False)
                loss, parts = compute_losses(
                    outputs,
                    depths,
                    boxes,
                    device,
                    topk,
                    cls_weight,
                    box_weight,
                    depth_weight,
                    focal_alpha,
                    focal_gamma,
                )
                batch_size = images.shape[0]
                sample_count += batch_size
                for name, value in {"loss": loss, **parts}.items():
                    sums[name] = sums.get(name, 0.0) + float(value) * batch_size
                valid = depths > 0
                if valid.any():
                    depth_abs_error += float(
                        (outputs["depth"][:, 0][valid] - depths[valid])
                        .abs()
                        .sum()
                        .item()
                    )
                    depth_valid_pixels += int(valid.sum().item())
                predictions.extend(
                    _collect_detections(
                        outputs, conf_threshold, iou_threshold, max_detections
                    )
                )
                for ground_truth in boxes:
                    if ground_truth.numel():
                        targets.append(
                            (
                                ground_truth[:, :4].to(device),
                                ground_truth[:, 4].to(device).long(),
                            )
                        )
                    else:
                        targets.append(
                            (
                                torch.zeros((0, 4), device=device),
                                torch.zeros((0,), dtype=torch.long, device=device),
                            )
                        )
    if sample_count == 0:
        raise ValueError(f"{description} loader is empty.")
    metrics: dict[str, Any] = {
        name: value / sample_count for name, value in sums.items()
    }
    if depth_valid_pixels == 0:
        raise ValueError(f"{description} contains no valid depth pixels.")
    metrics["depth_mae_m"] = depth_abs_error / depth_valid_pixels * depth_max_m
    metrics["depth_valid_pixels"] = float(depth_valid_pixels)
    metrics.update(
        _evaluate_detection(predictions, targets, len(classes), tuple(classes))
    )
    return metrics


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


class ModelEMA:
    """Exponential moving average of the model weights used for evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.updates = 0
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def _build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    backbone_lr_mult: float,
) -> torch.optim.Optimizer:
    """AdamW with a smaller learning rate for the pretrained trunk and no decay on 1-D params."""
    trunk_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone.") or name.startswith("neck."):
            trunk_parameters.append(parameter)
        else:
            head_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    for parameters, learning_rate_scale in (
        (head_parameters, 1.0),
        (trunk_parameters, backbone_lr_mult),
    ):
        if not parameters:
            continue
        group_lr = learning_rate * learning_rate_scale
        decayed = [parameter for parameter in parameters if parameter.ndim > 1]
        plain = [parameter for parameter in parameters if parameter.ndim <= 1]
        if decayed:
            groups.append(
                {"params": decayed, "lr": group_lr, "weight_decay": weight_decay}
            )
        if plain:
            groups.append({"params": plain, "lr": group_lr, "weight_decay": 0.0})
    if not groups:
        raise ValueError("The model has no trainable parameters.")
    return torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warm-up followed by cosine decay to ``min_lr_ratio`` of the base LR."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / float(max(1, warmup_steps))
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / float(span)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_sampling_weights(
    records: list[Record],
    num_classes: int,
    threshold: float,
    cap: float,
    depth_only_weight: float,
) -> list[float]:
    """Repeat-factor sampling weights so rare classes are seen more often."""
    if not records:
        return []
    class_presence = [0] * num_classes
    record_classes: list[set[int]] = []
    for record in records:
        classes_present = {int(box[4]) for box in record.boxes}
        record_classes.append(classes_present)
        for class_id in classes_present:
            class_presence[class_id] += 1
    repeat_factors = [1.0] * num_classes
    for class_id in range(num_classes):
        frequency = class_presence[class_id] / len(records)
        if frequency <= 0:
            continue
        repeat_factors[class_id] = min(
            cap, max(1.0, math.sqrt(threshold / frequency))
        )
    weights: list[float] = []
    for classes_present in record_classes:
        if not classes_present:
            weights.append(depth_only_weight)
            continue
        weights.append(max(repeat_factors[class_id] for class_id in classes_present))
    return weights


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    classes: tuple[str, ...],
    image_size: tuple[int, int],
    depth_max_m: float,
    width_mult: float,
    depth_mult: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    ema_state_dict: dict[str, Tensor] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "kind": "arsitektur_multitask",
        "architecture_id": ARCHITECTURE_ID,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "classes": list(classes),
        "imgsz": list(image_size),
        "depth_max_m": depth_max_m,
        "width_mult": width_mult,
        "depth_mult": depth_mult,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "split_seed": seed,
    }
    if ema_state_dict is not None:
        checkpoint["ema_state_dict"] = ema_state_dict
    if extra:
        checkpoint.update(extra)
    return checkpoint


_YOLO26_PRETRAINED_LAYERS = (
    ("backbone.stem", 0),
    ("backbone.l2.0", 1),
    ("backbone.l2.1", 2),
    ("backbone.l3", 3),
    ("backbone.l4", 4),
    ("backbone.l5", 5),
    ("backbone.l6", 6),
    ("backbone.l7", 7),
    ("backbone.l8", 8),
    ("backbone.l9", 9),
    ("backbone.l10", 10),
    ("neck.n13", 13),
    ("neck.n16", 16),
    ("neck.d17", 17),
    ("neck.n19", 19),
    ("neck.d20", 20),
    ("neck.n22", 22),
)


def _transfer_yolo26_backbone_neck(
    model: nn.Module, pretrained_model: nn.Module
) -> int:
    layers = getattr(pretrained_model, "model", None)
    if not isinstance(layers, (nn.Sequential, nn.ModuleList)) or len(layers) <= 22:
        raise ValueError("Pretrained weights must contain the YOLO26n detection graph.")

    transferred = 0
    for target_name, source_index in _YOLO26_PRETRAINED_LAYERS:
        target = model.get_submodule(target_name)
        source_state = layers[source_index].state_dict()
        mapped_state = {
            ".".join(
                {
                    "m": "blocks",
                    "attn": "attention",
                    "pe": "position",
                }.get(part, part)
                for part in key.split(".")
            ): value
            for key, value in source_state.items()
        }
        target_state = target.state_dict()
        missing = target_state.keys() - mapped_state.keys()
        unexpected = mapped_state.keys() - target_state.keys()
        if missing or unexpected:
            raise ValueError(
                f"Pretrained layer {source_index} does not match {target_name}: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}."
            )
        mismatched = [
            key
            for key, value in mapped_state.items()
            if target_state[key].shape != value.shape
        ]
        if mismatched:
            raise ValueError(
                f"Pretrained layer {source_index} has incompatible tensor shapes "
                f"for {target_name}: {mismatched}."
            )
        target.load_state_dict(mapped_state, strict=True)
        transferred += len(mapped_state)
    return transferred


def initialize_pretrained_yolo26(model: nn.Module, weights: str) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Pretrained YOLO26 initialization requires Ultralytics. "
            "Install ultralytics==8.4.175 in the training environment."
        ) from exc

    pretrained_model = YOLO(weights).model
    transferred = _transfer_yolo26_backbone_neck(model, pretrained_model)
    print(
        f"Initialized the arsitektur.py backbone and neck from {weights} "
        f"({transferred:,} tensors). Detection and depth heads remain task-specific."
    )


def initialize_from_checkpoint(
    model: nn.Module, checkpoint_path: Path, expected: dict[str, Any]
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint is not a valid training checkpoint: {checkpoint_path}")
    if checkpoint.get("kind") != "arsitektur_multitask":
        raise ValueError(f"Checkpoint is not an arsitektur.py checkpoint: {checkpoint_path}")
    if checkpoint.get("architecture_id") not in (None, ARCHITECTURE_ID):
        raise ValueError("Cannot initialize from a different architecture checkpoint.")
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Cannot initialize from checkpoint: {key}={checkpoint.get(key)!r}, "
                f"current value is {value!r}."
            )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no model weights: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    print(
        f"Initialized model weights from {checkpoint_path}; "
        "optimizer and learning-rate schedule will start fresh."
    )


def train(args: argparse.Namespace) -> None:
    if args.img_size[0] < 32 or args.img_size[1] < 32:
        raise ValueError("Both image dimensions must be at least 32.")
    if any(value % 32 for value in args.img_size):
        raise ValueError("--img-size dimensions must both be divisible by 32.")
    if (
        args.val_fraction <= 0
        or args.test_fraction <= 0
        or args.val_fraction + args.test_fraction >= 1
    ):
        raise ValueError(
            "--val-fraction and --test-fraction must be positive and sum to less than 1."
        )
    if not 0 <= args.coco_ratio < 0.9:
        raise ValueError("--coco-ratio must be between 0 and 0.9.")
    if args.depth_max_m <= 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Depth range, epoch count, and batch size must be positive.")
    if args.topk_assign < 1:
        raise ValueError("--topk-assign must be at least 1.")
    if args.backbone_lr_mult <= 0:
        raise ValueError("--backbone-lr-mult must be positive.")
    if not 0.0 <= args.min_lr_ratio <= 1.0:
        raise ValueError("--min-lr-ratio must be between 0 and 1.")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs cannot be negative.")
    if not 0.0 <= args.scale_jitter < 1.0:
        raise ValueError("--scale-jitter must be in [0, 1).")
    if not 0.0 <= args.translate_jitter < 1.0:
        raise ValueError("--translate-jitter must be in [0, 1).")
    if args.hsv_jitter < 0:
        raise ValueError("--hsv-jitter cannot be negative.")
    if not 0.0 <= args.conf_threshold <= 1.0:
        raise ValueError("--conf-threshold must be between 0 and 1.")
    if not 0.0 <= args.nms_iou <= 1.0:
        raise ValueError("--nms-iou must be between 0 and 1.")
    if args.ema_decay < 0 or args.ema_decay > 1:
        raise ValueError("--ema-decay must be between 0 and 1.")
    if args.balance_cap < 1.0 or args.depth_only_weight <= 0:
        raise ValueError("--balance-cap must be >= 1 and --depth-only-weight must be > 0.")
    initialization_modes = sum(
        (
            args.resume is not None,
            args.pretrained_yolo26 is not None,
            args.finetune_from is not None,
        )
    )
    if initialization_modes > 1:
        raise ValueError(
            "Choose only one of --resume, --pretrained-yolo26, or --finetune-from."
        )

    classes = tuple(name.strip().lower() for name in args.classes if name.strip())
    if not classes or len({_class_key(name) for name in classes}) != len(classes):
        raise ValueError("--classes must contain distinct, non-empty class names.")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch install.")
    use_amp = device.type == "cuda" and args.amp
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    download_dir = args.download_dir.expanduser().resolve()
    sun_root = (
        args.sun_root.expanduser().resolve()
        if args.sun_root is not None
        else download_dir / "SUNRGBD"
    )
    ensure_sun_dataset(sun_root, download_dir, allow_download=not args.no_download)
    sun_records = discover_sun_records(
        sun_root,
        classes,
        args.depth_max_m,
        args.sun_depth_rotation,
    )
    train_records, val_records, test_records = split_sun_records(
        sun_records, args.val_fraction, args.test_fraction, args.seed
    )
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]
    if args.max_test_samples is not None:
        test_records = test_records[: args.max_test_samples]
    if not train_records or not val_records or not test_records:
        raise ValueError("Train/validation/test split is empty after applying sample limits.")

    if args.coco_ratio > 0:
        coco_root = (
            args.coco_root.expanduser().resolve()
            if args.coco_root is not None
            else download_dir / "coco"
        )
        if args.coco_annotations is not None:
            coco_annotations = args.coco_annotations.expanduser().resolve()
            if not coco_annotations.is_file():
                raise FileNotFoundError(
                    f"COCO annotation file not found: {coco_annotations}"
                )
        else:
            coco_annotations = ensure_coco_annotations(
                coco_root, download_dir, allow_download=not args.no_download
            )
        target_train_coco = round(
            len(train_records) * args.coco_ratio / (1 - args.coco_ratio)
        )
        if target_train_coco == 0:
            print("COCO ratio and SUN train size produce zero COCO images; skipping COCO.")
        else:
            train_fraction = 1 - args.val_fraction - args.test_fraction
            target_count = max(3, math.ceil(target_train_coco / train_fraction))
            coco_records = discover_coco_records(
                coco_root,
                coco_annotations,
                classes,
                max_records=target_count,
                seed=args.seed,
                allow_missing_images=not args.no_download,
                local_only=args.coco_local_only,
            )
            if not args.no_download and not args.coco_local_only:
                download_coco_images(coco_records)
            coco_train, coco_val, coco_test = split_coco_records(
                coco_records, args.val_fraction, args.test_fraction, args.seed
            )
            train_records.extend(coco_train)
            val_records.extend(coco_val)
            test_records.extend(coco_test)
            print(
                f"Split {len(coco_records)} COCO images into "
                f"train/validation/test={len(coco_train)}/{len(coco_val)}/{len(coco_test)} "
                f"(target COCO share of training data {args.coco_ratio:.0%})."
            )

    def count_boxes(records: list[Record]) -> dict[str, int]:
        counts = [0] * len(classes)
        for record in records:
            for *_, class_id in record.boxes:
                counts[class_id] += 1
        return dict(zip(classes, counts))

    class_counts = count_boxes(train_records)
    if not any(class_counts.values()):
        raise ValueError(
            f"No training boxes matched classes {classes}. "
            "Check class spelling and SUN RGB-D annotations."
        )

    split_summary = {
        "train": {
            "sun_rgbd_records": sum(
                record.depth_path is not None for record in train_records
            ),
            "coco_records": sum(
                record.depth_path is None for record in train_records
            ),
            "total_records": len(train_records),
            "boxes_by_class": count_boxes(train_records),
        },
        "validation": {
            "sun_rgbd_records": sum(
                record.depth_path is not None for record in val_records
            ),
            "coco_records": sum(
                record.depth_path is None for record in val_records
            ),
            "total_records": len(val_records),
            "boxes_by_class": count_boxes(val_records),
        },
        "test": {
            "sun_rgbd_records": sum(
                record.depth_path is not None for record in test_records
            ),
            "coco_records": sum(
                record.depth_path is None for record in test_records
            ),
            "total_records": len(test_records),
            "boxes_by_class": count_boxes(test_records),
        },
    }
    print("Split summary:")
    print(json.dumps(split_summary, indent=2))
    print(
        "Split sizes "
        + " | ".join(
            f"{name}={summary['total_records']} "
            f"(SUN RGB-D={summary['sun_rgbd_records']}, COCO={summary['coco_records']})"
            for name, summary in split_summary.items()
        )
    )

    train_dataset = MultitaskDataset(
        train_records,
        tuple(args.img_size),
        True,
        args.depth_max_m,
        args.seed,
        args.scale_jitter,
        args.translate_jitter,
        args.hsv_jitter,
    )
    val_dataset = MultitaskDataset(
        val_records,
        tuple(args.img_size),
        False,
        args.depth_max_m,
        args.seed,
    )
    test_dataset = MultitaskDataset(
        test_records,
        tuple(args.img_size),
        False,
        args.depth_max_m,
        args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda" and not args.no_pin_memory,
        "collate_fn": collate_multitask,
    }
    train_sampler: WeightedRandomSampler | None = None
    if args.balanced_sampling:
        sampling_weights = _compute_sampling_weights(
            train_records,
            len(classes),
            args.balance_threshold,
            args.balance_cap,
            args.depth_only_weight,
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        train_sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        print(
            "Class-balanced sampling enabled over "
            f"{len(sampling_weights)} training records "
            f"(weight {min(sampling_weights):.2f}-{max(sampling_weights):.2f})."
        )
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=False,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = build_model(
        num_classes=len(classes),
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
    )
    if args.pretrained_yolo26 is not None:
        initialize_pretrained_yolo26(model, args.pretrained_yolo26)
    if args.finetune_from is not None:
        expected_checkpoint = {
            "kind": "arsitektur_multitask",
            "classes": list(classes),
            "imgsz": list(args.img_size),
            "depth_max_m": args.depth_max_m,
            "width_mult": args.width_mult,
            "depth_mult": args.depth_mult,
        }
        initialize_from_checkpoint(model, args.finetune_from, expected_checkpoint)
    model = model.to(device)
    optimizer = _build_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
        args.backbone_lr_mult,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    scheduler = _build_scheduler(
        optimizer, warmup_steps, total_steps, args.min_lr_ratio
    )
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_val_loss = math.inf
    best_map50 = -1.0
    best_selection_loss = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("architecture_id") not in (None, ARCHITECTURE_ID):
            raise ValueError("Cannot resume: checkpoint uses a different architecture.")
        expected = {
            "kind": "arsitektur_multitask",
            "classes": list(classes),
            "imgsz": list(args.img_size),
            "depth_max_m": args.depth_max_m,
            "width_mult": args.width_mult,
            "depth_mult": args.depth_mult,
        }
        for key, value in expected.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"Cannot resume: checkpoint {key}={checkpoint.get(key)!r}, "
                    f"current value is {value!r}."
                )
        expected_split = {
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "split_seed": args.seed,
        }
        for key, value in expected_split.items():
            if checkpoint.get(key) != value:
                raise ValueError(
                    f"Cannot resume: checkpoint {key}={checkpoint.get(key)!r}, "
                    f"current value is {value!r}. A new split requires a fresh run."
                )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
        best_map50 = float(checkpoint.get("best_map50", best_map50))
        best_selection_loss = float(
            checkpoint.get("best_selection_loss", best_selection_loss)
        )
        if ema is not None and isinstance(checkpoint.get("ema_state_dict"), dict):
            ema.model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
            ema.updates = int(checkpoint.get("ema_updates", ema.updates))
        print(f"Resuming after epoch {start_epoch} from {args.resume}.")

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    if args.resume is not None and start_epoch >= args.epochs:
        raise ValueError(
            f"Checkpoint already completed {start_epoch} epochs; increase --epochs."
        )

    for epoch in range(start_epoch, args.epochs):
        train_dataset.epoch = epoch
        model.train()
        train_rows: list[dict[str, float]] = []
        train_iterator = iter(train_loader)
        with ActivityIndicator(f"Epoch {epoch + 1}/{args.epochs}: training") as activity:
            for batch_index in range(len(train_loader)):
                activity.update(
                    f"Epoch {epoch + 1}/{args.epochs}: "
                    f"loading train batch {batch_index + 1}/{len(train_loader)}"
                )
                images, depths, boxes = next(train_iterator)
                activity.update(
                    f"Epoch {epoch + 1}/{args.epochs}: "
                    f"training batch {batch_index + 1}/{len(train_loader)}"
                )
                images = images.to(device, non_blocking=True)
                depths = depths.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    outputs = model(images, use_nms=False)
                    loss, parts = compute_losses(
                        outputs,
                        depths,
                        boxes,
                        device,
                        args.topk_assign,
                        args.cls_weight,
                        args.box_weight,
                        args.depth_weight,
                        args.focal_alpha,
                        args.focal_gamma,
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}."
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if ema is not None:
                    ema.update(model)
                train_rows.append(
                    {
                        "loss": float(loss.detach()),
                        **{name: float(value.detach()) for name, value in parts.items()},
                    }
                )

        evaluation_model = ema.model if ema is not None else model
        val_metrics = _evaluate_model(
            evaluation_model,
            val_loader,
            device,
            args.depth_max_m,
            f"Epoch {epoch + 1}/{args.epochs}: validation",
            classes,
            args.conf_threshold,
            args.nms_iou,
            args.topk_assign,
            args.cls_weight,
            args.box_weight,
            args.depth_weight,
            args.focal_alpha,
            args.focal_gamma,
            args.max_detections,
        )
        train_metrics = _mean_metrics(train_rows)
        depth_mae_m = float(val_metrics["depth_mae_m"])
        map50 = float(val_metrics.get("map50", 0.0))
        val_loss = float(val_metrics["loss"])
        vram_note = ""
        if device.type == "cuda":
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            free_gib = torch.cuda.mem_get_info()[0] / 1024**3
            vram_note = f" vram_peak={peak_gib:.2f}GiB vram_free={free_gib:.2f}GiB"
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_det_cls={val_metrics['classification']:.4f} "
            f"val_box={val_metrics['box']:.4f} "
            f"val_depth={val_metrics['depth']:.4f} "
            f"val_depth_MAE_m={depth_mae_m:.3f} "
            f"val_mAP50={map50:.4f} "
            f"val_mAP50_95={float(val_metrics.get('map50_95', 0.0)):.4f} "
            f"val_P={float(val_metrics.get('precision', 0.0)):.4f} "
            f"val_R={float(val_metrics.get('recall', 0.0)):.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
            f"{vram_note}"
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        completed_epoch = epoch + 1
        use_map_selection = args.best_metric == "map50" and (
            map50 > 0.0 or best_map50 > 0.0
        )
        if use_map_selection:
            is_best = map50 > best_map50
            if is_best:
                best_map50 = map50
        else:
            is_best = val_loss < best_selection_loss
        best_val_loss = min(best_val_loss, val_loss)
        best_selection_loss = min(best_selection_loss, val_loss)
        checkpoint = _checkpoint(
            model,
            optimizer,
            scheduler,
            completed_epoch,
            best_val_loss,
            classes,
            tuple(args.img_size),
            args.depth_max_m,
            args.width_mult,
            args.depth_mult,
            args.val_fraction,
            args.test_fraction,
            args.seed,
            ema_state_dict=ema.model.state_dict() if ema is not None else None,
            extra={
                "ema_updates": ema.updates if ema is not None else 0,
                "best_map50": best_map50,
                "best_selection_loss": best_selection_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "val_depth_mae_m": depth_mae_m,
            },
        )
        _atomic_torch_save(checkpoint, last_path)
        if is_best:
            _atomic_torch_save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_path}")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    ema_state = best_checkpoint.get("ema_state_dict")
    if isinstance(ema_state, dict) and ema_state:
        model.load_state_dict(ema_state, strict=True)
        print("Evaluated the EMA weights stored in the best checkpoint.")
    test_metrics = _evaluate_model(
        model,
        test_loader,
        device,
        args.depth_max_m,
        "Held-out test",
        classes,
        args.conf_threshold,
        args.nms_iou,
        args.topk_assign,
        args.cls_weight,
        args.box_weight,
        args.depth_weight,
        args.focal_alpha,
        args.focal_gamma,
        args.max_detections,
    )
    print(
        "Test metrics "
        f"loss={test_metrics['loss']:.4f} "
        f"det_cls={test_metrics['classification']:.4f} "
        f"box={test_metrics['box']:.4f} "
        f"depth={test_metrics['depth']:.4f} "
        f"depth_MAE_m={test_metrics['depth_mae_m']:.3f} "
        f"mAP50={test_metrics['map50']:.4f} "
        f"mAP50_95={test_metrics['map50_95']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )
    report = {
        "classes": list(classes),
        "seed": args.seed,
        "split_fractions": {
            "train": 1 - args.val_fraction - args.test_fraction,
            "validation": args.val_fraction,
            "test": args.test_fraction,
        },
        "best_checkpoint": str(best_path.resolve()),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_loss": float(best_checkpoint["best_val_loss"]),
        "best_epoch_metrics": {
            "train": best_checkpoint["train_metrics"],
            "validation": best_checkpoint["val_metrics"],
            "validation_depth_mae_m": float(best_checkpoint["val_depth_mae_m"]),
        },
        "splits": split_summary,
        "test_metrics": test_metrics,
        "test_metrics_note": (
            "Detection metrics (mAP@0.5, mAP@0.5:0.95, precision, recall) are "
            "computed on held-out SUN RGB-D and selected local COCO records with "
            "class-wise NMS; COCO records come from instances_train2017 and this "
            "is not the official COCO validation/test benchmark. "
            "Depth MAE is measured in metres on valid SUN RGB-D depth pixels; "
            "COCO images have no depth labels and do not contribute to it."
        ),
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": list(args.img_size),
            "learning_rate": args.learning_rate,
            "backbone_lr_mult": args.backbone_lr_mult,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
            "min_lr_ratio": args.min_lr_ratio,
            "ema_decay": args.ema_decay,
            "topk_assign": args.topk_assign,
            "cls_weight": args.cls_weight,
            "box_weight": args.box_weight,
            "depth_weight": args.depth_weight,
            "focal_alpha": args.focal_alpha,
            "focal_gamma": args.focal_gamma,
            "scale_jitter": args.scale_jitter,
            "translate_jitter": args.translate_jitter,
            "hsv_jitter": args.hsv_jitter,
            "balanced_sampling": args.balanced_sampling,
            "coco_ratio": args.coco_ratio,
            "best_metric": args.best_metric,
        },
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Evaluation report: {report_path}")
    print(f"Training complete. Best checkpoint: {best_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train arsitektur.py on SUN RGB-D and optional COCO instances."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("datasets"),
        help="Folder for downloaded archives and default dataset directories.",
    )
    parser.add_argument(
        "--sun-root",
        type=Path,
        help="SUN RGB-D directory; defaults to <download-dir>/SUNRGBD.",
    )
    parser.add_argument("--coco-root", type=Path)
    parser.add_argument("--coco-annotations", type=Path)
    parser.add_argument(
        "--coco-local-only",
        action="store_true",
        help="Use only annotated COCO images already present under --coco-root.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading missing SUN RGB-D/COCO data.",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--img-size", nargs=2, type=int, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned host memory for the data loaders.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--sun-depth-rotation", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--coco-ratio",
        type=float,
        default=0.5,
        help=(
            "Target fraction of COCO records in the training split; COCO records "
            "are independently divided into train/validation/test."
        ),
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--depth-mult", type=float, default=0.5)
    parser.add_argument(
        "--cls-weight",
        type=float,
        default=1.0,
        help="Weight of the focal classification loss.",
    )
    parser.add_argument(
        "--box-weight",
        type=float,
        default=7.5,
        help="Weight of the bounding-box regression loss.",
    )
    parser.add_argument(
        "--depth-weight",
        type=float,
        default=0.5,
        help="Weight of the depth loss (keeps the shared trunk from over-focusing on depth).",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--topk-assign",
        type=int,
        default=3,
        help="Number of positive detector cells assigned to every ground-truth box.",
    )
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="Learning-rate multiplier for the pretrained backbone/neck.",
    )
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.02)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Exponential moving average decay; 0 disables the EMA weights.",
    )
    parser.add_argument("--scale-jitter", type=float, default=0.2)
    parser.add_argument("--translate-jitter", type=float, default=0.08)
    parser.add_argument("--hsv-jitter", type=float, default=1.0)
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Sample training images with repeat factors so rare classes appear more often.",
    )
    parser.add_argument("--balance-threshold", type=float, default=0.1)
    parser.add_argument("--balance-cap", type=float, default=8.0)
    parser.add_argument(
        "--depth-only-weight",
        type=float,
        default=0.3,
        help="Sampling weight for images without any detection label (depth-only).",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="Minimum confidence used when collecting detections for mAP.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument(
        "--max-detections",
        type=int,
        default=300,
        help="Maximum number of detections kept per image when computing mAP.",
    )
    parser.add_argument(
        "--best-metric",
        choices=("map50", "loss"),
        default="map50",
        help="Metric used to select the best checkpoint.",
    )
    parser.add_argument(
        "--pretrained-yolo26",
        help="Optional YOLO26n weights for initializing the matching backbone/neck.",
    )
    parser.add_argument(
        "--finetune-from",
        type=Path,
        help="Initialize model weights from a project checkpoint with fresh optimizer/scheduler.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/arsitektur"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
