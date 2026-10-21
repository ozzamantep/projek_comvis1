"""Train the standalone YOLO26n-Depth blueprint on SUN RGB-D and optional COCO."""

from __future__ import annotations

import argparse
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
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from arsitektur import ARCHITECTURE_ID, build_model


DEFAULT_CLASSES = ("person", "laptop", "cell phone", "bottle")
DEPTH_ROTATION_MASK = 0xFFFF
DETECTION_LEVELS = ((8, "p3"), (16, "p4"), (32, "p5"))
SUN_RGBD_URL = "https://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
COCO_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/images.cocodataset.org/"
    "annotations/annotations_trainval2017.zip"
)
COCO_IMAGES_URL = "https://s3.amazonaws.com/images.cocodataset.org/train2017"


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

    class_to_id = {name.casefold(): index for index, name in enumerate(classes)}
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


def split_sun_records(
    records: list[Record], val_fraction: float, seed: int
) -> tuple[list[Record], list[Record]]:
    groups: dict[str, list[Record]] = {}
    for record in records:
        groups.setdefault(record.group, []).append(record)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 2:
        raise ValueError("Need at least two SUN RGB-D groups to make a train/val split.")
    target_val = max(1, round(len(records) * val_fraction))
    val_groups: list[str] = []
    val_count = 0
    for key in keys:
        if val_count >= target_val or len(val_groups) == len(keys) - 1:
            break
        val_groups.append(key)
        val_count += len(groups[key])
    val_keys = set(val_groups)
    train = [record for record in records if record.group not in val_keys]
    val = [record for record in records if record.group in val_keys]
    if not train or not val:
        raise ValueError("SUN RGB-D split is empty; check the dataset's scene grouping.")
    return train, val


def discover_coco_records(
    root: Path,
    annotation_path: Path | None,
    classes: tuple[str, ...],
    min_box_px: float = 4.0,
    max_records: int | None = None,
    seed: int = 42,
    allow_missing_images: bool = False,
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

    try:
        with annotations_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read COCO annotations {annotations_path}") from exc

    class_to_id = {name.casefold(): index for index, name in enumerate(classes)}
    category_to_class = {
        int(category["id"]): class_to_id[category["name"].casefold()]
        for category in data.get("categories", [])
        if category.get("name", "").casefold() in class_to_id
    }
    if not category_to_class:
        available = sorted(category.get("name", "") for category in data.get("categories", []))
        raise ValueError(
            f"None of --classes match COCO categories. Requested {classes}; "
            f"available examples: {available[:20]}"
        )
    image_info = {int(image["id"]): image for image in data.get("images", [])}
    by_image: dict[int, list[tuple[float, float, float, float, int]]] = {}
    for annotation in data.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        class_id = category_to_class.get(int(annotation["category_id"]))
        image_id = int(annotation["image_id"])
        image = image_info.get(image_id)
        if class_id is None or image is None:
            continue
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        x1 = max(x, 0.0)
        y1 = max(y, 0.0)
        x2 = min(x + box_width, float(image["width"]) - 1.0)
        y2 = min(y + box_height, float(image["height"]) - 1.0)
        if x2 - x1 < min_box_px or y2 - y1 < min_box_px:
            continue
        by_image.setdefault(image_id, []).append((x1, y1, x2, y2, class_id))

    candidates: list[Record] = []
    for image_id, boxes in by_image.items():
        image = image_info[image_id]
        file_name = image.get("file_name") or f"{image_id:012d}.jpg"
        image_path = images_dir / file_name
        candidates.append(
            Record(
                image_path=image_path,
                depth_path=None,
                boxes=tuple(boxes),
                group=f"coco/{image_id}",
            )
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


def prepare_sample(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    boxes: tuple[tuple[float, float, float, float, int], ...],
    image_size: tuple[int, int],
    train: bool,
    rng: random.Random,
    depth_max_m: float,
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
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.train = train
        self.depth_max_m = depth_max_m
        self.seed = seed
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
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    """Assign each GT box to one free cell across the three output scales."""
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
            assignment: tuple[int, int, int, int] | None = None
            for level in _candidate_levels(width, height):
                stride, prefix = DETECTION_LEVELS[level]
                grid_height, grid_width = outputs[f"{prefix}_cls"].shape[-2:]
                center_col = min(int(center_x / stride), grid_width - 1)
                center_row = min(int(center_y / stride), grid_height - 1)
                candidates = [
                    (row_index, col_index)
                    for row_index in range(
                        max(0, center_row - 2), min(grid_height, center_row + 3)
                    )
                    for col_index in range(
                        max(0, center_col - 2), min(grid_width, center_col + 3)
                    )
                ]
                candidates.sort(
                    key=lambda cell: (
                        ((cell[1] + 0.5) * stride - center_x) ** 2
                        + ((cell[0] + 0.5) * stride - center_y) ** 2
                    )
                )
                for row_index, col_index in candidates:
                    if not positive_masks[level][batch_index, row_index, col_index]:
                        assignment = (level, row_index, col_index, stride)
                        break
                if assignment is not None:
                    break
            if assignment is None:
                raise RuntimeError(
                    "Could not assign a ground-truth box to an unused detector cell; "
                    "try a larger input size or inspect crowded annotations."
                )

            level, row_index, col_index, stride = assignment
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


def _focal_classification_loss(logits: Tensor, targets: Tensor) -> Tensor:
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = 0.25 * targets + 0.75 * (1.0 - targets)
    focal_loss = alpha_t * (1.0 - p_t).square() * cross_entropy
    positive = targets > 0
    negative = ~positive
    positive_loss = (
        focal_loss[positive].mean() if positive.any() else focal_loss.new_zeros(())
    )
    negative_loss = (
        focal_loss[negative].mean() if negative.any() else focal_loss.new_zeros(())
    )
    return positive_loss + negative_loss


def compute_losses(
    outputs: dict[str, Tensor],
    target_depth: Tensor,
    boxes_per_image: list[Tensor],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    class_targets, box_targets, positive_masks = build_detection_targets(
        outputs, boxes_per_image, device
    )
    classification_loss = outputs["p3_cls"].new_zeros(())
    regression_parts: list[Tensor] = []
    for level, (_, prefix) in enumerate(DETECTION_LEVELS):
        logits = outputs[f"{prefix}_cls"]
        classification_loss = classification_loss + _focal_classification_loss(
            logits, class_targets[level]
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
    total = classification_loss + 5.0 * box_loss + depth_loss
    return total, {
        "classification": classification_loss,
        "box": box_loss,
        "depth": depth_loss,
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


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
) -> dict[str, Any]:
    return {
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
    }


def train(args: argparse.Namespace) -> None:
    if args.img_size[0] < 32 or args.img_size[1] < 32:
        raise ValueError("Both image dimensions must be at least 32.")
    if any(value % 32 for value in args.img_size):
        raise ValueError("--img-size dimensions must both be divisible by 32.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if not 0 <= args.coco_ratio < 0.9:
        raise ValueError("--coco-ratio must be between 0 and 0.9.")
    if args.depth_max_m <= 0 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Depth range, epoch count, and batch size must be positive.")

    classes = tuple(name.strip().lower() for name in args.classes if name.strip())
    if not classes or len(set(classes)) != len(classes):
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
    train_records, val_records = split_sun_records(
        sun_records, args.val_fraction, args.seed
    )
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]
    if not train_records or not val_records:
        raise ValueError("Training/validation split is empty after applying sample limits.")

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
        target_count = round(
            len(train_records) * args.coco_ratio / (1 - args.coco_ratio)
        )
        if target_count == 0:
            print("COCO ratio and SUN train size produce zero COCO images; skipping COCO.")
        else:
            coco_records = discover_coco_records(
                coco_root,
                coco_annotations,
                classes,
                max_records=target_count,
                seed=args.seed,
                allow_missing_images=not args.no_download,
            )
            if not args.no_download:
                download_coco_images(coco_records)
            train_records.extend(coco_records)
            print(
                f"Using {len(coco_records)} COCO images (target ratio "
                f"{args.coco_ratio:.0%})."
            )

    class_counts = [0] * len(classes)
    for record in train_records:
        for *_, class_id in record.boxes:
            class_counts[class_id] += 1
    if not any(class_counts):
        raise ValueError(
            f"No training boxes matched classes {classes}. "
            "Check class spelling and SUN RGB-D annotations."
        )
    print(
        "Train/val records:",
        len(train_records),
        "/",
        len(val_records),
        "| boxes:",
        dict(zip(classes, class_counts)),
    )

    train_dataset = MultitaskDataset(
        train_records,
        tuple(args.img_size),
        True,
        args.depth_max_m,
        args.seed,
    )
    val_dataset = MultitaskDataset(
        val_records,
        tuple(args.img_size),
        False,
        args.depth_max_m,
        args.seed,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multitask,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=False, **loader_options
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    model = build_model(
        num_classes=len(classes),
        width_mult=args.width_mult,
        depth_mult=args.depth_mult,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_val_loss = math.inf
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
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        best_val_loss = float(checkpoint["best_val_loss"])
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
                    outputs = model(images)
                    loss, parts = compute_losses(outputs, depths, boxes, device)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite training loss at epoch {epoch + 1}."
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                train_rows.append(
                    {
                        "loss": float(loss.detach()),
                        **{name: float(value.detach()) for name, value in parts.items()},
                    }
                )

        model.eval()
        val_rows: list[dict[str, float]] = []
        depth_abs_error = depth_valid_pixels = 0.0
        with torch.inference_mode():
            val_iterator = iter(val_loader)
            with ActivityIndicator(f"Epoch {epoch + 1}/{args.epochs}: validation") as activity:
                for batch_index in range(len(val_loader)):
                    activity.update(
                        f"Epoch {epoch + 1}/{args.epochs}: "
                        f"loading validation batch {batch_index + 1}/{len(val_loader)}"
                    )
                    images, depths, boxes = next(val_iterator)
                    activity.update(
                        f"Epoch {epoch + 1}/{args.epochs}: "
                        f"validating batch {batch_index + 1}/{len(val_loader)}"
                    )
                    images = images.to(device, non_blocking=True)
                    depths = depths.to(device, non_blocking=True)
                    outputs = model(images)
                    loss, parts = compute_losses(outputs, depths, boxes, device)
                    val_rows.append(
                        {
                            "loss": float(loss),
                            **{name: float(value) for name, value in parts.items()},
                        }
                    )
                    valid = depths > 0
                    if valid.any():
                        depth_abs_error += float(
                            (outputs["depth"][:, 0][valid] - depths[valid])
                            .abs()
                            .sum()
                            .item()
                        )
                        depth_valid_pixels += int(valid.sum().item())
        scheduler.step()
        train_metrics = _mean_metrics(train_rows)
        val_metrics = _mean_metrics(val_rows)
        depth_mae_m = (
            depth_abs_error / depth_valid_pixels * args.depth_max_m
            if depth_valid_pixels
            else float("nan")
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_det_cls={val_metrics['classification']:.4f} "
            f"val_box={val_metrics['box']:.4f} "
            f"val_depth={val_metrics['depth']:.4f} "
            f"val_depth_MAE_m={depth_mae_m:.3f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        completed_epoch = epoch + 1
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
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
        )
        _atomic_torch_save(checkpoint, last_path)
        if is_best:
            _atomic_torch_save(checkpoint, best_path)
            print(f"Saved best checkpoint: {best_path}")
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
        "--no-download",
        action="store_true",
        help="Fail instead of downloading missing SUN RGB-D/COCO data.",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--img-size", nargs=2, type=int, default=(480, 640), metavar=("H", "W"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--depth-max-m", type=float, default=10.0)
    parser.add_argument("--sun-depth-rotation", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--coco-ratio", type=float, default=0.3)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--width-mult", type=float, default=0.25)
    parser.add_argument("--depth-mult", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/arsitektur"))
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    train(_parse_args())
