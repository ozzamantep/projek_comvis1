"""Render qualitative samples from the held-out 275-image test split (4 classes + failure cases)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from architecture_pt import load_architecture_torch_model  # noqa: E402
from main import (  # noqa: E402
    _draw_detection_overlay,
    get_depth_in_bbox,
    normalize_depth_for_display,
)
from multitask_pt import load_multitask_torch_model  # noqa: E402
from train_arsitektur import (  # noqa: E402
    _read_depth_m,
    _split_records_by_group,
    discover_coco_records,
    discover_sun_records,
)

CLASSES = ("person", "laptop", "cell phone", "bottle")
SEED = 42
MAX_DEPTH_M = 10.0


def _make_side_by_side(
    bgr: np.ndarray,
    dets: np.ndarray,
    class_names: tuple[str, ...],
    depth_m: np.ndarray,
    title: str,
    gt_boxes: tuple[tuple[float, float, float, float, int], ...],
    gt_depth: np.ndarray | None,
) -> np.ndarray:
    ann = _draw_detection_overlay(
        bgr.copy(), dets, class_names, depth_m, 15, MAX_DEPTH_M, 0.25, None, False, "m"
    )
    depth_color = cv2.applyColorMap(normalize_depth_for_display(depth_m), cv2.COLORMAP_INFERNO)
    depth_ann = _draw_detection_overlay(
        depth_color, dets, class_names, depth_m, 15, MAX_DEPTH_M, 0.25, None, False, "m"
    )
    # Resize to standard 480x640 panel height if needed
    target_h = 420
    scale = target_h / ann.shape[0]
    target_w = int(round(ann.shape[1] * scale))
    ann_r = cv2.resize(ann, (target_w, target_h))
    dep_r = cv2.resize(depth_ann, (target_w, target_h))
    panel = np.hstack((ann_r, dep_r))

    # Header bar
    header = np.full((42, panel.shape[1], 3), (15, 23, 42), dtype=np.uint8)
    gt_info = f"GT boxes: {len(gt_boxes)}"
    if gt_depth is not None and np.any(gt_depth > 0):
        gt_med = float(np.median(gt_depth[gt_depth > 0]))
        gt_info += f" | GT median depth: {gt_med:.2f} m"
    cv2.putText(header, f"{title} ({gt_info})", (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack((header, panel))


def main() -> None:
    out_dir = ROOT / "docs" / "figures" / "testset_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    sun_records = discover_sun_records(
        root=ROOT / "datasets" / "SUNRGBD",
        classes=CLASSES,
        max_depth_m=MAX_DEPTH_M,
        depth_rotation="auto",
        min_box_px=4.0,
    )
    coco_records = discover_coco_records(
        root=ROOT / "datasets" / "coco",
        annotation_path=None,
        classes=CLASSES,
        min_box_px=4.0,
        max_records=None,
        seed=SEED,
        allow_missing_images=False,
        local_only=True,
    )
    _, _, sun_te = _split_records_by_group(sun_records, 0.2, 0.1, SEED, "SUN RGB-D")
    _, _, coco_te = _split_records_by_group(coco_records, 0.2, 0.1, SEED, "COCO")
    test_records = sun_te + coco_te

    base_model = load_architecture_torch_model(ROOT / "training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt")
    legacy_model = load_multitask_torch_model(ROOT / "multitask_best.pt")

    selected: dict[str, dict] = {}
    summary = []

    # Find representative sample for each class (prefer SUN RGB-D where available so GT depth exists)
    for cls_idx, cls_name in enumerate(CLASSES):
        candidates = [r for r in sun_te if any(b[4] == cls_idx for b in r.boxes)]
        if not candidates:
            candidates = [r for r in coco_te if any(b[4] == cls_idx for b in r.boxes)]
        best_rec = None
        best_dets = None
        best_depth = None
        best_score = -1.0
        for rec in candidates[:40]:
            bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            dets, depth_m = base_model.predict(bgr)
            cls_dets = dets[(dets[:, 5] == cls_idx) & (dets[:, 4] >= 0.25)] if len(dets) else np.zeros((0, 6))
            if len(cls_dets) > 0:
                top_conf = float(cls_dets[:, 4].max())
                if top_conf > best_score:
                    best_score = top_conf
                    best_rec = rec
                    best_dets = dets
                    best_depth = depth_m
        if best_rec is None and candidates:
            best_rec = candidates[0]
            bgr = cv2.imread(str(best_rec.image_path), cv2.IMREAD_COLOR)
            best_dets, best_depth = base_model.predict(bgr)

        if best_rec is not None:
            bgr = cv2.imread(str(best_rec.image_path), cv2.IMREAD_COLOR)
            gt_depth = _read_depth_m(best_rec.depth_path, best_rec.depth_rotation, MAX_DEPTH_M) if best_rec.depth_path else None
            # Compute predicted vs GT median depth for first GT box of this class if SUN RGB-D
            gt_cls_boxes = [b for b in best_rec.boxes if b[4] == cls_idx]
            gt_box_dist = None
            pred_box_dist = None
            if gt_depth is not None and gt_cls_boxes:
                try:
                    gt_box_dist = round(get_depth_in_bbox(gt_depth, gt_cls_boxes[0][:4], 15), 3)
                except ValueError:
                    pass
                try:
                    pred_box_dist = round(get_depth_in_bbox(best_depth, gt_cls_boxes[0][:4], 15), 3)
                except ValueError:
                    pass

            safe_cls = cls_name.replace(" ", "_")
            img_base = _make_side_by_side(
                bgr, best_dets, base_model.class_names, best_depth,
                f"Test Split: Kelas {cls_name} - Baseline (YOLO26nDepth)", best_rec.boxes, gt_depth
            )
            out_path = out_dir / f"testset_{safe_cls}_baseline.png"
            cv2.imwrite(str(out_path), img_base)

            # Also render legacy multitask_best.pt on the exact same image for visual comparison
            l_dets, l_depth = legacy_model.predict(bgr)
            img_leg = _make_side_by_side(
                bgr, l_dets, legacy_model.class_names, l_depth,
                f"Test Split: Kelas {cls_name} - Legacy (multitask_best.pt)", best_rec.boxes, gt_depth
            )
            out_leg = out_dir / f"testset_{safe_cls}_legacy.png"
            cv2.imwrite(str(out_leg), img_leg)

            summary.append({
                "scenario": f"class_{cls_name}",
                "image": str(best_rec.image_path.name),
                "source": "SUN RGB-D" if best_rec.depth_path else "COCO",
                "gt_box_depth_m": gt_box_dist,
                "pred_box_depth_m": pred_box_dist,
                "baseline_file": str(out_path.relative_to(ROOT)),
                "legacy_file": str(out_leg.relative_to(ROOT)),
            })

    # Failure case 1: Missed small cell phone (GT has cell phone, Baseline conf>=0.25 misses it)
    for rec in coco_te:
        if any(b[4] == 2 for b in rec.boxes):
            bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            dets, depth_m = base_model.predict(bgr)
            phone_hits = dets[(dets[:, 5] == 2) & (dets[:, 4] >= 0.25)] if len(dets) else np.zeros((0, 6))
            if len(phone_hits) == 0:
                img_fail = _make_side_by_side(
                    bgr, dets, base_model.class_names, depth_m,
                    "Kasus Gagal: Small Object Missed (cell phone)", rec.boxes, None
                )
                out_fail = out_dir / "testset_fail_small_cellphone.png"
                cv2.imwrite(str(out_fail), img_fail)
                summary.append({
                    "scenario": "fail_small_cellphone",
                    "image": rec.image_path.name,
                    "source": "COCO",
                    "file": str(out_fail.relative_to(ROOT)),
                })
                break

    # Failure case 2: Multi-object / clutter / occlusion in SUN RGB-D (>=3 GT boxes)
    for rec in sun_te:
        if len(rec.boxes) >= 3:
            bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            dets, depth_m = base_model.predict(bgr)
            gt_depth = _read_depth_m(rec.depth_path, rec.depth_rotation, MAX_DEPTH_M)
            img_multi = _make_side_by_side(
                bgr, dets, base_model.class_names, depth_m,
                "Kasus Uji: Multi-Objek & Clutter (SUN RGB-D)", rec.boxes, gt_depth
            )
            out_multi = out_dir / "testset_multi_object_clutter.png"
            cv2.imwrite(str(out_multi), img_multi)
            summary.append({
                "scenario": "multi_object_clutter",
                "image": rec.image_path.name,
                "source": "SUN RGB-D",
                "gt_boxes": len(rec.boxes),
                "file": str(out_multi.relative_to(ROOT)),
            })
            break

    with open(out_dir / "samples_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Rendered test set samples:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

