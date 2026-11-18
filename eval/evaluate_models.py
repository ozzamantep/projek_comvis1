"""Evaluate Baseline (YOLO26nDepth) and Legacy (multitask_best.pt) on the held-out 275-image test split."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arsitektur import build_model  # noqa: E402
from arsitektur_noattn import build_baseline_model  # noqa: E402
from train_arsitektur import (  # noqa: E402
    _collect_detections,
    _evaluate_detection,
    _read_depth_m,
    _split_records_by_group,
    discover_coco_records,
    discover_sun_records,
    prepare_sample,
)

CLASSES = ("person", "laptop", "cell phone", "bottle")
SEED = 42
MAX_DEPTH_M = 10.0


def _compute_depth_metrics(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict[str, float]:
    abs_err_sum = 0.0
    absrel_sum = 0.0
    sq_err_sum = 0.0
    delta1_hits = 0
    valid_pixels = 0

    for pred, gt in zip(preds, gts):
        mask = (gt > 0.0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0.0)
        if not np.any(mask):
            continue
        p = pred[mask].astype(np.float64)
        g = gt[mask].astype(np.float64)
        diff = np.abs(p - g)
        abs_err_sum += float(diff.sum())
        absrel_sum += float((diff / g).sum())
        sq_err_sum += float((diff ** 2).sum())
        ratio = np.maximum(p / g, g / p)
        delta1_hits += int((ratio < 1.25).sum())
        valid_pixels += int(p.size)

    if valid_pixels == 0:
        return {"depth_mae_m": float("nan"), "abs_rel": float("nan"), "rmse_m": float("nan"), "delta1": float("nan"), "valid_pixels": 0}
    return {
        "depth_mae_m": abs_err_sum / valid_pixels,
        "abs_rel": absrel_sum / valid_pixels,
        "rmse_m": float(np.sqrt(sq_err_sum / valid_pixels)),
        "delta1": delta1_hits / valid_pixels,
        "valid_pixels": valid_pixels,
    }


def eval_arsitektur_ckpt(test_records, ckpt_path: Path, device: torch.device, model_builder=build_model, label: str = "Baseline (YOLO26nDepth, best.pt Ep.56)") -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    img_h, img_w = tuple(int(v) for v in ckpt.get("imgsz", (384, 640)))
    depth_max_m = float(ckpt.get("depth_max_m", MAX_DEPTH_M))
    model = model_builder(
        num_classes=len(CLASSES),
        width_mult=float(ckpt.get("width_mult", 0.25)),
        depth_mult=float(ckpt.get("depth_mult", 0.50)),
    ).to(device)
    state_dict = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    predictions = []
    targets = []
    depth_preds = []
    depth_gts = []

    rng = random.Random(SEED)
    with torch.inference_mode():
        for rec in test_records:
            bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gt_depth = (
                _read_depth_m(rec.depth_path, rec.depth_rotation, depth_max_m)
                if rec.depth_path is not None
                else np.zeros(rgb.shape[:2], dtype=np.float32)
            )
            img_t, depth_t, box_t = prepare_sample(
                rgb, gt_depth, rec.boxes, (img_h, img_w),
                train=False, rng=rng, depth_max_m=depth_max_m,
            )
            out = model(img_t.unsqueeze(0).to(device), use_nms=False)
            det_boxes, det_scores, det_classes = _collect_detections(
                out, conf_threshold=0.001, iou_threshold=0.7, max_detections=300
            )[0]
            predictions.append((det_boxes.cpu(), det_scores.cpu(), det_classes.cpu()))

            if box_t.numel():
                targets.append((box_t[:, :4].clone(), box_t[:, 4].long().clone()))
            else:
                targets.append((torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)))

            if rec.depth_path is not None:
                pred_depth = out["depth"][0, 0].detach().cpu().numpy() * depth_max_m
                gt_canvas = depth_t.numpy() * depth_max_m
                depth_preds.append(pred_depth)
                depth_gts.append(gt_canvas)

    det_metrics = _evaluate_detection(predictions, targets, len(CLASSES), CLASSES)
    depth_metrics = _compute_depth_metrics(depth_preds, depth_gts)
    return {
        "model": label,
        "img_size": f"{img_h}x{img_w}",
        **det_metrics,
        **depth_metrics,
    }


def eval_legacy(test_records, ckpt_path: Path) -> dict:
    from multitask_pt import load_multitask_torch_model
    mt = load_multitask_torch_model(ckpt_path)
    # Note: multitask_best.pt class order is ['person', 'laptop', 'cell phone', 'bottle']
    predictions = []
    targets = []
    depth_preds = []
    depth_gts = []

    for rec in test_records:
        bgr = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        dets, depth_m = mt.predict(bgr)
        if dets.shape[0] > 0:
            boxes = torch.from_numpy(dets[:, :4].astype(np.float32))
            scores = torch.from_numpy(dets[:, 4].astype(np.float32))
            cls_ids = torch.from_numpy(dets[:, 5].astype(np.int64))
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            scores = torch.zeros((0,), dtype=torch.float32)
            cls_ids = torch.zeros((0,), dtype=torch.int64)
        predictions.append((boxes, scores, cls_ids))

        if rec.boxes:
            gt_arr = np.array(rec.boxes, dtype=np.float32)
            targets.append((torch.from_numpy(gt_arr[:, :4]), torch.from_numpy(gt_arr[:, 4].astype(np.int64))))
        else:
            targets.append((torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)))

        if rec.depth_path is not None:
            gt_depth = _read_depth_m(rec.depth_path, rec.depth_rotation, MAX_DEPTH_M)
            depth_preds.append(depth_m)
            depth_gts.append(gt_depth)

    det_metrics = _evaluate_detection(predictions, targets, len(CLASSES), CLASSES)
    depth_metrics = _compute_depth_metrics(depth_preds, depth_gts)
    return {
        "model": "Legacy Runtime (multitask_best.pt, Ep.10)",
        "img_size": f"{mt.input_height}x{mt.input_width}",
        **det_metrics,
        **depth_metrics,
    }


def main() -> None:
    eval_dir = ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Evaluating on held-out test split: {len(test_records)} images ({len(sun_te)} SUN + {len(coco_te)} COCO)")

    plus_attn_res = eval_arsitektur_ckpt(
        test_records,
        ROOT / "training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt",
        device,
        model_builder=build_model,
        label="+Attention (YOLO26nDepth, best.pt Ep.56)",
    )
    print("+Attention result:", json.dumps(plus_attn_res, indent=2))

    cached_path = eval_dir / "evaluation_results.json"
    cached_legacy = None
    if cached_path.is_file():
        try:
            cached_legacy = json.loads(cached_path.read_text(encoding="utf-8")).get("legacy")
        except Exception:
            cached_legacy = None
    try:
        legacy_res = eval_legacy(test_records, ROOT / "multitask_best.pt")
        print("Legacy result:", json.dumps(legacy_res, indent=2))
    except Exception as exc:
        if cached_legacy is not None:
            print(f"Using cached legacy result ({exc})")
            legacy_res = cached_legacy
        else:
            raise

    results = {"plus_attention": plus_attn_res, "legacy": legacy_res}
    noattn_ckpt = ROOT / "training_runs/persist/rtx4050_noattn_70_20_10/best.pt"
    base_res = None
    if noattn_ckpt.is_file():
        base_res = eval_arsitektur_ckpt(
            test_records,
            noattn_ckpt,
            device,
            model_builder=build_baseline_model,
            label="Baseline (Tanpa Attention, NoAttn)",
        )
        results["baseline_noattn"] = base_res
        print("Baseline (NoAttn) result:", json.dumps(base_res, indent=2))

    # Save JSON
    with open(eval_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save accuracy_comparison.csv
    with open(eval_dir / "accuracy_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Input", "mAP@0.5 (%)", "mAP@0.5:0.95 (%)", "Precision (%)", "Recall (%)",
                         "Depth MAE (m)", "AbsRel", "RMSE (m)", "delta1 (%)"])
        if base_res is None:
            writer.writerow([
                "Baseline (Tanpa Attention, NoAttn)", "384x640",
                "sedang dilatih", "sedang dilatih", "sedang dilatih", "sedang dilatih",
                "sedang dilatih", "sedang dilatih", "sedang dilatih", "sedang dilatih",
            ])
        rows_to_write = ([base_res] if base_res else []) + [plus_attn_res, legacy_res]
        for r in rows_to_write:
            writer.writerow([
                r["model"], r["img_size"],
                f"{r['map50']*100:.2f}", f"{r['map50_95']*100:.2f}",
                f"{r['precision']*100:.2f}", f"{r['recall']*100:.2f}",
                f"{r['depth_mae_m']:.3f}", f"{r['abs_rel']:.3f}",
                f"{r['rmse_m']:.3f}", f"{r['delta1']*100:.2f}",
            ])

    # Save per_class_ap.csv
    with open(eval_dir / "per_class_ap.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Kelas", "Baseline (NoAttn) AP@0.5 (%)", "+Attention AP@0.5 (%)", "Delta", "Legacy AP@0.5 (%)"])
        for cls in CLASSES:
            a_ap = plus_attn_res["per_class_ap50"].get(cls, 0.0) * 100
            l_ap = legacy_res["per_class_ap50"].get(cls, 0.0) * 100
            if base_res is not None:
                b_ap = base_res["per_class_ap50"].get(cls, 0.0) * 100
                writer.writerow([cls, f"{b_ap:.2f}", f"{a_ap:.2f}", f"{a_ap - b_ap:+.2f}", f"{l_ap:.2f}"])
            else:
                writer.writerow([cls, "sedang dilatih", f"{a_ap:.2f}", "menunggu", f"{l_ap:.2f}"])

    print("Saved eval/evaluation_results.json, eval/accuracy_comparison.csv, eval/per_class_ap.csv")


if __name__ == "__main__":
    main()
