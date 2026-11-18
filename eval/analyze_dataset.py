"""Analyze SUN RGB-D + COCO dataset statistics using train_arsitektur.py loaders."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_arsitektur import (  # noqa: E402
    _read_depth_m,
    _split_records_by_group,
    discover_coco_records,
    discover_sun_records,
)

CLASSES = ("person", "laptop", "cell phone", "bottle")
MAX_DEPTH_M = 10.0
SEED = 42


def _sun_sensor_subset(image_path: Path) -> str:
    parts = image_path.parts
    for i, part in enumerate(parts):
        if part.lower() == "sunrgbd" and i + 2 < len(parts):
            return f"{parts[i + 1]}/{parts[i + 2]}"
    return "unknown"


def _resolution_stats(records) -> dict:
    dims = []
    for rec in records:
        img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        dims.append((w, h))
    counts = Counter(dims)
    areas = [w * h for w, h in dims]
    min_dim = dims[int(np.argmin(areas))] if dims else (0, 0)
    max_dim = dims[int(np.argmax(areas))] if dims else (0, 0)
    mode_dim, mode_count = counts.most_common(1)[0] if counts else ((0, 0), 0)
    return {
        "count": len(dims),
        "min_wh": list(min_dim),
        "max_wh": list(max_dim),
        "mode_wh": list(mode_dim),
        "mode_count": mode_count,
        "unique_resolutions": {f"{w}x{h}": c for (w, h), c in counts.most_common(10)},
    }


def main() -> None:
    eval_dir = ROOT / "eval"
    fig_dir = ROOT / "docs" / "figures"
    eval_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sun_root = ROOT / "datasets" / "SUNRGBD"
    coco_root = ROOT / "datasets" / "coco"

    print("Discovering SUN RGB-D records...")
    sun_records = discover_sun_records(
        root=sun_root,
        classes=CLASSES,
        max_depth_m=MAX_DEPTH_M,
        depth_rotation="auto",
        min_box_px=4.0,
    )
    print(f"Total SUN RGB-D records: {len(sun_records)}")

    print("Discovering local COCO records...")
    coco_records = discover_coco_records(
        root=coco_root,
        annotation_path=None,
        classes=CLASSES,
        min_box_px=4.0,
        max_records=None,
        seed=SEED,
        allow_missing_images=False,
        local_only=True,
    )
    print(f"Total COCO records: {len(coco_records)}")

    sun_tr, sun_va, sun_te = _split_records_by_group(
        sun_records, val_fraction=0.2, test_fraction=0.1, seed=SEED, dataset_name="SUN RGB-D"
    )
    coco_tr, coco_va, coco_te = _split_records_by_group(
        coco_records, val_fraction=0.2, test_fraction=0.1, seed=SEED, dataset_name="COCO"
    )

    # Sensor subsets in SUN RGB-D
    sensor_counts = Counter(_sun_sensor_subset(r.image_path) for r in sun_records)

    # Class box counts per source & split
    def _count_boxes(recs):
        c = Counter()
        for r in recs:
            for b in r.boxes:
                c[CLASSES[b[4]]] += 1
        return {cls: c[cls] for cls in CLASSES}

    splits_summary = {
        "train": {
            "sun_rgbd": len(sun_tr),
            "coco": len(coco_tr),
            "total": len(sun_tr) + len(coco_tr),
            "sun_boxes": _count_boxes(sun_tr),
            "coco_boxes": _count_boxes(coco_tr),
            "total_boxes": _count_boxes(sun_tr + coco_tr),
        },
        "validation": {
            "sun_rgbd": len(sun_va),
            "coco": len(coco_va),
            "total": len(sun_va) + len(coco_va),
            "sun_boxes": _count_boxes(sun_va),
            "coco_boxes": _count_boxes(coco_va),
            "total_boxes": _count_boxes(sun_va + coco_va),
        },
        "test": {
            "sun_rgbd": len(sun_te),
            "coco": len(coco_te),
            "total": len(sun_te) + len(coco_te),
            "sun_boxes": _count_boxes(sun_te),
            "coco_boxes": _count_boxes(coco_te),
            "total_boxes": _count_boxes(sun_te + coco_te),
        },
    }

    # Resolution stats
    print("Computing image resolution stats...")
    sun_res = _resolution_stats(sun_records)
    coco_res = _resolution_stats(coco_records)

    # Ground-truth depth stats across all SUN RGB-D records (subsampled pixels for exact percentiles/histogram)
    print("Computing SUN RGB-D ground-truth depth statistics...")
    rng = np.random.default_rng(SEED)
    sampled_depths = []
    img_means = []
    valid_pixel_total = 0
    total_pixel_total = 0
    global_min = float("inf")
    global_max = 0.0
    depth_sum = 0.0

    for rec in sun_records:
        depth_m = _read_depth_m(rec.depth_path, rec.depth_rotation, MAX_DEPTH_M)
        valid = depth_m[depth_m > 0]
        total_pixel_total += depth_m.size
        if valid.size == 0:
            continue
        valid_pixel_total += valid.size
        vmin = float(valid.min())
        vmax = float(valid.max())
        if vmin < global_min:
            global_min = vmin
        if vmax > global_max:
            global_max = vmax
        depth_sum += float(valid.sum())
        img_means.append(float(valid.mean()))
        # Subsample 2000 pixels per image for memory-safe percentile & histogram
        if valid.size > 2000:
            idx = rng.choice(valid.size, size=2000, replace=False)
            sampled_depths.append(valid[idx])
        else:
            sampled_depths.append(valid)

    all_sample = np.concatenate(sampled_depths)
    depth_stats = {
        "valid_pixels": int(valid_pixel_total),
        "total_pixels": int(total_pixel_total),
        "valid_ratio": round(valid_pixel_total / max(1, total_pixel_total), 4),
        "cap_m": MAX_DEPTH_M,
        "min_m": round(global_min, 4),
        "max_m": round(global_max, 4),
        "mean_m": round(depth_sum / max(1, valid_pixel_total), 4),
        "median_m": round(float(np.median(all_sample)), 4),
        "p2_m": round(float(np.percentile(all_sample, 2)), 4),
        "p98_m": round(float(np.percentile(all_sample, 98)), 4),
        "std_m": round(float(np.std(all_sample)), 4),
    }

    # Plot histogram of GT depth
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
    ax.hist(
        all_sample,
        bins=50,
        range=(0.0, MAX_DEPTH_M),
        color="#2563EB",
        edgecolor="#1E3A8A",
        alpha=0.82,
        density=True,
    )
    ax.axvline(depth_stats["mean_m"], color="#DC2626", linestyle="--", linewidth=1.6,
               label=f"Mean = {depth_stats['mean_m']:.2f} m")
    ax.axvline(depth_stats["median_m"], color="#059669", linestyle="-.", linewidth=1.6,
               label=f"Median = {depth_stats['median_m']:.2f} m")
    ax.axvspan(depth_stats["p2_m"], depth_stats["p98_m"], color="#93C5FD", alpha=0.22,
               label=f"P2–P98 = {depth_stats['p2_m']:.2f}–{depth_stats['p98_m']:.2f} m")
    ax.set_title("Distribusi Kedalaman Ground Truth SUN RGB-D (Cap 10,0 m)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Jarak Kedalaman (meter)", fontsize=10)
    ax.set_ylabel("Kepadatan Probabilitas (Density)", fontsize=10)
    ax.set_xlim(0, MAX_DEPTH_M)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(fig_dir / "hist_depth_gt.png", dpi=300)
    fig.savefig(fig_dir / "hist_depth_gt.svg")
    plt.close(fig)

    # Save CSV distribution per class (template style)
    total_all_boxes = sum(
        splits_summary[sp]["total_boxes"][c] for sp in ("train", "validation", "test") for c in CLASSES
    )
    csv_path = eval_dir / "dataset_distribution.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Kelas", "Sumber", "Train", "Validation", "Test", "Total", "Proporsi (%)"])
        for cls in CLASSES:
            tr_sun = splits_summary["train"]["sun_boxes"][cls]
            tr_coco = splits_summary["train"]["coco_boxes"][cls]
            va_sun = splits_summary["validation"]["sun_boxes"][cls]
            va_coco = splits_summary["validation"]["coco_boxes"][cls]
            te_sun = splits_summary["test"]["sun_boxes"][cls]
            te_coco = splits_summary["test"]["coco_boxes"][cls]
            tr_t = tr_sun + tr_coco
            va_t = va_sun + va_coco
            te_t = te_sun + te_coco
            tot = tr_t + va_t + te_t
            prop = 100.0 * tot / max(1, total_all_boxes)
            writer.writerow([
                cls,
                f"SUN RGB-D ({tr_sun+va_sun+te_sun}) + COCO ({tr_coco+va_coco+te_coco})",
                tr_t,
                va_t,
                te_t,
                tot,
                f"{prop:.2f}",
            ])
        tr_tot = sum(splits_summary["train"]["total_boxes"].values())
        va_tot = sum(splits_summary["validation"]["total_boxes"].values())
        te_tot = sum(splits_summary["test"]["total_boxes"].values())
        writer.writerow(["Total Kotak (Bbox)", "SUN RGB-D + COCO", tr_tot, va_tot, te_tot, total_all_boxes, "100.00"])
        writer.writerow([
            "Total Citra",
            f"SUN RGB-D ({len(sun_records)}) + COCO ({len(coco_records)})",
            splits_summary["train"]["total"],
            splits_summary["validation"]["total"],
            splits_summary["test"]["total"],
            len(sun_records) + len(coco_records),
            "100.00",
        ])

    summary = {
        "classes": list(CLASSES),
        "sun_total_records": len(sun_records),
        "coco_total_records": len(coco_records),
        "grand_total_records": len(sun_records) + len(coco_records),
        "sun_sensor_subsets": dict(sorted(sensor_counts.items())),
        "splits": splits_summary,
        "resolutions": {
            "sun_rgbd": sun_res,
            "coco": coco_res,
        },
        "depth_gt_stats_m": depth_stats,
    }
    with open(eval_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved eval/dataset_stats.json, eval/dataset_distribution.csv, docs/figures/hist_depth_gt.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
