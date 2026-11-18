"""Webcam side-by-side capture tool: runs Baseline (NoAttn) and +Attention on the same frame.

Usage example:
    python scripts/capture_compare.py --scenario jarak_0.5m_terang_person --actual-dist 0.50
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from architecture_pt import ArchitectureTorchModel, load_architecture_torch_model  # noqa: E402
from arsitektur_noattn import build_baseline_model  # noqa: E402
from main import (  # noqa: E402
    DepthEMASmoother,
    _draw_detection_overlay,
    calibrate_depth_map,
    get_depth_in_bbox,
    normalize_depth_for_display,
)
from multitask_pt import load_multitask_torch_model  # noqa: E402


def _load_any_model(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and ckpt.get("kind") == "arsitektur_multitask":
        state = ckpt.get("ema_state_dict") or ckpt["model_state_dict"]
        # Auto-detect Baseline (NoAttn) vs +Attention (YOLO26nDepth) from state_dict keys
        has_l10 = any(k.startswith("backbone.l10.") for k in state)
        if not has_l10:
            model = build_baseline_model(
                num_classes=len(ckpt.get("classes", ("person", "laptop", "cell phone", "bottle"))),
                width_mult=float(ckpt.get("width_mult", 0.25)),
                depth_mult=float(ckpt.get("depth_mult", 0.50)),
            )
            model.load_state_dict(state, strict=True)
            img_h, img_w = tuple(int(v) for v in ckpt.get("imgsz", (384, 640)))
            return ArchitectureTorchModel(
                model=model,
                class_names=tuple(ckpt.get("classes", ("person", "laptop", "cell phone", "bottle"))),
                input_height=img_h,
                input_width=img_w,
                depth_max_m=float(ckpt.get("depth_max_m", 10.0)),
            )
        return load_architecture_torch_model(path)
    return load_multitask_torch_model(path)


def _validate_scenario_distance(scenario: str, actual_dist: float | None) -> None:
    match = re.search(r"jarak_([0-9]+(?:\.[0-9]+)?)m", scenario.lower())
    if match and actual_dist is not None:
        expected = float(match.group(1))
        if abs(expected - actual_dist) > 0.01:
            print(
                f"[PERINGATAN] Jarak pada nama skenario ({expected:.2f} m di '{scenario}') "
                f"tidak cocok dengan --actual-dist ({actual_dist:.2f} m)!"
            )


def _extract_top_prediction(
    dets: np.ndarray,
    depth_cal: np.ndarray,
    class_names: tuple[str, ...],
    conf: float,
) -> tuple[str, float | None]:
    if dets is None or len(dets) == 0:
        return "", None
    valid = dets[dets[:, 4] >= conf]
    if len(valid) == 0:
        return "", None
    best = valid[int(np.argmax(valid[:, 4]))]
    cls_idx = int(best[5])
    cls_name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else str(cls_idx)
    try:
        dist_m = float(get_depth_in_bbox(depth_cal, tuple(float(v) for v in best[:4]), 15))
    except ValueError:
        dist_m = None
    return cls_name, dist_m


def _next_repetition(csv_path: Path, scenario: str) -> int:
    if not csv_path.is_file():
        return 1
    count = 0
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("skenario") == scenario:
                count += 1
    return count + 1


def _append_measurement_csv(
    csv_path: Path,
    scenario: str,
    rep: int,
    actual_dist: float | None,
    pred_a: float | None,
    pred_b: float | None,
    cls_a: str,
    cls_b: str,
    filename: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()
    fieldnames = [
        "skenario",
        "ulangan",
        "jarak_aktual_m",
        "prediksi_baseline_m",
        "prediksi_attention_m",
        "error_baseline_m",
        "error_attention_m",
        "kelas_baseline",
        "kelas_attention",
        "nama_file",
        "timestamp",
    ]
    err_a = f"{abs(pred_a - actual_dist):.4f}" if (actual_dist is not None and pred_a is not None) else ""
    err_b = f"{abs(pred_b - actual_dist):.4f}" if (actual_dist is not None and pred_b is not None) else ""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "skenario": scenario,
            "ulangan": rep,
            "jarak_aktual_m": f"{actual_dist:.4f}" if actual_dist is not None else "",
            "prediksi_baseline_m": f"{pred_a:.4f}" if pred_a is not None else "",
            "prediksi_attention_m": f"{pred_b:.4f}" if pred_b is not None else "",
            "error_baseline_m": err_a,
            "error_attention_m": err_b,
            "kelas_baseline": cls_a,
            "kelas_attention": cls_b,
            "nama_file": filename,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        })


def _render_panel(
    frame: np.ndarray,
    model,
    smoother: DepthEMASmoother,
    title: str,
    conf: float,
    scale: float,
    offset_m: float,
    show_depth_strip: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dets, depth_m = model.predict(frame)
    depth_cal = calibrate_depth_map(smoother.update(depth_m), scale=scale, offset_m=offset_m)
    rgb_ann = _draw_detection_overlay(
        frame.copy(), dets, model.class_names, depth_cal, 15, 10.0, conf, None, False, "m"
    )
    depth_color = cv2.applyColorMap(normalize_depth_for_display(depth_cal), cv2.COLORMAP_INFERNO)
    depth_ann = _draw_detection_overlay(
        depth_color, dets, model.class_names, depth_cal, 15, 10.0, conf, None, False, "m"
    )
    cv2.putText(rgb_ann, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    panel = np.vstack((rgb_ann, depth_ann)) if show_depth_strip else rgb_ann
    return panel, dets, depth_cal


def main() -> None:
    parser = argparse.ArgumentParser(description="Side-by-side webcam capture for Baseline (NoAttn) vs +Attention.")
    parser.add_argument(
        "--model-a",
        type=Path,
        default=ROOT / "training_runs/persist/rtx4050_noattn_70_20_10/best.pt",
        help="Path to Baseline (NoAttn) checkpoint.",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        default=ROOT / "training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt",
        help="Path to +Attention checkpoint.",
    )
    parser.add_argument("--label-a", default="Baseline")
    parser.add_argument("--label-b", default="+Attention")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--scenario", default="uji_realtime")
    parser.add_argument("--actual-dist", type=float, default=None, help="Jarak aktual meteran (meter).")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures" / "webcam_captures")
    parser.add_argument("--csv-path", type=Path, default=ROOT / "eval" / "webcam_measurements.csv")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--offset-m", type=float, default=0.0)
    args = parser.parse_args()

    _validate_scenario_distance(args.scenario, args.actual_dist)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_a = _load_any_model(args.model_a)
    model_b = _load_any_model(args.model_b)
    smoother_a = DepthEMASmoother(alpha=0.35)
    smoother_b = DepthEMASmoother(alpha=0.35)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak dapat membuka kamera indeks {args.camera}.")

    show_depth = True
    save_idx = 0
    print("Kontrol: [SPACE/c] Simpan screenshot | [s] Toggle depth map | [q] Keluar")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            panel_a, dets_a, depth_a = _render_panel(
                frame, model_a, smoother_a, args.label_a, args.conf, args.scale, args.offset_m, show_depth
            )
            panel_b, dets_b, depth_b = _render_panel(
                frame, model_b, smoother_b, args.label_b, args.conf, args.scale, args.offset_m, show_depth
            )
            combined = np.hstack((panel_a, panel_b))

            if args.actual_dist is not None:
                banner = f"Skenario: {args.scenario} | Jarak Meteran Aktual: {args.actual_dist:.2f} m"
            else:
                banner = f"Skenario: {args.scenario} | Jarak Meteran: belum diisi (--actual-dist)"
            cv2.rectangle(combined, (0, combined.shape[0] - 34), (combined.shape[1], combined.shape[0]), (15, 23, 42), -1)
            cv2.putText(
                combined, banner, (12, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow("Compare: Baseline vs +Attention", combined)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_depth = not show_depth
            elif key in (ord(" "), ord("c")):
                save_idx += 1
                rep = _next_repetition(args.csv_path, args.scenario)
                fname = f"{args.scenario}_{save_idx:02d}_{int(time.time())}.png"
                out_path = args.out_dir / fname
                cv2.imwrite(str(out_path), combined)
                cls_a, pred_a = _extract_top_prediction(dets_a, depth_a, model_a.class_names, args.conf)
                cls_b, pred_b = _extract_top_prediction(dets_b, depth_b, model_b.class_names, args.conf)
                _append_measurement_csv(
                    args.csv_path,
                    args.scenario,
                    rep,
                    args.actual_dist,
                    pred_a,
                    pred_b,
                    cls_a,
                    cls_b,
                    fname,
                )
                print(
                    f"Tersimpan: {out_path} | ulangan={rep} | "
                    f"pred_base={pred_a} ({cls_a}) | pred_attn={pred_b} ({cls_b}) -> {args.csv_path}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
