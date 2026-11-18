"""Benchmark efficiency (Params, GFLOPs, Size MB, CPU/GPU Latency, FPS) for Baseline, +CBAM, Ablation, and Legacy."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arsitektur import YOLO26nDepth  # noqa: E402
from arsitektur_attention import CBAMBlock, YOLO26nDepthCBAM  # noqa: E402
from multitask_pt import load_multitask_torch_model  # noqa: E402


class YOLO26nDepthCBAMNeckOnly(YOLO26nDepth):
    """Ablation variant: +CBAM only on Neck outputs (P3/P4/P5)."""

    def __init__(self) -> None:
        super().__init__()
        self.cbam_p3 = CBAMBlock(64, reduction=16)
        self.cbam_p4 = CBAMBlock(128, reduction=16)
        self.cbam_p5 = CBAMBlock(256, reduction=16)

    def forward(self, x: torch.Tensor, use_nms: bool = True):
        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)
        p3_att = self.cbam_p3(p3_out)
        p4_att = self.cbam_p4(p4_out)
        p5_att = self.cbam_p5(p5_out)
        p3_reg, p3_cls = self.detect_p3(p3_att)
        p4_reg, p4_cls = self.detect_p4(p4_att)
        p5_reg, p5_cls = self.detect_p5(p5_att)
        depth_map = self.depth_head((p3_att, p4_att, p5_att), x.shape[-2:])
        return {
            "p3_bbox": p3_reg, "p3_cls": p3_cls,
            "p4_bbox": p4_reg, "p4_cls": p4_cls,
            "p5_bbox": p5_reg, "p5_cls": p5_cls,
            "depth": depth_map, "nms_applied": use_nms,
        }


class YOLO26nDepthCBAMDepthOnly(YOLO26nDepth):
    """Ablation variant: +CBAM only on DepthHead fusion."""

    def __init__(self) -> None:
        super().__init__()
        self.cbam_depth = CBAMBlock(32 * 3, reduction=16)

    def forward(self, x: torch.Tensor, use_nms: bool = True):
        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)
        p3_reg, p3_cls = self.detect_p3(p3_out)
        p4_reg, p4_cls = self.detect_p4(p4_out)
        p5_reg, p5_cls = self.detect_p5(p5_out)
        target_size = x.shape[-2:]
        up_p3 = torch.nn.functional.interpolate(
            self.depth_head.conv_p3(p3_out), size=target_size, mode="bilinear", align_corners=False
        )
        up_p4 = torch.nn.functional.interpolate(
            self.depth_head.conv_p4(p4_out), size=target_size, mode="bilinear", align_corners=False
        )
        up_p5 = torch.nn.functional.interpolate(
            self.depth_head.conv_p5(p5_out), size=target_size, mode="bilinear", align_corners=False
        )
        fused = self.cbam_depth(torch.cat((up_p3, up_p4, up_p5), dim=1))
        depth_map = self.depth_head.refine(fused)
        return {
            "p3_bbox": p3_reg, "p3_cls": p3_cls,
            "p4_bbox": p4_reg, "p4_cls": p4_cls,
            "p5_bbox": p5_reg, "p5_cls": p5_cls,
            "depth": depth_map, "nms_applied": use_nms,
        }


def _measure_flops(fn, x: torch.Tensor) -> float:
    with FlopCounterMode(display=False) as fc:
        with torch.inference_mode():
            _ = fn(x)
    return fc.get_total_flops() / 1e9


def _measure_latency(fn, x_cpu: torch.Tensor, x_gpu: torch.Tensor | None, warmup: int = 50, iters: int = 200):
    # CPU
    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn(x_cpu)
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x_cpu)
        cpu_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return cpu_ms


def _measure_gpu_latency(model_move_fn, fn_factory, x_gpu: torch.Tensor, warmup: int = 50, iters: int = 200) -> float:
    if not torch.cuda.is_available():
        return float("nan")
    model_move_fn(torch.device("cuda"))
    fn = fn_factory()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = fn(x_gpu)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fn(x_gpu)
        torch.cuda.synchronize()
        gpu_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return gpu_ms


def _state_dict_mb(state_dict) -> float:
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tmp_path = tf.name
    try:
        torch.save(state_dict, tmp_path)
        return os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _get_cpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True,
            timeout=5,
        ).strip()
        return out or platform.processor()
    except Exception:
        return platform.processor()


def main() -> None:
    eval_dir = ROOT / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_info = {
        "os": platform.platform(),
        "cpu": _get_cpu_name(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "opencv": cv2.__version__,
        "warmup_iters": 50,
        "measure_iters": 200,
        "batch_size": 1,
    }
    print("Environment:", json.dumps(env_info, indent=2))

    rows = []

    configs = [
        ("Baseline (YOLO26nDepth)", lambda: YOLO26nDepth().eval(), (480, 640),
         ROOT / "training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt"),
        ("Baseline (YOLO26nDepth, 384x640)", lambda: YOLO26nDepth().eval(), (384, 640),
         ROOT / "training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt"),
        ("+Attention Neck Only (+CBAM P3-P5)", lambda: YOLO26nDepthCBAMNeckOnly().eval(), (480, 640), None),
        ("+Attention Depth Only (+CBAM Depth)", lambda: YOLO26nDepthCBAMDepthOnly().eval(), (480, 640), None),
        ("Full Usulan (+CBAM Neck + Depth)", lambda: YOLO26nDepthCBAM().eval(), (480, 640), None),
        ("Full Usulan (+CBAM, 384x640)", lambda: YOLO26nDepthCBAM().eval(), (384, 640), None),
    ]

    for name, builder, (h, w), ckpt_path in configs:
        print(f"Benchmarking {name} @ {h}x{w}...")
        model = builder()
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        size_mb = (
            os.path.getsize(ckpt_path) / (1024 * 1024)
            if ckpt_path and ckpt_path.exists()
            else _state_dict_mb(model.state_dict())
        )
        x_cpu = torch.randn(1, 3, h, w)
        gflops = _measure_flops(model, x_cpu)
        cpu_ms = _measure_latency(model, x_cpu, None, warmup=50, iters=200)

        gpu_ms = float("nan")
        if torch.cuda.is_available():
            model_gpu = model.to(device_gpu)
            x_gpu = torch.randn(1, 3, h, w, device=device_gpu)
            with torch.inference_mode():
                for _ in range(50):
                    _ = model_gpu(x_gpu)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    _ = model_gpu(x_gpu)
                torch.cuda.synchronize()
                gpu_ms = (time.perf_counter() - t0) * 1000.0 / 200
            model_gpu.cpu()
            torch.cuda.empty_cache()

        fps_gpu = 1000.0 / gpu_ms if gpu_ms == gpu_ms and gpu_ms > 0 else float("nan")
        fps_cpu = 1000.0 / cpu_ms if cpu_ms > 0 else float("nan")
        rows.append({
            "model": name,
            "input_res": f"{h}x{w}",
            "params_m": round(params_m, 3),
            "gflops": round(gflops, 3),
            "size_mb": round(size_mb, 2),
            "latency_cpu_ms": round(cpu_ms, 2),
            "latency_gpu_ms": round(gpu_ms, 2),
            "fps_cpu": round(fps_cpu, 1),
            "fps_gpu": round(fps_gpu, 1),
        })

    # Also benchmark legacy multitask_best.pt (detector + depth_head)
    print("Benchmarking Legacy Runtime (multitask_best.pt) @ 480x640...")
    mt = load_multitask_torch_model(ROOT / "multitask_best.pt")
    mt.detector.eval()
    mt.depth_head.eval()
    params_mt = (sum(p.numel() for p in mt.detector.parameters()) + sum(p.numel() for p in mt.depth_head.parameters())) / 1e6
    size_mt = os.path.getsize(ROOT / "multitask_best.pt") / (1024 * 1024)

    def _mt_forward(x: torch.Tensor):
        mt._features.clear()
        det_out = mt.detector(x)
        feats = (mt._features[16], mt._features[19], mt._features[22])
        depth_out = mt.depth_head(feats)
        return det_out, depth_out

    mt.detector.cpu()
    mt.depth_head.cpu()
    x_cpu = torch.randn(1, 3, 480, 640)
    gflops_mt = _measure_flops(_mt_forward, x_cpu)
    cpu_ms_mt = _measure_latency(_mt_forward, x_cpu, None, warmup=50, iters=200)

    gpu_ms_mt = float("nan")
    if torch.cuda.is_available():
        mt.detector.to(device_gpu)
        mt.depth_head.to(device_gpu)
        x_gpu = torch.randn(1, 3, 480, 640, device=device_gpu)
        with torch.inference_mode():
            for _ in range(50):
                _ = _mt_forward(x_gpu)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                _ = _mt_forward(x_gpu)
            torch.cuda.synchronize()
            gpu_ms_mt = (time.perf_counter() - t0) * 1000.0 / 200

    rows.append({
        "model": "Legacy Runtime (multitask_best.pt)",
        "input_res": "480x640",
        "params_m": round(params_mt, 3),
        "gflops": round(gflops_mt, 3),
        "size_mb": round(size_mt, 2),
        "latency_cpu_ms": round(cpu_ms_mt, 2),
        "latency_gpu_ms": round(gpu_ms_mt, 2),
        "fps_cpu": round(1000.0 / cpu_ms_mt, 1),
        "fps_gpu": round(1000.0 / gpu_ms_mt, 1) if gpu_ms_mt == gpu_ms_mt else float("nan"),
    })

    csv_path = eval_dir / "efficiency_benchmark.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(eval_dir / "efficiency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump({"environment": env_info, "results": rows}, f, indent=2)

    print("Saved eval/efficiency_benchmark.csv and eval/efficiency_benchmark.json")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

