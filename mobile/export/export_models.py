"""Export YOLO26n detector and local YOLO26nDepth to LiteRT/TFLite FP16."""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEPTH_HEIGHT = 240
DEPTH_WIDTH = 320


class DepthOnly(nn.Module):
    """Expose the depth tensor as the only model output for export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["depth"]


def resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def load_depth_model(weights_path: Path, device: torch.device) -> nn.Module:
    from penting.arsitektur_yolo_depth import (
        YOLO26nDepth,
        load_depth_checkpoint,
    )

    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    load_depth_checkpoint(model, weights_path, device)
    model.eval()
    return model


def validation_indices(
    sample_count: int,
    max_samples: int,
    requested_count: int,
) -> np.ndarray:
    count = sample_count if max_samples <= 0 else min(sample_count, max_samples)
    if count < 2:
        raise ValueError("The NYU dataset must contain at least two samples.")
    indices = np.arange(count)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    split = max(1, int(count * 0.9))
    validation = indices[split:]
    if validation.size == 0:
        raise ValueError("The NYU validation split is empty.")
    number_to_use = min(requested_count, int(validation.size))
    if number_to_use < 2:
        raise ValueError(
            "At least two validation inputs are required for export verification."
        )
    positions = np.linspace(0, validation.size - 1, number_to_use, dtype=int)
    return validation[positions]


def load_validation_tensors(
    nyu_path: Path,
    max_samples: int,
    requested_count: int,
) -> tuple[list[torch.Tensor], list[int]]:
    import h5py
    import cv2

    tensors: list[torch.Tensor] = []
    with h5py.File(nyu_path, "r") as dataset_file:
        sample_count = int(dataset_file["images"].shape[0])
        indices = validation_indices(sample_count, max_samples, requested_count)
        for index in indices:
            image_rgb = np.asarray(dataset_file["images"][int(index)]).transpose(
                2, 1, 0
            )
            image_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            image_bgr = cv2.resize(
                image_bgr,
                (DEPTH_WIDTH, DEPTH_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_tensor = (
                torch.from_numpy(image_rgb.transpose(2, 0, 1).copy())
                .float()
                .div_(255.0)
            )
            tensors.append(image_tensor.unsqueeze(0))
    return tensors, [int(index) for index in indices]


def export_depth_onnx(
    model: nn.Module,
    output_path: Path,
    device: torch.device,
    opset: int,
) -> None:
    wrapper = DepthOnly(model).eval()
    dummy = torch.zeros(
        (1, 3, DEPTH_HEIGHT, DEPTH_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth"],
            dynamic_axes=None,
            dynamo=False,
        )

    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)


def convert_onnx_to_fp16_tflite(onnx_path: Path, output_dir: Path) -> Path:
    import onnx2tf
    import tensorflow as tf

    saved_model_dir = output_dir / "depth_saved_model"
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        non_verbose=True,
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_bytes = converter.convert()

    tflite_path = output_dir / "yolo26n_depth_fp16.tflite"
    with tflite_path.open("wb") as output:
        output.write(tflite_bytes)
    return tflite_path


def run_onnx(session: Any, image: torch.Tensor) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run(
        [output_name],
        {input_name: image.detach().cpu().numpy().astype(np.float32)},
    )[0]
    return to_nchw(np.asarray(output, dtype=np.float32))


def to_nchw(output: np.ndarray) -> np.ndarray:
    if output.ndim != 4:
        raise ValueError(f"Expected a 4D depth tensor; got shape {output.shape}.")
    if output.shape[1] == 1:
        return output
    if output.shape[-1] == 1:
        return output.transpose(0, 3, 1, 2)
    raise ValueError(
        f"Could not identify single-channel depth output layout: {output.shape}."
    )


def run_tflite(interpreter: Any, image: torch.Tensor) -> np.ndarray:
    input_details = interpreter.get_input_details()[0]
    input_shape = tuple(int(size) for size in input_details["shape"])
    nchw = image.detach().cpu().numpy().astype(np.float32)
    if input_shape == nchw.shape:
        model_input = nchw
    elif input_shape == (
        1,
        DEPTH_HEIGHT,
        DEPTH_WIDTH,
        3,
    ):
        model_input = nchw.transpose(0, 2, 3, 1)
    else:
        raise ValueError(
            f"Unexpected TFLite input shape {input_shape}; expected NCHW or NHWC."
        )

    if input_details["dtype"] == np.float16:
        model_input = model_input.astype(np.float16)
    else:
        model_input = model_input.astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], model_input)
    interpreter.invoke()

    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    return to_nchw(np.asarray(output, dtype=np.float32))


def max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"Output shapes differ: {left.shape} and {right.shape}.")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def benchmark(
    name: str,
    run: Callable[[], Any],
    warmup: int,
    repetitions: int,
    device: torch.device | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        run()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)

    durations_ms: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        run()
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    result = {
        "mean_ms": float(statistics.mean(durations_ms)),
        "median_ms": float(statistics.median(durations_ms)),
        "min_ms": float(min(durations_ms)),
        "max_ms": float(max(durations_ms)),
    }
    print(
        f"{name} latency: mean={result['mean_ms']:.3f} ms, "
        f"median={result['median_ms']:.3f} ms "
        f"(warmup={warmup}, runs={repetitions})"
    )
    return result


def export_detector(
    weights_path: Path,
    output_dir: Path,
    image_size: int,
) -> Path:
    import ultralytics
    from ultralytics import YOLO

    version_parts = tuple(
        int(part) for part in ultralytics.__version__.split(".")[:3]
    )
    if version_parts < (8, 4, 175):
        raise RuntimeError(
            "Detector export requires Ultralytics >=8.4.175 for "
            "format='litert', quantize=16."
        )

    model = YOLO(str(weights_path))
    exported_path = model.export(
        format="litert",
        quantize=16,
        imgsz=image_size,
        batch=1,
        project=str(output_dir),
        name="detector",
    )
    result_path = Path(exported_path)
    if not result_path.is_file() or result_path.suffix.lower() != ".tflite":
        raise RuntimeError(
            f"Ultralytics did not return an existing .tflite file: {exported_path}"
        )
    canonical_path = output_dir / "yolo26n_detector_fp16.tflite"
    shutil.copy2(result_path, canonical_path)
    return canonical_path


def file_size_report(paths: dict[str, Path]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, path in paths.items():
        if path.is_file():
            sizes[name] = path.stat().st_size
            print(f"{name} size: {sizes[name]:,} bytes ({path})")
    return sizes


def export_and_verify(args: argparse.Namespace) -> dict[str, Any]:
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        raise RuntimeError(
            "Use Python 3.10, 3.11, or 3.12 for the pinned export toolchain; "
            f"current Python is {sys.version.split()[0]}."
        )

    import onnxruntime as ort
    import tensorflow as tf
    import ultralytics

    output_dir = resolve_project_path(args.output_dir)
    detector_weights = resolve_project_path(args.detector_weights)
    depth_weights = resolve_project_path(args.depth_weights)
    nyu_path = resolve_project_path(args.nyu_path)

    require_file(detector_weights, "Ultralytics detector checkpoint")
    require_file(depth_weights, "Local depth checkpoint")
    require_file(nyu_path, "NYU Depth V2 MAT dataset")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Python: {__import__('sys').version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Ultralytics: {ultralytics.__version__}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"onnx2tf: {getattr(__import__('onnx2tf'), '__version__', 'version unavailable')}")
    print(f"Device: {device}")

    detector_tflite_path = export_detector(
        detector_weights,
        output_dir,
        args.detector_imgsz,
    )

    model = load_depth_model(depth_weights, device)
    wrapper = DepthOnly(model).eval()
    onnx_path = output_dir / "yolo26n_depth.onnx"
    export_depth_onnx(model, onnx_path, device, args.opset)
    depth_tflite_path = convert_onnx_to_fp16_tflite(onnx_path, output_dir)

    images, source_indices = load_validation_tensors(
        nyu_path,
        args.max_samples,
        args.verify_inputs,
    )
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    interpreter = tf.lite.Interpreter(model_path=str(depth_tflite_path))
    interpreter.allocate_tensors()

    error_rows: list[dict[str, float | int]] = []
    for image, source_index in zip(images, source_indices):
        with torch.inference_mode():
            torch_output = wrapper(image.to(device)).detach().cpu().numpy()
        onnx_output = run_onnx(session, image)
        tflite_output = run_tflite(interpreter, image)
        torch_output = to_nchw(np.asarray(torch_output, dtype=np.float32))
        onnx_output = to_nchw(onnx_output)
        tflite_output = to_nchw(tflite_output)

        row = {
            "nyu_source_index": source_index,
            "pytorch_vs_onnx_max_abs": max_abs_error(
                torch_output, onnx_output
            ),
            "pytorch_vs_tflite_max_abs": max_abs_error(
                torch_output, tflite_output
            ),
            "onnx_vs_tflite_max_abs": max_abs_error(
                onnx_output, tflite_output
            ),
        }
        error_rows.append(row)
        print(
            f"NYU[{source_index}] max abs error: "
            f"PyTorch/ONNX={row['pytorch_vs_onnx_max_abs']:.8g}, "
            f"PyTorch/TFLite={row['pytorch_vs_tflite_max_abs']:.8g}, "
            f"ONNX/TFLite={row['onnx_vs_tflite_max_abs']:.8g}"
        )

    first_image = images[0].to(device)
    torch_latency = benchmark(
        "PyTorch",
        lambda: wrapper(first_image),
        args.warmup,
        args.benchmark_runs,
        device,
    )
    onnx_latency = benchmark(
        "ONNX Runtime CPU",
        lambda: run_onnx(session, images[0]),
        args.warmup,
        args.benchmark_runs,
    )
    tflite_latency = benchmark(
        "TFLite FP16 CPU",
        lambda: run_tflite(interpreter, images[0]),
        args.warmup,
        args.benchmark_runs,
    )

    report: dict[str, Any] = {
        "versions": {
            "pytorch": str(torch.__version__),
            "ultralytics": str(ultralytics.__version__),
            "onnxruntime": ort.__version__,
            "tensorflow": tf.__version__,
            "onnx2tf": str(getattr(__import__("onnx2tf"), "__version__", "unknown")),
        },
        "device": str(device),
        "depth_input_shape_nchw": [1, 3, DEPTH_HEIGHT, DEPTH_WIDTH],
        "depth_output_shape_nchw": [1, 1, DEPTH_HEIGHT, DEPTH_WIDTH],
        "fp16_conversion": (
            "TensorFlow Lite Converter from the onnx2tf SavedModel, "
            "Optimize.DEFAULT with supported_types=[tf.float16]"
        ),
        "depth_verification_inputs": len(images),
        "nyu_source_indices": source_indices,
        "max_abs_errors": error_rows,
        "latency": {
            "pytorch": torch_latency,
            "onnxruntime_cpu": onnx_latency,
            "tflite_fp16_cpu": tflite_latency,
        },
        "files_bytes": file_size_report(
            {
                "detector_litert_fp16": detector_tflite_path,
                "depth_onnx": onnx_path,
                "depth_tflite_fp16": depth_tflite_path,
            }
        ),
        "detector_tflite_path": str(detector_tflite_path),
        "depth_onnx_path": str(onnx_path),
        "depth_tflite_path": str(depth_tflite_path),
    }
    report_path = output_dir / "export_verification.json"
    with report_path.open("w", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(f"Verification report: {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detector/depth TFLite FP16 and compare depth outputs."
    )
    parser.add_argument("--detector-weights", default="data/yolo26n.pt")
    parser.add_argument(
        "--depth-weights",
        default="data/yolo26n_depth_nyu_full.pt",
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument("--output-dir", default="mobile/export/artifacts")
    parser.add_argument("--detector-imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-inputs", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--benchmark-runs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.verify_inputs < 2:
        parser.error("--verify-inputs must be at least 2.")
    if args.detector_imgsz < 32:
        parser.error("--detector-imgsz must be at least 32.")
    if args.opset < 11:
        parser.error("--opset must be at least 11.")
    if args.warmup < 0 or args.benchmark_runs < 1:
        parser.error("--warmup must be non-negative and --benchmark-runs positive.")
    return args


if __name__ == "__main__":
    export_and_verify(parse_args())
