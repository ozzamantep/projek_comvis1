"""Train Baseline YOLO26nDepthNoAttn (without C2PSA L10 and without Neck n22 attention) with identical settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_arsitektur as trainer  # noqa: E402
from arsitektur_noattn import build_baseline_model  # noqa: E402


def _transfer_noattn_backbone_neck(model, pretrained_model) -> int:
    """Transfer all matching YOLO26n weights, skipping removed L10 C2PSA and n22 PSABlock."""
    layers = getattr(pretrained_model, "model", None)
    transferred = 0
    for target_name, source_index in trainer._YOLO26_PRETRAINED_LAYERS:
        if target_name == "backbone.l10":
            continue
        target = model.get_submodule(target_name)
        source_state = layers[source_index].state_dict()
        mapped_state = {
            ".".join(
                {"m": "blocks", "attn": "attention", "pe": "position"}.get(part, part)
                for part in key.split(".")
            ): value
            for key, value in source_state.items()
        }
        target_state = target.state_dict()
        compatible = {
            k: v for k, v in mapped_state.items()
            if k in target_state and target_state[k].shape == v.shape
        }
        target.load_state_dict(compatible, strict=(target_name != "neck.n22"))
        transferred += len(compatible)
    return transferred


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Baseline (NoAttn) with identical settings to rtx4050_arsitektur_70_20_10.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training_runs" / "persist" / "rtx4050_noattn_70_20_10")
    parser.add_argument("--resume", type=Path, default=None)
    args, remaining = parser.parse_known_args()

    trainer.build_model = build_baseline_model
    trainer._transfer_yolo26_backbone_neck = _transfer_noattn_backbone_neck
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "train_arsitektur.py",
        "--sun-root", str(ROOT / "datasets" / "SUNRGBD" / "SUNRGBD"),
        "--coco-root", str(ROOT / "datasets" / "coco"),
        "--coco-local-only",
        "--no-download",
        "--classes", "person", "laptop", "cell phone", "bottle",
        "--img-size", "384", "640",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--learning-rate", "1e-3",
        "--backbone-lr-mult", "0.1",
        "--weight-decay", "1e-4",
        "--warmup-epochs", "3.0",
        "--min-lr-ratio", "0.02",
        "--ema-decay", "0.999",
        "--cls-weight", "1.0",
        "--box-weight", "7.5",
        "--depth-weight", "0.5",
        "--focal-alpha", "0.25",
        "--focal-gamma", "2.0",
        "--topk-assign", "3",
        "--scale-jitter", "0.2",
        "--translate-jitter", "0.08",
        "--hsv-jitter", "1.0",
        "--balanced-sampling",
        "--coco-ratio", "0.5",
        "--val-fraction", "0.2",
        "--test-fraction", "0.1",
        "--seed", "42",
        "--conf-threshold", "0.001",
        "--nms-iou", "0.7",
        "--max-detections", "300",
        "--best-metric", "map50",
        "--device", "cuda",
        "--amp",
        "--output-dir", str(args.output_dir),
    ]
    last_ckpt = args.output_dir / "last.pt"
    if args.resume is not None:
        sys.argv.extend(["--resume", str(args.resume)])
    elif last_ckpt.is_file():
        sys.argv.extend(["--resume", str(last_ckpt)])
    else:
        sys.argv.extend(["--pretrained-yolo26", str(ROOT / "yolo26n.pt")])

    sys.argv.extend(remaining)
    trainer.train(trainer._parse_args())


if __name__ == "__main__":
    main()
