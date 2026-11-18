"""Train YOLO26nDepthCBAM (or CBAM Neck Only) using train_arsitektur.py with identical baseline settings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_arsitektur as trainer  # noqa: E402
from arsitektur import YOLO26nDepth  # noqa: E402
from arsitektur_attention import CBAMBlock, build_proposed_model  # noqa: E402


class YOLO26nDepthCBAMNeckOnly(YOLO26nDepth):
    """Ablation variant: +CBAM only on Neck outputs (P3/P4/P5)."""

    def __init__(self, num_classes: int = 4, width_mult: float = 0.25, depth_mult: float = 0.50, max_channels: int = 1024) -> None:
        super().__init__(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult, max_channels=max_channels)
        self.cbam_p3 = CBAMBlock(64, reduction=16)
        self.cbam_p4 = CBAMBlock(128, reduction=16)
        self.cbam_p5 = CBAMBlock(256, reduction=16)

    def forward(self, x, use_nms: bool = True):
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


def build_neck_only_model(num_classes: int = 4, width_mult: float = 0.25, depth_mult: float = 0.50):
    return YOLO26nDepthCBAMNeckOnly(num_classes=num_classes, width_mult=width_mult, depth_mult=depth_mult)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train +Attention variant with identical baseline settings.")
    parser.add_argument("--variant", choices=("full", "neck_only"), default="full")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    args, remaining = parser.parse_known_args()

    if args.variant == "full":
        trainer.build_model = build_proposed_model
        default_out = ROOT / "training_runs" / "persist" / "rtx4050_cbam_70_20_10"
    else:
        trainer.build_model = build_neck_only_model
        default_out = ROOT / "training_runs" / "persist" / "rtx4050_cbam_neck_70_20_10"

    out_dir = args.output_dir or default_out
    out_dir.mkdir(parents=True, exist_ok=True)

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
        "--output-dir", str(out_dir),
    ]
    last_ckpt = out_dir / "last.pt"
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

