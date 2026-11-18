"""Baseline YOLO26n-Depth WITHOUT attention modules (L10 C2PSA removed, Neck n22 attention=False)."""

from __future__ import annotations

import argparse

import torch
from torch import nn

from arsitektur import C3k2, YOLO26nDepth, _scaled_channels


class YOLO26nDepthNoAttn(YOLO26nDepth):
    """YOLO26n-Depth Baseline without C2PSA (L10) and without attention in Neck n22."""

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 0.25,
        depth_mult: float = 0.50,
        max_channels: int = 1024,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width_mult=width_mult,
            depth_mult=depth_mult,
            max_channels=max_channels,
        )
        c_p4 = _scaled_channels(512, width_mult, max_channels)
        c_p5 = _scaled_channels(1024, width_mult, max_channels)

        # Remove L10 C2PSA (identity bypass keeps channel width c_p5=256 and layer numbering 0-22 aligned)
        self.backbone.l10 = nn.Identity()
        # Replace Neck L22 (n22) with C3k2(attention=False)
        self.neck.n22 = C3k2(
            c_p4 + c_p5,
            c_p5,
            c3k=True,
            expansion=0.5,
            attention=False,
        )


def build_baseline_model(
    num_classes: int = 4,
    width_mult: float = 0.25,
    depth_mult: float = 0.50,
) -> YOLO26nDepthNoAttn:
    """Build the Baseline YOLO26n-Depth model without attention modules."""
    return YOLO26nDepthNoAttn(
        num_classes=num_classes,
        width_mult=width_mult,
        depth_mult=depth_mult,
        max_channels=1024,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        base = build_baseline_model().eval()
        plus_attn = YOLO26nDepth().eval()
        p_base = sum(p.numel() for p in base.parameters())
        p_attn = sum(p.numel() for p in plus_attn.parameters())
        l10_params = sum(p.numel() for p in plus_attn.backbone.l10.parameters())
        n22_diff = sum(p.numel() for p in plus_attn.neck.n22.parameters()) - sum(
            p.numel() for p in base.neck.n22.parameters()
        )
        print(f"Baseline (NoAttn) params: {p_base:,}")
        print(f"+Attention (YOLO26nDepth) params: {p_attn:,}")
        print(f"Diff (+Attention - Baseline): +{p_attn - p_base:,} (L10 C2PSA={l10_params:,}, n22 attn={n22_diff:,})")
        x = torch.zeros(1, 3, 384, 640)
        with torch.inference_mode():
            out = base(x)
        for k in ("p3_bbox", "p3_cls", "p4_bbox", "p4_cls", "p5_bbox", "p5_cls", "depth"):
            print(f"{k}: {tuple(out[k].shape)}")

