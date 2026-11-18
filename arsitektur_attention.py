"""YOLO26n-Depth + CBAM Attention (Usulan) built by extending arsitektur.py without modifying baseline."""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from arsitektur import YOLO26nDepth, _scaled_channels


class ChannelAttention(nn.Module):
    """CBAM Channel Attention using parallel global average and max pooling."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        scale = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * scale


class SpatialAttentionCBAM(nn.Module):
    """CBAM Spatial Attention using channel-wise avg/max pooling + 7x7 conv."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (Channel -> Spatial) with residual option."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction=reduction)
        self.spatial_attn = SpatialAttentionCBAM(kernel_size=kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class YOLO26nDepthCBAM(YOLO26nDepth):
    """YOLO26n-Depth + CBAM attention at Neck outputs (P3/P4/P5) and DepthHead fusion."""

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 0.25,
        depth_mult: float = 0.50,
        max_channels: int = 1024,
        reduction: int = 16,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            width_mult=width_mult,
            depth_mult=depth_mult,
            max_channels=max_channels,
        )
        c_p3 = _scaled_channels(256, width_mult, max_channels)
        c_p4 = _scaled_channels(512, width_mult, max_channels)
        c_p5 = _scaled_channels(1024, width_mult, max_channels)

        self.cbam_p3 = CBAMBlock(c_p3, reduction=reduction)
        self.cbam_p4 = CBAMBlock(c_p4, reduction=reduction)
        self.cbam_p5 = CBAMBlock(c_p5, reduction=reduction)
        self.cbam_depth = CBAMBlock(32 * 3, reduction=reduction)

    def forward(self, x: Tensor, use_nms: bool = True) -> dict[str, Tensor | bool]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must have shape [batch, 3, height, width].")
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            raise ValueError("Input height and width must each be at least 32.")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32.")

        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)

        p3_att = self.cbam_p3(p3_out)
        p4_att = self.cbam_p4(p4_out)
        p5_att = self.cbam_p5(p5_out)

        p3_reg, p3_cls = self.detect_p3(p3_att)
        p4_reg, p4_cls = self.detect_p4(p4_att)
        p5_reg, p5_cls = self.detect_p5(p5_att)

        # DepthHead fusion with CBAM on concatenated 96-ch multi-scale features
        target_size = x.shape[-2:]
        up_p3 = F.interpolate(
            self.depth_head.conv_p3(p3_att), size=target_size, mode="bilinear", align_corners=False
        )
        up_p4 = F.interpolate(
            self.depth_head.conv_p4(p4_att), size=target_size, mode="bilinear", align_corners=False
        )
        up_p5 = F.interpolate(
            self.depth_head.conv_p5(p5_att), size=target_size, mode="bilinear", align_corners=False
        )
        fused = self.cbam_depth(torch.cat((up_p3, up_p4, up_p5), dim=1))
        depth_map = self.depth_head.refine(fused)

        return {
            "p3_bbox": p3_reg,
            "p3_cls": p3_cls,
            "p4_bbox": p4_reg,
            "p4_cls": p4_cls,
            "p5_bbox": p5_reg,
            "p5_cls": p5_cls,
            "depth": depth_map,
            "nms_applied": use_nms,
        }


def build_proposed_model(
    num_classes: int = 4,
    width_mult: float = 0.25,
    depth_mult: float = 0.50,
) -> YOLO26nDepthCBAM:
    """Build YOLO26n-Depth + CBAM proposed model."""
    return YOLO26nDepthCBAM(
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
        base = YOLO26nDepth().eval()
        prop = build_proposed_model().eval()
        x = torch.zeros(1, 3, 480, 640)
        with torch.inference_mode():
            out = prop(x)
        base_params = sum(p.numel() for p in base.parameters())
        prop_params = sum(p.numel() for p in prop.parameters())
        print(f"Baseline params: {base_params:,}")
        print(f"Proposed (+CBAM) params: {prop_params:,} (+{prop_params - base_params:,})")
        for k in ("p3_bbox", "p3_cls", "p4_bbox", "p4_cls", "p5_bbox", "p5_cls", "depth"):
            print(f"{k}: {tuple(out[k].shape)}")

