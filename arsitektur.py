"""Untrained YOLO26n-Depth architecture based on Sections 3.1 and 3.2.

This model is independent of the trained ONNX model used by the camera app.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _scaled_channels(
    base_channels: int,
    width_mult: float,
    minimum: int,
    max_channels: int,
) -> int:
    return max(minimum, min(int(base_channels * width_mult), max_channels))


class ConvBlock(nn.Module):
    """Convolution, batch normalization, and SiLU activation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
    ) -> None:
        super().__init__()
        padding = k // 2 if p is None else p
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=padding,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.cv1 = ConvBlock(ch, ch, k=3, s=1, p=1)
        self.cv2 = ConvBlock(ch, ch, k=3, s=1, p=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.cv2(self.cv1(x))


class C3k2(nn.Module):
    """Split-feature C3k2 block with a stack of residual bottlenecks."""

    def __init__(self, c1: int, c2: int, n: int = 1) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("C3k2 repeat count must be positive.")
        hidden = max(1, c2 // 2)
        self.cv1 = ConvBlock(c1, hidden, k=1, s=1, p=0)
        self.cv2 = ConvBlock(c1, hidden, k=1, s=1, p=0)
        self.blocks = nn.Sequential(*(Bottleneck(hidden) for _ in range(n)))
        self.cv3 = ConvBlock(2 * hidden, c2, k=1, s=1, p=0)

    def forward(self, x: Tensor) -> Tensor:
        branch_a = self.cv1(x)
        branch_b = self.blocks(self.cv2(x))
        return self.cv3(torch.cat((branch_a, branch_b), dim=1))


class C2PSA(nn.Module):
    """Channel gate used for C2PSA in the report's Section 3.2 implementation."""

    def __init__(self, c1: int, c2: int) -> None:
        super().__init__()
        self.cv1 = ConvBlock(c1, c2, k=1, s=1, p=0)
        hidden = max(1, c2 // 4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.cv1(x)
        return features * self.attn(features)


class SPPF(nn.Module):
    """Fast spatial pyramid pooling with three serial max-pooling passes."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 5) -> None:
        super().__init__()
        hidden_ch = max(1, in_ch // 2)
        self.cv1 = ConvBlock(in_ch, hidden_ch, k=1, s=1, p=0)
        self.cv2 = ConvBlock(hidden_ch * 4, out_ch, k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class Backbone(nn.Module):
    """Build hierarchical P2/4-P5/32 features from the report's backbone."""

    def __init__(
        self,
        c_stem: int,
        channels: tuple[int, int, int, int],
        n_repeat: int,
    ) -> None:
        super().__init__()
        c_p2, c_p3, c_p4, c_p5 = channels
        self.stem = ConvBlock(3, c_stem, k=3, s=2, p=1)
        self.l2 = nn.Sequential(
            ConvBlock(c_stem, c_p2, k=3, s=2, p=1),
            C3k2(c_p2, c_p2, n=n_repeat),
        )
        self.l3 = ConvBlock(c_p2, c_p3, k=3, s=2, p=1)
        self.l4 = C3k2(c_p3, c_p3, n=n_repeat)
        self.l5 = ConvBlock(c_p3, c_p4, k=3, s=2, p=1)
        self.l6 = C3k2(c_p4, c_p4, n=n_repeat)
        self.l7 = ConvBlock(c_p4, c_p5, k=3, s=2, p=1)
        self.l8 = C3k2(c_p5, c_p5, n=n_repeat)
        self.l9 = SPPF(c_p5, c_p5, k=5)
        self.l10 = C2PSA(c_p5, c_p5)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.l2(self.stem(x))
        p3 = self.l4(self.l3(x))
        p4 = self.l6(self.l5(p3))
        p5 = self.l10(self.l9(self.l8(self.l7(p4))))
        return p3, p4, p5


def _upsample_like(x: Tensor, reference: Tensor) -> Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="nearest")


class Neck(nn.Module):
    """Top-down FPN and bottom-up PAN producing P3_out, P4_out, and P5_out."""

    def __init__(self, channels: tuple[int, int, int], n_repeat: int) -> None:
        super().__init__()
        c_p3, c_p4, c_p5 = channels
        self.n13 = C3k2(c_p5 + c_p4, c_p4, n=n_repeat)
        self.n16 = C3k2(c_p4 + c_p3, c_p3, n=n_repeat)
        self.d17 = ConvBlock(c_p3, c_p3, k=3, s=2, p=1)
        self.n19 = C3k2(c_p3 + c_p4, c_p4, n=n_repeat)
        self.d20 = ConvBlock(c_p4, c_p4, k=3, s=2, p=1)
        self.n22 = C3k2(c_p4 + c_p5, c_p5, n=n_repeat)

    def forward(
        self, p3: Tensor, p4: Tensor, p5: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        p4_fused = self.n13(torch.cat((_upsample_like(p5, p4), p4), dim=1))
        p3_out = self.n16(torch.cat((_upsample_like(p4_fused, p3), p3), dim=1))
        p4_out = self.n19(torch.cat((self.d17(p3_out), p4_fused), dim=1))
        p5_out = self.n22(torch.cat((self.d20(p4_out), p5), dim=1))
        return p3_out, p4_out, p5_out


class DetectionHead(nn.Module):
    """Report-style independent bounding-box regression and class branches."""

    def __init__(self, in_ch: int, num_classes: int = 4) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        self.reg = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, 4, kernel_size=1),
        )
        self.cls = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.reg(x), self.cls(x)


class DepthHead(nn.Module):
    """Fuse P3/P4/P5 at input resolution into a relative depth map in [0, 1]."""

    def __init__(self, channels: tuple[int, int, int]) -> None:
        super().__init__()
        self.conv_p3 = ConvBlock(channels[0], 32, k=1, s=1, p=0)
        self.conv_p4 = ConvBlock(channels[1], 32, k=1, s=1, p=0)
        self.conv_p5 = ConvBlock(channels[2], 32, k=1, s=1, p=0)
        self.refine = nn.Sequential(
            ConvBlock(32 * 3, 32, k=3, s=1, p=1),
            ConvBlock(32, 16, k=3, s=1, p=1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, features: tuple[Tensor, Tensor, Tensor], target_size: tuple[int, int]
    ) -> Tensor:
        p3, p4, p5 = features
        up_p3 = F.interpolate(
            self.conv_p3(p3),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        up_p4 = F.interpolate(
            self.conv_p4(p4),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        up_p5 = F.interpolate(
            self.conv_p5(p5),
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.refine(torch.cat((up_p3, up_p4, up_p5), dim=1))


class YOLO26nDepth(nn.Module):
    """Report-aligned detector/depth blueprint; it has no trained weights."""

    def __init__(
        self,
        num_classes: int = 4,
        width_mult: float = 0.25,
        depth_mult: float = 0.50,
        max_channels: int = 1024,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        if width_mult <= 0 or depth_mult <= 0 or max_channels < 1:
            raise ValueError("Scaling factors and max_channels must be positive.")

        c_stem = _scaled_channels(64, width_mult, 32, max_channels)
        c_p2 = _scaled_channels(128, width_mult, 32, max_channels)
        c_p3 = _scaled_channels(256, width_mult, 64, max_channels)
        c_p4 = _scaled_channels(512, width_mult, 128, max_channels)
        c_p5 = _scaled_channels(1024, width_mult, 256, max_channels)
        n_repeat = max(1, round(2 * depth_mult))

        self.backbone = Backbone(
            c_stem,
            (c_p2, c_p3, c_p4, c_p5),
            n_repeat,
        )
        self.neck = Neck((c_p3, c_p4, c_p5), n_repeat)
        self.detect_p3 = DetectionHead(c_p3, num_classes)
        self.detect_p4 = DetectionHead(c_p4, num_classes)
        self.detect_p5 = DetectionHead(c_p5, num_classes)
        self.depth_head = DepthHead((c_p3, c_p4, c_p5))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must have shape [batch, 3, height, width].")
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            raise ValueError("Input height and width must each be at least 32.")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32.")

        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)
        p3_reg, p3_cls = self.detect_p3(p3_out)
        p4_reg, p4_cls = self.detect_p4(p4_out)
        p5_reg, p5_cls = self.detect_p5(p5_out)
        depth_map = self.depth_head((p3_out, p4_out, p5_out), x.shape[-2:])
        return {
            "p3_bbox": p3_reg,
            "p3_cls": p3_cls,
            "p4_bbox": p4_reg,
            "p4_cls": p4_cls,
            "p5_bbox": p5_reg,
            "p5_cls": p5_cls,
            "depth": depth_map,
        }


def build_model(
    num_classes: int = 4,
    width_mult: float = 0.25,
    depth_mult: float = 0.50,
) -> YOLO26nDepth:
    """Build the report's Nano-scale model for the project's four object classes."""
    return YOLO26nDepth(
        num_classes=num_classes,
        width_mult=width_mult,
        depth_mult=depth_mult,
        max_channels=1024,
    )


def _verify() -> None:
    model = build_model().eval()
    images = torch.zeros(1, 3, 640, 640)
    with torch.inference_mode():
        output = model(images)
    print(f"Input: {tuple(images.shape)}")
    for key in ("p3_bbox", "p3_cls", "p4_bbox", "p4_cls", "p5_bbox", "p5_cls", "depth"):
        print(f"{key}: {tuple(output[key].shape)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the report-aligned, untrained YOLO26n-Depth architecture."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a forward pass on the report's 640x640 input size.",
    )
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify to run the architecture smoke test.")
    _verify()
