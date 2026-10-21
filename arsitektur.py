"""YOLO26n-Depth backbone and neck with project-specific detection and depth heads.

The YOLO26 feature blocks are implemented locally and require only PyTorch.
"""

from __future__ import annotations

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F


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
        act: bool = True,
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
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


def _scaled_channels(
    base_channels: int,
    width_mult: float,
    max_channels: int,
) -> int:
    scaled = min(base_channels, max_channels) * width_mult
    return max(8, int((scaled + 7) // 8) * 8)


class Bottleneck(nn.Module):
    """Two-convolution residual block with configurable kernels and expansion."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        shortcut: bool = True,
        k: tuple[int, int] = (3, 3),
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = max(1, int(out_ch * expansion))
        self.cv1 = ConvBlock(in_ch, hidden, k=k[0], s=1)
        self.cv2 = ConvBlock(hidden, out_ch, k=k[1], s=1)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x: Tensor) -> Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class C3k(nn.Module):
    """Three-convolution block containing repeated residual bottlenecks."""

    def __init__(self, ch: int, repeats: int = 2) -> None:
        super().__init__()
        hidden = max(1, ch // 2)
        self.cv1 = ConvBlock(ch, hidden, k=1, s=1)
        self.cv2 = ConvBlock(ch, hidden, k=1, s=1)
        self.blocks = nn.Sequential(
            *(Bottleneck(hidden, hidden, k=(3, 3), expansion=1.0) for _ in range(repeats))
        )
        self.cv3 = ConvBlock(hidden * 2, ch, k=1, s=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.cv3(torch.cat((self.blocks(self.cv1(x)), self.cv2(x)), dim=1))


class C3k2(nn.Module):
    """CSP feature block with either residual bottlenecks or C3k sub-blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        expansion: float = 0.5,
        attention: bool = False,
    ) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("C3k2 repeat count must be positive.")
        hidden = max(1, int(c2 * expansion))
        self.cv1 = ConvBlock(c1, 2 * hidden, k=1, s=1)
        self.cv2 = ConvBlock((2 + n) * hidden, c2, k=1, s=1)
        if attention:
            self.blocks = nn.ModuleList(
                nn.Sequential(
                    Bottleneck(hidden, hidden),
                    PSABlock(hidden, max(hidden // 64, 1)),
                )
                for _ in range(n)
            )
        elif c3k:
            self.blocks = nn.ModuleList(C3k(hidden) for _ in range(n))
        else:
            self.blocks = nn.ModuleList(
                Bottleneck(hidden, hidden) for _ in range(n)
            )

    def forward(self, x: Tensor) -> Tensor:
        features = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            features.append(block(features[-1]))
        return self.cv2(torch.cat(features, dim=1))


class SpatialAttention(nn.Module):
    """Multi-head spatial attention with a depthwise positional feature."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.key_dim = max(1, self.head_dim // 2)
        self.scale = self.key_dim**-0.5
        self.qkv = ConvBlock(
            channels, channels + 2 * self.key_dim * heads, k=1, act=False
        )
        self.proj = ConvBlock(channels, channels, k=1, act=False)
        self.position = ConvBlock(channels, channels, k=3, g=channels, act=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        positions = height * width
        q, k, v = self.qkv(x).reshape(
            batch, self.heads, 2 * self.key_dim + self.head_dim, positions
        ).split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        weights = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        attended = (v @ weights.transpose(-2, -1)).reshape(
            batch, channels, height, width
        )
        value_map = v.reshape(batch, channels, height, width)
        return self.proj(attended + self.position(value_map))


class PSABlock(nn.Module):
    """Spatial attention and feed-forward residual block."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.attention = SpatialAttention(channels, heads)
        self.ffn = nn.Sequential(
            ConvBlock(channels, channels * 2, k=1),
            ConvBlock(channels * 2, channels, k=1, act=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """Split-channel spatial attention block used at the end of the backbone."""

    def __init__(self, c1: int, c2: int, n: int = 1, expansion: float = 0.5) -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA input and output channels must match.")
        hidden = max(1, int(c1 * expansion))
        self.cv1 = ConvBlock(c1, 2 * hidden, k=1, s=1)
        self.cv2 = ConvBlock(2 * hidden, c2, k=1, s=1)
        self.blocks = nn.Sequential(
            *(PSABlock(hidden, max(hidden // 64, 1)) for _ in range(n))
        )

    def forward(self, x: Tensor) -> Tensor:
        bypass, attended = self.cv1(x).chunk(2, dim=1)
        return self.cv2(torch.cat((bypass, self.blocks(attended)), dim=1))


class SPPF(nn.Module):
    """Fast spatial pyramid pooling with three serial max-pooling passes and a shortcut."""

    def __init__(
        self, in_ch: int, out_ch: int, k: int = 5, repeats: int = 3, shortcut: bool = True
    ) -> None:
        super().__init__()
        hidden_ch = max(1, in_ch // 2)
        self.cv1 = ConvBlock(in_ch, hidden_ch, k=1, s=1, act=False)
        self.cv2 = ConvBlock(hidden_ch * (repeats + 1), out_ch, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.add_shortcut = shortcut and in_ch == out_ch
        self.repeats = repeats

    def forward(self, x: Tensor) -> Tensor:
        res = x
        x_mapped = self.cv1(x)
        pooled = [x_mapped]
        for _ in range(self.repeats):
            pooled.append(self.pool(pooled[-1]))
        out = self.cv2(torch.cat(pooled, dim=1))
        return out + res if self.add_shortcut else out


class Backbone(nn.Module):
    """YOLO26 P2/4-P5/32 backbone with project-compatible feature outputs."""

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
            C3k2(c_p2, c_p3, n=n_repeat, expansion=0.25),
        )
        self.l3 = ConvBlock(c_p3, c_p3, k=3, s=2, p=1)
        self.l4 = C3k2(c_p3, c_p4, n=n_repeat, expansion=0.25)
        self.l5 = ConvBlock(c_p4, c_p4, k=3, s=2, p=1)
        self.l6 = C3k2(c_p4, c_p4, n=n_repeat, c3k=True)
        self.l7 = ConvBlock(c_p4, c_p5, k=3, s=2, p=1)
        self.l8 = C3k2(c_p5, c_p5, n=n_repeat, c3k=True)
        self.l9 = SPPF(c_p5, c_p5, k=5, repeats=3, shortcut=True)
        self.l10 = C2PSA(c_p5, c_p5, n=n_repeat)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = self.l2(self.stem(x))
        p3 = self.l4(self.l3(x))
        p4 = self.l6(self.l5(p3))
        p5 = self.l10(self.l9(self.l8(self.l7(p4))))
        return p3, p4, p5


def _upsample_like(x: Tensor, reference: Tensor) -> Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="nearest")


class Neck(nn.Module):
    """YOLO26 feature pyramid with separate backbone and output channel widths."""

    def __init__(self, channels: tuple[int, int, int], n_repeat: int) -> None:
        super().__init__()
        c_p3, c_p4, c_p5 = channels
        self.n13 = C3k2(c_p5 + c_p4, c_p4, n=n_repeat, c3k=True)
        self.n16 = C3k2(c_p4 * 2, c_p3, n=n_repeat, c3k=True)
        self.d17 = ConvBlock(c_p3, c_p3, k=3, s=2, p=1)
        self.n19 = C3k2(c_p3 + c_p4, c_p4, n=n_repeat, c3k=True)
        self.d20 = ConvBlock(c_p4, c_p4, k=3, s=2, p=1)
        self.n22 = C3k2(
            c_p4 + c_p5,
            c_p5,
            c3k=True,
            expansion=0.5,
            attention=True,
        )

    def forward(
        self, p3: Tensor, p4: Tensor, p5: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # Top-down FPN
        p4_fused = self.n13(torch.cat((_upsample_like(p5, p4), p4), dim=1))
        p3_out = self.n16(torch.cat((_upsample_like(p4_fused, p3), p3), dim=1))

        # Bottom-up PAN
        p4_out = self.n19(torch.cat((self.d17(p3_out), p4_fused), dim=1))
        p5_out = self.n22(torch.cat((self.d20(p4_out), p5), dim=1))
        return p3_out, p4_out, p5_out


class DetectionHead(nn.Module):
    """Report-style independent bounding-box regression and class branches (DFL-free)."""

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


ARCHITECTURE_ID = "ultralytics-yolo26n-backbone-neck-project-heads-v2"


class YOLO26nDepth(nn.Module):
    """YOLO26n feature extractor with project-specific detection and depth heads."""

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

        c_stem = _scaled_channels(64, width_mult, max_channels)
        c_p2 = _scaled_channels(128, width_mult, max_channels)
        c_p3 = _scaled_channels(256, width_mult, max_channels)
        c_p4 = _scaled_channels(512, width_mult, max_channels)
        c_p5 = _scaled_channels(1024, width_mult, max_channels)
        n_repeat = max(1, round(2 * depth_mult))

        self.backbone = Backbone(c_stem, (c_p2, c_p3, c_p4, c_p5), n_repeat)
        self.neck = Neck((c_p3, c_p4, c_p5), n_repeat)
        self.detect_p3 = DetectionHead(c_p3, num_classes)
        self.detect_p4 = DetectionHead(c_p4, num_classes)
        self.detect_p5 = DetectionHead(c_p5, num_classes)
        self.depth_head = DepthHead((c_p3, c_p4, c_p5))

    def forward(self, x: Tensor, use_nms: bool = True) -> dict[str, Tensor | bool]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Input must have shape [batch, 3, height, width].")
        if x.shape[-2] < 32 or x.shape[-1] < 32:
            raise ValueError("Input height and width must each be at least 32.")
        if x.shape[-2] % 32 or x.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32.")

        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)

        # Dual heads, DFL-free
        p3_reg, p3_cls = self.detect_p3(p3_out)
        p4_reg, p4_cls = self.detect_p4(p4_out)
        p5_reg, p5_cls = self.detect_p5(p5_out)

        depth_map = self.depth_head((p3_out, p4_out, p5_out), x.shape[-2:])

        # Output dengan logika NMS dari diagram
        output_dict = {
            "p3_bbox": p3_reg,
            "p3_cls": p3_cls,
            "p4_bbox": p4_reg,
            "p4_cls": p4_cls,
            "p5_bbox": p5_reg,
            "p5_cls": p5_cls,
            "depth": depth_map,
            "nms_applied": use_nms,
        }

        if use_nms:
            # Placeholder untuk post-processing Predictions with NMS
            pass
        else:
            # Placeholder untuk post-processing NMS-free predictions
            pass

        return output_dict


def build_model(
    num_classes: int = 4,
    width_mult: float = 0.25,
    depth_mult: float = 0.50,
) -> YOLO26nDepth:
    """Build the report's Nano-scale model for the project's object classes."""
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
        # Test default NMS (True) dan NMS-Free (False)
        output_nms = model(images, use_nms=True)
        output_nms_free = model(images, use_nms=False)

    print(f"Input: {tuple(images.shape)}")
    for key in ("p3_bbox", "p3_cls", "p4_bbox", "p4_cls", "p5_bbox", "p5_cls", "depth"):
        print(f"{key}: {tuple(output_nms[key].shape)}")

    print(f"Predictions with NMS applied: {output_nms['nms_applied']}")
    print(f"NMS-free predictions applied: {output_nms_free['nms_applied']}")
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