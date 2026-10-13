"""Independent PyTorch blueprint inspired by the shared detection/depth diagram.

This is a new, untrained implementation. It does not recreate the trained ONNX
weights or guarantee internal layer-for-layer compatibility with that export.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _scaled_channels(channels: int, width_mult: float) -> int:
    return max(8, int(round(channels * width_mult / 8) * 8))


def _scaled_repeats(repeats: int, depth_mult: float) -> int:
    return max(1, int(round(repeats * depth_mult)))


def _attention_heads(channels: int) -> int:
    return max(1, min(4, channels // 32))


class ConvBNAct(nn.Module):
    """Conv2d + BatchNorm + SiLU block used throughout the blueprint."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            ConvBNAct(channels, channels, 3),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.layers(x)


class C3k2(nn.Module):
    """Two-branch C3k2-style block with a compact bottleneck path."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 1) -> None:
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.branch_a = ConvBNAct(in_channels, hidden, 1)
        self.branch_b = ConvBNAct(in_channels, hidden, 1)
        self.blocks = nn.Sequential(*(Bottleneck(hidden) for _ in range(repeats)))
        self.fuse = ConvBNAct(hidden * 2, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fuse(torch.cat((self.branch_a(x), self.blocks(self.branch_b(x))), dim=1))


class SpatialAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        heads = _attention_heads(channels)
        while channels % heads:
            heads -= 1
        self.heads = heads
        self.head_channels = channels // heads
        self.scale = self.head_channels**-0.5
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.projection = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = height * width
        qkv = self.qkv(x).reshape(
            batch, 3, self.heads, self.head_channels, tokens
        )
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attended = torch.matmul(weights.softmax(dim=-1), value)
        attended = attended.transpose(-2, -1).reshape(batch, channels, height, width)
        return self.projection(attended)


class PSABlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.attention = SpatialAttention(channels)
        self.feed_forward = nn.Sequential(
            ConvBNAct(channels, channels * 2, 1),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(x)
        return self.act(x + self.feed_forward(x))


class C2PSA(nn.Module):
    """Partial self-attention block on the high-level backbone feature."""

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 1) -> None:
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.project_in = ConvBNAct(in_channels, hidden * 2, 1)
        self.attention_blocks = nn.Sequential(*(PSABlock(hidden) for _ in range(repeats)))
        self.project_out = ConvBNAct(hidden * 2, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        bypass, attended = self.project_in(x).chunk(2, dim=1)
        return self.project_out(torch.cat((bypass, self.attention_blocks(attended)), dim=1))


class SPPF(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden = max(8, in_channels // 2)
        self.reduce = ConvBNAct(in_channels, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
        self.fuse = ConvBNAct(hidden * 4, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.reduce(x)
        pooled_1 = self.pool(x)
        pooled_2 = self.pool(pooled_1)
        pooled_3 = self.pool(pooled_2)
        return self.fuse(torch.cat((x, pooled_1, pooled_2, pooled_3), dim=1))


class Backbone(nn.Module):
    """Stem and P2/4-P5/32 stages, including SPPF then C2PSA at P5."""

    def __init__(self, channels: Sequence[int], repeats: Sequence[int]) -> None:
        super().__init__()
        c2, c3, c4, c5 = channels
        r2, r3, r4, r5 = repeats
        self.stem = ConvBNAct(3, max(8, c2 // 2), 3, 2)
        self.stage_p2 = nn.Sequential(
            ConvBNAct(max(8, c2 // 2), c2, 3, 2),
            C3k2(c2, c2, r2),
        )
        self.stage_p3 = nn.Sequential(
            ConvBNAct(c2, c3, 3, 2),
            C3k2(c3, c3, r3),
        )
        self.stage_p4 = nn.Sequential(
            ConvBNAct(c3, c4, 3, 2),
            C3k2(c4, c4, r4),
        )
        self.stage_p5 = nn.Sequential(
            ConvBNAct(c4, c5, 3, 2),
            C3k2(c5, c5, r5),
            SPPF(c5, c5),
            C2PSA(c5, c5),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        p2 = self.stage_p2(self.stem(x))
        p3 = self.stage_p3(p2)
        p4 = self.stage_p4(p3)
        p5 = self.stage_p5(p4)
        return p2, p3, p4, p5


def _upsample_like(x: Tensor, reference: Tensor) -> Tensor:
    return F.interpolate(x, size=reference.shape[-2:], mode="nearest")


class Neck(nn.Module):
    """Top-down FPN followed by bottom-up PAN, returning P3/P4/P5."""

    def __init__(self, channels: Sequence[int], repeats: int) -> None:
        super().__init__()
        c3, c4, c5 = channels
        self.fpn_p4 = C3k2(c5 + c4, c4, repeats)
        self.fpn_p3 = C3k2(c4 + c3, c3, repeats)
        self.pan_p4_down = ConvBNAct(c3, c3, 3, 2)
        self.pan_p4 = C3k2(c3 + c4, c4, repeats)
        self.pan_p5_down = ConvBNAct(c4, c4, 3, 2)
        self.pan_p5 = C3k2(c4 + c5, c5, repeats)

    def forward(
        self, p3: Tensor, p4: Tensor, p5: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        fpn_p4 = self.fpn_p4(torch.cat((_upsample_like(p5, p4), p4), dim=1))
        fpn_p3 = self.fpn_p3(torch.cat((_upsample_like(fpn_p4, p3), p3), dim=1))
        pan_p4 = self.pan_p4(torch.cat((self.pan_p4_down(fpn_p3), fpn_p4), dim=1))
        pan_p5 = self.pan_p5(torch.cat((self.pan_p5_down(pan_p4), p5), dim=1))
        return fpn_p3, pan_p4, pan_p5


class DetectionLevel(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.box = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, 4, 1),
        )
        self.classes = nn.Sequential(
            ConvBNAct(channels, channels, 3),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.cat((self.box(x), self.classes(x)), dim=1)


class DetectionHead(nn.Module):
    """Separate one-to-many/one-to-one branches; decode only one-to-one in eval."""

    def __init__(
        self,
        channels: Sequence[int],
        num_classes: int,
        max_detections: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.max_detections = max_detections
        self.strides = (8, 16, 32)
        self.one_to_many = nn.ModuleList(
            DetectionLevel(channel, num_classes) for channel in channels
        )
        self.one_to_one = nn.ModuleList(
            DetectionLevel(channel, num_classes) for channel in channels
        )

    def forward(
        self, features: Sequence[Tensor], input_hw: tuple[int, int]
    ) -> dict[str, list[Tensor] | Tensor]:
        one_to_one_features = (
            [feature.detach() for feature in features] if self.training else features
        )
        one_to_one = [
            head(feature)
            for head, feature in zip(self.one_to_one, one_to_one_features)
        ]
        if self.training:
            one_to_many = [
                head(feature) for head, feature in zip(self.one_to_many, features)
            ]
            return {
                "one_to_many": one_to_many,
                "one_to_one": one_to_one,
            }
        return {"decoded": self._decode(one_to_one, input_hw)}

    def _decode(
        self, raw_outputs: Sequence[Tensor], input_hw: tuple[int, int]
    ) -> Tensor:
        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        for raw, stride in zip(raw_outputs, self.strides):
            batch, _, height, width = raw.shape
            box_logits, class_logits = raw.split((4, self.num_classes), dim=1)
            distances = F.softplus(box_logits).permute(0, 2, 3, 1).reshape(
                batch, height * width, 4
            ) * stride
            class_scores = class_logits.sigmoid().permute(0, 2, 3, 1).reshape(
                batch, height * width, self.num_classes
            )
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=raw.device, dtype=raw.dtype),
                torch.arange(width, device=raw.device, dtype=raw.dtype),
                indexing="ij",
            )
            centers = torch.stack((grid_x + 0.5, grid_y + 0.5), dim=-1)
            centers = centers.reshape(1, height * width, 2) * stride
            top_left = centers - distances[..., :2]
            bottom_right = centers + distances[..., 2:]
            boxes = torch.cat((top_left, bottom_right), dim=-1)
            all_boxes.append(boxes)
            all_scores.append(class_scores)

        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        flat_scores, flat_indices = scores.flatten(1).topk(
            min(self.max_detections, scores.shape[1] * scores.shape[2]), dim=1
        )
        box_indices = torch.div(flat_indices, self.num_classes, rounding_mode="floor")
        class_ids = flat_indices.remainder(self.num_classes).to(boxes.dtype)
        selected_boxes = boxes.gather(
            1, box_indices.unsqueeze(-1).expand(-1, -1, 4)
        )
        input_height, input_width = input_hw
        selected_boxes[..., (0, 2)] = selected_boxes[..., (0, 2)].clamp(
            0, input_width
        )
        selected_boxes[..., (1, 3)] = selected_boxes[..., (1, 3)].clamp(
            0, input_height
        )
        return torch.cat(
            (selected_boxes, flat_scores.unsqueeze(-1), class_ids.unsqueeze(-1)),
            dim=-1,
        )


class DepthHead(nn.Module):
    """Fuse neck features with P2 and predict a positive full-resolution depth map."""

    def __init__(
        self,
        p2_channels: int,
        neck_channels: Sequence[int],
        hidden_channels: int,
        max_depth_m: float,
    ) -> None:
        super().__init__()
        c3, c4, c5 = neck_channels
        self.max_depth_m = max_depth_m
        self.project_p5 = ConvBNAct(c5, hidden_channels, 1)
        self.project_p4 = ConvBNAct(c4, hidden_channels, 1)
        self.project_p3 = ConvBNAct(c3, hidden_channels, 1)
        self.project_p2 = ConvBNAct(p2_channels, hidden_channels, 1)
        self.fuse_p4 = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.fuse_p3 = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.fuse_p2 = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.refine = ConvBNAct(hidden_channels, hidden_channels, 3)
        self.output = nn.Conv2d(hidden_channels, 1, 1)

    def forward(
        self,
        p2: Tensor,
        features: Sequence[Tensor],
        output_hw: tuple[int, int],
    ) -> Tensor:
        p3, p4, p5 = features
        x = self.project_p5(p5)
        x = self.fuse_p4(_upsample_like(x, p4) + self.project_p4(p4))
        x = self.fuse_p3(_upsample_like(x, p3) + self.project_p3(p3))
        x = self.fuse_p2(_upsample_like(x, p2) + self.project_p2(p2))
        x = self.refine(x)
        x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False)
        return self.output(x).sigmoid() * self.max_depth_m


class YOLO26nMultitaskBlueprint(nn.Module):
    """Diagram-inspired detector plus monocular-depth model, without pretrained weights."""

    def __init__(
        self,
        num_classes: int = 4,
        max_detections: int = 300,
        width_mult: float = 0.25,
        depth_mult: float = 0.5,
        max_depth_m: float = 10.0,
    ) -> None:
        super().__init__()
        if num_classes < 1 or max_detections < 1:
            raise ValueError("num_classes and max_detections must be positive.")
        if width_mult <= 0 or depth_mult <= 0 or max_depth_m <= 0:
            raise ValueError("width_mult, depth_mult, and max_depth_m must be positive.")

        channels = tuple(
            _scaled_channels(base, width_mult) for base in (64, 128, 256, 512)
        )
        repeats = tuple(
            _scaled_repeats(base, depth_mult) for base in (1, 2, 2, 1)
        )
        self.backbone = Backbone(channels, repeats)
        neck_channels = channels[1:]
        self.neck = Neck(neck_channels, _scaled_repeats(2, depth_mult))
        self.detection_head = DetectionHead(
            neck_channels, num_classes, max_detections
        )
        self.depth_head = DepthHead(
            channels[0],
            neck_channels,
            hidden_channels=channels[1],
            max_depth_m=max_depth_m,
        )

    def forward(self, images: Tensor) -> dict[str, Tensor | list[Tensor]]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width].")
        if images.shape[-2] < 32 or images.shape[-1] < 32:
            raise ValueError("image height and width must each be at least 32.")

        input_hw = (images.shape[-2], images.shape[-1])
        p2, backbone_p3, backbone_p4, backbone_p5 = self.backbone(images)
        features = self.neck(backbone_p3, backbone_p4, backbone_p5)
        detection = self.detection_head(features, input_hw)
        depth = self.depth_head(p2, features, input_hw)
        if self.training:
            return {
                "det_one_to_many": detection["one_to_many"],
                "det_one_to_one": detection["one_to_one"],
                "depth": depth,
            }
        return {"det": detection["decoded"], "depth": depth}


def build_model(
    num_classes: int = 4,
    max_detections: int = 300,
    width_mult: float = 0.25,
    depth_mult: float = 0.5,
) -> YOLO26nMultitaskBlueprint:
    return YOLO26nMultitaskBlueprint(
        num_classes=num_classes,
        max_detections=max_detections,
        width_mult=width_mult,
        depth_mult=depth_mult,
    )


def _verify() -> None:
    model = build_model().eval()
    images = torch.zeros(1, 3, 128, 160)
    with torch.inference_mode():
        output = model(images)
    print(f"Input: {tuple(images.shape)}")
    print(f"Detections: {tuple(output['det'].shape)}")
    print(f"Depth: {tuple(output['depth'].shape)}")
    print(f"Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify the untrained diagram-inspired PyTorch architecture."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a forward-pass smoke test on a small rectangular input.",
    )
    args = parser.parse_args()
    if not args.verify:
        parser.error("Use --verify to run the architecture smoke test.")
    _verify()
