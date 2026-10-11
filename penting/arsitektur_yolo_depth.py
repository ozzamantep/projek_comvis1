import os
import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================
# 1. Basic blocks
# ==============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cv1 = ConvBlock(ch, ch, k=3, s=1, p=1)
        self.cv2 = ConvBlock(ch, ch, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.cv2 = ConvBlock(c1, c_, k=1, s=1, p=0)
        self.m = nn.ModuleList([Bottleneck(c_) for _ in range(n)])
        self.cv3 = ConvBlock(2 * c_, c2, k=1, s=1, p=0)

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        for m in self.m:
            y2 = m(y2)
        return self.cv3(torch.cat((y1, y2), dim=1))


class C2PSA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = ConvBlock(c1, c2, k=1, s=1, p=0)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, 1),
            nn.SiLU(),
            nn.Conv2d(c2 // 4, c2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cv1(x)
        return x * self.attn(x)


class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = ConvBlock(in_ch, hidden_ch, k=1, s=1, p=0)
        self.cv2 = ConvBlock(hidden_ch * 4, out_ch, k=1, s=1, p=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


# ==============================================================
# 2. Detection head (multi-scale)
# ==============================================================
class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_classes=80):
        super().__init__()
        self.reg = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, 4, kernel_size=1),
        )
        self.cls = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.reg(x), self.cls(x)


# ==============================================================
# 3. Depth head
# ==============================================================
class DepthHead(nn.Module):
    def __init__(self, channels=(64, 128, 256)):
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

    def forward(self, x_list, target_size):
        p3, p4, p5 = x_list

        up_p3 = F.interpolate(self.conv_p3(p3), size=target_size, mode='bilinear', align_corners=False)
        up_p4 = F.interpolate(self.conv_p4(p4), size=target_size, mode='bilinear', align_corners=False)
        up_p5 = F.interpolate(self.conv_p5(p5), size=target_size, mode='bilinear', align_corners=False)

        fused = torch.cat([up_p3, up_p4, up_p5], dim=1)
        return self.refine(fused)


# ==============================================================
# 4. YOLOv26n-depth-inspired model
# ==============================================================
class YOLO26nDepth(nn.Module):
    def __init__(self, num_classes=80, width_mult=0.25, depth_mult=0.50):
        super().__init__()

        c_stem = max(32, int(64 * width_mult))
        c_p2 = max(32, int(128 * width_mult))
        c_p3 = max(64, int(256 * width_mult))
        c_p4 = max(128, int(512 * width_mult))
        c_p5 = max(256, int(1024 * width_mult))
        n_repeat = max(1, round(2 * depth_mult))

        # Backbone
        self.stem = ConvBlock(3, c_stem, k=3, s=2, p=1)
        self.l2 = C3k2(c_stem, c_p2, n=n_repeat)
        self.l3 = ConvBlock(c_p2, c_p3, k=3, s=2, p=1)
        self.l4 = C3k2(c_p3, c_p3, n=n_repeat)
        self.l5 = ConvBlock(c_p3, c_p4, k=3, s=2, p=1)
        self.l6 = C3k2(c_p4, c_p4, n=n_repeat)
        self.l7 = ConvBlock(c_p4, c_p5, k=3, s=2, p=1)
        self.l8 = C3k2(c_p5, c_p5, n=n_repeat)
        self.l9 = SPPF(c_p5, c_p5, k=5)
        self.l10 = C2PSA(c_p5, c_p5)

        # Neck (FPN + PAN)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.n13 = C3k2(c_p5 + c_p4, c_p4, n=n_repeat)
        self.n16 = C3k2(c_p4 + c_p3, c_p3, n=n_repeat)
        self.d17 = ConvBlock(c_p3, c_p3, k=3, s=2, p=1)
        self.n19 = C3k2(c_p3 + c_p4, c_p4, n=n_repeat)
        self.d20 = ConvBlock(c_p4, c_p4, k=3, s=2, p=1)
        self.n22 = C3k2(c_p4 + c_p5, c_p5, n=n_repeat)

        # Multi-scale detection heads
        self.detect_p3 = DetectionHead(c_p3, num_classes=num_classes)
        self.detect_p4 = DetectionHead(c_p4, num_classes=num_classes)
        self.detect_p5 = DetectionHead(c_p5, num_classes=num_classes)

        # Depth head
        self.depth_head = DepthHead(channels=(c_p3, c_p4, c_p5))

    def forward(self, x):
        h, w = x.shape[-2], x.shape[-1]

        # Backbone
        x = self.stem(x)
        x = self.l2(x)
        x = self.l3(x)
        p3 = self.l4(x)

        x = self.l5(p3)
        p4 = self.l6(x)

        x = self.l7(p4)
        x = self.l8(x)
        x = self.l9(x)
        p5 = self.l10(x)

        # Neck: FPN + PAN
        p4_fused = self.n13(torch.cat([self.up(p5), p4], dim=1))
        p3_out = self.n16(torch.cat([self.up(p4_fused), p3], dim=1))

        p4_out = self.n19(torch.cat([self.d17(p3_out), p4_fused], dim=1))
        p5_out = self.n22(torch.cat([self.d20(p4_out), p5], dim=1))

        # Detection outputs at three scales, similar to YOLO's P3/P4/P5 heads
        p3_reg, p3_cls = self.detect_p3(p3_out)
        p4_reg, p4_cls = self.detect_p4(p4_out)
        p5_reg, p5_cls = self.detect_p5(p5_out)

        # Depth map from fused multi-scale features
        depth_map = self.depth_head([p3_out, p4_out, p5_out], target_size=(h, w))

        return {
            'p3_bbox': p3_reg,
            'p3_cls': p3_cls,
            'p4_bbox': p4_reg,
            'p4_cls': p4_cls,
            'p5_bbox': p5_reg,
            'p5_cls': p5_cls,
            'depth': depth_map,
        }


def load_depth_checkpoint(
    model: nn.Module,
    weights_path: str | os.PathLike[str],
    device: torch.device,
) -> None:
    """Load only weights compatible with the active YOLO26n-Depth architecture."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Depth checkpoint not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state_dict checkpoint at {weights_path}; "
            f"got {type(state_dict).__name__}."
        )
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as error:
        raise ValueError(
            f"Checkpoint '{weights_path}' is incompatible with YOLO26nDepth "
            "defined in arsitektur_yolo_depth.py (the architecture in the PDF). "
            "Use weights trained with that model; do not bypass this check."
        ) from error


class NYUDepthV2Dataset:
    def __init__(self, mat_path, indices=None, image_size=(320, 240)):
        self.mat_path = mat_path
        self.image_size = image_size
        self.file = h5py.File(mat_path, 'r')
        self.images = self.file['images']
        self.depths = self.file['depths']
        sample_count = self.images.shape[0]
        self.indices = list(range(sample_count)) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image_rgb = np.asarray(self.images[index]).transpose(2, 1, 0)
        depth_map = np.asarray(self.depths[index]).transpose(1, 0).astype(np.float32)

        image_bgr = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        image_bgr = cv2.resize(image_bgr, self.image_size, interpolation=cv2.INTER_AREA)
        depth_map = cv2.resize(depth_map, self.image_size, interpolation=cv2.INTER_NEAREST)
        depth_map = np.clip(depth_map / 10.0, 0.0, 1.0)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
        depth_tensor = torch.from_numpy(depth_map).unsqueeze(0).float()
        return image_tensor, depth_tensor

    def close(self):
        self.file.close()


def train_nyu_depth_model(mat_path, epochs=5, learning_rate=1e-4, save_path='data/yolo26n_depth_nyu.pt', max_samples=0, image_size=(320, 240)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with h5py.File(mat_path, 'r') as dataset_file:
        sample_count = dataset_file['images'].shape[0]

    if max_samples > 0:
        sample_count = min(sample_count, max_samples)

    indices = np.arange(sample_count)
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    split = max(1, int(sample_count * 0.9))
    train_set = NYUDepthV2Dataset(mat_path, indices[:split], image_size=image_size)
    val_set = NYUDepthV2Dataset(mat_path, indices[split:], image_size=image_size)
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss(reduction='none')

    print(f'Memulai training NYU Depth V2: {len(train_set)} train, {len(val_set)} validasi.')
    print(f'Device aktif: {device}; input: {image_size[0]}x{image_size[1]}')

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for index in range(len(train_set)):
                image_tensor, target = train_set[index]
                image_tensor = image_tensor.unsqueeze(0).to(device)
                target = target.unsqueeze(0).to(device)

                optimizer.zero_grad()
                prediction = model(image_tensor)['depth']
                valid = target > 0
                loss = criterion(prediction, target)[valid].mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for index in range(len(val_set)):
                    image_tensor, target = val_set[index]
                    prediction = model(image_tensor.unsqueeze(0).to(device))['depth']
                    target = target.unsqueeze(0).to(device)
                    valid = target > 0
                    validation_loss += criterion(prediction, target)[valid].mean().item()

            train_avg = train_loss / max(1, len(train_set))
            val_avg = validation_loss / max(1, len(val_set))
            print(f'Epoch [{epoch + 1}/{epochs}] - Train: {train_avg:.6f} - Val: {val_avg:.6f}')
    finally:
        train_set.close()
        val_set.close()

    torch.save(model.state_dict(), save_path)
    print(f'Model NYU berhasil disimpan ke: {save_path}')
    return model


# ==============================================================
# 6. CLI runner
# ==============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify or train the YOLO26n-Depth architecture on NYU."
    )
    parser.add_argument(
        "--mode",
        choices=("verify", "train_nyu"),
        default="verify",
    )
    parser.add_argument("--nyu-path", default="data/nyu_depth_v2_labeled.mat")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", default="data/yolo26n_depth_nyu.pt")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=240)
    args = parser.parse_args()

    if args.mode == "train_nyu":
        if not os.path.exists(args.nyu_path):
            parser.error(f"File NYU tidak ditemukan: {args.nyu_path}")
        train_nyu_depth_model(
            mat_path=args.nyu_path,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_path=args.save_path,
            max_samples=args.max_samples,
            image_size=(args.input_width, args.input_height),
        )
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = YOLO26nDepth(num_classes=80, width_mult=0.25, depth_mult=0.50).to(device)
        model.eval()

        x = torch.randn(2, 3, 640, 640, device=device)
        with torch.no_grad():
            out = model(x)

        print('=== VERIFIKASI YOLO26N-DEPTH (MIRIP DIAGRAM) ===')
        print(f'Input                : {tuple(x.shape)}')
        print(f'P3 bbox              : {tuple(out["p3_bbox"].shape)}')
        print(f'P3 class             : {tuple(out["p3_cls"].shape)}')
        print(f'P4 bbox              : {tuple(out["p4_bbox"].shape)}')
        print(f'P4 class             : {tuple(out["p4_cls"].shape)}')
        print(f'P5 bbox              : {tuple(out["p5_bbox"].shape)}')
        print(f'P5 class             : {tuple(out["p5_cls"].shape)}')
        print(f'Depth map            : {tuple(out["depth"].shape)}')
        print(f'Device               : {device}')
        print(f'Param count          : {sum(p.numel() for p in model.parameters()):,}')
        print('Status               : RUNNING OK')
