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


def load_model_with_weights(weights_path='data/yolo26n_depth_nyu_full.pt'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    load_depth_checkpoint(model, weights_path, device)
    print(f"Loaded weights from: {weights_path}")
    model.eval()
    return model, device


def preprocess_image_for_model(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
    return img.unsqueeze(0)


def infer_depth_from_image(model, image_bgr, device):
    img = preprocess_image_for_model(image_bgr).to(device)

    with torch.no_grad():
        pred = model(img)

    depth_map = pred['depth'][0, 0].cpu().numpy()
    depth_norm = normalize_depth_for_display(depth_map)
    depth_vis = (depth_norm * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
    return depth_color


def normalize_depth_for_display(depth_map):
    low = float(np.percentile(depth_map, 2.0))
    high = float(np.percentile(depth_map, 98.0))
    depth_range = high - low

    if depth_range < 0.01:
        return np.zeros_like(depth_map, dtype=np.float32)

    return np.clip((depth_map - low) / depth_range, 0.0, 1.0).astype(np.float32)


def run_image_inference(image_path, weights_path='data/yolo26n_depth_trained.pt'):
    model, device = load_model_with_weights(weights_path)
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image_bgr is None:
        print(f"Gagal membaca gambar: {image_path}")
        return

    depth_color = infer_depth_from_image(model, image_bgr, device)
    h, w = image_bgr.shape[:2]
    depth_color = cv2.resize(depth_color, (w, h))

    combined = np.hstack([image_bgr, depth_color])
    cv2.imshow("Image + Depth Map (YOLO26n-depth)", combined)
    print(f"Menampilkan hasil inferensi dari: {image_path}")
    print("Tekan tombol apapun pada jendela untuk menutup.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_first_image_path(image_dir=None):
    candidates = []
    if image_dir and os.path.isdir(image_dir):
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    candidates.append(os.path.join(root, file))

    if candidates:
        return sorted(candidates)[0]

    for folder in ['data/train/images', 'data/val/images', 'data']:
        if os.path.isdir(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        return os.path.join(root, file)
    return None


def run_live_camera(weights_path='data/yolo26n_depth_trained.pt'):
    model, device = load_model_with_weights(weights_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak tersedia atau tidak bisa dibuka.")
        print("Coba jalankan mode image untuk melihat hasil depth tanpa webcam.")
        return

    print(f"Model berjalan pada device: {device}")
    print("Tekan 'q' untuk keluar dari jendela kamera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb.transpose(2, 0, 1)).float() / 255.0
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img)

        depth_map = pred['depth'][0, 0].cpu().numpy()
        depth_norm = normalize_depth_for_display(depth_map)
        depth_vis = (depth_norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        h, w = frame.shape[:2]
        depth_color = cv2.resize(depth_color, (w, h))

        raw_range = float(depth_map.max() - depth_map.min())
        if raw_range < 0.01:
            cv2.putText(
                depth_color,
            'DEPTH BELUM VALID - TRAIN DENGAN DATA RGB-D NYATA',
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        combined = np.hstack([frame, depth_color])
        cv2.imshow("Webcam + Depth Map (YOLO26n-depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ==============================================================
# 5. Training pipeline (pseudo-depth / synthetic depth)
# ==============================================================

def make_synthetic_depth_from_image(image_bgr):
    """Membuat label kedalaman sintetis dari citra RGB sebagai fallback saat dataset depth tidak tersedia."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    normalized = gray.astype(np.float32) / 255.0

    center_bias = 1.0 - np.abs(np.linspace(-1, 1, normalized.shape[1], dtype=np.float32))
    center_bias = np.tile(center_bias.reshape(1, -1), (normalized.shape[0], 1))
    synthetic = normalized * 0.7 + center_bias * 0.3
    synthetic = np.clip(synthetic, 0.0, 1.0)
    return synthetic.astype(np.float32)


def collect_training_files(image_dir, depth_dir=None):
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    image_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                image_files.append(os.path.join(root, file))
    image_files = sorted(image_files)

    if depth_dir is None:
        return image_files, None

    depth_files = []
    for root, _, files in os.walk(depth_dir):
        for file in files:
            if file.lower().endswith(valid_ext):
                depth_files.append(os.path.join(root, file))
    depth_files = sorted(depth_files)
    return image_files, depth_files


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


def train_depth_model(image_dir, depth_dir=None, epochs=2, batch_size=1, learning_rate=1e-3, save_path='yolo26n_depth_trained.pt'):
    """Melatih model depth dengan label pseudo-depth jika dataset depth tidak ada."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    image_files, depth_files = collect_training_files(image_dir, depth_dir)

    if len(image_files) == 0:
        print("Tidak ada file gambar yang ditemukan di folder input.")
        return model

    if depth_files is not None and len(depth_files) != len(image_files):
        print(f"Jumlah file depth ({len(depth_files)}) tidak sama dengan gambar ({len(image_files)}). Akan menggunakan pseudo-depth otomatis.")
        depth_files = None

    print(f"Memulai training depth pada {len(image_files)} gambar dengan {epochs} epoch.")
    print(f"Device aktif: {device}")

    for epoch in range(epochs):
        running_loss = 0.0
        count = 0

        for idx in range(0, len(image_files), batch_size):
            batch_paths = image_files[idx:idx + batch_size]
            batch_images = []
            batch_targets = []

            for path in batch_paths:
                img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(device)
                batch_images.append(img_tensor)

                if depth_files is not None:
                    depth_path = depth_files[image_files.index(path)] if path in image_files else None
                    if depth_path is not None:
                        depth_bgr = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                        if depth_bgr is not None:
                            depth_map = depth_bgr.astype(np.float32) / 255.0
                        else:
                            depth_map = make_synthetic_depth_from_image(img_bgr)
                    else:
                        depth_map = make_synthetic_depth_from_image(img_bgr)
                else:
                    depth_map = make_synthetic_depth_from_image(img_bgr)

                target = torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).float().to(device)
                batch_targets.append(target)

            if len(batch_images) == 0:
                continue

            x = torch.cat(batch_images, dim=0)
            y = torch.cat(batch_targets, dim=0)

            optimizer.zero_grad()
            output = model(x)
            pred_depth = output['depth']
            loss = criterion(pred_depth, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            count += 1

        avg_loss = running_loss / max(1, count)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model berhasil dilatih dan disimpan ke: {save_path}")
    return model


# ==============================================================
# 6. CLI runner
# ==============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='YOLO26n-depth training + webcam demo')
    parser.add_argument('--mode', type=str, choices=['camera', 'train', 'train_nyu', 'verify', 'image'], default='camera')
    parser.add_argument('--image-dir', type=str, default='')
    parser.add_argument('--depth-dir', type=str, default='')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='yolo26n_depth_trained.pt')
    parser.add_argument('--weights', type=str, default='data/yolo26n_depth_trained.pt')
    parser.add_argument('--image-path', type=str, default='')
    parser.add_argument('--nyu-path', type=str, default='data/nyu_depth_v2_labeled.mat')
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--input-width', type=int, default=320)
    parser.add_argument('--input-height', type=int, default=240)
    args = parser.parse_args()

    if args.mode == 'train':
        if not args.image_dir:
            print("Untuk mode train, gunakan --image-dir <folder_gambar>.")
            print("Contoh: python arsitektur_yolo_depth.py --mode train --image-dir data/train")
            raise SystemExit(1)
        train_depth_model(
            image_dir=args.image_dir,
            depth_dir=args.depth_dir if args.depth_dir else None,
            epochs=args.epochs,
            batch_size=1,
            learning_rate=args.lr,
            save_path=args.save_path,
        )
    elif args.mode == 'train_nyu':
        if not os.path.exists(args.nyu_path):
            print(f'File NYU tidak ditemukan: {args.nyu_path}')
            raise SystemExit(1)
        train_nyu_depth_model(
            mat_path=args.nyu_path,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_path=args.save_path,
            max_samples=args.max_samples,
            image_size=(args.input_width, args.input_height),
        )
    elif args.mode == 'image':
        image_path = args.image_path or find_first_image_path(args.image_dir or '')
        if not image_path:
            print("Mode image membutuhkan --image-path <gambar> atau folder gambar yang valid.")
            raise SystemExit(1)
        run_image_inference(image_path=image_path, weights_path=args.weights)
    elif args.mode == 'verify':
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
    else:
        run_live_camera(weights_path=args.weights)
