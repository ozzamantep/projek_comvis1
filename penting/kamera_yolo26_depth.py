import argparse
import os

import cv2
import numpy as np
import torch

if __package__:
    from .arsitektur_yolo_depth import YOLO26nDepth, load_depth_checkpoint
else:
    from arsitektur_yolo_depth import YOLO26nDepth, load_depth_checkpoint


def normalize_depth(depth_map):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    low = float(np.percentile(depth_map, 2.0))
    high = float(np.percentile(depth_map, 98.0))
    if high - low < 0.01:
        return np.zeros_like(depth_map, dtype=np.uint8)
    normalized = np.clip((depth_map - low) / (high - low), 0.0, 1.0)
    normalized = 1.0 - normalized
    return (normalized * 255.0).astype(np.uint8)


def colorize_depth(depth_map):
    return cv2.applyColorMap(normalize_depth(depth_map), cv2.COLORMAP_INFERNO)


def load_local_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    load_depth_checkpoint(model, weights_path, device)
    model.eval()
    print(f'Backend: local PyTorch | Device: {device}')
    return model, device


def run_local_camera(weights_path, camera_index):
    model, device = load_local_model(weights_path)
    camera = open_camera(camera_index)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print('Gagal membaca frame dari kamera.')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255.0)
            image = image.unsqueeze(0).to(device)
            with torch.no_grad():
                depth = model(image)['depth'][0, 0].cpu().numpy()

            depth_color = cv2.resize(colorize_depth(depth), (frame.shape[1], frame.shape[0]))
            show_camera_result(frame, depth_color, 'YOLO26n-depth Local')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        close_camera(camera)


def run_ultralytics_camera(weights_path, camera_index):
    from ultralytics import YOLO

    model = YOLO(weights_path)
    camera = open_camera(camera_index)
    print(f'Backend: Ultralytics | Model: {weights_path}')

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print('Gagal membaca frame dari kamera.')
                break

            result = model(frame, verbose=False)[0]
            depth_result = getattr(result, 'depth', None)
            if depth_result is None:
                raise RuntimeError(
                    'Model tidak menghasilkan result.depth. Pastikan checkpoint adalah model depth Ultralytics.'
                )

            depth = depth_result.data.detach().cpu().numpy()
            if depth.ndim == 3:
                depth = depth[0]
            depth_color = cv2.resize(colorize_depth(depth), (frame.shape[1], frame.shape[0]))
            show_camera_result(frame, depth_color, 'YOLO26n-depth Ultralytics')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        close_camera(camera)


def open_camera(camera_index):
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f'Kamera index {camera_index} tidak dapat dibuka.')
    print("Tekan 'q' untuk keluar.")
    return camera


def close_camera(camera):
    camera.release()
    cv2.destroyAllWindows()


def show_camera_result(frame, depth_color, title):
    combined = np.hstack((frame, depth_color))
    cv2.imshow(title, combined)


def detect_backend(weights_path, requested_backend):
    if requested_backend != 'auto':
        return requested_backend
    filename = os.path.basename(weights_path).lower()
    if filename in {'best.pt', 'yolo26n-depth.pt'}:
        return 'ultralytics'
    return 'local'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified YOLO26n-depth camera runner')
    parser.add_argument('--weights', default='data/yolo26n_depth_nyu_full.pt')
    parser.add_argument('--backend', choices=['auto', 'local', 'ultralytics'], default='auto')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit(f'Checkpoint tidak ditemukan: {args.weights}')

    backend = detect_backend(args.weights, args.backend)
    if backend == 'local':
        run_local_camera(args.weights, args.camera)
    else:
        run_ultralytics_camera(args.weights, args.camera)