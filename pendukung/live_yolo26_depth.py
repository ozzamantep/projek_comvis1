import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO


def colorize_depth(depth_map):
    depth_map = np.asarray(depth_map, dtype=np.float32)
    low = float(np.percentile(depth_map, 2.0))
    high = float(np.percentile(depth_map, 98.0))
    if high - low < 1e-6:
        normalized = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        normalized = np.clip((depth_map - low) / (high - low), 0.0, 1.0)
        normalized = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def main(weights_path, camera_index):
    model = YOLO(weights_path)
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f'Kamera index {camera_index} tidak dapat dibuka.')

    print(f'Model: {weights_path}')
    print("Tekan 'q' untuk keluar.")

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
                    'Checkpoint atau versi Ultralytics ini tidak menghasilkan result.depth. '
                    'Pastikan memakai yolo26n-depth.pt dan checkpoint hasil training depth.'
                )

            depth_map = depth_result.data.detach().cpu().numpy()
            if depth_map.ndim == 3:
                depth_map = depth_map[0]
            depth_color = colorize_depth(depth_map)
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

            combined = np.hstack((frame, depth_color))
            cv2.imshow('YOLO26n-depth Camera', combined)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live camera inference for Ultralytics YOLO26 depth')
    default_weights = next(
        (
            path for path in [
                'data/best.pt',
                'data/pendukung/yolo26n-depth.pt',
            ]
            if os.path.exists(path)
        ),
        'data/best.pt',
    )
    parser.add_argument('--weights', default=default_weights, help='Path ke best.pt hasil training Ultralytics')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    main(args.weights, args.camera)
