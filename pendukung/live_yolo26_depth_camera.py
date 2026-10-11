import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "penting"))
from arsitektur_yolo_depth import YOLO26nDepth


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO26nDepth(width_mult=0.25, depth_mult=0.50).to(device)
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera tidak tersedia atau tidak bisa dibuka.")
        print("Pastikan webcam terhubung dan izin kamera sudah diberikan.")
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

        depth_map = pred[0, 0].cpu().numpy()
        depth_min = float(depth_map.min())
        depth_max = float(depth_map.max())
        if depth_max - depth_min < 1e-6:
            depth_norm = np.zeros_like(depth_map, dtype=np.float32)
        else:
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)

        depth_vis = (depth_norm * 255.0).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # Resize agar tampil padat di layar
        h, w = frame.shape[:2]
        depth_color = cv2.resize(depth_color, (w, h))

        combined = np.hstack([frame, depth_color])
        cv2.imshow("Webcam + Depth Map (YOLO26n-depth)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
