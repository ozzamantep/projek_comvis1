import cv2
import numpy as np
import torch

def letterbox(img, target_size=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    rasio = min(target_h / h, target_w / w)
    
    new_w = int(round(w * rasio))
    new_h = int(round(h * rasio))

    citra_skala = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_w - new_w
    dh = target_h - new_h
    atas, bawah = dh // 2, dh - (dh // 2)
    kiri, kanan = dw // 2, dw - (dw // 2)

    citra_padded = cv2.copyMakeBorder(
        citra_skala, atas, bawah, kiri, kanan,
        cv2.BORDER_CONSTANT, value=color
    )
    return citra_padded

def pipeline_pra_pemrosesan(citra_bgr, use_fp16=False):
    citra_kotak = letterbox(citra_bgr, target_size=(640, 640))
    citra_rgb = cv2.cvtColor(citra_kotak, cv2.COLOR_BGR2RGB)
    citra_norm = citra_rgb.astype(np.float32) / 255.0
    tensor_rgb = torch.from_numpy(citra_norm).permute(2, 0, 1)

    if use_fp16:
        tensor_rgb = tensor_rgb.half()

    return tensor_rgb

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    citra_uji_1 = np.random.randint(0, 256, (480, 720, 3), dtype=np.uint8)
    citra_uji_2 = np.random.randint(0, 256, (1080, 1080, 3), dtype=np.uint8)

    tensor_1 = pipeline_pra_pemrosesan(citra_uji_1, use_fp16=True)
    tensor_2 = pipeline_pra_pemrosesan(citra_uji_2, use_fp16=True)

    batch_rgb = torch.stack([tensor_1, tensor_2]).to(device)

    print("=== VERIFIKASI PRA-PEMROSESAN CITRA LIGHTWEIGHT YOLO26 ===")
    print(f"Dimensi Tensor Batch RGB : {batch_rgb.shape} -> [Batch, Channel, Height, Width]")
    print(f"Tipe Data Tensor         : {batch_rgb.dtype}")
    print(f"Rentang Nilai Piksel     : Min = {batch_rgb.min():.2f}, Max = {batch_rgb.max():.2f}")
    print(f"Perangkat Aktif          : {batch_rgb.device}")