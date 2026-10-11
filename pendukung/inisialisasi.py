import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Mengunci nilai acak agar hasil eksperimen konsisten dan terulang (reproducible)
torch.manual_seed(42)
np.random.seed(42)

# Konfigurasi optimasi komputasi cuDNN dan presisi FP16 untuk Edge-AI
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
use_fp16 = True

# Deteksi ketersediaan akselerator GPU CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=== VERIFIKASI LINGKUNGAN SISTEM LIGHTWEIGHT YOLO26 ===")
print(f"Versi Python        : {sys.version.split()}")
print(f"Versi PyTorch       : {torch.__version__}")
print(f"Versi OpenCV        : {cv2.__version__}")
print(f"Perangkat Komputasi : {device}")
print(f"Optimasi Edge FP16  : {'Aktif' if use_fp16 else 'Nonaktif'}")