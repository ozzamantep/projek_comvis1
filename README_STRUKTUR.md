# Struktur proyek

- `main.py`: aplikasi kamera terminal untuk deteksi objek dan estimasi depth.
- `arsitektur.py`: blueprint PyTorch mandiri, belum dilatih dan tidak digunakan
  oleh kamera.
- `requirements.txt`: dependensi minimum aplikasi kamera.
- `requirements-architecture.txt`: dependensi opsional untuk blueprint PyTorch.
- `tests/`: tes pemrosesan depth, tampilan kamera, dan inferensi ONNX.
- `yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/`: model ONNX
  aktif beserta metadata kontrak input/output.

`main.py` menangani preprocessing frame, pemanggilan graph ONNX, pemulihan
koordinat, dan tampilan. Backbone, neck, detection head, serta depth head
dieksekusi di dalam graph ONNX; detail arsitekturnya bukan layer PyTorch di
`main.py`. Kontrak model aktif menerima `[1, 3, 480, 640]` dan menghasilkan
deteksi serta peta depth.

`arsitektur.py` adalah implementasi PyTorch baru yang terpisah dan belum
memiliki bobot terlatih. Ia tidak kompatibel sebagai pengganti langsung untuk
ONNX kamera.

Jalankan dari root proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` pada jendela kamera untuk keluar.
