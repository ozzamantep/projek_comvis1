# Struktur proyek

- `main.py`: aplikasi kamera terminal untuk deteksi objek dan estimasi depth.
- `requirements.txt`: dependensi minimum aplikasi kamera.
- `tests/`: tes pemrosesan depth, tampilan kamera, dan inferensi ONNX.
- `yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/`: model ONNX
  aktif beserta metadata kontrak input/output.

Jalankan dari root proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` pada jendela kamera untuk keluar.
