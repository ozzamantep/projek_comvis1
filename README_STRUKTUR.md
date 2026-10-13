# Struktur proyek

- `main.py`: aplikasi kamera terminal untuk deteksi objek dan estimasi depth.
- `arsitektur.py`: implementasi PyTorch YOLO26n-Depth dari bagian 3.1-3.2
  laporan; belum dilatih dan tidak digunakan oleh kamera.
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

`arsitektur.py` menghasilkan head bbox/kelas mentah pada P3/8, P4/16, dan
P5/32, serta depth relatif `[0, 1]` pada resolusi input 640×640. Ia belum
memiliki bobot terlatih dan bukan pengganti ONNX kamera, yang menghasilkan
deteksi terdekode serta depth bermeter pada input 640×480.

Jalankan dari root proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` pada jendela kamera untuk keluar.
