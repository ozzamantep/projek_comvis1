# Struktur workspace

- `main.py` adalah entry point terminal. Tanpa argumen, ia langsung membuka
  kamera; tampilan membagi deteksi/jarak dan peta depth menjadi dua panel.
- `yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/` menyediakan
  model ONNX dan metadata yang dipakai kamera.
- `penting/` menyimpan trainer, alat diagnosis, eksperimen, dan tes.
- `pendukung/` menyimpan prototipe, runner alternatif, dan alat dataset.

Jalankan dari root repository di terminal VS Code:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` di jendela kamera untuk keluar. Perintah analisis terminal dan tes:

```powershell
.\.venv\Scripts\python.exe main.py evaluate --help
.\.venv\Scripts\python.exe main.py validate --help
.\.venv\Scripts\python.exe main.py calibrate --help
.\.venv\Scripts\python.exe -m unittest discover
```

Kamera menggunakan ONNX Runtime lokal; ONNX tetap diperlukan untuk inferensi
karena checkpoint `.pt` yang tersedia tidak bisa dimuat oleh aplikasi kamera
sebagai model PyTorch inference.
