# Struktur proyek

- `main.py`: aplikasi kamera terminal untuk deteksi objek dan estimasi depth.
- `multitask_pt.py`: pemuatan checkpoint multitask `.pt` dan inferensi PyTorch.
- `arsitektur.py`: implementasi PyTorch YOLO26n-Depth dari bagian 3.1-3.2
  laporan; belum dilatih dan tidak digunakan oleh kamera.
- `requirements.txt`: dependensi aplikasi kamera berbasis PyTorch.
- `requirements-architecture.txt`: dependensi PyTorch untuk blueprint laporan.
- `tests/`: tes pemrosesan depth, tampilan kamera, dan preprocessing PyTorch.

`multitask_pt.py` memuat checkpoint `multitask_best.pt` secara strict,
membangun detector YOLO26n dari konfigurasi Ultralytics, dan menjalankan depth
head dengan bobot pada checkpoint yang sama. Kontrak model menerima
`[1, 3, 480, 640]`, lalu menghasilkan deteksi terdekode dan depth bermeter.
Checkpoint `.pt` harus tersedia di root proyek atau diberikan lewat argumen
`--model`.

`arsitektur.py` menghasilkan head bbox/kelas mentah pada P3/8, P4/16, dan
P5/32, serta depth relatif `[0, 1]` pada resolusi input 640×640. Ia belum
memiliki bobot terlatih dan bukan model yang dipakai kamera.

Jalankan dari root proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` pada jendela kamera untuk keluar.
