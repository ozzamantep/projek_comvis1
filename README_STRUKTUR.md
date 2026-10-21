# Struktur proyek

- `main.py`: aplikasi kamera terminal untuk deteksi objek dan estimasi depth.
- `architecture_pt.py`: pemuatan checkpoint training `arsitektur.py` dan
  decoding output deteksi/depth untuk aplikasi kamera.
- `multitask_pt.py`: pemuatan checkpoint multitask lama `.pt` dan inferensi
  PyTorch.
- `arsitektur.py`: implementasi custom YOLO26n-Depth dari bagian 3.1-3.2
  laporan, dengan backbone, neck FPN/PAN, detection head per skala, dan depth head.
- `train_arsitektur.py`: pipeline training terpisah untuk `arsitektur.py`,
  memakai SUN RGB-D dan opsional anotasi COCO instances 2017.
- `requirements.txt`: dependensi aplikasi kamera berbasis PyTorch.
- `requirements-architecture.txt`: dependensi untuk verifikasi/training model.
- `tests/`: tes pemrosesan depth, tampilan kamera, dan preprocessing PyTorch.

`multitask_pt.py` memuat checkpoint `multitask_best.pt` secara strict,
membangun detector YOLO26n dari konfigurasi Ultralytics, dan menjalankan depth
head dengan bobot pada checkpoint yang sama. Kontrak model menerima
`[1, 3, 480, 640]`, lalu menghasilkan deteksi terdekode dan depth bermeter.
Checkpoint `.pt` harus tersedia di root proyek atau diberikan lewat argumen
`--model`.

`arsitektur.py` menghasilkan prediksi bbox/kelas mentah pada P3/8, P4/16, dan
P5/32. Depth head memproyeksikan ketiga fitur ke 32 channel, menggabungkannya
pada ukuran input, dan menghasilkan depth relatif `[0, 1]`. Checkpoint hasil
`train_arsitektur.py` dapat dipakai kamera melalui `architecture_pt.py`; depth
yang tampil tetap perkiraan dan belum terkalibrasi terhadap kamera.

`train_arsitektur.py` memakai assignment target dan loss deteksi custom sesuai
blueprint, ditambah loss depth. SUN RGB-D memberi target box, kelas, dan depth;
COCO (opsional) memberi target box/kelas saja. Checkpoint dari trainer ini hanya
dapat dimuat dengan arsitektur yang sama dan tidak dapat dibuka oleh
`multitask_pt.py`. Jika data tidak ditemukan, trainer mengunduh SUN RGB-D ke
`datasets/SUNRGBD` dan subset gambar COCO dengan kelas target ke `datasets/coco`.

Jalankan dari root proyek:

```powershell
.\.venv\Scripts\python.exe main.py --model .\best.pt
```

Checkpoint lama tetap dapat dijalankan dengan `--model .\multitask_best.pt`.
Tekan `q` pada jendela kamera untuk keluar.
