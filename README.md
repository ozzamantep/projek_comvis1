# YOLO26n Kamera: Deteksi dan Depth

Aplikasi desktop ini membuka kamera langsung dari terminal VS Code. Satu model
multitask mendeteksi objek dan memperkirakan depth, lalu menampilkan deteksi
beserta jaraknya di panel kiri dan peta depth di panel kanan.

## Menjalankan

Pasang dependensi sekali:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jalankan kamera dari root folder proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` untuk keluar, `s` untuk mengganti tampilan, `d` untuk menyalakan atau
mematikan overlay, dan klik gambar untuk membaca jarak pada titik tersebut.
Kamera selain perangkat default dapat dipilih, misalnya:

```powershell
.\.venv\Scripts\python.exe main.py --camera 1
```

## Model

Inferensi memakai ONNX Runtime dan model ONNX multitask berikut:

`yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/multitask.onnx`

Metadata model berada di folder `inference/` yang sama. Model mendeteksi kelas
`person`, `laptop`, `cell phone`, dan `bottle`; output depth memakai satuan
meter. Checkpoint `.pt`, dataset, dan berkas training tidak diperlukan untuk
menjalankan kamera.

## Alur terhadap diagram arsitektur

`main.py` merupakan runner inferensi, bukan implementasi layer YOLO26n.
Backbone, neck, dan kedua head dijalankan bersama di dalam graph ONNX:

1. Kamera menghasilkan frame BGR.
2. `MultitaskOnnxModel.predict` melakukan resize dengan rasio aspek terjaga,
   padding ke ukuran input model, konversi BGR ke RGB, dan normalisasi.
3. ONNX Runtime menjalankan satu graph multitask untuk deteksi dan depth.
4. Kode mengembalikan kotak deteksi dan peta depth ke ukuran frame kamera;
   median depth di dalam kotak dipakai sebagai jarak objek.

Ukuran input model yang sebenarnya adalah `[1, 3, 480, 640]` dalam urutan
`NCHW`, bukan gambar persegi `640 x 640` pada diagram. Output graph adalah
deteksi `[1, 300, 6]` dan depth `[1, 1, 480, 640]`. Diagram menunjukkan jalur
detection YOLO26n; aplikasi ini mempertahankan jalur tersebut sekaligus output
depth tambahan. Nama blok internal pada diagram tidak didefinisikan ulang di
`main.py`; saat runtime, implementasinya berasal dari graph ONNX terlatih.

### Blueprint PyTorch terpisah

`arsitektur.py` berisi rancangan mandiri yang mengikuti alur diagram: backbone
C3k2 dengan SPPF/C2PSA, neck FPN+PAN, head deteksi one-to-many/one-to-one, dan
depth head tambahan. Detail channel/repeat adalah konfigurasi blueprint karena
diagram tidak memuat seluruh parameter internal. Model baru ini tidak memiliki
bobot terlatih, bukan salinan layer persis model ONNX, dan tidak menggantikan
model yang dipakai kamera.

Opsional, pasang PyTorch lalu jalankan pemeriksaan forward pass:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-architecture.txt
.\.venv\Scripts\python.exe arsitektur.py --verify
```

## Tes

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```
