# YOLO26n Kamera: Deteksi dan Depth

Aplikasi desktop ini membuka kamera langsung dari terminal VS Code. Satu model
multitask mendeteksi objek dan memperkirakan depth, lalu menampilkan deteksi
beserta jarak perkiraannya di panel kiri dan peta depth di panel kanan.

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

Kalibrasi kamera saat ini menggunakan transformasi `jarak x 0.9936 - 1.0 m`.
Offset dinaikkan 20 cm setelah pembacaan 40 cm pada jarak aktual 60 cm.
Kalibrasi ini hanya menambahkan koreksi pada titik pengukuran tersebut;
pengukuran lain yang pernah dicatat belum konsisten, jadi akurasi pada jarak
lain belum tervalidasi. Offset dapat disetel dengan `--depth-offset-m`,
sedangkan faktor skala dapat disetel dengan `--depth-scale`.

## Model

Inferensi kamera mendukung dua format checkpoint PyTorch. Untuk melihat hasil
training dari notebook Colab, salin `best.pt` ke root proyek lalu jalankan:

```powershell
.\.venv\Scripts\python.exe main.py --model .\best.pt
```

Checkpoint training `best.pt` dibangun ulang dari `arsitektur.py` lewat
`architecture_pt.py`; checkpoint lama `multitask_best.pt` tetap didukung
lewat `multitask_pt.py`. Jenis checkpoint dikenali otomatis. `best.pt`
mendeteksi `person`, `laptop`, `cell phone`, dan `bottle`. Depth ditampilkan
sebagai perkiraan meter dari output model dan belum dikalibrasi terhadap kamera;
kalibrasi webcam checkpoint lama tidak diterapkan secara default. Atur koreksi
secara manual dengan `--depth-scale` dan `--depth-offset-m` jika diperlukan.
Bobot dimuat aman (`weights_only=True`) dan divalidasi secara strict.

## Alur terhadap diagram arsitektur

`main.py` menangani kamera dan tampilan. Format checkpoint menentukan model
inferensi yang digunakan:

1. Kamera menghasilkan frame BGR.
2. Untuk `best.pt`, `architecture_pt.py` merekonstruksi `build_model()` dari
   `arsitektur.py`; untuk `multitask_best.pt`, `multitask_pt.py` memuat model
   lama.
3. Kedua model memproses frame, menghasilkan deteksi dan peta depth, lalu
   mengembalikan keduanya ke ukuran kamera. Median depth dalam kotak dipakai
   sebagai perkiraan jarak objek.

Checkpoint Colab menyimpan ukuran input, kelas, faktor skala arsitektur, dan
batas depth sehingga tidak perlu mengatur ulang konfigurasi model.

### Model checkpoint lama

`multitask_best.pt` menggunakan implementasi dan kontrak output yang berbeda
dari blueprint training. Model tersebut dimuat lewat `multitask_pt.py`.
Beberapa detail model runtime yang terverifikasi:

- Backbone memakai konfigurasi YOLO26n dari Ultralytics dan menghasilkan fitur
  P3/8, P4/16, dan P5/32. Depth head mengambil fitur dari keluaran layer 16, 19,
  dan 22.
- Neck menggabungkan P3/P4/P5 dengan alur FPN top-down dan PAN bottom-up.
- Detection head one-to-one menghasilkan hingga 300 prediksi terdekode,
  masing-masing dengan koordinat kotak, confidence, dan class ID.
- Depth head memproyeksikan fitur P3/P4/P5 masing-masing ke 256 channel,
  menggabungkannya secara top-down, lalu mengeluarkan depth meter terkalibrasi.
- Bobot detection head menggunakan saluran internal kelas berukuran 80 dari
  model pralatih, dengan classifier akhir empat kelas dari checkpoint.

Kesetaraan rekonstruksi sudah dibandingkan dengan ekspor resmi dari run yang
sama: bobot deteksi dimuat strict; selisih maksimum depth sekitar `1.5e-6 m`
pada input uji, dan selisih confidence deteksi maksimum sekitar `7e-8`.
File model ONNX dan metadata ekspor lama tidak dibutuhkan.

### Training dengan blueprint `arsitektur.py`

`train_arsitektur.py` adalah pipeline terpisah yang melatih blueprint custom
YOLO26n-Depth di `arsitektur.py`, bukan checkpoint yang dipakai aplikasi kamera.
Model mengikuti blok backbone, neck FPN/PAN, detection head per skala, serta
depth head yang dijelaskan pada bagian 3.1-3.2 laporan. Pipeline ini membaca
struktur resmi SUN RGB-D (`image/`, `depth/` atau `depth_bfx/`,
`annotation2Dfinal/index.json`) dan, opsional, anotasi COCO instances 2017.
Default kelas mengikuti eksperimen notebook lama: `person`, `laptop`,
`cell phone`, dan `bottle`. COCO hanya memberi supervision deteksi; piksel
depth-nya dimask karena COCO tidak menyediakan depth map.

Install dependensi lalu jalankan dari root proyek. Untuk training lokal di
Windows, gunakan [`train_arsitektur_laptop.ipynb`](./train_arsitektur_laptop.ipynb)
dan pilih kernel `.venv` proyek. Pastikan kernel memakai build PyTorch CUDA;
installer PyTorch yang sesuai dapat dipilih di
[halaman resmi PyTorch](https://pytorch.org/get-started/locally/). Notebook
memeriksa CUDA sebelum memulai dan memakai batch size 2, AMP, serta `last.pt`
untuk resume—konfigurasi awal yang lebih sesuai untuk GPU laptop 6 GB.

Jika dataset belum ada,
trainer otomatis mengunduh SUN RGB-D lengkap dan menaruhnya di `datasets\`
di dalam folder proyek. Karena `--coco-ratio` default-nya 0.3, trainer juga
mengunduh anotasi COCO 2017 lalu hanya gambar train yang diperlukan dan
memiliki kelas target—bukan semua gambar COCO. Jika unduhan anotasi gagal
karena sertifikat HTTPS pada `images.cocodataset.org`, trainer menggunakan
endpoint HTTPS path-style Amazon S3 untuk bucket COCO yang sama, dengan
verifikasi sertifikat tetap aktif. Notebook laptop juga menerima file anotasi
lokal `instances_train2017.json`; trainer CLI menerima path alternatif melalui
`--coco-annotations`.

Untuk menjalankannya di Google Colab, buka
[`train_arsitektur_colab.ipynb`](./train_arsitektur_colab.ipynb), jalankan cell
berurutan, lalu unggah `arsitektur.py` dan `train_arsitektur.py` terbaru saat
diminta. Notebook memakai Drive untuk menyimpan arsip dataset dan checkpoint.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-architecture.txt
.\.venv\Scripts\python.exe train_arsitektur.py `
  --epochs 50 --batch-size 2 --workers 0 --coco-ratio 0.3 --device cuda --amp
```

Unduhan SUN RGB-D sekitar 6,5 GB terkompresi dan dataset juga perlu diekstrak,
jadi pastikan koneksi internet serta ruang disk mencukupi. Unduhan bisa
dilanjutkan dengan menjalankan perintah yang sama lagi jika koneksi terputus.
Lokasi default arsip adalah `datasets\SUNRGBD.zip`, sedangkan data hasil
ekstraksi berada di `datasets\SUNRGBD\`. COCO memakai anotasi
`datasets\coco\annotations\instances_train2017.json` dan gambar terpilih di
`datasets\coco\train2017\`. Gunakan `--download-dir` untuk mengubah folder
tersebut, `--sun-root` atau `--coco-root` untuk memakai lokasi dataset lain,
dan `--no-download` agar trainer gagal dengan pesan jelas jika data belum ada.
COCO dapat dimatikan dengan `--coco-ratio 0`. Gunakan `--classes` untuk
mengganti kelas, dan
`--max-train-samples 200 --max-val-samples 40 --epochs 1` untuk uji singkat
pipeline. Resolusi `--img-size H W` harus kelipatan 32.

`runs/arsitektur/last.pt` menyimpan state untuk resume dengan `--resume
runs/arsitektur/last.pt`; `best.pt` dipilih dari validation loss gabungan.
`best.pt` hasil pipeline ini dapat langsung dipakai pada aplikasi kamera
dengan perintah di atas. Checkpoint blueprint lama yang memiliki struktur dan
parameter model sama tetap didukung; `multitask_best.pt` menggunakan jalur
runtime terpisah.
Depth dilatih sebagai nilai relatif `meter / --depth-max-m` karena output
depth head memakai sigmoid. Checkpoint ini **tidak kompatibel** dengan
`multitask_pt.py` atau `multitask_best.pt`. Pipeline memakai assignment target
per skala dan loss deteksi custom seperti blueprint, serta mencetak loss
deteksi/depth dan MAE depth validasi; metrik mAP belum dihitung oleh trainer.
Loss klasifikasi memisahkan rata-rata lokasi positif dan negatif agar label
objek yang jarang tidak tenggelam oleh jutaan lokasi background. Perubahan ini
tidak kompatibel secara makna dengan loss/checkpoint training sebelumnya;
notebook laptop terbaru memakai `runs/arsitektur-laptop-v2/` dan memulai
training baru, tidak melanjutkan checkpoint lama.

## Tes

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```
