# Projek Computer Vision — YOLO26n-Depth

Proyek tugas Computer Vision ini mengembangkan pipeline berbasis YOLO26n-Depth
untuk estimasi kedalaman dan jarak metrik (meter/cm), dengan sasaran porting
dan inferensi pada perangkat mobile. Pipeline desktop saat ini memisahkan
deteksi objek COCO (Ultralytics) dari prediksi depth lokal yang dilatih pada
NYU Depth V2. Backbone, neck FPN+PAN, detection head multi-skala, dan cabang
DepthHead mengikuti arsitektur pada diagram laporan PDF.

`data/depthnet_best.pt` tidak memakai arsitektur `YOLO26nDepth` pada laporan:
nama kunci tensor checkpoint tersebut tidak cocok dengan model proyek. Karena
itu checkpoint tersebut tidak dipasang ke model dan loader akan menolak bobot
yang arsitekturnya berbeda. Untuk mempertahankan arsitektur PDF, latih ulang
dengan trainer proyek dan simpan hasilnya ke file terpisah:

```powershell
.\.venv\Scripts\python.exe penting\train_nyu_colab.py `
  --nyu-path data\nyu_depth_v2_labeled.mat `
  --output data\yolo26n_depth_pdf_best.pt `
  --loss-plot data\yolo26n_depth_pdf_loss.png
```

Setelah training, berikan `data\yolo26n_depth_pdf_best.pt` lewat opsi
`--depth-weights` pada runner kamera atau `--weights` pada evaluator/diagnostik.
Checkpoint PDF lama yang kompatibel tetap tersedia; training baru tidak
menimpa file tersebut.

Konversi depth lokal saat ini menggunakan asumsi sementara
`raw_depth × 10 meter`; angka tersebut bukan jaminan akurasi jarak kamera tanpa
validasi dan kalibrasi untuk kamera yang digunakan.

## Struktur

- `penting/`: arsitektur `YOLO26nDepth`, trainer NYU, runner kamera, helper depth,
  evaluasi NYU, validasi foto jarak, kalibrasi, diagnostik preprocessing, dan tes.
- `pendukung/`: skrip simulasi/prototipe, runner alternatif, dan alat inspeksi
  MATLAB.
- `data/`: checkpoint dan data lokal yang diperlukan oleh beberapa perintah.
- `data/pendukung/`: checkpoint eksperimental dan model alternatif.
- `README_STRUKTUR.md`: daftar fungsi file dan perintah kerja lebih lengkap.

## Persiapan environment

Disarankan Python 3.10 atau lebih baru. Dari PowerShell pada Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install numpy opencv-python h5py matplotlib ultralytics
```

Untuk GPU NVIDIA, pilih instalasi PyTorch yang sesuai dengan versi CUDA dari
halaman resmi PyTorch: <https://pytorch.org/get-started/locally/>.
Deteksi COCO membutuhkan paket `ultralytics`; visualisasi/training menggunakan
`matplotlib`; akses NYU MAT memakai `h5py`.

Jalankan perintah dari direktori root proyek agar path default `data\...`
berfungsi, contohnya:

```powershell
.\.venv\Scripts\python.exe penting\camera_yolo26_depth_metric.py --help
.\.venv\Scripts\python.exe penting\evaluate_depth.py --help
.\.venv\Scripts\python.exe -m unittest discover
```

Di workspace ini, hook Git lokal mencoba menjalankan `git push origin main`
setelah setiap commit; jika push gagal, commit tetap berhasil dan hook memberi
peringatan.

## Dataset NYU Depth V2

Dataset NYU Depth V2 (`.mat`, sekitar 2.8 GiB) **tidak disertakan di repository**.
Unduh dataset dari halaman resmi NYU Depth V2:
<https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>

Letakkan file `nyu_depth_v2_labeled.mat` di folder `data/` lokal. Semua file
`.mat` diabaikan Git oleh `.gitignore`; checkpoint model `.pt` yang sudah ada
tetap dikelola sebagai file proyek karena ukuran masing-masing di bawah 100 MB.

## Catatan mobile

Target implementasi mobile yang direncanakan adalah Android Native (Kotlin)
dengan LiteRT. Ekspor dan validasi model mobile belum dinyatakan selesai hanya
dengan adanya kode desktop di repository.
