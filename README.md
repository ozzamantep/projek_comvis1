# Projek Computer Vision — YOLO26n-Depth

Proyek ini memakai satu model YOLO26n multitask untuk depth metrik dan deteksi.
Satu gambar mengalir melalui backbone YOLO26n dan neck FPN/PAN bersama, lalu
bercabang ke detection head (kotak/kelas) dan depth head (peta jarak). Desktop
dan Android menjalankan graph deployment yang sama, `data/multitask.onnx`,
sehingga deteksi dan depth dihasilkan oleh satu inferensi.

Checkpoint terlatih `data/multitask_best.pt` diekspor sebagai
`data/multitask.onnx`; `data/model_metadata.json` menjelaskan kontrak model.
Input ONNX adalah float32 NCHW `[1,3,480,640]`. Output `det` berbentuk
`[1,300,6]` dan output `depth` berbentuk `[1,1,480,640]` dalam meter.
Validasi ekspor ONNX pada metadata menunjukkan perbedaan kecil dari PyTorch,
tetapi pengukuran jarak tetap perlu diuji pada kamera target.

## Struktur

- `main.py`: satu entry point desktop untuk kamera, evaluasi NYU, validasi foto,
  kalibrasi, serta helper inference/depth.
- `penting/`: trainer, eksperimen, diagnostik, dan tes yang bukan jalur harian.
- `mobile/android_app/`: semua class Kotlin aplikasi berada di satu file
  `MainActivity.kt`; tes Android tetap terpisah.
- `pendukung/`: contoh/prototipe mandiri dan alat inspeksi MATLAB, bukan jalur
  inferensi utama.
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
python -m pip install numpy opencv-python h5py matplotlib ultralytics onnxruntime
```

Untuk GPU NVIDIA, pilih instalasi PyTorch yang sesuai dengan versi CUDA dari
halaman resmi PyTorch: <https://pytorch.org/get-started/locally/>.
Inferensi desktop multitask memakai `onnxruntime`; trainer dan eksperimen
memakai `ultralytics`; visualisasi memakai `matplotlib`; akses NYU MAT memakai
`h5py`.

Jalankan perintah dari direktori root proyek agar path default `data\...`
berfungsi, contohnya:

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py camera --help
.\.venv\Scripts\python.exe main.py evaluate --help
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

Android memakai sesi ONNX Runtime yang sama dengan desktop. Gradle menyalin
`data/multitask.onnx` ke assets saat build; file model tetap lokal dan tidak
perlu diduplikasi ke source tree.
