# Projek Computer Vision — YOLO26n-Depth

Proyek ini memakai satu model YOLO26n multitask untuk depth metrik dan deteksi.
Satu gambar mengalir melalui backbone YOLO26n dan neck FPN/PAN bersama, lalu
bercabang ke detection head (kotak/kelas) dan depth head (peta jarak). Desktop
dan Android menjalankan graph deployment yang sama dari hasil training terbaru,
`yolo26n_multitask_balanced_v1_20261013-095519-344395/inference/multitask.onnx`,
sehingga deteksi dan depth dihasilkan oleh satu inferensi.

Metadata kontrak model ada di folder `inference/` yang sama. Input ONNX adalah
float32 NCHW `[1,3,480,640]`. Output `det` berbentuk
`[1,300,6]` dan output `depth` berbentuk `[1,1,480,640]` dalam meter.
Validasi ekspor ONNX pada metadata menunjukkan perbedaan kecil dari PyTorch,
tetapi pengukuran jarak tetap perlu diuji pada kamera target.

Evaluasi export baru melaporkan pada split test: depth AbsRel `0.140`,
RMSE `0.481 m`, delta1 `0.819`; deteksi mAP50 `0.313` dan mAP50-95 `0.208`.
Angka ini berasal dari satu run dan split grouped, jadi tidak langsung sebanding
dengan baseline validasi lama. Export memakai checkpoint balanced epoch 17;
checkpoint depth epoch 25 punya AbsRel validasi lebih rendah, tetapi export
deployment dipilih memakai metrik gabungan.

## Aplikasi web lokal

Jalankan server dari root proyek, lalu buka alamat localhost di browser:

```powershell
.\.venv\Scripts\python.exe main.py
```

Secara default aplikasi tersedia di `http://127.0.0.1:8000`. Browser akan
meminta izin kamera; bisa juga pilih foto untuk diuji. Frame diproses oleh
ONNX hasil training terbaru melalui ONNX Runtime lokal. Server hanya bind ke
`127.0.0.1`, jadi tidak dibuka ke jaringan.

Perintah yang tersedia:

```powershell
.\.venv\Scripts\python.exe main.py web --port 8000
.\.venv\Scripts\python.exe main.py camera --help
.\.venv\Scripts\python.exe main.py evaluate --help
.\.venv\Scripts\python.exe main.py validate --help
.\.venv\Scripts\python.exe main.py calibrate --help
.\.venv\Scripts\python.exe -m unittest discover
```

`main.py` menjadi entry point aplikasi Python. `penting/` tetap berisi trainer,
eksperimen, dan tes; `pendukung/` berisi prototipe. Aplikasi Android masih ada
sebagai target opsional, tetapi tidak diperlukan untuk menjalankan web lokal.

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
ONNX hasil training terbaru ke assets saat build; file model tetap lokal dan
tidak perlu diduplikasi ke source tree.
