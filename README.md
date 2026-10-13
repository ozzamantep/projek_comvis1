# Projek Computer Vision — YOLO26n-Depth

Proyek ini menjalankan kamera langsung dari terminal VS Code memakai satu model
YOLO26n multitask untuk depth metrik dan deteksi. Satu gambar mengalir melalui
backbone YOLO26n dan neck FPN/PAN bersama, lalu bercabang ke detection head
(kotak/kelas) dan depth head (peta jarak). Kamera memakai graph deployment hasil
training terbaru, yaitu
`yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/multitask.onnx`,
sehingga deteksi dan depth dihasilkan oleh satu inferensi lokal.

Metadata kontrak model ada di folder `inference/` yang sama. Input ONNX adalah
float32 NCHW `[1,3,480,640]`. Output `det` berbentuk
`[1,300,6]` dan output `depth` berbentuk `[1,1,480,640]` dalam meter.
Model deteksi ini memiliki empat kelas: `person`, `laptop`, `cell phone`, dan
`bottle`.
Validasi ekspor ONNX pada metadata menunjukkan perbedaan kecil dari PyTorch,
tetapi pengukuran jarak tetap perlu diuji pada kamera target.

Laporan run `coco4_v1` mencatat pada split test: depth AbsRel `0.160`,
RMSE `0.542 m`, delta1 `0.770`; deteksi mAP50 `0.365` dan mAP50-95 `0.275`.
Export deployment menggunakan checkpoint metrik gabungan epoch 10 dari training
yang berhenti awal pada epoch 18. Angka ini berasal dari satu run dan split
grouped, serta jumlah/kelas deteksinya berbeda dari run sebelumnya; jangan
membandingkannya langsung dengan baseline validasi lama.

## Menjalankan kamera dari terminal VS Code

```powershell
.\.venv\Scripts\python.exe main.py
```

Tanpa argumen, program langsung membuka kamera desktop. Tampilan awal dibagi
dua: deteksi dan jarak tiap objek di kiri, peta depth dan jarak tiap objek di
kanan. Tekan `s` untuk mengganti tampilan, klik gambar untuk membaca jarak
titik, dan `q` untuk keluar. Pilih kamera lain dengan `--camera 1`.

Perintah terminal lain:

```powershell
.\.venv\Scripts\python.exe main.py camera --help
.\.venv\Scripts\python.exe main.py evaluate --help
.\.venv\Scripts\python.exe main.py validate --help
.\.venv\Scripts\python.exe main.py calibrate --help
.\.venv\Scripts\python.exe -m unittest discover
```

Jarak titik tengah juga ditampilkan di panel depth.
Kamera memakai faktor kalibrasi `0.4968` (median rasio) dari sepuluh pasangan
pengukuran yang diberikan, termasuk pengukuran terbaru yang dicatat setelah
faktor sebelumnya diterapkan. Bacaan baru 85–86 cm untuk jarak asli 50 cm hanya
sedikit mengubah faktor dari `0.5000`, sehingga hasilnya masih sekitar 85 cm,
bukan 50 cm. Ini mengoreksi skala rata-rata saja, bukan respons jarak yang
berubah secara tidak konsisten: bacaan 150 cm sebelumnya lebih kecil daripada
bacaan 100 cm. Jarak hasilnya masih perkiraan dan faktor ini khusus untuk setup
kamera tersebut. Faktor dapat diganti dengan
`--depth-scale`, misalnya
`.\.venv\Scripts\python.exe main.py camera --depth-scale 0.60`.

`main.py` adalah entry point terminal. Kode trainer, eksperimen, dan tes ada di
`penting/`; prototipe dan runner tambahan ada di `pendukung/`.

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

ONNX Runtime dan file ONNX multitask tetap diperlukan untuk inferensi kamera
terminal. Checkpoint `.pt` multitask hanya menyediakan bobot untuk arsitektur
training yang tidak disertakan sebagai model PyTorch inference di proyek ini.

Di workspace ini, hook Git lokal mencoba menjalankan `git push origin main`
setelah setiap commit; jika push gagal, commit tetap berhasil dan hook memberi
peringatan.

## Dataset NYU Depth V2

Dataset NYU Depth V2 (`.mat`, sekitar 2.8 GiB) **tidak disertakan di repository**.
Unduh dataset dari halaman resmi NYU Depth V2:
<https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html>

Letakkan file `nyu_depth_v2_labeled.mat` di folder `data/` lokal. Semua file
`.mat` diabaikan Git oleh `.gitignore`. Checkpoint `.pt` dipakai oleh trainer
atau alat diagnosis tertentu, bukan oleh kamera multitask live.
