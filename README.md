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

Kalibrasi kamera saat ini menggunakan transformasi `jarak x 0.9936 - 1.2 m`,
berdasarkan pengukuran 60 cm → 90 cm dan 80 cm → 100 cm sebelum kalibrasi ini.
Offset dapat disetel dengan `--depth-offset-m`, sedangkan faktor skala dapat
disetel dengan `--depth-scale`. Kalibrasi ini khusus untuk kamera dan setup
pengukuran tersebut.

## Model

Inferensi kamera langsung memakai checkpoint PyTorch `multitask_best.pt` di
folder proyek. File checkpoint disediakan terpisah dan tidak disertakan dalam
repository; salin file yang kamu unggah ke root proyek. Checkpoint ini berisi
kedua head terlatih (deteksi dan depth),
nama kelas, ukuran input 480×640, serta kalibrasi depth bawaan model. Letakkan
file checkpoint di root proyek atau berikan jalurnya melalui `--model`:

```powershell
.\.venv\Scripts\python.exe main.py --model .\multitask_best.pt
```

Model mendeteksi `person`, `laptop`, `cell phone`, dan `bottle`; output depth
memakai meter. Kode memuat bobot secara aman (`weights_only=True`) dan
memvalidasi semua key secara strict terhadap arsitektur YOLO26n dan depth head.
ONNX Runtime tidak digunakan oleh aplikasi.

## Alur terhadap diagram arsitektur

`main.py` menangani kamera dan tampilan. `multitask_pt.py` membangun arsitektur
yang sesuai dengan checkpoint terlatih:

1. Kamera menghasilkan frame BGR.
2. `MultitaskTorchModel.predict` melakukan resize dengan rasio aspek terjaga,
   padding ke ukuran input model, konversi BGR ke RGB, dan normalisasi.
3. PyTorch menjalankan backbone/neck YOLO26n, detection head, dan depth head
   dari checkpoint `.pt`.
4. Kode mengembalikan kotak deteksi dan peta depth ke ukuran frame kamera;
   median depth di dalam kotak dipakai sebagai jarak objek.

Ukuran input model adalah `[1, 3, 480, 640]` dalam urutan `NCHW`. Model
menghasilkan deteksi `[1, 300, 6]` dan depth `[1, 1, 480, 640]`.

### Implementasi arsitektur dari laporan

`arsitektur.py` tetap merupakan blueprint laporan yang terpisah dan belum
dilatih. Kamera sekarang menggunakan model terlatih yang direkonstruksi dari
checkpoint lewat `multitask_pt.py`, bukan blueprint laporan tersebut. Beberapa
detail model runtime yang terverifikasi:

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
File model ONNX dan metadata ekspor lama tidak dibutuhkan dan sudah dihapus
dari artefak proyek yang dilacak Git.

## Tes

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```
