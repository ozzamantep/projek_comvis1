# Struktur workspace

Kode aplikasi utama diringkas agar mudah ditemukan:

- `main.py` adalah satu entry point Python. Secara default, ia menjalankan
  aplikasi web localhost dengan kamera browser/foto upload dan inferensi ONNX.
  Perintah evaluasi, validasi, kalibrasi, dan kamera desktop juga tersedia.
- `mobile/android_app/app/src/main/java/id/ac/comvis/yolo26depth/MainActivity.kt`
  memuat class Kotlin aplikasi Android dalam satu file.
- `penting/` masih menyimpan trainer arsitektur PDF, alat diagnosis, eksperimen,
  dan tes. Kode itu bukan bagian dari alur aplikasi harian.
- `pendukung/` berisi prototipe, contoh terpisah, runner alternatif, dan
  pemeriksa dataset MATLAB.
- `mobile/export/` berisi alat ekspor model, terpisah dari aplikasi Android.
- `yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/` berisi
  model ONNX deployment dan metadata hasil training terbaru. Model dan checkpoint
  tidak diduplikasi di source tree Android.

## Menjalankan aplikasi web lokal

Jalankan dari root repository agar path relatif `data/` benar. Server hanya
menerima koneksi localhost.

```powershell
.\.venv\Scripts\python.exe main.py
```

Buka `http://127.0.0.1:8000`, izinkan kamera atau pilih foto. Untuk port lain:

```powershell
.\.venv\Scripts\python.exe main.py web --port 8080
```

Perintah lain:

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py camera --help
.\.venv\Scripts\python.exe main.py evaluate --help
.\.venv\Scripts\python.exe main.py validate --help
.\.venv\Scripts\python.exe main.py calibrate --help
```

Jalankan perintah berikut jika ingin menggunakan CLI desktop/analisis:

```powershell
.\.venv\Scripts\python.exe main.py camera
.\.venv\Scripts\python.exe main.py evaluate
.\.venv\Scripts\python.exe main.py validate
.\.venv\Scripts\python.exe main.py calibrate --csv data\metric_validation.csv
```

Perintah evaluasi membutuhkan dataset NYU, validasi membutuhkan foto, dan
kalibrasi membutuhkan CSV pengukuran. Tes tetap dipisah sebagai kode tes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

Trainer dan eksperimen arsitektur PDF yang masih diperlukan dijalankan secara
terpisah, misalnya:

```powershell
.\.venv\Scripts\python.exe penting\train_nyu_colab.py --help
.\.venv\Scripts\python.exe penting\arsitektur_yolo_depth.py --help
```

## Aplikasi Android

Class implementasi Android digabung di `MainActivity.kt`. Unit test Android
tetap berada di `mobile/android_app/app/src/test/` supaya tidak ikut masuk ke
APK. Build debug dari folder `mobile/android_app/`:

```powershell
.\gradlew.bat :app:testDebugUnitTest :app:assembleDebug
```
