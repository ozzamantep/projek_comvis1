# Struktur workspace

Kode aplikasi utama diringkas agar mudah ditemukan:

- `main.py` adalah satu entry point Python untuk aplikasi desktop. Helper ONNX,
  sampling/smoothing depth, kamera, evaluasi, validasi foto, dan kalibrasi
  berada di file yang sama.
- `mobile/android_app/app/src/main/java/id/ac/comvis/yolo26depth/MainActivity.kt`
  memuat class Kotlin aplikasi Android dalam satu file.
- `penting/` masih menyimpan trainer arsitektur PDF, alat diagnosis, eksperimen,
  dan tes. Kode itu bukan bagian dari alur aplikasi harian.
- `pendukung/` berisi prototipe, contoh terpisah, runner alternatif, dan
  pemeriksa dataset MATLAB.
- `mobile/export/` berisi alat ekspor model, terpisah dari aplikasi Android.
- `data/multitask.onnx` adalah model deployment lokal. Model dan checkpoint
  tidak diduplikasi di source tree Android.

## Menjalankan aplikasi desktop

Jalankan dari root repository agar path relatif `data/` benar.

```powershell
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe main.py camera --help
.\.venv\Scripts\python.exe main.py evaluate --help
.\.venv\Scripts\python.exe main.py validate --help
.\.venv\Scripts\python.exe main.py calibrate --help
```

`camera` adalah perintah default. Contoh menjalankan fitur:

```powershell
.\.venv\Scripts\python.exe main.py
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
