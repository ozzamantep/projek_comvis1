# YOLO26n multitask Android

Aplikasi web localhost adalah antarmuka utama yang disarankan untuk penggunaan
di komputer. Android di bawah ini merupakan target opsional yang masih
dipertahankan.

Aplikasi Android menjalankan satu graph ONNX multitask dari
`data/multitask.onnx`, hasil export terbaru dari `data/multitask_best.pt`.
Satu input gambar menghasilkan dua keluaran:
`det` untuk bounding box/kelas dan `depth` untuk peta jarak. Nilai jarak objek
diambil sebagai median depth di bagian tengah bounding box, lalu diratakan
secara temporal.

## Model yang dipakai

- Checkpoint sumber: `data/multitask_best.pt`.
- ONNX deployment: `data/multitask.onnx` (disalin Gradle ke assets).
- Input: `images`, float32 NCHW `[1, 3, 480, 640]`, RGB dengan nilai `/255`.
- Keluaran deteksi: `det`, float32 `[1, 300, 6]`, baris berformat
  `[x1, y1, x2, y2, confidence, class_id]`; seleksi dan NMS sudah berada
  di dalam graph model.
- Keluaran depth: `depth`, float32 `[1, 1, 480, 640]`. Aplikasi memakai
  skala meter 1.0 dari konfigurasi, membuang padding letterbox, lalu sampling
  median depth di bagian tengah bbox.
- Nama kelas dan urutannya ada di `app/src/main/assets/model_config.json` dan
  sama dengan daftar 30 kelas di `data/model_metadata.json`.

Android menggunakan satu sesi ONNX Runtime untuk meminta kedua output dalam
satu inferensi per frame. Deteksi dan depth bukan lagi dua interpreter/model
terpisah pada jalur Android.

Implementasi class aplikasi Android disatukan di
`app/src/main/java/id/ac/comvis/yolo26depth/MainActivity.kt` agar mudah
ditemukan. Unit test tetap di folder `app/src/test/` dan tidak masuk ke APK.

## Kalibrasi jarak

Kalibrasi linear dan marker QR tetap merupakan post-processing atas keluaran
depth model. Untuk pasangan manual, masukkan `jarak_asli_m,jarak_model_m`;
kolom kedua harus bacaan depth sebelum kalibrasi. Ukur hasil di kamera dan
perangkat yang akan dipakai. Kalibrasi tidak mengubah bobot ONNX.

Marker QR 10 cm harus dicetak pada ukuran sebenarnya (tanpa fit-to-page).
Aplikasi hanya menyimpan auto-kalibrasi jika jumlah sampel, rentang, dan mutu
fit memenuhi ambang yang ditentukan.

## Build

1. Buka `mobile/android_app/` di Android Studio.
2. Gunakan JDK 17 dan Android SDK 35, lalu Gradle Sync.
3. Jalankan `app` pada perangkat Android (minSdk 26) dan izinkan kamera.
4. Untuk APK debug, dari folder ini jalankan
   `gradlew.bat :app:assembleDebug`.

Aplikasi menjalankan ONNX Runtime CPU saat ini. FPS dan akurasi jarak tetap
perlu dicek pada HP target; integrasi model tidak menjamin hasil metrik
selalu tepat di semua kamera atau kondisi ruangan.
