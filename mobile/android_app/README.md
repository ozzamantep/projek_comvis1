# YOLO26n-Depth Android

Prototipe Android Native (Kotlin) untuk menggabungkan detector COCO TFLite dan
model depth lokal TFLite. Frame kamera diproses pada perangkat; aplikasi
menampilkan bbox dan kelas pada kamera maupun peta depth Inferno berdampingan.
Jarak per objek diambil sebagai median area bagian tengah bbox; tap-to-measure
tetap tersedia di tampilan kamera.

## Isi proyek

- `app/src/main/java/id/ac/comvis/yolo26depth/MainActivity.kt`: izin kamera,
  CameraX, kontrol confidence/kelas/kalibrasi, tampilan FPS dan latency.
- `InferenceEngine.kt`: pemanggilan dua interpreter LiteRT dan asosiasi depth
  ke area tengah bbox, dengan median nilai dan smoothing tujuh frame.
- `CameraFrameConverter.kt`, `ImagePreprocessor.kt`: konversi frame kamera dan
  preprocessing sesuai kontrak tensor model.
- `YoloPostprocessor.kt`: decoder output detector `[1,84,8400]` dan NMS.
- `DetectionOverlayView.kt`, `DepthMapView.kt`, `DepthMath.kt`: UI, visualisasi,
  sampling median patch dan tap-to-measure.
- `app/src/main/assets/model_config.json`: nama file model, ukuran tensor,
  faktor skala, kalibrasi, dan status verifikasi decoder.
- `app/src/main/assets/detector_model.tflite`: detector COCO yang dipakai aplikasi.
- `data/yolo26n-depth-nyu-calibrated.tflite`: model depth terkalibrasi yang
  disalin ke generated assets saat Gradle build; file lokal ini tidak diduplikasi
  ke repositori.

## Kontrak model dan batasan

- Depth: NCHW `[1,3,640,640]` float32 RGB `/255`; output NCHW
  `[1,1,640,640]` float32, sesuai tensor TFLite hasil ekspor checkpoint. Input
  mempertahankan aspek frame dan memakai letterbox
  berwarna 114; output depth langsung dalam meter dari head terkalibrasi, tanpa
  faktor `×10`. Aplikasi membuang padding sebelum menyelaraskan depth dengan
  frame kamera. Kalibrasi kamera tetap bisa diterapkan dari UI.
- Detector: NCHW `[1,3,640,640]` float32; keluaran `[1,84,8400]`. Decoder dan
  NMS dalam aplikasi masih **belum diverifikasi** dengan gambar nyata terhadap
  keluaran Ultralytics di PC. Banner aplikasi menyatakan status verifikasi ini
  dan menyarankan kalibrasi tambahan untuk kamera yang dipakai.
- Decoder mendeteksi koordinat kandidat ternormalisasi atau pixel, membalik
  letterbox ke koordinat frame portrait, lalu menjalankan class-aware greedy NMS
  berdasarkan urutan confidence menurun. Logcat `YoloDecode` mencatat top raw
  candidates dan jumlah box sebelum/sesudah NMS; `YoloBoxes` mencatat top box
  setelah NMS dalam koordinat frame dan koordinat view Android. Pengamatan pada
  perangkat yang tersedia menunjukkan raw xywh sekitar 0–1, sehingga ditafsirkan
  sebagai koordinat ternormalisasi dan dikalikan 640 sebelum inverse letterbox;
  ini bukan pembandingan hasil terhadap Ultralytics di PC.
- FPS dan latency yang ditampilkan adalah pengukuran runtime perangkat ketika
  aplikasi berjalan. Angka Colab bukan benchmark HP.
- Implementasi mencoba LiteRT GPU delegate dan menyediakan fallback CPU/XNNPACK
  bila inisialisasi GPU gagal. Performa dan kompatibilitas perlu diuji di HP.
- Preview dan ImageAnalysis berbagi CameraX ViewPort/crop. Peta depth dibentangkan
  ke aspek frame upright yang sama saat digambar; area ini hanya menyelaraskan
  koordinat visual, bukan mengubah geometri input/training model.

## Hubungan dengan arsitektur baseline

Arsitektur **YOLO26n-Depth pada checkpoint depth tidak diubah** oleh aplikasi ini:
model depth yang sudah diekspor tetap menerima gambar dan menghasilkan peta
depth. Baseline backbone, neck FPN/PAN, dan `DepthHead` tetap berada dalam
grafik model yang menjadi sumber checkpoint depth.

Perubahan yang ditambahkan berada pada **pipeline deployment**, bukan mengganti
blok internal baseline:

```text
frame kamera
  ├─> detector COCO TFLite (model kedua) ─> bbox + kelas
  └─> YOLO26n-Depth TFLite ─> kedalaman terkalibrasi (meter) ─> median patch di bbox ─> label jarak
```

Model detector dipisah karena model depth baseline tidak dilatih dengan anotasi
bbox COCO. Jadi, dibanding diagram satu model di laporan, aplikasi menjalankan
dua inferensi paralel-sekuensial pada frame yang sama: ini tambahan arsitektur
pipeline dengan biaya latency dan ukuran model, **bukan perubahan backbone,
neck, atau `DepthHead`**. Tidak ada CBAM atau gradient-loss fine-tuning di
checkpoints/aplikasi saat ini. Jika kelak CBAM ditambahkan ke `DepthHead`, itu
perubahan arsitektur model dan perlu retraining serta ekspor ulang.

Checkpoint depth baru sudah menyimpan kalibrasi skala yang dipilih pada data
validasi dan menghasilkan nilai kedalaman dalam meter. Ini tidak menjamin
akurasi pengukuran dunia nyata: NYU indoor dan kamera/objek tangan pada jarak
dekat dapat berbeda domain, sehingga tetap validasikan hasil di kamera target.

Kalibrasi jarak menerima beberapa baris `jarak_asli_m,jarak_model_m`, lalu
meng-fit `jarak_terkalibrasi = a × jarak_model + b` dan menyimpan `a`, `b`, serta
pasangan di SharedPreferences. Masukkan pembacaan model sebelum kalibrasi sebagai
kolom kedua. Model dilatih untuk memprediksi depth langsung yang dinormalisasi
dengan membagi meter terhadap 10, bukan disparity/inverse depth; checkpoint
ekspor mengembalikan unit meter melalui kalibrasi yang tersimpan. Tap membaca
median patch 15×15 dan median temporal tujuh
frame; jarak objek memakai median di area tengah bbox dan median temporal tujuh
frame per track berbasis kelas/IoU. Tombol Ekspor CSV membuat tabel pasangan
validasi dengan jarak asli, nilai model, dan hasil kalibrasi.

## Build dan jalankan

1. Buka folder `mobile/android_app/` di Android Studio.
2. Gunakan JDK 17 dan Android SDK platform 35, lalu jalankan Gradle Sync.
3. Pilih perangkat Android dengan kamera (minSdk 26), lalu jalankan konfigurasi
   `app`. Berikan izin kamera saat diminta.
4. Untuk APK debug dari terminal Windows, jalankan
   `.\gradlew.bat :app:assembleDebug` dari folder proyek ini; output umumnya
   berada di `app/build/outputs/apk/debug/app-debug.apk`.

Build APK dan startup model sudah diverifikasi pada satu perangkat Xiaomi.
Inferensi kamera setelah izin diberikan, kompatibilitas semua operator LiteRT,
FPS HP, dan akurasi decoder belum diverifikasi pada tahap ini.
