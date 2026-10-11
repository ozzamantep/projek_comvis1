# YOLO26n-Depth Android

Prototipe Android Native (Kotlin) untuk menggabungkan detector COCO TFLite dan
model depth lokal TFLite. Frame kamera diproses pada perangkat; aplikasi
menampilkan bbox, kelas, estimasi jarak dari median depth patch di tengah bbox,
Inferno depth map, dan pengukuran titik melalui tap.

## Isi proyek

- `app/src/main/java/id/ac/comvis/yolo26depth/MainActivity.kt`: izin kamera,
  CameraX, kontrol confidence/kelas/kalibrasi, tampilan FPS dan latency.
- `InferenceEngine.kt`: pemanggilan dua interpreter LiteRT dan asosiasi depth
  ke titik tengah bbox.
- `CameraFrameConverter.kt`, `ImagePreprocessor.kt`: konversi frame kamera dan
  preprocessing sesuai kontrak tensor model.
- `YoloPostprocessor.kt`: decoder output detector `[1,84,8400]` dan NMS.
- `DetectionOverlayView.kt`, `DepthMapView.kt`, `DepthMath.kt`: UI, visualisasi,
  sampling median patch dan tap-to-measure.
- `app/src/main/assets/model_config.json`: nama file model, ukuran tensor,
  faktor skala, kalibrasi, dan status verifikasi decoder.
- `app/src/main/assets/depth_model.tflite` dan
  `app/src/main/assets/detector_model.tflite`: salinan model yang dipakai aplikasi.

## Kontrak model dan batasan

- Depth: NHWC `[1,240,320,3]` float32 RGB `/255`; output NHWC
  `[1,240,320,1]` float32. Nilai raw dikalikan 10 sebagai meter, sesuai asumsi
  sementara proyek. Ini belum menjamin jarak metrik akurat di kamera lain.
- Detector: NCHW `[1,3,640,640]` float32; keluaran `[1,84,8400]`. Decoder dan
  NMS dalam aplikasi masih **belum diverifikasi** dengan gambar nyata terhadap
  keluaran Ultralytics di PC. Label peringatan tetap ditampilkan di aplikasi.
- FPS dan latency yang ditampilkan adalah pengukuran runtime perangkat ketika
  aplikasi berjalan. Angka Colab bukan benchmark HP.
- Implementasi mencoba LiteRT GPU delegate dan menyediakan fallback CPU/XNNPACK
  bila inisialisasi GPU gagal. Performa dan kompatibilitas perlu diuji di HP.

## Build dan jalankan

1. Buka folder `mobile/android_app/` di Android Studio.
2. Gunakan JDK 17, Gradle 8.9 atau lebih baru, dan Android SDK platform 35,
   lalu jalankan Gradle Sync. Jika memakai Android Studio, pilih instalasi
   Gradle lokal karena Gradle wrapper belum disertakan.
3. Pilih perangkat Android dengan kamera (minSdk 26), lalu jalankan konfigurasi
   `app`. Berikan izin kamera saat diminta.
4. Untuk APK debug, jalankan task Gradle `:app:assembleDebug`; output umumnya
   berada di `app/build/outputs/apk/debug/app-debug.apk`.

Gradle wrapper belum disertakan. APK, pemasangan ke perangkat, kompatibilitas
operator LiteRT, FPS HP, dan akurasi decoder belum diverifikasi pada tahap ini.
