# Ekspor model Android: YOLO26n detector + YOLO26n-Depth

Tahap ini menghasilkan model detector COCO dan model depth lokal dalam format
`.tflite` dengan bobot FP16. Model depth dipertahankan terpisah dari detector:

- `data/yolo26n.pt`: checkpoint deteksi COCO Ultralytics.
- `data/yolo26n_depth_nyu_full.pt`: `state_dict` lokal `YOLO26nDepth`.
- Input depth statis NCHW `1×3×240×320`, RGB float32 `/255`.
- Output depth `1×1×240×320` dengan target training `clip(depth_meter / 10, 0, 1)`.
  Konversi sementara ke jarak menggunakan `raw × 10 m`; ini belum menjamin
  akurasi metrik pada kamera HP.

## Jalur konversi dan versi

Detector memakai API Ultralytics modern:
`YOLO(...).export(format="litert", quantize=16)`. Ekspor ini menghasilkan
FlatBuffer `.tflite` FP16. Skrip mengharuskan Ultralytics **>=8.4.175**;
perintah instalasi memasang rilis tersebut sebagai batas minimum.

Depth menempuh jalur eksplisit PyTorch → ONNX opset 17 → SavedModel melalui
`onnx2tf==1.28.6` → TFLite FP16 melalui TensorFlow Lite Converter
`tensorflow==2.16.2`. Versi onnx2tf tersebut menyatakan dukungan Python >=3.10;
TensorFlow 2.16.2 menyediakan runtime Python 3.10, 3.11, dan 3.12. Pin ini
memilih versi converter yang menerima rentang Python yang diminta, alih-alih
onnx2tf terbaru yang metadata paketnya telah menaikkan minimum Python.

FP16 diterapkan pada bobot dengan `Optimize.DEFAULT` dan
`target_spec.supported_types=[tf.float16]`; input/output invocation tetap
float32. INT8 sengaja belum diaktifkan. Ultralytics telah mengarahkan ekspor
TFLite terbaru ke nama format LiteRT; hasilnya tetap file `.tflite` untuk
LiteRT Android.

## Menjalankan di Google Colab

1. Buat notebook Colab dengan runtime Python 3.10–3.12. Periksa runtime:

   ```python
   import sys
   print(sys.version)
   ```

2. Clone repo atau unggah folder proyek ke `/content/projek_comvis1`. Contoh
   untuk repo GitHub yang dapat diakses runtime:

   ```python
   !git clone https://github.com/ozzamantep/projek_comvis1.git /content/projek_comvis1
   %cd /content/projek_comvis1
   ```

   Jika repo private atau tidak ingin memakai `git clone`, unggah arsip repo
   dan ekstrak ke path tersebut. Jangan taruh token GitHub di notebook.

3. Unduh `nyu_depth_v2_labeled.mat` secara terpisah dari halaman resmi
   [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html), lalu
   unggah/salin file tersebut ke `/content/projek_comvis1/data/`. Dataset
   `.mat` tidak disertakan di repository.

4. Pasang dependensi di runtime:

   ```python
   %cd /content/projek_comvis1
   !python -m pip install --upgrade pip
   !python -m pip install "ultralytics>=8.4.175" "litert-torch" \
       "onnx2tf==1.28.6" "tensorflow==2.16.2" \
       "onnx==1.17.0" "onnxruntime==1.20.1" \
       "opencv-python" "h5py"
   ```

   Colab dapat memberi peringatan konflik versi dari paket bawaan runtime.
   Jika import TensorFlow/ONNX gagal setelah instalasi, restart runtime lalu
   jalankan sel dari atas.

5. Jalankan ekspor dan verifikasi:

   ```python
   %cd /content/projek_comvis1
   !python mobile/export/export_models.py \
       --detector-weights data/yolo26n.pt \
       --depth-weights data/yolo26n_depth_nyu_full.pt \
       --nyu-path data/nyu_depth_v2_labeled.mat \
       --output-dir mobile/export/artifacts \
       --verify-inputs 5
   ```

   Jika direktori `mobile/export/artifacts` sudah ada, skrip berhenti supaya
   tidak menimpa hasil lama. Gunakan path output baru untuk ekspor berikutnya.
   Skrip membandingkan PyTorch, ONNX Runtime CPU, dan TFLite FP16 pada beberapa
   sampel val NYU deterministik; ia mencetak max absolute error, latency desktop
   yang benar-benar terukur, versi tool, serta ukuran file. Hasil ringkas juga
   disimpan ke `mobile/export/artifacts/export_verification.json`.

6. Unduh dua model melalui panel Files Colab (klik kanan → Download), atau
   gunakan sel:

   ```python
   from google.colab import files

   files.download("mobile/export/artifacts/yolo26n_detector_fp16.tflite")
   files.download("mobile/export/artifacts/yolo26n_depth_fp16.tflite")
   ```

   Skrip menyalin hasil detector ke nama stabil
   `mobile/export/artifacts/yolo26n_detector_fp16.tflite`; file hasil antara
   Ultralytics juga tetap disimpan di subfolder `detector/`.

## Operator dan pemecahan masalah

- **Bilinear interpolation / `Resize`**: DepthHead melakukan resize bilinear
  (`align_corners=False`) untuk menyamakan skala fitur sebelum fusi dan
  menghasilkan map pada ukuran input. PyTorch mengekspor operasi resize ke
  ONNX; variasi `Resize`/ukuran tetap atau semantik koordinatnya dapat membuat
  konverter menolak graph atau menimbulkan selisih output. Input dibuat statis,
  ONNX diperiksa oleh `onnx.checker`, lalu hasil TFLite dibandingkan per sampel.
  Jika operasi `Resize` tidak didukung, lihat node graph ONNX dan versi
  onnx2tf; jangan menghapus resize atau mengganti alignment tanpa pengujian
  ulang.
- **SiLU**: ONNX merepresentasikan SiLU sebagai operasi elementwise/aktivasi;
  bila konverter tidak dapat memetakan bentuk graph tertentu, ganti SiLU
  khusus ekspor hanya setelah pembandingan akurasi dan runtime, bukan diam-diam
  mengubah arsitektur training.
- **Operator Flex/Select TF**: Converter dibatasi ke built-in LiteRT agar
  model hasil tidak memerlukan Flex delegate tambahan di Android. Jika konversi
  gagal pada operator, periksa traceback dan graph; jangan mengaktifkan
  `SELECT_TF_OPS` tanpa menyiapkan dependensi runtime Android yang sesuai.
- **Perbedaan output**: Skrip melaporkan selisih, bukan menyatakan model lolos
  ambang yang belum disepakati. Bandingkan nilai max absolute error dan
  validasi metrik depth sebelum memilih model untuk aplikasi.
- **Latency**: Pengukuran di Colab adalah latency desktop/runtime host, bukan
  benchmark HP. Catat perangkat, delegate, warmup, serta durasi inferensi
  CameraX secara terpisah untuk laporan.
