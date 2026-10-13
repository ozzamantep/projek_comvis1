# Struktur workspace `code`

Jalankan perintah dari root workspace `code` agar path relatif ke `data/`
berfungsi. Di Windows, contoh perintah memakai `.\.venv\Scripts\python.exe`.
Kode dipisah menurut fitur dan class: satu modul Python per tanggung jawab dan
satu file Kotlin per class/komponen. Runner desktop dan Android tetap memakai
satu model multitask; file-file kecil memisahkan preprocessing, inferensi,
sampling depth, UI, dan validasi, bukan model yang berbeda.

## Tata letak

```text
code/
├── penting/
│   ├── arsitektur_yolo_depth.py
│   ├── calibrate_depth.py
│   ├── camera_yolo26_depth_metric.py
│   ├── depth_utils.py
│   ├── diagnose_depth_inference.py
│   ├── diagnose_depth_input.py
│   ├── evaluate_depth.py
│   ├── kamera_yolo26_depth.py
│   ├── multitask_onnx.py
│   ├── train_nyu_colab.py
│   ├── validate_metric_photos.py
│   └── tests/
├── pendukung/
├── data/
│   └── pendukung/
├── .venv/
└── README_STRUKTUR.md
```

Laporan PDF, `.venv/`, dataset, checkpoint utama, serta file `best.pt` di root
tidak dipindahkan. Backup sebelum perapian ada di `_backup_sebelum_rapi/`;
backup tersebut sengaja tidak memuat `.venv/` atau file `.pt`.

## Isi `penting/`

| File | Fungsi |
|---|---|
| `camera_yolo26_depth_metric.py` | Runner desktop utama; satu inferensi multitask menghasilkan deteksi dan depth. |
| `multitask_onnx.py` | Memvalidasi metadata dan graph ONNX, preprocessing, lalu mengembalikan kedua output dalam satu panggilan. |
| `depth_utils.py` | Sampling depth di titik/bbox, smoothing temporal, konversi meter-ke-cm, dan visualisasi. |
| `evaluate_depth.py` | Evaluasi output depth model multitask pada split validasi NYU. |
| `validate_metric_photos.py` | Pemeriksaan jarak dari foto dan titik yang dipilih. |
| `calibrate_depth.py` | Fit koreksi kamera dari hasil pengukuran CSV. |
| `arsitektur_yolo_depth.py` | Arsitektur eksperimen/PDF dan dataset NYU untuk trainer legacy; bukan graph ONNX deployment. |
| `train_nyu_colab.py` | Trainer untuk arsitektur eksperimen di atas; terpisah dari model multitask terbaru. |
| `kamera_yolo26_depth.py` | Runner baseline depth-only untuk checkpoint eksperimen/Ultralytics. |
| `diagnose_depth_inference.py`, `diagnose_depth_input.py` | Alat diagnosis checkpoint dan preprocessing eksperimen; tidak dipanggil runner utama. |
| `tests/` | Tes helper, evaluasi, kalibrasi, kompatibilitas checkpoint, dan output multitask. |

## Isi `pendukung/`

| File | Fungsi / alasan dipisahkan |
|---|---|
| `evaluasi_loss_gradien.py` | Prototipe simulasi deteksi 2D dan pemeriksaan gradien, bukan trainer depth NYU. |
| `training_and_evaluation.py` | Simulasi training deteksi memakai tensor dummy, bukan alur model depth utama. |
| `metrics_evaluation.py` | Contoh metrik IoU/precision/recall deteksi sintetis yang tidak dipakai runner utama. |
| `verifikasi_arsitektur.py` | Model tiruan untuk inspeksi bentuk tensor, bukan definisi `YOLO26nDepth`. |
| `inisialisasi.py` | Pemeriksaan environment/perangkat mandiri. |
| `pra_pemrosesan_data.py` | Contoh letterbox 640×640 mandiri; bukan preprocessing NYU 320×240 pada alur depth utama. |
| `live_yolo26_depth.py` | Runner alternatif Ultralytics yang mengharapkan `result.depth`. |
| `live_yolo26_depth_camera.py` | Runner depth-only alternatif; bukan jalur aplikasi multitask. |
| `inspect_nyu.m` | Inspeksi MATLAB struktur dan rentang file NYU. |
| `inspect_nyu_short.m` | Varian ringkas inspeksi NYU. |
| `inspect_ranges.m` | Inspeksi shape/rentang image dan depth dengan MATLAB. |

Model deployment terbaru adalah `data/multitask.onnx`; metadata bentuk input,
output, dan urutan kelas ada di `data/model_metadata.json`. File model/checkpoint
tetap lokal dan tidak disalin ke source Android atau dimasukkan ke Git.

Checkpoint eksperimen di `data/pendukung/`:

- `yolo26n-depth.pt`: varian Ultralytics depth; tidak dipakai pada pipeline multitask
  utama, tetapi tetap dapat diberikan eksplisit ke runner alternatif.
- `yolo26n_depth_nyu_smoke.pt`, `argparse_fix.pt`, `colab_import_fix.pt`,
  `colab_smoke.pt`, dan `notebook_fix.pt`: checkpoint eksperimental/smoke yang
  tidak menjadi default pipeline utama.

Arsitektur dan trainer NYU lokal tetap tersedia untuk eksperimen yang mengikuti
PDF. Keduanya bukan sumber model inference multitask terbaru dan tidak perlu
dijalankan untuk memakai aplikasi.

## Perintah menjalankan

Semua path di bawah ini relatif terhadap root workspace. Ganti
`.\.venv\Scripts\python.exe` dengan `python` jika memakai interpreter lain.

### Jalur utama: model multitask

```powershell
.\.venv\Scripts\python.exe penting\camera_yolo26_depth_metric.py --help
.\.venv\Scripts\python.exe penting\evaluate_depth.py --help
.\.venv\Scripts\python.exe penting\validate_metric_photos.py --help
.\.venv\Scripts\python.exe penting\calibrate_depth.py --help
```

Contoh operasi yang benar-benar menjalankan workload:

```powershell
.\.venv\Scripts\python.exe penting\camera_yolo26_depth_metric.py
.\.venv\Scripts\python.exe penting\evaluate_depth.py
.\.venv\Scripts\python.exe penting\validate_metric_photos.py
.\.venv\Scripts\python.exe penting\calibrate_depth.py --csv data\metric_validation.csv
```

Runner kamera membutuhkan kamera/perangkat aktif; validasi foto membutuhkan
foto dan catatan jarak. Evaluasi membutuhkan dataset NYU. Perintah `--help`
tidak menjalankan inferensi kamera.

### Eksperimen arsitektur PDF dan diagnosis checkpoint lama

```powershell
.\.venv\Scripts\python.exe penting\arsitektur_yolo_depth.py --help
.\.venv\Scripts\python.exe penting\train_nyu_colab.py --help
.\.venv\Scripts\python.exe penting\kamera_yolo26_depth.py --help
.\.venv\Scripts\python.exe penting\diagnose_depth_inference.py --help
.\.venv\Scripts\python.exe penting\diagnose_depth_input.py --help
```

### Unit test

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```

### Skrip pendukung

```powershell
.\.venv\Scripts\python.exe pendukung\evaluasi_loss_gradien.py
.\.venv\Scripts\python.exe pendukung\training_and_evaluation.py
.\.venv\Scripts\python.exe pendukung\metrics_evaluation.py
.\.venv\Scripts\python.exe pendukung\verifikasi_arsitektur.py
.\.venv\Scripts\python.exe pendukung\inisialisasi.py
.\.venv\Scripts\python.exe pendukung\pra_pemrosesan_data.py
.\.venv\Scripts\python.exe pendukung\live_yolo26_depth.py --help
.\.venv\Scripts\python.exe pendukung\live_yolo26_depth_camera.py
matlab -batch "run(fullfile('pendukung','inspect_nyu.m'))"
matlab -batch "run(fullfile('pendukung','inspect_nyu_short.m'))"
matlab -batch "run(fullfile('pendukung','inspect_ranges.m'))"
```

Untuk menggunakan model Ultralytics depth pada runner alternatif:

```powershell
.\.venv\Scripts\python.exe pendukung\live_yolo26_depth.py --weights data\pendukung\yolo26n-depth.pt
```
