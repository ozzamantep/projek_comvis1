# Struktur workspace `code`

Jalankan perintah dari root workspace `code` agar argumen relatif seperti
`data/nyu_depth_v2_labeled.mat` dan `data/yolo26n_depth_nyu_full.pt` tetap
mengarah ke folder data root. Dengan virtual environment proyek di Windows,
contoh perintah memakai `.\.venv\Scripts\python.exe`.

## Tata letak

```text
code/
├── penting/
│   ├── arsitektur_yolo_depth.py
│   ├── calibrate_depth.py
│   ├── camera_yolo26_depth_metric.py
│   ├── depth_utils.py
│   ├── diagnose_depth_inference.py
│   ├── evaluate_depth.py
│   ├── kamera_yolo26_depth.py
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
| `arsitektur_yolo_depth.py` | Definisi arsitektur `YOLO26nDepth`, dataset NYU bersama, pemeriksaan checkpoint, dan mode verifikasi model. |
| `kamera_yolo26_depth.py` | Runner kamera baseline yang dibahas pada Bab 2, dengan pilihan backend lokal atau Ultralytics. |
| `train_nyu_colab.py` | Trainer NYU Depth V2 30 epoch yang memakai arsitektur lokal. |
| `camera_yolo26_depth_metric.py` | Runner multitask satu-model: detection head dan depth head dari graph ONNX yang sama. |
| `multitask_onnx.py` | Memvalidasi metadata/model, melakukan preprocessing letterbox, dan mengambil output deteksi serta depth dalam satu inference. |
| `depth_utils.py` | Konversi output depth ke meter/cm, sampling bbox/titik, smoothing, dan normalisasi untuk visualisasi. |
| `evaluate_depth.py` | Evaluasi metrik depth pada split validasi NYU. |
| `validate_metric_photos.py` | Validasi jarak dari foto dengan input pengukuran meteran. |
| `calibrate_depth.py` | Fit koreksi rasio atau regresi linear dari CSV validasi. |
| `diagnose_depth_inference.py` | Perbandingan preprocessing NYU/kamera dan pemeriksaan output depth pada banyak input. |
| `tests/` | Unit test untuk utilitas depth, evaluator, kalibrasi, dan kompatibilitas checkpoint. |

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
| `live_yolo26_depth_camera.py` | Runner depth-only alternatif; bukan runner metrik dua-model kanonik. |
| `inspect_nyu.m` | Inspeksi MATLAB struktur dan rentang file NYU. |
| `inspect_nyu_short.m` | Varian ringkas inspeksi NYU. |
| `inspect_ranges.m` | Inspeksi shape/rentang image dan depth dengan MATLAB. |

Checkpoint di `data/pendukung/`:

- `yolo26n-depth.pt`: varian Ultralytics depth; tidak dipakai pada pipeline dua-model
  utama, tetapi tetap dapat diberikan eksplisit ke runner alternatif.
- `yolo26n_depth_nyu_smoke.pt`, `argparse_fix.pt`, `colab_import_fix.pt`,
  `colab_smoke.pt`, dan `notebook_fix.pt`: checkpoint eksperimental/smoke yang
  tidak menjadi default pipeline utama.

Checkpoint utama `data/yolo26n.pt` (detector) dan
`data/yolo26n_depth_nyu_full.pt` (depth lokal) tetap di `data/`. `data/` dan
dataset `.mat` tidak dipindahkan. Training utama dijalankan melalui
`train_nyu_colab.py`; file arsitektur hanya menyediakan dataset, model,
pemeriksaan checkpoint, dan verifikasi.

## Perintah menjalankan

Semua path di bawah ini relatif terhadap root workspace. Ganti
`.\.venv\Scripts\python.exe` dengan `python` jika memakai interpreter lain.

### CLI dan alur utama

```powershell
.\.venv\Scripts\python.exe penting\arsitektur_yolo_depth.py --help
.\.venv\Scripts\python.exe penting\arsitektur_yolo_depth.py --mode verify
.\.venv\Scripts\python.exe penting\kamera_yolo26_depth.py --help
.\.venv\Scripts\python.exe penting\train_nyu_colab.py --help
.\.venv\Scripts\python.exe penting\camera_yolo26_depth_metric.py --help
.\.venv\Scripts\python.exe penting\evaluate_depth.py --help
.\.venv\Scripts\python.exe penting\validate_metric_photos.py --help
.\.venv\Scripts\python.exe penting\calibrate_depth.py --help
.\.venv\Scripts\python.exe penting\diagnose_depth_inference.py --help
```

Contoh operasi yang benar-benar menjalankan workload:

```powershell
.\.venv\Scripts\python.exe penting\camera_yolo26_depth_metric.py
.\.venv\Scripts\python.exe penting\train_nyu_colab.py --nyu-path data\nyu_depth_v2_labeled.mat
.\.venv\Scripts\python.exe penting\evaluate_depth.py
.\.venv\Scripts\python.exe penting\validate_metric_photos.py
.\.venv\Scripts\python.exe penting\calibrate_depth.py --csv data\metric_validation.csv
.\.venv\Scripts\python.exe penting\diagnose_depth_inference.py --camera-frame path\ke\frame.png
```

Runner kamera membutuhkan kamera/perangkat aktif; validasi foto membutuhkan
foto dan catatan jarak; diagnosis membutuhkan frame atau kamera, dataset NYU,
dan checkpoint. Perintah `--help` tidak menjalankan inferensi kamera.

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
