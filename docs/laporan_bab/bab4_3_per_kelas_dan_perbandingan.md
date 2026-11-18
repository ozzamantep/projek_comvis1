## 4.3 Analisis Performa per Kelas dan Skenario Pengujian Kuantitatif

Seluruh pengujian kuantitatif pada sub-bab ini dijalankan pada ***held-out test split* yang sama (275 citra: 200 SUN RGB-D + 75 MS COCO)** menggunakan `seed=42`, *1 seed* (evaluasi deterministik), dengan skrip `eval/evaluate_models.py` dan `eval/benchmark_efficiency.py`.

> **Catatan Definisi**: **Baseline (Tanpa Attention)** adalah `YOLO26nDepthNoAttn` (`arsitektur_noattn.py`, tanpa `C2PSA` L10 dan `n22 attn=False`). **+Attention (Usulan)** adalah `YOLO26nDepth` (`arsitektur.py`, *checkpoint* `rtx4050_arsitektur_70_20_10/best.pt` Ep.56 dengan `C2PSA` L10 dan `C3k2 attn=True` L22). Sebagai catatan pembanding tambahan, disertakan pula *legacy runtime* `multitask_best.pt` (Ep.10, `480×640`, bobot pralatih COCO penuh + `TrainedDepthHead` 256 kanal).

---

### A. Performa Deteksi per Kelas (Test Split 275 Citra)

**Tabel 4.4 Nilai AP@0.5 per Kelas pada Data Uji (Test Split 275 Citra)**

| Kelas Objek | Baseline (Tanpa Attention) `AP@0.5` (%) ↑ | +Attention (`YOLO26nDepth`, Ep.56) `AP@0.5` (%) ↑ | $\Delta$ (+Attn vs Baseline) | Penyebab Utama |
|-------------|------------------------------------------:|--------------------------------------------------:|-----------------------------:|----------------|
| `person` | 47,19% | **51,76%** | **+4,57** | Ukuran objek besar & sampel latih terbanyak (1.377 bbox) |
| `laptop` | 29,12% | **43,12%** | **+14,00** | Struktur tepi dan bentuk persegi panjang jelas (48 bbox uji) |
| `bottle` | **21,51%** | 18,32% | -3,19 | Ukuran relatif kecil & variasi bentuk/transparansi tinggi |
| `cell phone` | 8,33% | **10,27%** | **+1,94** | Ukuran sangat kecil & sampel latih paling sedikit (210 bbox; hanya 36 bbox uji) |
| **Rata-rata (mAP@0.5)** | **26,54%** | **30,87%** | **+4,33** | Kenaikan didorong oleh `laptop` (+14,00) dan `person` (+4,57) |

*\*Catatan pembanding legacy (`multitask_best.pt`, 480×640): `person` 74,30%, `laptop` 77,54%, `bottle` 51,08%, `cell phone` 59,92% (`mAP@0.5` 65,71%).*

*Sumber: Penulis (`eval/per_class_ap.csv`)*

**Kesimpulan Tabel 4.4**: Pada pengujian **1 seed (`seed=42`)** terhadap **275 citra uji**, model **+Attention (`YOLO26nDepth`, Ep.56)** mencatat `mAP@0.5` lebih tinggi **`+4,33` poin (`+16,3%` relatif)** dibandingkan **Baseline Tanpa Attention (Ep.33)** (`30,87%` vs `26,54%`). Kenaikan terbesar terjadi pada kelas `laptop` (`+14,00` poin, `48` bbox uji), diikuti `person` (`+4,57` poin, `204` bbox uji) dan `cell phone` (`+1,94` poin, hanya `36` bbox uji sehingga bersifat indikatif), sedangkan kelas `bottle` lebih rendah (`-3,19` poin). Karena pengujian hanya menggunakan 1 seed pada *test set* berukuran 275 citra, selisih ini belum dapat dinyatakan signifikan secara statistik tanpa uji multi-*seed*.

---

### B. Perbandingan Akurasi Deteksi & Kedalaman

**Tabel 4.5a Perbandingan Kinerja Akurasi Deteksi dan Estimasi Kedalaman (Test Split 275 Citra)**

*Keterangan: ↑ semakin besar semakin baik, ↓ semakin kecil semakin baik. Evaluasi pada 275 citra test split (538 GT boxes; 35,69M piksel depth valid pada 384×640); 1 seed (`seed=42`).*

| Model | Input | mAP@0.5 (%) ↑ | mAP@0.5:0.95 (%) ↑ | Precision (%) ↑ | Recall (%) ↑ | Depth MAE (m) ↓ | AbsRel ↓ | RMSE (m) ↓ | $\delta_1$ (%) ↑ |
|-------|-------|--------------:|-------------------:|----------------:|-------------:|----------------:|---------:|-----------:|-----------------:|
| Baseline (Tanpa Attention, Ep.33) | 384×640 | 26,54% | 13,97% | **0,87%** | 77,51% | 0,543 | 0,273 | 0,760 | 58,42% |
| **+Attention (`YOLO26nDepth`, Ep.56)** | **384×640** | **30,87%** | **17,96%** | **0,86%** | **78,62%** | **0,533** | **0,273** | **0,742** | **60,61%** |
| *Catatan Legacy (`multitask_best.pt`, Ep.10)* | *480×640* | *65,71%* | *43,90%* | *0,61%* | *93,68%* | *0,354* | *0,158* | *0,556* | *77,59%* |

*Sumber: Penulis (`eval/accuracy_comparison.csv`)*

**Kesimpulan Tabel 4.5a**: Pada evaluasi **1 seed (`seed=42`)** di *test split* (275 citra), model **+Attention (`YOLO26nDepth`, Ep.56)** memperoleh `mAP@0.5` lebih tinggi (`30,87%` vs `26,54%`, `+4,33` poin / `+16,3%` relatif), `mAP@0.5:0.95` lebih tinggi (`17,96%` vs `13,97%`, `+3,99` poin / `+28,6%` relatif), serta `Recall` lebih tinggi (`78,62%` vs `77,51%`, `+1,11` poin) dibanding **Baseline Tanpa Attention (Ep.33)**. Pada estimasi kedalaman, **+Attention** juga mencatat `Depth MAE` sedikit lebih rendah (`0,533 m` vs `0,543 m`, `-0,010 m` / `-1,8%` relatif), `RMSE` lebih rendah (`0,742 m` vs `0,760 m`, `-0,018 m`), dan $\delta_1$ lebih tinggi (`60,61%` vs `58,42%`, `+2,19` poin), dengan `AbsRel` identik (`0,273`). Seluruh selisih ini merupakan hasil pengukuran 1 *seed* pada *test set* kecil (275 citra) sehingga belum disertai uji signifikansi statistik multi-*seed*.

---

### C. Perbandingan Efisiensi Komputasi & Latensi Real-Time

**Tabel 4.5b Perbandingan Efisiensi Komputasi pada Laptop (CPU & GPU RTX 4050)**

*Protokol: Warm-up 50 iterasi, rata-rata 200 iterasi, `batch size = 1`, PyTorch 2.5.1+cu121. Ukuran model diukur dari bobot `state_dict`.*

| Model | Resolusi | Params (M) ↓ | GFLOPs ↓ | Ukuran (MB) ↓ | Latency CPU (ms) ↓ | Latency GPU (ms) ↓ | FPS GPU ↑ |
|-------|----------|-------------:|---------:|--------------:|-------------------:|-------------------:|----------:|
| Baseline (Tanpa Attention) | 384×640 | 3,529 | 20,291 | 13,68 | 169,11 | 16,17* | 61,8* |
| **+Attention (`YOLO26nDepth`)** | **384×640** | **3,863** | **20,449** | **14,98** | **131,75** | **9,91** | **100,9** |
| Baseline (Tanpa Attention) | 480×640 | 3,529 | 25,364 | 13,68 | 199,12 | 9,63 | 103,8 |
| **+Attention (`YOLO26nDepth`)** | **480×640** | **3,863** | **25,562** | **14,98** | **151,44** | **11,48** | **87,1** |
| *Catatan Legacy (`multitask_best.pt`)* | *480×640* | *5,448* | *25,041* | *21,14* | *165,36* | *16,50* | *60,6* |

*\*Catatan: Pada pengukuran `Baseline 384×640` (`16,17 ms`), sesi GPU berada pada kondisi awal setelah training panjang; pada resolusi lebih besar `480×640`, GPU terukur stabil di `9,63 ms` (`103,8 FPS`).*

*Sumber: Penulis (`eval/efficiency_benchmark.csv`)*

**Kesimpulan Tabel 4.5b**: Penambahan `C2PSA` (L10) dan `C3k2 attn=True` (L22) pada **+Attention** hanya menambah **`+333.952` parameter (`+9,46%`)**, **`+0,158` s.d. `+0,198` GFLOPs (`+0,78%`)**, dan **`+1,30 MB`** ukuran bobot. Pada resolusi standar `480×640`, terdapat *trade-off* latensi GPU yang kecil sebesar **`+1,85 ms`** (`9,63 ms` / `103,8 FPS` → `11,48 ms` / `87,1 FPS`), sehingga kedua varian sama-sama berjalan *real-time* jauh di atas `30 FPS` pada GPU RTX 4050 (sementara pada CPU berada di kisaran `5,0–7,6 FPS`).

---

### D. Uji Ketahanan (Robustness) Real-Time (Webcam & Meteran Fisik)

**Tabel 4.6 Uji Ketahanan (Robustness) Real-Time (Minimal 3 Pengulangan, Mean ± Std)**

*Protokol: Error = `|prediksi - jarak_meteran_aktual|` (meter), minimal 3 pengulangan per skenario menggunakan `scripts/capture_compare.py` (otomatis tercatat ke `eval/webcam_measurements.csv`).*

| Skenario Uji | Jarak Aktual (m) | Error Baseline (m) ↓ | Error +Attention (m) ↓ |
|--------------|-----------------:|---------------------:|-----------------------:|
| **A. Pencahayaan Terang (`jarak_*_terang_person`)** | | | |
| Jarak Dekat (Terang, `person`) | 0,50 | 1,078 ± 0,076 *(n=3)* | **1,048 ± 0,010 *(n=3)*** |
| Jarak Sedang (Terang, `person`) | 1,00 | **0,493 *(n=1)*** | 0,806 *(n=1)* |
| Jarak Jauh (Terang, `person`) | 2,00 | 0,549 *(n=1)* | **0,476 *(n=1)*** |
| **B. Variasi Pencahayaan: Gelap (`jarak_*_gelap_person`)** | | | |
| Jarak Dekat (Gelap, `person`) | 0,50 | tidak terdeteksi *(n=1)* | **1,165 *(n=1)*** |
| Jarak Sedang (Gelap, `person`) | 1,00 | tidak terdeteksi *(n=1)* | **0,951 *(n=1)*** |
| Jarak Jauh (Gelap, `person`) | 2,00 | 0,273 *(n=1)* | **0,033 *(n=1)*** |
| **C. Kondisi Khusus & Objek Target Lain** | | | |
| *Occlusion* (Terhalang Sebagian, `person`) | 1,00 | **0,312 *(n=1)*** | 0,524 *(n=1)* |
| Dinding Polos (*Low-texture*) | 1,50 | tidak terdeteksi *(n=1)* | tidak terdeteksi *(n=1)* |
| Objek `bottle` (Terang, 0,50 m)* | 0,50 | 0,892 ± 0,142 *(n=4)* | 1,208 ± 0,008 *(n=4)* |
| Objek `bottle` (Terang, 1,00 m)* | 1,00 | 0,570 ± 0,004 *(n=2)* | 1,017 ± 0,012 *(n=2)* |
| Objek `laptop` (Terang, 1,00 m) | 1,00 | **0,190 *(n=1)*** | 0,612 *(n=1)* (`person`) |
| Objek `cell phone` (Terang, 0,50 m)* | 0,50 | 1,105 ± 0,018 *(n=3)* | 1,217 ± 0,022 *(n=3)* |

*\*Catatan: Pada skenario `bottle` dan `cell phone` di depan webcam, kotak dengan confidence tertinggi yang terekam pada `eval/webcam_measurements.csv` adalah `person` (pemegang objek) atau `laptop`, bukan kelas objek kecil itu sendiri. Empat baris uji coba lama tanggal 2026-11-17 diabaikan.*

*Sumber: Penulis (`eval/tabel_4_6_summary.csv` & `eval/webcam_measurements.csv` tanggal 2026-11-18)*

**Kesimpulan Tabel 4.6**:
1. **Ketahanan Pencahayaan (Terang vs Gelap)**: Pada kondisi gelap di jarak `0,50 m` dan `1,00 m`, **Baseline (Tanpa Attention) gagal mendeteksi objek `person` sama sekali**, sedangkan **+Attention** tetap mampu mendeteksi `person` (`error` `1,165 m` dan `0,951 m`), dan pada jarak `2,00 m` gelap mencatat `error` `0,033 m` (*n=1*) dibanding Baseline `0,273 m`. Karena sebagian skenario gelap baru diukur `n=1`, temuan ini bersifat indikatif.
2. **Bias Skala Jarak Dekat (Uncalibrated Webcam)**: Pada jarak dekat `0,50 m` (terang), kedua model menunjukkan *error* besar yang konsisten (`1,078 ± 0,076 m` vs `1,048 ± 0,010 m`, prediksi ~`1,55–1,62 m`), disebabkan oleh perbedaan panjang fokal (*focal length*) webcam laptop terhadap kamera Kinect v1 SUN RGB-D tanpa kalibrasi *offset* (`scale=1,0, offset=0,0`).
3. **Kasus Khusus & Objek Kecil**: Pada *occlusion* (`1,00 m`), Baseline mencatat `error` `0,312 m` vs +Attention `0,524 m`. Pada dinding polos (`1,50 m`), kedua model tidak menghasilkan deteksi dan peta kedalaman *INFERNO* menunjukkan artefak kedalaman di area minim tekstur. Pada skenario `bottle` dan `cell phone` yang dipegang di depan kamera, deteksi masih didominasi kelas `person`.
