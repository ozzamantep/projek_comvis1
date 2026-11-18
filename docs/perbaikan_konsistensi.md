# Daftar Perbaikan Konsistensi Laporan (Bab 1–4) & Penomoran

Dokumen ini mencatat seluruh kalimat pada draf laporan asli ([Kelompok 1.1_Final_Comvis C.docx](file:///c:/Users/ACER/Downloads/Comvis/code/Kelompok%201.1_Final_Comvis%20C.docx)) yang perlu disesuaikan dengan definisi model terbaru (**Baseline = Tanpa Attention**, **+Attention = YOLO26nDepth dengan C2PSA L10 & L22 attn=True**), resolusi aktual, dan ruang lingkup desktop PyTorch.

## 1. Perbaikan Konsistensi Model, Judul Sub-bab, Resolusi, & Ruang Lingkup

| Bagian | Kalimat / Judul Lama (Tidak Sesuai) | Ketidaksesuaian | Usulan Kalimat / Judul Pengganti |
|--------|-------------------------------------|-----------------|----------------------------------|
| **Bab 1.2** | *"menggabungkan detektor yolo26n dan modul pengukur kedalaman (depth head) di dalam satu file model (multitask_best.pt)... resolusi masukan standar 480 x 640 piksel"* | Mencampuradukkan runtime *legacy* `multitask_best.pt` (`480×640`) dengan arsitektur `YOLO26nDepth` (`384×640`) yang dievaluasi pada Bab 4. | *"Pada sistem ini, arsitektur `YOLO26nDepth` (`arsitektur.py`, model **+Attention**) dan varian **Baseline tanpa attention** (`arsitektur_noattn.py`) dilatih dan dievaluasi pada resolusi `384 × 640` piksel, sedangkan pembanding legacy (`multitask_best.pt`) dan uji efisiensi standar juga diuji pada `480 × 640` piksel."* |
| **Bab 1.2 (Gambar 1.1)** | *"Gambar 1.1 Ilustrasi umum RGB dan monocular depth pada perangkat mobile..."* | Menyebut perangkat *mobile*. | *"Gambar 1.1 Ilustrasi konsep umum citra RGB dan peta kedalaman monokular (*monocular depth*); bukan tangkapan layar aplikasi desktop proyek."* |
| **Bab 2.1 & 2.2** | *"model = load_multitask_torch_model('multitask_best.pt') ... memproses fitur layer 16, 19, dan 22"* | Hanya menampilkan *loader* legacy `multitask_pt.py`. | *"Aplikasi kamera (`main.py`) mendukung *loader* `load_architecture_torch_model()` (`architecture_pt.py`) untuk checkpoint `YOLO26nDepth` (`384 × 640`) serta `load_multitask_torch_model()` (`multitask_pt.py`) untuk pembanding legacy `multitask_best.pt` (`480 × 640`)."* |
| **Bab 3.1 (Judul)** | ***"3.1 Spesifikasi Arsitektur Baseline (YOLO26n Multitask)"*** | Pada definisi baru, arsitektur YOLO26n lengkap dengan `C2PSA` (L10) dan `C3k2 attn=True` (L22) adalah model **+Attention (Usulan)**, bukan Baseline tanpa attention. | Ganti judul menjadi ***"3.1 Spesifikasi Arsitektur Model +Attention (YOLO26n-Depth)"*** dan tambahkan penjelasan bahwa varian **Baseline** menghapus `C2PSA` (L10) dan mengatur `attn=False` pada `n22` (L22). |
| **Bab 3.1 (Tabel 3.1 & 3.1.1)** | *"260 layer, 2.572.280 parameter, 6.1 GFLOPS"* dan *"256 channel... deconvolution"* | Angka `2,57M / 6,1 GFLOPs` milik `yolo26n` murni (tanpa `DepthHead`), dan *256 channel + deconvolution* milik `multitask_pt.py`. | *"Model **+Attention (`YOLO26nDepth`)** memiliki **3.863.209 parameter** (`20,449 GFLOPs` @ `384×640`), sedangkan **Baseline (Tanpa Attention)** memiliki **3.529.257 parameter** (`-333.952` parameter). `DepthHead` memproyeksikan P3/P4/P5 ke 32 kanal dengan interpolasi bilinear."* |
| **Bab 3.3.1 & 3.3.2** | *"pendekatan end-to-end (NMS-free)"* dan *"kalibrasi `cal_a` dan `cal_b`"* | `YOLO26nDepth` memakai *class-aware NMS* (`IoU = 0,45`) dan `Sigmoid × depth_max_m (10,0 m)`. | *"Detection Head `YOLO26nDepth` bekerja secara *anchor-free* DFL-*free* dengan *class-aware NMS* (`IoU = 0,45`), sedangkan Depth Head menghasilkan depth relatif `[0, 1]` via `Sigmoid` lalu dikalikan `depth_max_m = 10,0 m`."* |
| **Bab 4.1–4.3** | Angka `30,85% mAP` / `0,533 m MAE` disebut sebagai *"Baseline"* | Angka tersebut berasal dari `rtx4050_arsitektur_70_20_10/best.pt` yang sudah memiliki `C2PSA` (L10) & `C3k2 attn=True` (L22). | Ubah label angka `rtx4050_arsitektur_70_20_10/best.pt` menjadi **`+Attention`**, dan bandingkan terhadap **`Baseline (Tanpa Attention)`** (`rtx4050_noattn_70_20_10/best.pt`). |

## 2. Catatan Penomoran Tabel dan Gambar

| Elemen | Penomoran di Draf Baru (`docs/laporan_bab/`) | Keterangan |
|--------|----------------------------------------------|------------|
| Sub-bab 1.3.2 | **Tabel 1.1**, **Gambar 1.5** (`hist_depth_gt.png`) | Melanjutkan Gambar 1.1–1.4 pada Bab 1 asli |
| Sub-bab 3.1.5 | **Gambar 3.4** (`arsitektur_baseline_noattn.png`), **Gambar 3.5** (`arsitektur_plus_attention.png`) | Diadaptasi dari Hidayatullah & Tubagus (2026) |
| Sub-bab 4.1–4.3 | **Tabel 4.1**, **Gambar 4.1**, **Tabel 4.2**, **Tabel 4.3**, **Gambar 4.2**, **Tabel 4.4**, **Tabel 4.5a**, **Tabel 4.5b**, **Tabel 4.6** | Selaras dengan penomoran laporan asli |
| Sub-bab 4.4 | **Gambar 4.3 – Gambar 4.8** (*Test Split*), **Gambar 4.9 – Gambar 4.16** (*Placeholder* Webcam) | Melanjutkan Gambar 4.2 secara berurutan |
