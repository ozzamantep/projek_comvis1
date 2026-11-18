# Catatan Asumsi dan Keputusan Teknis

Dokumen ini mencatat seluruh asumsi dan keputusan teknis yang diambil secara otonom agar pengerjaan tetap konsisten, jujur, dan dapat direproduksi.

1. **Definisi Resmi Baseline vs +Attention (Revisi Final)**:
   - **Baseline (Tanpa Attention)**: `YOLO26nDepthNoAttn` (`arsitektur_noattn.py`), yaitu arsitektur `YOLO26nDepth` dengan **penghapusan modul attention**: Layer 10 (`C2PSA` pada *Backbone*) diganti `nn.Identity()` (*bypass*) dan Layer 22 (`C3k2` pada *Neck* `n22`) menggunakan `attention=False` (tanpa `PSABlock`). Total parameter: **3.529.257**.
   - **+Attention (Usulan)**: `YOLO26nDepth` (`arsitektur.py`), yaitu arsitektur YOLO26n *dual-head* sesuai **Gambar 3.1** (Hidayatullah & Tubagus, 2026): *Backbone* L0–10 dengan **`C2PSA` pada Layer 10** (`249.728` parameter) dan *Neck* L11–22 dengan **`C3k2(..., attention=True)` pada Layer 22** (`+84.224` parameter dari `PSABlock`), ditambah *Detection Head* 3 skala dan *Depth Head*. Total parameter: **3.863.209** (`+333.952` atau `+9,46%` dibanding Baseline).
   - **Eksperimen CBAM (Diarsipkan)**: `YOLO26nDepthCBAM` (`arsitektur_attention.py`) tidak lagi menjadi model usulan utama dan dipindahkan ke [lampiran_eksperimen_cbam.md](file:///c:/Users/ACER/Downloads/Comvis/code/docs/laporan_bab/lampiran_eksperimen_cbam.md).

2. **Penanganan Penomoran Layer pada Varian Baseline (Tanpa L10)**:
   - Agar penomoran layer `0–22` tetap konsisten 1-to-1 antara *Baseline*, *+Attention*, dokumen laporan, dan kode, Layer 10 pada `YOLO26nDepthNoAttn` dipertahankan sebagai slot `self.backbone.l10 = nn.Identity()` dan ditandai **"DIHAPUS (Bypass)"** pada diagram arsitektur (`arsitektur_baseline_noattn.png`). Dengan cara ini, dimensi kanal keluaran `c_p5 = 256` dan penomoran `L11–L22` pada *Neck* tidak bergeser.

3. **Pemetaan Checkpoint Resmi**:
   - **+Attention (Usulan)**: `training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt` (Epoch 56, `384×640`, `mAP@0.5 = 30,87%`, `Depth MAE = 0,533 m`). Seluruh angka yang sebelumnya berlabel "Baseline" dari run ini kini berlabel **"+Attention"**.
   - **Baseline (Tanpa Attention)**: `training_runs/persist/rtx4050_noattn_70_20_10/best.pt`, dilatih menggunakan `scripts/train_noattn.py` dengan setelan identik (`seed=42`, split `70:20:10`, `384×640`, `batch=4`, `AdamW`, EMA `0,999`, 100 *epoch*).
   - **Legacy Runtime (Pembanding Tambahan)**: `multitask_best.pt` (Epoch 10, `480×640`, `5,448 M` parameter, `TrainedDepthHead` 256 kanal, `COCO_MIX=0.3`, `N_VAL=500/N_TEST=500`). Hanya dicantumkan sebagai catatan pembanding.

4. **Pengukuran Fisik Real-Time (Tabel 4.6 & Screenshot Webcam)**:
   - Pengukuran jarak fisik dengan meteran, kondisi pencahayaan terang dan gelap (`0,5 / 1,0 / 2,0 m`), dan pengambilan screenshot webcam langsung dengan objek nyata di depan kamera tidak dapat dilakukan secara otomatis oleh agen. Sel terkait diisi `"belum diukur"` dan dilengkapi skrip `scripts/capture_compare.py` serta `docs/checklist_screenshot.md`.

5. **Environment Python**:
   - Evaluasi, training, dan benchmark GPU menggunakan Python sistem (`python` 3.10.0, PyTorch 2.5.1+cu121, CUDA RTX 4050 Laptop GPU); `.venv` (Python 3.14.6 CPU-only) digunakan untuk `python-docx`.
