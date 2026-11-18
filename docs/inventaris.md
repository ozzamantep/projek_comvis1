# Inventaris Proyek: Lightweight YOLO26n-Depth

Dokumen ini mencatat hasil inventarisasi seluruh artefak, checkpoint, skrip, dataset, diagram, dan draf laporan sesuai **definisi model terbaru (Baseline = Tanpa Attention, +Attention = YOLO26nDepth dengan C2PSA L10 & C3k2 attn L22)**.

## 1. Tabel Inventaris Kebutuhan

| No | Kebutuhan | Sudah ada? | Lokasi | Tindakan |
|----|-----------|------------|--------|----------|
| 1 | Model & Checkpoint **+Attention (Usulan)** | Ya | `arsitektur.py`, `training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt` (Ep.56) | Ditetapkan sebagai **+Attention**; `state_dict` & layer 0–22 terverifikasi di `docs/verifikasi_layer.md` |
| 2 | Model & Checkpoint **Baseline (Tanpa Attention)** | Ya (Kode) / Sedang Dilatih | `arsitektur_noattn.py`, `scripts/train_noattn.py`, `training_runs/persist/rtx4050_noattn_70_20_10/` | Dilatih 100 epoch dengan setelan identik +Attention; evaluasi dijalankan setelah selesai |
| 3 | Arsip Eksperimen CBAM | Ya | `arsitektur_attention.py`, `training_runs/persist/rtx4050_cbam_70_20_10/`, `docs/figures/arsip_cbam/`, `docs/laporan_bab/lampiran_eksperimen_cbam.md` | Diarsipkan, dikeluarkan dari tabel utama |
| 4 | Checkpoint Legacy (Pembanding Tambahan) | Ya | `multitask_best.pt` (`multitask_pt.py`) | Hanya sebagai catatan pembanding |
| 5 | Skrip Evaluasi & Benchmark Efisiensi | Ya | `eval/evaluate_models.py`, `eval/benchmark_efficiency.py` | Perluas untuk mengevaluasi Baseline (NoAttn) bersama +Attention setelah training selesai |
| 6 | Diagram Arsitektur Gaya Figure 1 (Hidayatullah & Tubagus 2026) | Ya | `docs/figures/arsitektur_baseline_noattn.{png,svg}`, `docs/figures/arsitektur_plus_attention.{png,svg}` | Dibuat ulang sesuai gaya Figure 1 (layer 0–22, tensor shapes terverifikasi) |
| 7 | Statistik Dataset & Histogram Depth | Ya | `eval/dataset_stats.json`, `eval/dataset_distribution.csv`, `docs/figures/hist_depth_gt.png` | Sudah lengkap (2.750 citra: 2.001 SUN RGB-D + 749 COCO) |
| 8 | Skrip Capture Webcam & Checklist | Ya | `scripts/capture_compare.py`, `docs/checklist_screenshot.md` | Default diperbarui: `--model-a` = Baseline (NoAttn), `--model-b` = +Attention (`best.pt` Ep.56) + validasi jarak |
| 9 | Sampel Test Split & Draf Laporan | Ya | `docs/figures/testset_samples/`, `docs/laporan_bab/`, `docs/Bab_Lengkap.docx` | Diperbarui sesuai definisi baru dan dibangun ulang |
