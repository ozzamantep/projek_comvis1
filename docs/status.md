# Papan Progres Pengerjaan (Status Board)

Terakhir diperbarui: Pelatihan & Evaluasi **Baseline (NoAttn)** vs **+Attention** selesai seluruhnya

| Urutan | Tugas | Status | Keterangan / Artefak |
|--------|-------|--------|----------------------|
| 0 | Revisi Definisi, Verifikasi Layer 0–22, & `arsitektur_noattn.py` | SELESAI | `docs/verifikasi_layer.md`, `docs/asumsi.md`, `docs/perbaikan_konsistensi.md` |
| 1 | Arsip Eksperimen CBAM | SELESAI | `docs/laporan_bab/lampiran_eksperimen_cbam.md`, `docs/figures/arsip_cbam/` |
| 2 | Diagram Gaya Figure 1 (Hidayatullah & Tubagus 2026) | SELESAI | `docs/figures/arsitektur_baseline_noattn.{png,svg}` & `arsitektur_plus_attention.{png,svg}` |
| 3 | Training **Baseline (NoAttn)** 100 Epoch | SELESAI | `training_runs/persist/rtx4050_noattn_70_20_10/best.pt` (Ep.33) |
| 4 | Evaluasi & Benchmark Efisiensi Baseline vs +Attention | SELESAI | Baseline `26,54% mAP@0.5` / `0,543 m MAE` vs +Attention `30,87%` / `0,533 m` (`+4,33` poin, 1 seed, 275 citra) |
| 5 | Skrip Webcam & Tabel 4.6 | SELESAI (Skrip) / BELUM DIUKUR (Webcam) | `scripts/capture_compare.py`, `docs/checklist_screenshot.md` |
| 6 | `docs/Bab_Lengkap.docx` & Codebase Memory | SELESAI | Diperbarui dengan angka final Baseline vs +Attention |
