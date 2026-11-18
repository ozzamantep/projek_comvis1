# Lampiran: Eksperimen Tambahan Modul Attention Eksternal (CBAM)

> **Catatan Status**: Pada revisi definisi model terbaru, **CBAM bukan lagi model usulan utama** dan dikeluarkan dari tabel utama Bab 4. Seluruh kode (`arsitektur_attention.py`), diagram arsip (`docs/figures/arsip_cbam/`), *checkpoint* (`training_runs/persist/rtx4050_cbam_70_20_10/best.pt`), dan hasil evaluasinya disimpan di lampiran ini sebagai arsip eksperimen tambahan.

## 1. Konfigurasi Eksperimen CBAM
- **Arsitektur**: `YOLO26nDepthCBAM` (`arsitektur_attention.py`) — menambahkan 4 blok `CBAMBlock` (Channel + Spatial Attention, *reduction ratio* $r = 16$, konvolusi spasial $7 \times 7$) pada keluaran *Neck* (`cbam_p3`, `cbam_p4`, `cbam_p5`) dan fusi `DepthHead` (`cbam_depth`).
- **Parameter & Komputasi**: `3.875.505` parameter (`+12.296` / `+0,32%` dari `YOLO26nDepth`), `20,499 GFLOPs` @ `384×640` (`25,623 GFLOPs` @ `480×640`), ukuran bobot `15,03 MB`.
- **Pelatihan**: 100 *epoch* di RTX 4050 (`384×640`, `seed=42`, split 70:20:10), *best checkpoint* pada **Epoch 66**.

## 2. Ringkasan Hasil Evaluasi (275 Citra Test Split, 1 Seed)
- **Deteksi**: `mAP@0.5 = 32,35%` (`person`: `51,10%`, `laptop`: `41,29%`, `bottle`: `17,80%`, `cell phone`: `19,21%` pada 36 *box* uji), `mAP@0.5:0.95 = 18,44%`, `Recall = 75,84%`, `Precision = 0,83%` (`conf=0,001`).
- **Kedalaman**: `Depth MAE = 0,534 m`, `AbsRel = 0,269`, `RMSE = 0,743 m`, $\delta_1 = 59,87\%$.
- **Efisiensi & *Profiling* Latensi GPU RTX 4050**:
  - Pada `384×640`: `23,89 ms` (`41,9 FPS`); pada `480×640`: `31,22 ms` (`32,0 FPS`); CPU `480×640`: `236,65 ms` (`4,2 FPS`).
  - *Bottleneck* teridentifikasi pada **`cbam_depth.channel_attn` (`15,659 ms`)** dari total `18,555 ms` `cbam_depth` akibat *memory-bandwidth bottleneck* pada tensor `[1, 96, 480, 640]`, sedangkan `cbam_p3/p4/p5` hanya memakan total `1,156 ms` (`11,87 ms` / `84,3 FPS` untuk varian `+CBAM Neck Only`).

