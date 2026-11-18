# Tabel Verifikasi Arsitektur per Layer (Layer 0–22 + Dual-Head)

Verifikasi dilakukan langsung melalui `forward pass` nyata pada `arsitektur.py` (`YOLO26nDepth`, **+Attention**) dan `arsitektur_noattn.py` (`YOLO26nDepthNoAttn`, **Baseline**), serta pencocokan `state_dict` `strict=True` terhadap `training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt` (Epoch 56).

- **Total Parameter +Attention (`YOLO26nDepth`)**: `3.863.209`
- **Total Parameter Baseline (`YOLO26nDepthNoAttn`)**: `3.529.257`
- **Selisih Parameter**: `+333.952` parameter (`+9,46%`), terdiri atas **`L10 C2PSA = 249.728`** dan **`L22 C3k2 (PSABlock attn=True) = 84.224`**.

| Layer | Bagian | Modul (+Attention / `arsitektur.py`) | Modul (Baseline / `arsitektur_noattn.py`) | Tensor Keluaran (`384×640`) | Tensor Keluaran (`480×640`) | Status |
|------:|--------|---------------------------------------|------------------------------------------|-----------------------------|-----------------------------|--------|
| **0** | Backbone | `Conv` (`3→16, k=3, s=2`) | Sama | `[1, 16, 192, 320]` | `[1, 16, 240, 320]` | Identik |
| **1** | Backbone | `Conv` (`16→32, k=3, s=2`) | Sama | `[1, 32, 96, 160]` | `[1, 32, 120, 160]` | Identik |
| **2** | Backbone | `C3k2` (`32→64, n=1, c3k=False, e=0.25`) | Sama | `[1, 64, 96, 160]` | `[1, 64, 120, 160]` | Identik |
| **3** | Backbone | `Conv` (`64→64, k=3, s=2`) | Sama | `[1, 64, 48, 80]` | `[1, 64, 60, 80]` | Identik |
| **4 (P3)** | Backbone | `C3k2` (`64→128, n=1, c3k=False, e=0.25`) | Sama | `[1, 128, 48, 80]` | `[1, 128, 60, 80]` | Identik |
| **5** | Backbone | `Conv` (`128→128, k=3, s=2`) | Sama | `[1, 128, 24, 40]` | `[1, 128, 30, 40]` | Identik |
| **6 (P4)** | Backbone | `C3k2` (`128→128, n=1, c3k=True, e=0.5`) | Sama | `[1, 128, 24, 40]` | `[1, 128, 30, 40]` | Identik |
| **7** | Backbone | `Conv` (`128→256, k=3, s=2`) | Sama | `[1, 256, 12, 20]` | `[1, 256, 15, 20]` | Identik |
| **8** | Backbone | `C3k2` (`256→256, n=1, c3k=True, e=0.5`) | Sama | `[1, 256, 12, 20]` | `[1, 256, 15, 20]` | Identik |
| **9** | Backbone | `SPPF` (`256→256, k=5, repeats=3`) | Sama | `[1, 256, 12, 20]` | `[1, 256, 15, 20]` | Identik |
| **10 (P5)** | Backbone | **`C2PSA` (`256→256, n=1, e=0.5`)** | **`nn.Identity()` (Dihapus)** | `[1, 256, 12, 20]` | `[1, 256, 15, 20]` | **-249.728 param** |
| **11** | Neck (FPN) | `Upsample` (nearest, ke L6) | Sama | `[1, 256, 24, 40]` | `[1, 256, 30, 40]` | Identik |
| **12** | Neck (FPN) | `Concat` (`L11 + L6`) | Sama | `[1, 384, 24, 40]` | `[1, 384, 30, 40]` | Identik |
| **13** | Neck (FPN) | `C3k2` (`384→128, n=1, c3k=True, e=0.5`) | Sama | `[1, 128, 24, 40]` | `[1, 128, 30, 40]` | Identik |
| **14** | Neck (FPN) | `Upsample` (nearest, ke L4) | Sama | `[1, 128, 48, 80]` | `[1, 128, 60, 80]` | Identik |
| **15** | Neck (FPN) | `Concat` (`L14 + L4`) | Sama | `[1, 256, 48, 80]` | `[1, 256, 60, 80]` | Identik |
| **16 (P3_out)** | Neck (FPN) | `C3k2` (`256→64, n=1, c3k=True, e=0.5`) | Sama | `[1, 64, 48, 80]` | `[1, 64, 60, 80]` | Identik |
| **17** | Neck (PAN) | `Conv` (`64→64, k=3, s=2`) | Sama | `[1, 64, 24, 40]` | `[1, 64, 30, 40]` | Identik |
| **18** | Neck (PAN) | `Concat` (`L17 + L13`) | Sama | `[1, 192, 24, 40]` | `[1, 192, 30, 40]` | Identik |
| **19 (P4_out)** | Neck (PAN) | `C3k2` (`192→128, n=1, c3k=True, e=0.5`) | Sama | `[1, 128, 24, 40]` | `[1, 128, 30, 40]` | Identik |
| **20** | Neck (PAN) | `Conv` (`128→128, k=3, s=2`) | Sama | `[1, 128, 12, 20]` | `[1, 128, 15, 20]` | Identik |
| **21** | Neck (PAN) | `Concat` (`L20 + L10`) | Sama | `[1, 384, 12, 20]` | `[1, 384, 15, 20]` | Identik |
| **22 (P5_out)** | Neck (PAN) | **`C3k2` (`384→256, n=1, c3k=True, e=0.5, attn=True`)** | **`C3k2` (`384→256, n=1, c3k=True, e=0.5, attn=False`)** | `[1, 256, 12, 20]` | `[1, 256, 15, 20]` | **-84.224 param** |
| **Head A** | Detection | `DetectionHead` (P3/P4/P5, DFL-free, 4 kelas + NMS) | Sama | `P3:[1,4,48,80]`, `P4:[1,4,24,40]`, `P5:[1,4,12,20]` | `P3:[1,4,60,80]`, `P4:[1,4,30,40]`, `P5:[1,4,15,20]` | Identik |
| **Head B** | Depth | `DepthHead` (proj 32ch + Bilinear + Refine + Sigmoid) | Sama | `[1, 1, 384, 640]` | `[1, 1, 480, 640]` | Identik |

