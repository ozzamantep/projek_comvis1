# Data YOLO26n-depth

Simpan dataset NYU Depth V2 lokal di folder ini sebagai
`nyu_depth_v2_labeled.mat`. Dataset tidak disertakan di repository.

```text
data/
  nyu_depth_v2_labeled.mat
```

Untuk training dengan arsitektur YOLO26n-Depth, gunakan trainer NYU:

```powershell
.\.venv\Scripts\python.exe penting\train_nyu_colab.py `
  --nyu-path data\nyu_depth_v2_labeled.mat `
  --output data\yolo26n_depth_pdf_best.pt
```

Untuk verifikasi arsitektur:

```powershell
.\.venv\Scripts\python.exe penting\arsitektur_yolo_depth.py --mode verify
```
