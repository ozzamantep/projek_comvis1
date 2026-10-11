# Struktur dataset untuk YOLO26n-depth

Gunakan folder ini sebagai struktur dataset latih dan validasi.

```text
data/
  train/
    images/
    depth/
  val/
    images/
    depth/
```

Aturan:
- File RGB dan depth harus memiliki nama yang sama.
- Format gambar sesuai: .jpg, .png, .jpeg, .bmp, .webp
- Depth dapat berupa grayscale atau map kedalaman 1 channel.
- Jika depth tidak tersedia, program akan mencoba membuat pseudo-depth otomatis.

Contoh:
- data/train/images/frame_001.jpg
- data/train/depth/frame_001.png

Untuk mulai training:

```powershell
python .\arsitektur_yolo_depth.py --mode train --image-dir .\data\train\images --depth-dir .\data\train\depth --epochs 5 --lr 1e-3
```

Untuk verifikasi arsitektur:

```powershell
python .\arsitektur_yolo_depth.py --mode verify
```

Untuk demo kamera live:

```powershell
python .\arsitektur_yolo_depth.py --mode camera
```
