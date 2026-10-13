# YOLO26n Kamera: Deteksi dan Depth

Aplikasi desktop ini membuka kamera langsung dari terminal VS Code. Satu model
multitask mendeteksi objek dan memperkirakan depth, lalu menampilkan deteksi
beserta jaraknya di panel kiri dan peta depth di panel kanan.

## Menjalankan

Pasang dependensi sekali:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Jalankan kamera dari root folder proyek:

```powershell
.\.venv\Scripts\python.exe main.py
```

Tekan `q` untuk keluar, `s` untuk mengganti tampilan, `d` untuk menyalakan atau
mematikan overlay, dan klik gambar untuk membaca jarak pada titik tersebut.
Kamera selain perangkat default dapat dipilih, misalnya:

```powershell
.\.venv\Scripts\python.exe main.py --camera 1
```

## Model

Inferensi memakai ONNX Runtime dan model ONNX multitask berikut:

`yolo26n_multitask_coco4_v1_20261013-163528-539afa/inference/multitask.onnx`

Metadata model berada di folder `inference/` yang sama. Model mendeteksi kelas
`person`, `laptop`, `cell phone`, dan `bottle`; output depth memakai satuan
meter. Checkpoint `.pt`, dataset, dan berkas training tidak diperlukan untuk
menjalankan kamera.

## Tes

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```
