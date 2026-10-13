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

## Alur terhadap diagram arsitektur

`main.py` merupakan runner inferensi, bukan implementasi layer YOLO26n.
Backbone, neck, dan kedua head dijalankan bersama di dalam graph ONNX:

1. Kamera menghasilkan frame BGR.
2. `MultitaskOnnxModel.predict` melakukan resize dengan rasio aspek terjaga,
   padding ke ukuran input model, konversi BGR ke RGB, dan normalisasi.
3. ONNX Runtime menjalankan satu graph multitask untuk deteksi dan depth.
4. Kode mengembalikan kotak deteksi dan peta depth ke ukuran frame kamera;
   median depth di dalam kotak dipakai sebagai jarak objek.

Ukuran input model yang sebenarnya adalah `[1, 3, 480, 640]` dalam urutan
`NCHW`, bukan gambar persegi `640 x 640` pada diagram. Output graph adalah
deteksi `[1, 300, 6]` dan depth `[1, 1, 480, 640]`. Diagram menunjukkan jalur
detection YOLO26n; aplikasi ini mempertahankan jalur tersebut sekaligus output
depth tambahan. Nama blok internal pada diagram tidak didefinisikan ulang di
`main.py`; saat runtime, implementasinya berasal dari graph ONNX terlatih.

### Implementasi arsitektur dari laporan

`arsitektur.py` menerapkan rancangan pada bagian 3.1-3.2 laporan sebagai model
PyTorch terpisah:

- Backbone ber-skala YOLO26n (`width_mult=0.25`, `depth_mult=0.50`,
  `max_channels=1024`) menghasilkan fitur P3/8, P4/16, dan P5/32. Backbone
  memakai Conv, C3k2, SPPF, lalu C2PSA dengan channel gate sesuai kode bagian
  3.2 laporan. Sisipan attention opsional tambahan pada C3k2/neck tidak
  diaktifkan.
- Neck menggabungkan P3/P4/P5 dengan alur FPN top-down dan PAN bottom-up.
- Tiga detection head terpisah menghasilkan tensor regresi bbox dan logits
  kelas pada masing-masing skala. Ini adalah output mentah per skala; belum
  didekode menjadi daftar kotak akhir.
- DepthHead memproyeksikan fitur P3/P4/P5 masing-masing ke 32 channel,
  mengubahnya ke ukuran input, menggabungkannya, lalu menghasilkan peta satu
  channel melalui sigmoid. Nilainya relatif pada rentang `[0, 1]`, **bukan
  jarak meter terkalibrasi**.

Pemeriksaan `--verify` menggunakan input `[1, 3, 640, 640]`. Dengan empat kelas
yang digunakan aplikasi, bentuk outputnya ialah P3 `[1, 4, 80, 80]`, P4
`[1, 4, 40, 40]`, P5 `[1, 4, 20, 20]` untuk masing-masing bbox dan kelas, serta
depth `[1, 1, 640, 640]`.

Laporan menyebut 2.572.280 parameter dan 6,1 GFLOPs untuk baseline YOLO26n;
angka itu merujuk ke baseline, bukan implementasi multitask khusus di file ini.
Blueprint ini memiliki 3.766.825 parameter pada konfigurasi default yang
diverifikasi. Bagian laporan juga membedakan output Detect baseline end-to-end
dari contoh Section 3.2 yang mengeluarkan tensor mentah per skala; kode ini
mengikuti struktur terperinci Section 3.2.

Model PyTorch ini **belum dilatih dan tidak dipakai `main.py`**. Kamera tetap
memakai ONNX multitask terlatih pada input 640×480, dengan depth dalam meter.
Blueprint 640×640 ini bukan pengganti ONNX atau checkpoint terlatih.

Opsional, pasang PyTorch lalu jalankan pemeriksaan forward pass:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-architecture.txt
.\.venv\Scripts\python.exe arsitektur.py --verify
```

## Tes

```powershell
.\.venv\Scripts\python.exe -m unittest discover
```
