## 4.2 Analisis Pengujian Kualitatif Model Multitask

Berdasarkan hasil pengujian kualitatif pada antarmuka *real-time* aplikasi desktop PyTorch (Gambar 4.2 pada draf laporan) serta hasil rendering pada *test split* (Sub-bab 4.4), model *multitask dual-head* mampu menjalankan deteksi objek dan estimasi kedalaman secara simultan dalam satu kali *forward pass*.

Pada pengujian webcam *real-time* terhadap objek `person`, sistem menghasilkan estimasi jarak sebesar `0,65 m` (rentang `0,46–0,65 m`) dengan pembacaan titik tengah layar `0,58 m`. Visualisasi peta kedalaman menggunakan *colormap* `INFERNO` memperlihatkan pemisahan kontur tubuh objek `person` terhadap latar belakang ruangan secara jelas, yang menunjukkan bahwa cabang *Depth Head* mampu menangkap struktur kedalaman spasial relatif meskipun menggunakan arsitektur ringan berskala nano.

Verifikasi tambahan pada *held-out test split* SUN RGB-D untuk model **+Attention (`YOLO26nDepth`)** (Gambar 4.3–4.5) juga memperlihatkan keselarasan visual antara kotak deteksi dan peta kedalaman *INFERNO*: pada citra `NYU0850.jpg`, objek `person` terdeteksi dengan estimasi jarak `4,12 m` dibandingkan *ground truth* kotak `4,17 m` (selisih absolut `0,05 m`), sedangkan pada `NYU0146.jpg` objek `laptop` diprediksi `2,13 m` terhadap *ground truth* `1,89 m` (selisih `0,24 m`).
