# Checklist Pengambilan Screenshot & Pengukuran Fisik Manual (Webcam)

Dokumen ini berisi daftar skenario pengujian real-time yang harus diambil secara manual oleh pengguna menggunakan webcam dan meteran fisik.

## Persiapan
1. Gunakan Python sistem yang memiliki CUDA aktif (`python`, bukan `.venv` CPU-only) dari root folder proyek:
   `c:\Users\ACER\Downloads\Comvis\code`
2. Skrip `scripts/capture_compare.py` kini memakai **default**:
   - `--model-a`: `training_runs/persist/rtx4050_noattn_70_20_10/best.pt` (**Baseline**)
   - `--model-b`: `training_runs/persist/rtx4050_arsitektur_70_20_10/best.pt` (**+Attention**)
3. Tombol kontrol di jendela OpenCV:
   - `SPACE` atau `c`: simpan tangkapan layar side-by-side (`[Baseline | +Attention]`) ke `docs/figures/webcam_captures/` dan catat otomatis ke `eval/webcam_measurements.csv`
   - `s`: toggle tampilan depth map INFERNO
   - `q`: keluar dari skrip

## Daftar Skenario Wajib (Tabel 4.6 & Sub-bab 4.4)

| No | Skenario | Nama File Target | Kondisi | Perintah Persis | Status |
|----|----------|------------------|---------|-----------------|--------|
| 1 | Jarak Dekat 0,5 m (Terang) | `jarak_0.5m_terang_person_*.png` | Objek `person`, terang, meteran `0.50 m` | `python scripts/capture_compare.py --scenario jarak_0.5m_terang_person --actual-dist 0.50` | [ ] Belum |
| 2 | Jarak Sedang 1,0 m (Terang) | `jarak_1.0m_terang_person_*.png` | Objek `person`, terang, meteran `1.00 m` | `python scripts/capture_compare.py --scenario jarak_1.0m_terang_person --actual-dist 1.00` | [ ] Belum |
| 3 | Jarak Jauh 2,0 m (Terang) | `jarak_2.0m_terang_person_*.png` | Objek `person`, terang, meteran `2.00 m` | `python scripts/capture_compare.py --scenario jarak_2.0m_terang_person --actual-dist 2.00` | [ ] Belum |
| 4 | Jarak Dekat 0,5 m (Gelap) | `jarak_0.5m_gelap_person_*.png` | Objek `person`, gelap, meteran `0.50 m` | `python scripts/capture_compare.py --scenario jarak_0.5m_gelap_person --actual-dist 0.50` | [ ] Belum |
| 5 | Jarak Sedang 1,0 m (Gelap) | `jarak_1.0m_gelap_person_*.png` | Objek `person`, gelap, meteran `1.00 m` | `python scripts/capture_compare.py --scenario jarak_1.0m_gelap_person --actual-dist 1.00` | [ ] Belum |
| 6 | Jarak Jauh 2,0 m (Gelap) | `jarak_2.0m_gelap_person_*.png` | Objek `person`, gelap, meteran `2.00 m` | `python scripts/capture_compare.py --scenario jarak_2.0m_gelap_person --actual-dist 2.00` | [ ] Belum |
| 7 | Kasus Occlusion | `kasus_occlusion_*.png` | Objek terhalang sebagian, meteran `1.00 m` | `python scripts/capture_compare.py --scenario kasus_occlusion --actual-dist 1.00` | [ ] Belum |
| 8 | Dinding Polos | `kasus_dinding_polos_*.png` | Dinding polos minim tekstur, meteran `1.50 m` | `python scripts/capture_compare.py --scenario kasus_dinding_polos --actual-dist 1.50` | [ ] Belum |
| 9 | Objek `bottle` (0,5 m) | `jarak_0.5m_terang_bottle_*.png` | Botol minum, terang, meteran `0.50 m` | `python scripts/capture_compare.py --scenario jarak_0.5m_terang_bottle --actual-dist 0.50` | [ ] Belum |
| 10 | Objek `bottle` (1,0 m) | `jarak_1.0m_terang_bottle_*.png` | Botol minum, terang, meteran `1.00 m` | `python scripts/capture_compare.py --scenario jarak_1.0m_terang_bottle --actual-dist 1.00` | [ ] Belum |
| 11 | Objek `laptop` (1,0 m) | `jarak_1.0m_terang_laptop_*.png` | Laptop terbuka, terang, meteran `1.00 m` | `python scripts/capture_compare.py --scenario jarak_1.0m_terang_laptop --actual-dist 1.00` | [ ] Belum |
| 12 | Objek `cell phone` (0,5 m) | `jarak_0.5m_terang_cellphone_*.png` | Smartphone, terang, meteran `0.50 m` | `python scripts/capture_compare.py --scenario jarak_0.5m_terang_cellphone --actual-dist 0.50` | [ ] Belum |

## Protokol Pencatatan Tabel 4.6
- Lakukan minimal **3 kali pengulangan** (`SPACE`/`c`) per skenario.
- Setiap tangkapan otomatis tersimpan ke `eval/webcam_measurements.csv` beserta `error = |prediksi - jarak_aktual|`.
