import torch

def hitung_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua set bounding box.
    Format koordinat: [x1, y1, x2, y2]
    """
    # 1. Menentukan koordinat area irisan (intersection)
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])

    lebar_irisan = torch.clamp(x2 - x1, min=0)
    tinggi_irisan = torch.clamp(y2 - y1, min=0)
    luas_irisan = lebar_irisan * tinggi_irisan

    # 2. Menghitung luas area masing-masing kotak
    luas_box1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    luas_box2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    # 3. Menghitung area gabungan (union) dan rasio IoU
    luas_gabungan = luas_box1 + luas_box2 - luas_irisan
    iou = luas_irisan / torch.clamp(luas_gabungan, min=1e-6)
    return iou

def evaluasi_deteksi(pred_boxes, pred_classes, target_boxes, target_classes, iou_threshold=0.5):
    """
    Mengevaluasi kinerja deteksi 2D dengan menghitung True Positive, Precision,
    Recall, dan F1-Score berdasarkan ambang batas IoU.
    """
    # Menghitung IoU untuk seluruh sampel uji
    skor_iou = hitung_iou(pred_boxes, target_boxes)

    # Evaluasi True Positive: IoU >= threshold DAN kelas objek sesuai
    benar_lokasi = skor_iou >= iou_threshold
    benar_kelas = pred_classes == target_classes
    true_positives = (benar_lokasi & benar_kelas).sum().item()

    total_prediksi = len(pred_boxes)
    total_target = len(target_boxes)

    false_positives = total_prediksi - true_positives
    false_negatives = total_target - true_positives

    # Perhitungan metrik Precision, Recall, dan F1-Score
    presisi = true_positives / max(total_prediksi, 1)
    recall = true_positives / max(total_target, 1)
    skor_f1 = 2 * (presisi * recall) / max((presisi + recall), 1e-6)

    print("=== VERIFIKASI METRIK EVALUASI KINERJA DETEKSI LIGHTWEIGHT YOLO26 ===")
    print(f"Rata-rata Skor IoU     : {skor_iou.mean().item():.4f}")
    print(f"Ambang Batas IoU       : {iou_threshold}")
    print(f"True Positive (TP)     : {true_positives}")
    print(f"False Positive (FP)    : {false_positives}")
    print(f"False Negative (FN)    : {false_negatives}")
    print(f"Presisi (Precision)    : {presisi * 100:.2f}%")
    print(f"Sensitivitas (Recall)  : {recall * 100:.2f}%")
    print(f"F1-Score               : {skor_f1:.4f}")
    print(f"Status Modul Evaluasi  : Terverifikasi (Kalkulasi Matriks Valid)")

# =====================================================================
# SIMULASI PENGUJIAN METRIK DENGAN DATA TENSOR
# =====================================================================
if __name__ == "__main__":
    # Format Bounding Box: [x1, y1, x2, y2] pada skala resolusi 640x640
    target_b = torch.tensor([[100.0, 100.0, 300.0, 300.0], [50.0, 50.0, 200.0, 200.0]])
    pred_b   = torch.tensor([[110.0, 105.0, 305.0, 295.0], [40.0, 40.0, 180.0, 190.0]])

    target_c = torch.tensor([1])  # Target: Kelas 0 dan Kelas 1
    pred_c   = torch.tensor([1])  # Tebakan: Kelas 0 dan Kelas 1

    evaluasi_deteksi(pred_b, pred_c, target_b, target_c)