import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# SIMULASI ARSITEKTUR LIGHTWEIGHT YOLO26 (MURNI DETEKSI OBJEK 2D)
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LightweightYOLO26Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32, k=3, s=2, p=1)
        self.stage2 = ConvBlock(32, 64, k=3, s=2, p=1)
        self.stage3 = ConvBlock(64, 128, k=3, s=2, p=1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x

class LightweightYOLO26Head(nn.Module):
    def __init__(self, in_ch=128, num_classes=80):
        super().__init__()
        self.reg_conv = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, 4, kernel_size=1)
        )
        self.cls_conv = nn.Sequential(
            ConvBlock(in_ch, in_ch, k=3, s=1, p=1),
            nn.Conv2d(in_ch, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.reg_conv(x), self.cls_conv(x)

class LightweightYOLO26(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = LightweightYOLO26Backbone()
        self.head = LightweightYOLO26Head(in_ch=128, num_classes=num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

# =====================================================================
# PENGUJIAN FUNGSI KERUGIAN & ALIRAN GRADIEN
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightYOLO26(num_classes=80).to(device)

    # Dummy Input
    dummy_batch = torch.randn(2, 3, 640, 640, device=device)
    out_bbox, out_cls = model(dummy_batch)

    # 1. Menyiapkan Fungsi Kerugian (Loss Function) 2D
    kriteria_bbox = nn.SmoothL1Loss()         # Evaluasi regresi koordinat bounding box
    kriteria_kelas = nn.BCEWithLogitsLoss()   # Evaluasi klasifikasi kategori objek

    # 2. Menyiapkan Pengoptimal AdamW
    pengoptimal = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 3. Simulasi Target Ground Truth
    target_bbox = torch.randn_like(out_bbox)
    target_kelas = torch.zeros_like(out_cls)
    target_kelas[:, 0, :, :] = 1.0            # Objek aktif kelas ke-0

    # 4. Kalkulasi Total Loss
    loss_bbox = kriteria_bbox(out_bbox, target_bbox)
    loss_kelas = kriteria_kelas(out_cls, target_kelas)
    total_loss = loss_bbox + loss_kelas

    # 5. Backward Pass & Pembaruan Parameter
    pengoptimal.zero_grad()
    total_loss.backward()
    pengoptimal.step()

    # Memeriksa Keberhasilan Aliran Gradien ke Lapisan Pertama Backbone
    gradien_tersedia = model.backbone.stage1.conv.weight.grad is not None

    print("=== VERIFIKASI FUNGSI KERUGIAN DAN ALIRAN GRADIEN LIGHTWEIGHT YOLO26 ===")
    print(f"Nilai Box Loss       : {loss_bbox.item():.4f}")
    print(f"Nilai Class Loss     : {loss_kelas.item():.4f}")
    print(f"Total Loss           : {total_loss.item():.4f}")
    print(f"Gradien Terdeteksi   : {gradien_tersedia}")
    print(f"Status Pembaruan     : Berhasil (Optimizer Step Sukses)")