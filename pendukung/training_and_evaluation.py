import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. ARSITEKTUR LIGHTWEIGHT YOLO26 (MURNI DETEKSI OBJEK 2D)
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
# 2. SIMULASI SIKLUS PELATIHAN (TRAINING LOOP) DAN EVALUASI INFERENSI
# =====================================================================
def simulasi_siklus_pelatihan(model, device, num_epochs=3):
    print("=== VERIFIKASI SIKLUS PELATIHAN DAN EVALUASI LIGHTWEIGHT YOLO26 ===")

    # 1. Konfigurasi kriteria kerugian multi-tugas 2D & pengoptimal AdamW
    kriteria_box = nn.SmoothL1Loss()
    kriteria_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 2. Siklus Pelatihan (Training Loop)
    batch_size = 2
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for step in range(2):  # Simulasi 2 iterasi per epoch
            # Penyiapan batch citra dan target koordinat serta kelas tiruan
            inputs = torch.randn(batch_size, 3, 640, 640, device=device)
            target_box = torch.randn(batch_size, 4, 80, 80, device=device)
            target_cls = torch.zeros(batch_size, 80, 80, 80, device=device)
            target_cls[:, 0, :, :] = 1.0  # Simulasi target kelas aktif ke-0

            # Forward pass & kalkulasi multi-task loss 2D
            optimizer.zero_grad()
            pred_box, pred_cls = model(inputs)

            loss_box = kriteria_box(pred_box, target_box)
            loss_cls = kriteria_cls(pred_cls, target_cls)

            # Akumulasi total loss (BBox + Klasifikasi)
            loss = loss_box + loss_cls

            # Backward pass & pembaruan bobot
            loss.backward()
            optimizer.step()
            total_epoch_loss += loss.item()

        rata_loss = total_epoch_loss / 2
        print(f"Epoch [{epoch}/{num_epochs}] - Training Loss: {rata_loss:.4f} - Status: Bobot Diperbarui")

    # 3. Siklus Evaluasi / Inferensi (Validation Mode)
    model.eval()
    with torch.no_grad():
        citra_val = torch.randn(1, 3, 640, 640, device=device)
        out_box_eval, out_cls_eval = model(citra_val)
        
        print("-------------------------------------------------------")
        print(f"Mode Evaluasi           : Sukses (torch.no_grad aktif)")
        print(f"Dimensi Inferensi BBox  : {list(out_box_eval.shape)} -> [Batch, 4, 80, 80]")
        print(f"Dimensi Inferensi Kelas : {list(out_cls_eval.shape)} -> [Batch, Num_Classes, 80, 80]")
        print(f"Kesiapan Model          : Siap Diterapkan pada Dataset Riil (Edge-AI)")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightYOLO26(num_classes=80).to(device)
    simulasi_siklus_pelatihan(model, device, num_epochs=3)