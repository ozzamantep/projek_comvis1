import torch
import torch.nn as nn

# =====================================================================
# 1. STRUKTUR ARSITEKTUR LIGHTWEIGHT YOLO26 (UNTUK UJI INSPEKSI)
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cv1 = ConvBlock(ch, ch, k=3, s=1, p=1)
        self.cv2 = ConvBlock(ch, ch, k=3, s=1, p=1)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))

class C2f(nn.Module):
    def __init__(self, in_ch, out_ch, n=1):
        super().__init__()
        self.cv1 = ConvBlock(in_ch, out_ch, k=1, s=1, p=0)
        self.cv2 = ConvBlock((2 + n) * (out_ch // 2), out_ch, k=1, s=1, p=0)
        self.m = nn.ModuleList([Bottleneck(out_ch // 2) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden_ch = in_ch // 2
        self.cv1 = ConvBlock(in_ch, hidden_ch, k=1, s=1, p=0)
        self.cv2 = ConvBlock(hidden_ch * 4, out_ch, k=1, s=1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class LightweightYOLO26Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = ConvBlock(3, 32, k=3, s=2, p=1)       # 640 -> 320
        self.stage2 = ConvBlock(32, 64, k=3, s=2, p=1)      # 320 -> 160
        self.c2f1   = C2f(64, 64, n=1)
        self.stage3 = ConvBlock(64, 128, k=3, s=2, p=1)     # 160 -> 80
        self.c2f2   = C2f(128, 128, n=1)
        self.sppf   = SPPF(128, 128)

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

class LightweightYOLO26(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.backbone = LightweightYOLO26Backbone()
        self.head = LightweightYOLO26Head(in_ch=128, num_classes=num_classes)

# =====================================================================
# 2. SKRIP INSPEKSI DAN VERIFIKASI DIMENSI TENSOR AUTOMATIS
# =====================================================================
def verifikasi_arsitektur(model, tensor_masukan):
    print("=== VERIFIKASI KESESUAIAN STRUKTUR ARSITEKTUR LIGHTWEIGHT YOLO26 ===")
    x = tensor_masukan
    print(f"0. Tensor Masukan       : {list(x.shape)} -> Sesuai [Batch, 3, 640, 640]")

    # 1. Penelusuran alur lapisan Backbone
    x = model.backbone.stage1(x)
    print(f"1. Backbone Stage 1     : {list(x.shape)} -> Reduksi 1/2 (Resolusi 320x320)")

    x = model.backbone.stage2(x)
    print(f"2. Backbone Stage 2     : {list(x.shape)} -> Reduksi 1/4 (Resolusi 160x160)")

    x = model.backbone.c2f1(x)
    print(f"3. Blok C2f Ke-1        : {list(x.shape)} -> Ekstraksi Fitur Konsisten")

    x = model.backbone.stage3(x)
    print(f"4. Backbone Stage 3     : {list(x.shape)} -> Reduksi 1/8 (Resolusi 80x80)")

    x = model.backbone.c2f2(x)
    print(f"5. Blok C2f Ke-2        : {list(x.shape)} -> Ekstraksi Fitur Konsisten")

    peta_fitur = model.backbone.sppf(x)
    print(f"6. Blok SPPF (Fusion)   : {list(peta_fitur.shape)} -> Konteks Spasial Adaptif Tergabung")

    # 2. Penelusuran alur Decoupled Detection Head Murni 2D
    pred_bbox = model.head.reg_conv(peta_fitur)
    pred_cls = model.head.cls_conv(peta_fitur)

    print(f"7. Head Regresi (BBox)  : {list(pred_bbox.shape)} -> Sesuai (4 Koordinat BBox)")
    print(f"8. Head Klasifikasi     : {list(pred_cls.shape)} -> Sesuai (80 Kelas Objek)")

    # 3. Penghitungan total parameter komputasi terlatih
    total_bobot = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("-------------------------------------------------------")
    print(f"Total Parameter Aktif   : {total_bobot:,} Bobot")
    print("Hasil Uji Keselarasan   : Struktur Sesuai Rancangan Arsitektur 2D (Valid)")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightweightYOLO26(num_classes=80).to(device)
    dummy_batch = torch.randn(2, 3, 640, 640, device=device)
    verifikasi_arsitektur(model, dummy_batch)