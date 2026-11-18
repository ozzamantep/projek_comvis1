"""Generate publication-quality architecture and CBAM detail diagrams (PNG 300 dpi + SVG)."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _box(ax, x, y, w, h, title, subtitle="", facecolor="#EFF6FF", edgecolor="#2563EB",
         lw=1.5, title_color="#0F172A", sub_color="#334155", fontsize=8.5, subsize=7.0,
         linestyle="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, linestyle=linestyle,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=title_color)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=subsize, color=sub_color)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=title_color)


def _arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.3, style="-|>", ls="-", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=11,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


def make_arsitektur_usulan() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.6), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # Title
    ax.text(7.0, 7.32, "Arsitektur Model Usulan: YOLO26n-Depth + CBAM Attention (Input 480x640)",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0F172A")

    # Legend
    _box(ax, 0.3, 6.72, 1.8, 0.38, "Blok Baseline", facecolor="#EFF6FF", edgecolor="#2563EB", fontsize=8)
    _box(ax, 2.3, 6.72, 2.3, 0.38, "Attention Bawaan (C2PSA)", facecolor="#FEF3C7", edgecolor="#D97706", fontsize=8)
    _box(ax, 4.8, 6.72, 2.7, 0.38, "Attention Tambahan (+CBAM)", facecolor="#DCFCE7", edgecolor="#16A34A", lw=2.0, fontsize=8)
    _box(ax, 7.7, 6.72, 2.5, 0.38, "Post-Processing / Output", facecolor="#F3E8FF", edgecolor="#7C3AED", fontsize=8)

    # Group containers
    ax.add_patch(FancyBboxPatch((1.85, 1.0), 2.5, 5.45, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", linestyle="--", linewidth=1.0))
    ax.text(3.1, 6.28, "Backbone (Layer 0–10)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E293B")

    ax.add_patch(FancyBboxPatch((4.65, 1.0), 2.7, 5.45, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#F8FAFC", edgecolor="#94A3B8", linestyle="--", linewidth=1.0))
    ax.text(6.0, 6.28, "Neck FPN + PAN (Layer 11–22)", ha="center", va="center", fontsize=9, fontweight="bold", color="#1E293B")

    # 1. Input
    _box(ax, 0.25, 3.3, 1.3, 0.9, "Input RGB", "[1, 3, 480, 640]\n(Letterbox)",
         facecolor="#F1F5F9", edgecolor="#475569", fontsize=8.5, subsize=7.0)

    # 2. Backbone (L0-10)
    _box(ax, 2.0, 5.15, 2.2, 0.8, "L0–L2: Stem + C3k2", "P2/4: [1, 64, 120, 160]")
    _box(ax, 2.0, 4.05, 2.2, 0.8, "L3–L4: Conv + C3k2", "P3/8: [1, 128, 60, 80]")
    _box(ax, 2.0, 2.95, 2.2, 0.8, "L5–L6: Conv + C3k2", "P4/16: [1, 128, 30, 40]")
    _box(ax, 2.0, 1.85, 2.2, 0.8, "L7–L9: Conv+C3k2+SPPF", "P5/32: [1, 256, 15, 20]")
    _box(ax, 2.0, 1.12, 2.2, 0.58, "L10: C2PSA (Bawaan)", "[1, 256, 15, 20]",
         facecolor="#FEF3C7", edgecolor="#D97706", fontsize=8.0, subsize=6.8)

    _arrow(ax, 1.55, 3.75, 2.0, 5.55)
    _arrow(ax, 3.1, 5.15, 3.1, 4.85)
    _arrow(ax, 3.1, 4.05, 3.1, 3.75)
    _arrow(ax, 3.1, 2.95, 3.1, 2.65)
    _arrow(ax, 3.1, 1.85, 3.1, 1.70)

    # 3. Neck (L11-22)
    _box(ax, 4.8, 4.65, 2.4, 0.8, "L11–13: FPN Top-Down", "n13 C3k2: [1, 128, 30, 40]")
    _box(ax, 4.8, 3.55, 2.4, 0.8, "L14–16: FPN -> P3_out", "n16 C3k2: [1, 64, 60, 80]")
    _box(ax, 4.8, 2.40, 2.4, 0.8, "L17–19: PAN -> P4_out", "n19 C3k2: [1, 128, 30, 40]")
    _box(ax, 4.8, 1.25, 2.4, 0.85, "L20–22: PAN -> P5_out", "n22 C3k2+PSA (Bawaan)\n[1, 256, 15, 20]",
         facecolor="#FEF3C7", edgecolor="#D97706", fontsize=7.8, subsize=6.5)

    # Backbone -> Neck arrows
    _arrow(ax, 4.2, 1.41, 4.8, 5.05, color="#2563EB", rad=-0.12)
    _arrow(ax, 4.2, 3.35, 4.8, 4.90, color="#2563EB")
    _arrow(ax, 4.2, 4.45, 4.8, 3.95, color="#2563EB")
    _arrow(ax, 4.2, 1.41, 4.8, 1.65, color="#2563EB")
    _arrow(ax, 6.0, 4.65, 6.0, 4.35)
    _arrow(ax, 6.0, 3.55, 6.0, 3.20)
    _arrow(ax, 6.0, 2.40, 6.0, 2.10)

    # 4. +CBAM Insertion at Neck Outputs
    _box(ax, 7.65, 4.15, 1.55, 0.72, "+CBAM P3", "[1, 64, 60, 80]",
         facecolor="#DCFCE7", edgecolor="#16A34A", lw=2.0, fontsize=8.2)
    _box(ax, 7.65, 2.85, 1.55, 0.72, "+CBAM P4", "[1, 128, 30, 40]",
         facecolor="#DCFCE7", edgecolor="#16A34A", lw=2.0, fontsize=8.2)
    _box(ax, 7.65, 1.45, 1.55, 0.72, "+CBAM P5", "[1, 256, 15, 20]",
         facecolor="#DCFCE7", edgecolor="#16A34A", lw=2.0, fontsize=8.2)

    _arrow(ax, 7.2, 3.95, 7.65, 4.51, color="#16A34A", lw=1.6)
    _arrow(ax, 7.2, 2.80, 7.65, 3.21, color="#16A34A", lw=1.6)
    _arrow(ax, 7.2, 1.67, 7.65, 1.81, color="#16A34A", lw=1.6)

    # 5. Dual Heads
    # Detection Head
    _box(ax, 9.6, 4.35, 2.1, 1.45, "Detection Head (3 Skala)",
         "DFL-Free (reg + cls)\nP3: [1,4,60,80] | P4: [1,4,30,40]\nP5: [1,4,15,20]\n4 Kelas Target + NMS",
         facecolor="#EFF6FF", edgecolor="#2563EB", fontsize=8.3, subsize=6.7)

    # Depth Head
    _box(ax, 9.6, 2.35, 2.1, 0.95, "Depth Fusion (32ch x 3)",
         "Conv1x1 + Bilinear Upsample\nConcat -> [1, 96, 480, 640]",
         facecolor="#EFF6FF", edgecolor="#2563EB", fontsize=8.0, subsize=6.7)
    _box(ax, 9.6, 1.45, 2.1, 0.65, "+CBAM Depth", "[1, 96, 480, 640]",
         facecolor="#DCFCE7", edgecolor="#16A34A", lw=2.0, fontsize=8.2)
    _box(ax, 9.6, 0.35, 2.1, 0.82, "Depth Refine + Sigmoid",
         "Conv3x3 (96->32->16->1)\nDepth Map [1, 1, 480, 640]",
         facecolor="#EFF6FF", edgecolor="#2563EB", fontsize=8.0, subsize=6.7)

    # Arrows to Dual Heads
    _arrow(ax, 9.2, 4.51, 9.6, 5.20, color="#2563EB")
    _arrow(ax, 9.2, 3.21, 9.6, 4.95, color="#2563EB")
    _arrow(ax, 9.2, 1.81, 9.6, 4.70, color="#2563EB")

    _arrow(ax, 9.2, 4.40, 9.6, 2.95, color="#2563EB", ls="--")
    _arrow(ax, 9.2, 3.10, 9.6, 2.82, color="#2563EB", ls="--")
    _arrow(ax, 9.2, 1.70, 9.6, 2.65, color="#2563EB", ls="--")

    _arrow(ax, 10.65, 2.35, 10.65, 2.10, color="#16A34A", lw=1.6)
    _arrow(ax, 10.65, 1.45, 10.65, 1.17, color="#2563EB")

    # 6. ROI Mapping & Output
    _box(ax, 12.1, 2.85, 1.65, 1.35, "ROI Mapping 15x15",
         "Un-letterbox + EMA (0,35)\nMedian Patch 15x15\ndi Pusat Bbox",
         facecolor="#F3E8FF", edgecolor="#7C3AED", fontsize=8.2, subsize=6.7)
    _box(ax, 12.1, 1.15, 1.65, 1.05, "Output Akhir",
         "Bbox + Label Kelas\nDepth Map INFERNO\nJarak Objek (meter)",
         facecolor="#F3E8FF", edgecolor="#7C3AED", fontsize=8.3, subsize=6.8)

    _arrow(ax, 11.7, 5.05, 12.92, 4.20, color="#7C3AED")
    _arrow(ax, 11.7, 0.76, 12.1, 3.30, color="#7C3AED")
    _arrow(ax, 12.92, 2.85, 12.92, 2.20, color="#7C3AED")

    ax.text(7.0, 0.12,
            "Catatan: Modul +CBAM disisipkan pada 4 titik (P3_out, P4_out, P5_out, dan fusi DepthHead) dengan tambahan +12.296 parameter (+0,32% dari Baseline).",
            ha="center", va="center", fontsize=8.0, style="italic", color="#334155")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "arsitektur_usulan.png", dpi=300)
    fig.savefig(OUT_DIR / "arsitektur_usulan.svg")
    plt.close(fig)


def make_modul_attention_detail() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=300)
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(6.25, 4.92, "Detail Internal Modul Attention Tambahan: CBAM (Channel + Spatial Attention)",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#0F172A")

    # Container 1: Channel Attention
    ax.add_patch(FancyBboxPatch((2.1, 0.55), 4.4, 3.9, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#F0FDF4", edgecolor="#16A34A", linewidth=1.4))
    ax.text(4.3, 4.25, "1. Channel Attention Module (CAM)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#166534")

    # Container 2: Spatial Attention
    ax.add_patch(FancyBboxPatch((6.85, 0.55), 4.1, 3.9, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#EFF6FF", edgecolor="#2563EB", linewidth=1.4))
    ax.text(8.9, 4.25, "2. Spatial Attention Module (SAM)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#1E40AF")

    # Input X
    _box(ax, 0.25, 2.1, 1.5, 0.9, "Input Fitur X", "[B, C, H, W]\nC=64/128/256/96",
         facecolor="#F1F5F9", edgecolor="#475569", fontsize=8.2, subsize=6.8)

    # CAM blocks
    _box(ax, 2.3, 3.15, 1.55, 0.65, "Global AvgPool", "[B, C, 1, 1]", facecolor="#DCFCE7", edgecolor="#16A34A")
    _box(ax, 2.3, 1.35, 1.55, 0.65, "Global MaxPool", "[B, C, 1, 1]", facecolor="#DCFCE7", edgecolor="#16A34A")
    _box(ax, 4.1, 2.15, 1.35, 0.85, "Shared MLP", "Conv1x1 (C->C/16)\n+ SiLU + Conv1x1", facecolor="#DCFCE7", edgecolor="#16A34A", subsize=6.6)
    _box(ax, 5.65, 2.2, 0.7, 0.75, "Add +\nSigmoid", "Mc(X)", facecolor="#BBF7D0", edgecolor="#15803D", fontsize=7.5, subsize=6.5)

    _arrow(ax, 1.75, 2.65, 2.3, 3.47, color="#16A34A")
    _arrow(ax, 1.75, 2.45, 2.3, 1.67, color="#16A34A")
    _arrow(ax, 3.85, 3.47, 4.1, 2.75, color="#16A34A")
    _arrow(ax, 3.85, 1.67, 4.1, 2.35, color="#16A34A")
    _arrow(ax, 5.45, 2.57, 5.65, 2.57, color="#16A34A")

    # Skip multiply for X' = X * Mc(X)
    _arrow(ax, 1.0, 2.1, 6.0, 0.95, color="#475569", ls="--", rad=0.15)
    _box(ax, 5.45, 0.72, 1.1, 0.55, "X' = X ⊗ Mc", "[B, C, H, W]", facecolor="#FEF9C3", edgecolor="#CA8A04", fontsize=7.6, subsize=6.5)
    _arrow(ax, 6.0, 2.2, 6.0, 1.27, color="#15803D")

    # SAM blocks
    _box(ax, 7.05, 3.15, 1.55, 0.65, "Channel AvgPool", "[B, 1, H, W]", facecolor="#DBEAFE", edgecolor="#2563EB")
    _box(ax, 7.05, 1.35, 1.55, 0.65, "Channel MaxPool", "[B, 1, H, W]", facecolor="#DBEAFE", edgecolor="#2563EB")
    _box(ax, 8.85, 2.15, 1.15, 0.85, "Concat +\nConv 7x7", "[B, 2, H, W]\n-> [B, 1, H, W]", facecolor="#DBEAFE", edgecolor="#2563EB", subsize=6.4)
    _box(ax, 10.18, 2.22, 0.65, 0.7, "Sigmoid", "Ms(X')", facecolor="#BFDBFE", edgecolor="#1D4ED8", fontsize=7.4, subsize=6.5)

    _arrow(ax, 6.55, 1.0, 7.05, 3.47, color="#2563EB")
    _arrow(ax, 6.55, 1.0, 7.05, 1.67, color="#2563EB")
    _arrow(ax, 8.60, 3.47, 8.85, 2.75, color="#2563EB")
    _arrow(ax, 8.60, 1.67, 8.85, 2.35, color="#2563EB")
    _arrow(ax, 10.0, 2.57, 10.18, 2.57, color="#2563EB")

    # Output Y = X' * Ms(X')
    _box(ax, 11.15, 2.1, 1.15, 0.9, "Output Y", "Y = X' ⊗ Ms\n[B, C, H, W]",
         facecolor="#F3E8FF", edgecolor="#7C3AED", fontsize=8.2, subsize=6.8)
    _arrow(ax, 10.83, 2.57, 11.15, 2.57, color="#7C3AED", lw=1.6)
    _arrow(ax, 6.55, 0.95, 11.72, 2.1, color="#475569", ls="--", rad=0.18)

    ax.text(6.25, 0.18,
            "Dimensi tensor terjaga [B, C, H, W]: P3 (C=64, 60x80), P4 (C=128, 30x40), P5 (C=256, 15x20), dan Depth Fusion (C=96, 480x640); reduction ratio r = 16.",
            ha="center", va="center", fontsize=8.0, style="italic", color="#334155")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "modul_attention_detail.png", dpi=300)
    fig.savefig(OUT_DIR / "modul_attention_detail.svg")
    plt.close(fig)


if __name__ == "__main__":
    make_arsitektur_usulan()
    make_modul_attention_detail()
    print("Generated arsitektur_usulan.{png,svg} and modul_attention_detail.{png,svg}")

