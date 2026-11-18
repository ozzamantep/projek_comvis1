"""Generate Figure 1-style (Hidayatullah & Tubagus 2026) architecture diagrams for Baseline (NoAttn) and +Attention."""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "Conv": ("#D1FAE5", "#059669", "#065F46"),
    "C3k2": ("#FEE2E2", "#DC2626", "#7F1D1D"),
    "SPPF": ("#FEF3C7", "#D97706", "#78350F"),
    "C2PSA": ("#DBEAFE", "#2563EB", "#1E3A8A"),
    "C2PSA_HI": ("#FDE047", "#B45309", "#78350F"),
    "Attn_C3k2": ("#FEF08A", "#B45309", "#78350F"),
    "Removed": ("#F1F5F9", "#94A3B8", "#475569"),
    "Upsample": ("#FCE7F3", "#DB2777", "#831843"),
    "Concat": ("#EDE9FE", "#7C3AED", "#4C1D95"),
    "Detect": ("#DCFCE7", "#15803D", "#14532D"),
    "Depth": ("#E0F2FE", "#0284C7", "#0C4A6E"),
    "ROI": ("#F3E8FF", "#9333EA", "#581C87"),
}


def _block(ax, x, y, w, h, title, sub="", kind="Conv", layer_num=None, dashed=False, lw=1.6):
    fc, ec, tc = COLORS[kind]
    ls = "--" if dashed else "-"
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
    )
    ax.add_patch(patch)
    if sub:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=7.6, color="#1E293B")
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=9.2, fontweight="bold", color=tc)
    if layer_num is not None:
        ax.text(x + w + 0.14, y + h / 2, str(layer_num), ha="left", va="center",
                fontsize=8.5, fontweight="bold", color="#0F172A")


def _arrow(ax, x1, y1, x2, y2, label="", lx=None, ly=None, color="#334155", lw=1.4):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=11,
        color=color, linewidth=lw,
    )
    ax.add_patch(arr)
    if label:
        tx = lx if lx is not None else (x1 + x2) / 2
        ty = ly if ly is not None else (y1 + y2) / 2
        ax.text(tx, ty, label, ha="center", va="center", fontsize=7.2,
                color="#0F172A", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def _ortho_arrow(ax, pts, label="", lx=None, ly=None, color="#334155", lw=1.3):
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw)
    x1, y1 = pts[-2]
    x2, y2 = pts[-1]
    _arrow(ax, x1, y1, x2, y2, color=color, lw=lw)
    if label and lx is not None and ly is not None:
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.2,
                color="#0F172A", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.88))


def draw_architecture(plus_attention: bool, out_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(16.5, 11.2), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 11.2)
    ax.axis("off")

    title_str = (
        "Arsitektur +Attention (Usulan): YOLO26n-Depth dengan C2PSA (L10) & C3k2 attn=True (L22)"
        if plus_attention
        else "Arsitektur Baseline (Tanpa Attention): YOLO26n-Depth (L10 Dihapus & L22 attn=False)"
    )
    ax.text(8.25, 10.92, title_str, ha="center", va="center", fontsize=13.5, fontweight="bold", color="#0F172A")
    ax.text(8.25, 10.62,
            "Diadaptasi dari Hidayatullah & Tubagus (2026) | Resolusi Masukan: 384×640 (evaluasi) / 480×640 (standar) | d=0,50, w=0,25, mc=1024",
            ha="center", va="center", fontsize=8.8, style="italic", color="#475569")

    # Legend bar
    lx = 0.4
    leg_items = [
        ("Conv", "Conv"),
        ("C3k2", "C3k2"),
        ("SPPF", "SPPF"),
        ("Upsample", "Upsample"),
        ("Concat", "Concat"),
        ("Detect", "Detection Head"),
        ("Depth", "Depth Head"),
    ]
    if plus_attention:
        leg_items.append(("C2PSA_HI", "Attention (C2PSA L10 / L22 attn=True)"))
    else:
        leg_items.append(("Removed", "Dihapus / attn=False"))

    for kind, lbl in leg_items:
        w_l = 3.15 if kind in ("C2PSA_HI", "Removed") else 1.40
        _block(ax, lx, 10.08, w_l, 0.36, lbl, kind=kind, dashed=(kind == "Removed"))
        lx += w_l + 0.14

    # Dashed Group Containers
    # Backbone
    ax.add_patch(FancyBboxPatch((0.35, 0.45), 3.65, 9.40, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F8FAFC", edgecolor="#16A34A", linestyle="--", linewidth=1.6))
    ax.text(2.15, 9.68, "Backbone (Layer 0–10)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#15803D")

    # Neck
    ax.add_patch(FancyBboxPatch((4.45, 0.45), 6.20, 9.40, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F8FAFC", edgecolor="#2563EB", linestyle="--", linewidth=1.6))
    ax.text(7.55, 9.68, "Neck FPN + PAN (Layer 11–22)", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1D4ED8")

    # Dual-Head + Post-Processing
    ax.add_patch(FancyBboxPatch((11.05, 0.45), 5.10, 9.40, boxstyle="round,pad=0.04,rounding_size=0.14",
                                facecolor="#F8FAFC", edgecolor="#DC2626", linestyle="--", linewidth=1.6))
    ax.text(13.60, 9.68, "Dual-Head & ROI Mapping", ha="center", va="center", fontsize=10.5, fontweight="bold", color="#B91C1C")

    # === BACKBONE (x=0.75, w=2.7) ===
    bx, bw, bh = 0.75, 2.70, 0.56
    ax.text(bx + bw / 2, 9.35, "Input: [1, 3, 384, 640]", ha="center", va="center", fontsize=8.3, fontweight="bold")
    _block(ax, bx, 8.55, bw, bh, "Conv", "k=3, s=2 | 3→16", "Conv", "0 (P1)")
    _arrow(ax, bx + bw / 2, 9.22, bx + bw / 2, 9.11)

    _block(ax, bx, 7.65, bw, bh, "Conv", "k=3, s=2 | 16→32", "Conv", "1 (P2)")
    _arrow(ax, bx + bw / 2, 8.55, bx + bw / 2, 8.21, "[1,16,192,320]")

    _block(ax, bx, 6.75, bw, bh, "C3k2", "n=1, c3k=False, e=0.25", "C3k2", "2")
    _arrow(ax, bx + bw / 2, 7.65, bx + bw / 2, 7.31, "[1,32,96,160]")

    _block(ax, bx, 5.85, bw, bh, "Conv", "k=3, s=2 | 64→64", "Conv", "3 (P3)")
    _arrow(ax, bx + bw / 2, 6.75, bx + bw / 2, 6.41, "[1,64,96,160]")

    _block(ax, bx, 4.95, bw, bh, "C3k2", "n=1, c3k=False, e=0.25", "C3k2", "4 (P3)")
    _arrow(ax, bx + bw / 2, 5.85, bx + bw / 2, 5.51, "[1,64,48,80]")

    _block(ax, bx, 4.05, bw, bh, "Conv", "k=3, s=2 | 128→128", "Conv", "5 (P4)")
    _arrow(ax, bx + bw / 2, 4.95, bx + bw / 2, 4.61, "[1,128,48,80]")

    _block(ax, bx, 3.15, bw, bh, "C3k2", "n=1, c3k=True, e=0.5", "C3k2", "6 (P4)")
    _arrow(ax, bx + bw / 2, 4.05, bx + bw / 2, 3.71, "[1,128,24,40]")

    _block(ax, bx, 2.25, bw, bh, "Conv", "k=3, s=2 | 128→256", "Conv", "7 (P5)")
    _arrow(ax, bx + bw / 2, 3.15, bx + bw / 2, 2.81, "[1,128,24,40]")

    _block(ax, bx, 1.45, bw, bh, "C3k2", "n=1, c3k=True, e=0.5", "C3k2", "8")
    _arrow(ax, bx + bw / 2, 2.25, bx + bw / 2, 2.01, "[1,256,12,20]")

    _block(ax, bx, 0.72, bw, 0.50, "SPPF", "k=5, repeats=3, shortcut", "SPPF", "9")
    _arrow(ax, bx + bw / 2, 1.45, bx + bw / 2, 1.22, "[1,256,12,20]")

    # L10 C2PSA placed at bottom-left of Neck/Backbone bridge (just like Figure 1!)
    if plus_attention:
        _block(ax, 4.65, 0.72, 2.35, 0.58, "C2PSA (Attention)", "n=1, e=0.5 | 256→256", "C2PSA_HI", "10 (P5)", lw=2.2)
    else:
        _block(ax, 4.65, 0.72, 2.35, 0.58, "L10: C2PSA DIHAPUS", "nn.Identity() (Bypass)", "Removed", "10 (P5)", dashed=True, lw=1.8)
    _arrow(ax, bx + bw, 0.97, 4.65, 1.01, "[1,256,12,20]", lx=4.05, ly=1.18)

    # === NECK FPN (col 1: x=4.65) ===
    nx1, nw = 4.65, 2.35
    _block(ax, nx1, 2.15, nw, 0.48, "Upsample", "nearest (ke P4)", "Upsample", "11")
    _arrow(ax, nx1 + nw / 2, 1.30, nx1 + nw / 2, 2.15, "[1,256,12,20]")

    _block(ax, nx1, 3.15, nw, 0.50, "Concat", "L11 + L6 → 384ch", "Concat", "12")
    _arrow(ax, nx1 + nw / 2, 2.63, nx1 + nw / 2, 3.15, "[1,256,24,40]")
    _ortho_arrow(ax, [(bx + bw, 3.43), (4.20, 3.43), (4.20, 3.40), (nx1, 3.40)],
                 "[1,128,24,40]", lx=4.20, ly=3.70, color="#DC2626")

    _block(ax, nx1, 4.15, nw, 0.56, "C3k2", "n=1, c3k=True, e=0.5", "C3k2", "13")
    _arrow(ax, nx1 + nw / 2, 3.65, nx1 + nw / 2, 4.15, "[1,384,24,40]")

    _block(ax, nx1, 5.20, nw, 0.48, "Upsample", "nearest (ke P3)", "Upsample", "14")
    _arrow(ax, nx1 + nw / 2, 4.71, nx1 + nw / 2, 5.20, "[1,128,24,40]")

    _block(ax, nx1, 6.20, nw, 0.50, "Concat", "L14 + L4 → 256ch", "Concat", "15")
    _arrow(ax, nx1 + nw / 2, 5.68, nx1 + nw / 2, 6.20, "[1,128,48,80]")
    _ortho_arrow(ax, [(bx + bw, 5.23), (4.15, 5.23), (4.15, 6.45), (nx1, 6.45)],
                 "[1,128,48,80]", lx=4.15, ly=5.85, color="#DC2626")

    _block(ax, nx1, 7.35, nw, 0.58, "C3k2 (P3_out)", "n=1, c3k=True, e=0.5", "C3k2", "16 (P3)")
    _arrow(ax, nx1 + nw / 2, 6.70, nx1 + nw / 2, 7.35, "[1,256,48,80]")

    # === NECK PAN (col 2: x=7.90) ===
    nx2 = 7.90
    _block(ax, nx2, 5.75, nw, 0.56, "Conv", "k=3, s=2 | 64→64", "Conv", "17")
    _ortho_arrow(ax, [(nx1 + nw, 7.50), (9.07, 7.50), (9.07, 6.31)],
                 "[1,64,48,80]", lx=8.35, ly=7.25)

    _block(ax, nx2, 4.15, nw, 0.50, "Concat", "L17 + L13 → 192ch", "Concat", "18")
    _arrow(ax, nx2 + nw / 2, 5.75, nx2 + nw / 2, 4.65, "[1,64,24,40]")
    _arrow(ax, nx1 + nw, 4.43, nx2, 4.40, "[1,128,24,40]", lx=7.45, ly=4.58)

    _block(ax, nx2, 3.05, nw, 0.56, "C3k2 (P4_out)", "n=1, c3k=True, e=0.5", "C3k2", "19 (P4)")
    _arrow(ax, nx2 + nw / 2, 4.15, nx2 + nw / 2, 3.61, "[1,192,24,40]")

    _block(ax, nx2, 2.05, nw, 0.54, "Conv", "k=3, s=2 | 128→128", "Conv", "20")
    _arrow(ax, nx2 + nw / 2, 3.05, nx2 + nw / 2, 2.59, "[1,128,24,40]")

    _block(ax, nx2, 1.32, nw, 0.48, "Concat", "L20 + L10 → 384ch", "Concat", "21")
    _arrow(ax, nx2 + nw / 2, 2.05, nx2 + nw / 2, 1.80, "[1,128,12,20]")
    _ortho_arrow(ax, [(nx1 + nw, 1.01), (7.45, 1.01), (7.45, 1.56), (nx2, 1.56)],
                 "[1,256,12,20]", lx=7.45, ly=0.80)

    if plus_attention:
        _block(ax, nx2, 0.55, nw, 0.58, "C3k2 (P5_out, attn=True)", "n=1, c3k=True, e=0.5, PSABlock", "Attn_C3k2", "22 (P5)", lw=2.2)
    else:
        _block(ax, nx2, 0.55, nw, 0.58, "C3k2 (P5_out, attn=False)", "n=1, c3k=True, e=0.5, Tanpa Attn", "Removed", "22 (P5)", dashed=True, lw=1.8)
    _arrow(ax, nx2 + nw / 2, 1.32, nx2 + nw / 2, 1.13, "[1,384,12,20]")

    # === DUAL-HEAD & ROI MAPPING (x=11.35) ===
    hx, hw = 11.35, 2.40
    _block(ax, hx, 7.35, hw, 0.68, "Detection Head (P3/8)", "DFL-free (reg + cls)\n[1,4,48,80] | 4 Kelas", "Detect")
    _block(ax, hx, 5.55, hw, 0.68, "Detection Head (P4/16)", "DFL-free (reg + cls)\n[1,4,24,40] | 4 Kelas", "Detect")
    _block(ax, hx, 3.85, hw, 0.68, "Detection Head (P5/32)", "DFL-free (reg + cls)\n[1,4,12,20] | 4 Kelas", "Detect")

    _arrow(ax, nx1 + nw, 7.68, hx, 7.68, "[1,64,48,80]", lx=10.75, ly=7.92, color="#15803D")
    _ortho_arrow(ax, [(nx2 + nw, 3.33), (10.85, 3.33), (10.85, 5.89), (hx, 5.89)],
                 "[1,128,24,40]", lx=10.85, ly=5.00, color="#15803D")
    _ortho_arrow(ax, [(nx2 + nw, 0.84), (10.95, 0.84), (10.95, 4.19), (hx, 4.19)],
                 "[1,256,12,20]", lx=10.95, ly=2.50, color="#15803D")

    _block(ax, hx, 1.55, hw, 1.15, "Depth Head (32ch × 3)",
           "Conv1x1 (P3,P4,P5 → 32ch)\nBilinear Upsample + Concat [1,96,H,W]\nRefine Conv3x3 + Sigmoid → [1,1,384,640]", "Depth")
    _ortho_arrow(ax, [(nx1 + nw, 7.45), (10.72, 7.45), (10.72, 2.35), (hx, 2.35)], color="#0284C7")
    _ortho_arrow(ax, [(nx2 + nw, 3.20), (10.78, 3.20), (10.78, 2.15), (hx, 2.15)], color="#0284C7")
    _ortho_arrow(ax, [(nx2 + nw, 0.72), (10.82, 0.72), (10.82, 1.95), (hx, 1.95)], color="#0284C7")

    # Post-processing & Output
    px, pw = 14.15, 1.80
    _block(ax, px, 5.35, pw, 1.10, "Class-Aware NMS", "IoU=0,45 (eval 0,7)\nBbox (x1,y1,x2,y2)", "ROI")
    _arrow(ax, hx + hw, 7.68, px, 6.10, color="#7C3AED")
    _arrow(ax, hx + hw, 5.89, px, 5.90, color="#7C3AED")
    _arrow(ax, hx + hw, 4.19, px, 5.65, color="#7C3AED")

    _block(ax, px, 2.85, pw, 1.35, "ROI Mapping 15×15", "EMA (α=0,35)\nMedian Patch 15×15\ndi Pusat Bbox", "ROI")
    _arrow(ax, px + pw / 2, 5.35, px + pw / 2, 4.20, color="#7C3AED")
    _arrow(ax, hx + hw, 2.12, px, 3.45, "[1,1,384,640]", lx=13.95, ly=2.55, color="#0284C7")

    _block(ax, px, 1.15, pw, 1.05, "Output Akhir", "Bbox + Label Kelas\nDepth Map INFERNO\nJarak Objek (meter)", "ROI")
    _arrow(ax, px + pw / 2, 2.85, px + pw / 2, 2.20, color="#7C3AED")

    # Parameter note at bottom
    note = (
        "Total Parameter +Attention: 3.863.209 (+333.952 atau +9,46% dari Baseline; L10 C2PSA = 249.728, L22 PSABlock = 84.224)"
        if plus_attention
        else "Total Parameter Baseline (Tanpa Attention): 3.529.257 (L10 C2PSA dihapus, L22 C3k2 attn=False; dimensi tensor L0–L22 tetap identik)"
    )
    ax.text(8.25, 0.16, note, ha="center", va="center", fontsize=8.8, fontweight="bold", color="#334155")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=300)
    fig.savefig(OUT_DIR / f"{out_stem}.svg")
    plt.close(fig)


if __name__ == "__main__":
    draw_architecture(plus_attention=False, out_stem="arsitektur_baseline_noattn")
    draw_architecture(plus_attention=True, out_stem="arsitektur_plus_attention")
    print("Generated arsitektur_baseline_noattn.{png,svg} and arsitektur_plus_attention.{png,svg}")
