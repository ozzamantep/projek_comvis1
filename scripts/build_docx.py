"""Build docs/Bab_Lengkap.docx from all markdown chapters in docs/laporan_bab/."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
BAB_DIR = ROOT / "docs" / "laporan_bab"
OUT_DOCX = ROOT / "docs" / "Bab_Lengkap.docx"

CHAPTER_FILES = [
    "bab1_3_2_detail_dataset.md",
    "bab3_x_arsitektur_usulan.md",
    "bab4_1_evaluasi_kuantitatif.md",
    "bab4_2_kualitatif.md",
    "bab4_3_per_kelas_dan_perbandingan.md",
    "bab4_4_screenshot.md",
    "bab4_5_pembahasan.md",
    "bab4_6_kelemahan.md",
    "bab4_7_kesimpulan_saran.md",
    "lampiran_eksperimen_cbam.md",
]


def _set_cell_shading(cell, fill_hex: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def _add_formatted_runs(paragraph, text: str) -> None:
    """Parse **bold**, *italic*, and `code` inline markdown into docx runs."""
    pattern = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos : m.start()])
        tok = m.group(0)
        if tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2])
            r.bold = True
        elif tok.startswith("*") and tok.endswith("*"):
            r = paragraph.add_run(tok[1:-1])
            r.italic = True
        elif tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _is_separator_row(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in cells if c.strip())


def _add_markdown_table(doc: Document, table_lines: list[str]) -> None:
    rows: list[list[str]] = []
    for line in table_lines:
        raw = [c.strip() for c in line.strip().strip("|").split("|")]
        if _is_separator_row(raw):
            continue
        rows.append(raw)
    if not rows:
        return
    num_cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=num_cols)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row in enumerate(rows):
        is_header = r_idx == 0
        is_proposed = any("+Attention" in c or "Full Usulan" in c or "Total" in c for c in row)
        for c_idx in range(num_cols):
            cell = table.cell(r_idx, c_idx)
            val = row[c_idx] if c_idx < len(row) else ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            _add_formatted_runs(p, val)
            for run in p.runs:
                run.font.size = Pt(9)
                if is_header:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
            if is_header:
                _set_cell_shading(cell, "1E3A8A")
            elif is_proposed:
                _set_cell_shading(cell, "EFF6FF")


def _render_markdown_file(doc: Document, md_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped or stripped == "---":
            i += 1
            continue

        # Table block
        if stripped.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            _add_markdown_table(doc, tbl_lines)
            doc.add_paragraph().paragraph_format.space_after = Pt(4)
            continue

        # Image: ![alt](relpath)
        img_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if img_match:
            rel = img_match.group(2)
            img_path = (md_path.parent / rel).resolve()
            if img_path.is_file():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.add_run().add_picture(str(img_path), width=Inches(6.0))
            else:
                p_miss = doc.add_paragraph(f"[Gambar tidak ditemukan: {rel}]")
                p_miss.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # Headings
        if stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=4)
            i += 1
            continue
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
            i += 1
            continue
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
            i += 1
            continue
        if stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            _add_formatted_runs(p, stripped[2:])
            i += 1
            continue

        # Bullet / numbered list
        if stripped.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            _add_formatted_runs(p, stripped[2:])
            i += 1
            continue
        num_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if num_match:
            p = doc.add_paragraph(style="List Number")
            _add_formatted_runs(p, num_match.group(2))
            i += 1
            continue

        # Normal paragraph (or centered caption)
        p = doc.add_paragraph()
        if stripped.startswith("*Gambar ") or stripped.startswith("*Sumber:") or stripped.startswith("**Tabel "):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if stripped.startswith("*Gambar ") else WD_ALIGN_PARAGRAPH.LEFT
        _add_formatted_runs(p, stripped)
        i += 1


def main() -> None:
    doc = Document()
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(11)

    title = doc.add_heading("Laporan Akhir Computer Vision — Draf Lengkap Sub-bab", level=1)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Topik: Analisis Mendalam Lightweight YOLO26n-Depth (Kelompok 1.1, Kelas C)")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for idx, fname in enumerate(CHAPTER_FILES):
        md_path = BAB_DIR / fname
        if md_path.is_file():
            if idx > 0:
                doc.add_page_break()
            _render_markdown_file(doc, md_path)

    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT_DOCX)
    print(f"Saved {OUT_DOCX} ({OUT_DOCX.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

