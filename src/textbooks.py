"""Extract clean Unicode Sinhala text from textbook PDFs.

Government textbooks (Educational Publications Department) are typeset in legacy
"FM" fonts (FM Abhaya, FM Derana, FM Ganganee, ...). Their text layer is ASCII that
only looks like Sinhala through the font, e.g. "l%sia;shdks O¾uh" = "ක්‍රිස්තියානි ධර්මය".
All FM fonts share one key layout, so every FM span is converted with the FM Abhaya
mapping; spans in other fonts (Times New Roman digits, English) are kept as they are.

Usage:
    python -m src.textbooks data/textbooks/christianity_g8_si.pdf
"""

import functools
import json
import re
import sys
from collections import Counter
from pathlib import Path

# pymupdf and pandukabhaya are imported only when a PDF is converted, so load_pages() and
# passages() work on machines (e.g. Colab) that only read the extracted .jsonl.

# Glyph codes the FM Abhaya mapping misses, rewritten to codes it knows.
# Verified on the grade 8 Christianity book: uq`M = මුළු, ¥úf,a, = දූවිල්ල, 3(1¡13 = 3:1-13.
_BASE_FIXES = {"`M": "¿", "¥": "oQ", "¡": "-"}

# In FM fonts "`" before a letter gives its prenasalised form: fyd`Èka = හොඳින්, u`. = මඟ.
_PRENASAL = {"ද": "ඳ", "ග": "ඟ", "ඩ": "ඬ", "ඞ": "ඬ", "ජ": "ඦ", "බ": "ඹ"}


def _prenasal_fixes(conv) -> dict:
    """Map "`" + code to the single FM code of the prenasalised letter, from the converter's table."""
    single = {chr(c): conv.convert(chr(c)) for c in range(0x21, 0x100)}
    by_output = {out: ch for ch, out in single.items() if out}
    fixes = {}
    for ch, out in single.items():
        if out and out[0] in _PRENASAL and ch != "`":
            target = _PRENASAL[out[0]] + out[1:]
            if target in by_output:
                fixes["`" + ch] = by_output[target]
    return fixes


@functools.lru_cache(maxsize=None)
def _fm():
    """The FM Abhaya converter and all glyph fixes, built on first use."""
    from pandukabhaya import Converter

    conv = Converter("fm_abhaya")
    return conv, {**_BASE_FIXES, **_prenasal_fixes(conv)}


HEADING_SIZE = 20          # chapter titles are 24 pt, body text 12 pt, quotes 14 pt
PAGE_NUMBER = re.compile(r"^\s*(\d{1,3}|[ivxlc]{1,6})\s*$", re.I)
SINHALA = re.compile(r"[඀-෿]")


def is_legacy_font(font: str) -> bool:
    return font.split("+")[-1].upper().startswith("FM")


def fm_to_unicode(text: str) -> str:
    conv, fixes = _fm()
    for old, new in fixes.items():
        text = text.replace(old, new)
    text = conv.convert(text).replace("`", "")   # stray "`" before an already-prenasal letter
    return _DETACHED_E.sub(_reattach_e, text)


# The converter sometimes leaves FM's leading "ෙ" in front of its consonant
# (ෙදාළොස් for දොළොස්, පරිච්ෙඡ්ද for පරිච්ඡේද). Move it behind and merge with ා / ්.
_DETACHED_E = re.compile(r"(?<![ක-ෆ])ෙ([ක-ෆ](?:්‍[යර])?)([ා්]?)")


def _reattach_e(m) -> str:
    return m.group(1) + {"ා": "ො", "්": "ේ", "": "ෙ"}[m.group(2)]


def _line_text(spans) -> str:
    """Convert runs of consecutive FM spans together: a style change can split one word
    across spans, and FM puts the vowel sign "ෙ" before its consonant."""
    out, run = [], []
    for s in spans:
        if is_legacy_font(s["font"]):
            run.append(s["text"])
            continue
        if run:
            out.append(fm_to_unicode("".join(run)))
            run = []
        out.append(s["text"])
    if run:
        out.append(fm_to_unicode("".join(run)))
    return "".join(out)


def _clean(text: str) -> str:
    text = re.sub(r"[​‌﻿]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def page_blocks(page):
    """Yield (text, max_font_size) for each text block on the page, in reading order."""
    for b in page.get_text("dict", sort=True)["blocks"]:
        spans = [s for line in b.get("lines", []) for s in line["spans"] if s["text"].strip()]
        if not spans:
            continue
        lines = [_line_text(line["spans"]) for line in b["lines"]]
        text = _clean(" ".join(lines))
        if text:
            yield text, max(s["size"] for s in spans)


def first_content_page(doc) -> int:
    """0-based index of the first page numbered 1 (front matter uses roman numerals)."""
    for i, page in enumerate(doc):
        if any(text == "1" for text, _ in page_blocks(page)):
            return i
    return 0


def extract(pdf_path, start_page=None, repeat_ratio=0.3):
    """Return one record per content page: {"pdf_page", "chapter", "text"}.

    Drops front matter (before start_page), page numbers, and lines that repeat on more
    than `repeat_ratio` of pages (running headers/footers).
    """
    import pymupdf

    doc = pymupdf.open(pdf_path)
    start = first_content_page(doc) if start_page is None else start_page - 1
    pages = [(i, list(page_blocks(doc[i]))) for i in range(start, doc.page_count)]

    counts = Counter(t for _, blocks in pages for t in {t for t, _ in blocks})
    repeated = {t for t, n in counts.items() if n > repeat_ratio * len(pages)}

    records, chapter = [], None
    for i, blocks in pages:
        paras = []
        for text, size in blocks:
            if PAGE_NUMBER.match(text) or text in repeated:
                continue
            if size >= HEADING_SIZE:
                if not SINHALA.search(text):        # decorative glyphs set at heading size
                    continue
                chapter = re.sub(r"^\d+\s*|\s*\d+$", "", text)   # drop the chapter number
                paras.append(f"# {chapter}")
            else:
                paras.append(text)
        if paras:
            records.append({"pdf_page": i + 1, "chapter": chapter, "text": "\n".join(paras)})
    return records


def load_pages(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def passages(records, max_chars=1500):
    """Group page text into passages of whole paragraphs that never cross a chapter.

    Returns [{"chapter", "first_page", "text"}]; each is a unit for paraphrasing or MCQ writing.
    """
    out, buf = [], []
    state = {"chapter": None, "page": None}

    def flush():
        if buf:
            out.append({"chapter": state["chapter"], "first_page": state["page"], "text": "\n".join(buf)})
            buf.clear()

    for r in records:
        for para in r["text"].split("\n"):
            if para.startswith("# ") or r["chapter"] != state["chapter"]:
                flush()
                state["chapter"] = r["chapter"]
                if para.startswith("# "):
                    continue
            if buf and sum(len(p) + 1 for p in buf) + len(para) > max_chars:
                flush()
            if not buf:
                state["page"] = r["pdf_page"]
            buf.append(para)
    flush()
    return out


def legacy_residue(records) -> Counter:
    """Latin letters left in the output: a sign of unconverted or mis-converted text."""
    return Counter(ch for r in records for ch in re.findall(r"[A-Za-z]", r["text"]))


def save(records, out_stem):
    out_stem = Path(out_stem)
    with open(out_stem.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_stem.with_suffix(".txt").write_text("\n\n".join(r["text"] for r in records), encoding="utf-8")


if __name__ == "__main__":
    pdf = Path(sys.argv[1])
    recs = extract(pdf)
    save(recs, pdf.with_suffix(""))
    chars = sum(len(r["text"]) for r in recs)
    print(f"{pdf.name}: {len(recs)} pages, {chars:,} characters, "
          f"{len({r['chapter'] for r in recs})} chapters -> {pdf.with_suffix('.jsonl')}")
    print("Latin letters left:", legacy_residue(recs).most_common(10))
