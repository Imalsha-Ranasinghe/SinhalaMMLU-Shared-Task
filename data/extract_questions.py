"""Extract Sinhala MCQs (question, 4 options, answer) from term test paper PDFs.

Works on PDFs whose text is typed in FM-family legacy Sinhala fonts (FM Abhaya etc.): the text is
converted to Unicode Sinhala, the MCQ section is parsed, and the answer key is read from the
marking scheme when the PDF includes one. Scanned PDFs (no usable text) are listed for OCR.

Usage (from the repo root, after data/collect_papers.py):
    python data/extract_questions.py                         # all PDFs in data/raw_papers
    python data/extract_questions.py --limit 20              # try on a few first
    python data/extract_questions.py --pdf some_paper.pdf    # one file, prints what it found

Output in data/extracted/:
    questions_grade_10.json   questions with an answer, same format as the Dev Set files
    no_answer_grade_10.json   questions whose answer key couldn't be found (not for training)
    report.csv                one row per PDF: status, #questions, #answers
    needs_ocr.csv             scanned PDFs (no text layer, or an unreadable OCR layer)
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from pandukabhaya import Converter

logging.getLogger("pymupdf").setLevel(logging.ERROR)
pymupdf.TOOLS.mupdf_display_errors(False)

_fm = Converter("fm_abhaya")

CATEGORY = {
    "Science": "stem", "Agriculture and Food Technology": "stem", "Design and Construction Technology": "stem",
    "Sinhala Language and Literature": "language",
    "Citizenship Education": "social_science", "Geography": "social_science",
    "Health and Physical Science": "social_science", "Business and Accounting Studies": "social_science",
    "Entrepreneurship Studies": "social_science", "Home Economics": "social_science",
    "Communication and Media Studies": "social_science",
    # A/L subjects (grades 12-13)
    "Physics": "stem", "Chemistry": "stem", "Biology": "stem", "Biosystems Technology": "stem",
    "Economics": "social_science", "Political Science": "social_science",
}  # everything else (religions, arts, history, music, dancing, drama, Buddhist Civilization) -> humanities

ENGLISH_WORDS = {"the", "and", "of", "to", "is", "in", "for", "are", "which", "what", "answer", "question",
                 "following", "paper", "write", "from", "with", "this", "that", "by", "on", "an", "be"}

# Lines that are page furniture, not question text.
HEADER_RE = re.compile(r"ශ්‍රේණිය|කාලය|පැය\s*\d|දෙපාර්තමේන්තුව|පරීක්‍ෂණය|පරීක්ෂණය|past\s*papers|wiki|"
                       r"සියලු\s*හිමිකම්|all\s*rights|විභාග\s*අංකය|නම\s*[:/]|term\s*test|grade\s*\d", re.I)
# Where the MCQ part ends (Part II / answer sheet).
END_MCQ_RE = re.compile(r"(^|\s)(II|ii|2)\s*(වන\s*)?කොටස|කොටස\s*[-–:]?\s*(II|ii)\b|දෙවන\s*කොටස|"
                        r"part\s*-?\s*ii\b|පිළිතුරු\s*පත්‍රය|marking\s*scheme|ලකුණු\s*දීමේ", re.I)
ANSWER_START_RE = re.compile(r"පිළිතුරු|marking\s*scheme|ලකුණු\s*දීමේ|answer|(^|\s)I\s*කොටස", re.I)
# Questions that depend on a figure/map/table or a passage above them, which we can't extract.
NEEDS_IMAGE_RE = re.compile(r"රූප|සිතියම|ප්‍රස්තාර|රූපසටහන|වගුව|ඡායාරූප|දළ\s*සටහන|පහත\s*දැක්වෙන\s*සටහන|"
                            r"පහත\s*ආකාර|ඉහත(?!\s*(ප්‍රකාශ|වගන්ති))|තීරුව")   # "above statements" are inline

Q_START_RE = re.compile(r"^\s*(\()?(0)?(\d{1,2})(\))?\s*([.)'\-:])?(\s+|$)")
OPT_RE = re.compile(r"(?:(?<=\s)|^)\((\d)\)|(?:(?<=\s)|^)(\d)[.)](?=\s|$)")
ROMAN_OPT_RE = re.compile(r"(?:(?<=\s)|^)\((i{1,3}|iv|v)\)|(?:(?<=\s)|^)(i{1,3}|iv|v)[.)]?(?=\s|$)")
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}


def option_count(lines, opt_re):
    """4 or 5: A/L papers (grades 12-13) usually have five options per question.

    Decided per paper by how often an option-5 marker appears compared with option 4, so that in a
    4-option paper a "5." (question 5) is never mistaken for an option.
    """
    counts = Counter()
    for l in lines:
        for m in opt_re.finditer(l.text):
            g = m.group(1) or m.group(2)
            counts[ROMAN.get(g) or int(g)] += 1
    return 5 if counts[4] >= 8 and counts[5] >= 0.6 * counts[4] else 4


def option_style(lines):
    """Some papers number options (i)-(iv) instead of (1)-(4); use whichever the paper uses most."""
    roman = sum(bool(re.match(r"\s*\(?(i{1,3}|iv)[.)]?(\s|$)", l.text)) for l in lines)
    digits = sum(bool(re.match(r"\s*\(?[1-5][.)](\s|$)|\s*\([1-5]\)", l.text)) for l in lines)
    return "roman" if roman > digits else "digits"


def font_name(span):
    return span["font"].split("+")[-1].lower()


def looks_fm_encoded(text):
    """For fonts embedded under generic names (e.g. "CIDFont+F2"): is this FM-encoded Sinhala?

    Yes if it isn't English, isn't already Unicode Sinhala, and converts into clean Sinhala.
    """
    latin = re.findall(r"[A-Za-z]{2,}", text.lower())
    sinhala_now = sum("඀" <= c <= "෿" for c in text)
    if len(text.strip()) < 40 or sinhala_now > 0.2 * max(len(latin), 1):
        return False
    if latin and sum(w in ENGLISH_WORDS for w in latin) / len(latin) > 0.08:
        return False                       # real English
    conv = _fm.convert(text)
    sin = sum("඀" <= c <= "෿" for c in conv)
    letters = sin + sum(c.isascii() and c.isalpha() for c in conv)
    return letters > 0 and sin / letters >= 0.6 and broken_sinhala_ratio(conv) < 0.03


def fm_fonts(doc):
    """Fonts whose text must be converted: FM-named fonts, plus generically named ones that hold FM text."""
    texts = defaultdict(list)
    for page in doc:
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    texts[font_name(s)].append(s["text"])
    return {f for f, t in texts.items() if f.startswith("fm") or looks_fm_encoded(" ".join(t))}


# Where an option's real text ends: marks notes, page markers, lead-ins for the next questions.
OPTION_JUNK_RE = re.compile(r"\(\s*ලකුණු|\s-\s?\d{1,2}\s?-(\s|$)|(පහත|ඉහත)[^.?]{0,120}?ප්‍රශ්න\s*(අංක)?\s*\d|"
                            r"ප්‍රශ්න\s*අංක\s*\d+\s*(සිට|හා|-)|(I|II|11)\s*පත්‍රය")


def clean_option(text):
    m = OPTION_JUNK_RE.search(text)
    return norm(text[:m.start()] if m and m.start() > 0 else text)


def norm(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace("•", " ")
    return re.sub(r"\s+", " ", text).strip(" ,;")


# ---------------------------------------------------------------- reading PDFs

@dataclass
class Line:
    text: str
    page: int
    bbox: tuple


def read_pdf(path):
    """All text lines (converted to Unicode) plus per-page numeric tokens for answer tables."""
    doc = pymupdf.open(path)
    convert = fm_fonts(doc)
    lines, fm_chars, other_chars = [], 0, 0
    for pno, page in enumerate(doc):
        h = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                parts = []
                for s in l["spans"]:
                    if font_name(s) in convert:
                        fm_chars += len(s["text"].strip())
                        parts.append(_fm.convert(s["text"]))
                    else:
                        other_chars += len(s["text"].strip())
                        parts.append(s["text"])
                text = "".join(parts)
                y0, y1 = l["bbox"][1], l["bbox"][3]
                in_margin = y1 < 0.07 * h or y0 > 0.93 * h
                if not text.strip() or HEADER_RE.search(text) or (in_margin and re.fullmatch(r"\s*-?\s*\d{1,2}\s*-?\s*", text)):
                    continue
                lines.append(Line(text, pno, tuple(l["bbox"])))
    return lines, fm_chars, other_chars


_DEP = re.compile(r"[ංඃ්ා-ෟ]")     # signs that must follow a letter
_BASE = re.compile(r"[අ-ඖක-ෆ‍]")


def broken_sinhala_ratio(text):
    """Share of vowel signs that don't follow a letter. Clean text: < 1%; broken font tables: 9-25%."""
    bad = total = 0
    for i, ch in enumerate(text):
        if _DEP.match(ch):
            total += 1
            prev = text[i - 1] if i else " "
            bad += not (_BASE.match(prev) or _DEP.match(prev))
    return bad / total if total else 0.0


def classify(lines, fm_chars, other_chars):
    text = " ".join(l.text for l in lines)
    sinhala = sum("඀" <= c <= "෿" for c in text)
    if sinhala >= 300 and broken_sinhala_ratio(text) > 0.03:
        return "needs_ocr"                 # text layer exists but its Sinhala is garbled
    words = re.findall(r"[A-Za-z]+", text.lower())
    english = sum(w in ENGLISH_WORDS for w in words)
    if sinhala >= 300:
        return "sinhala_text"
    if words and english / max(len(words), 1) > 0.08 and english > 30:
        return "english_medium"
    return "needs_ocr"


# ---------------------------------------------------------------- MCQ parsing

@dataclass
class Q:
    n: int
    question: str = ""
    options: list = field(default_factory=list)
    line_idx: int = 0
    extra_lines: int = 0          # continuation lines added to the current field
    n_opts: int = 4               # options per question in this paper (4, or 5 in A/L papers)

    @property
    def next_opt(self):
        return len(self.options) + 1

    def add(self, text):
        text = text.strip()
        if not text:
            return
        if self.options:
            self.options[-1] += " " + text
        else:
            self.question += " " + text

    def complete(self):
        return len(self.options) == self.n_opts


def parse_mcqs(lines):
    questions, cur, expect = [], None, 1
    opt_re = ROMAN_OPT_RE if option_style(lines) == "roman" else OPT_RE
    n_opts = option_count(lines, opt_re)

    def finish():
        nonlocal cur, expect
        if cur is None:
            return
        opts = [clean_option(o) for o in cur.options]
        qtext = norm(cur.question)
        # An option holding a full sentence and then more text has swallowed the next question.
        merged = any(re.search(r"[.?]\s+\S.{30,}", o) for o in opts)
        if cur.complete() and len(qtext) >= 3 and not merged and all(0 < len(o) <= 250 for o in opts):
            questions.append({"n": cur.n, "question": qtext, "choices": opts, "line_idx": cur.line_idx})
        else:
            expect = cur.n            # malformed: allow this number to start again
        cur = None

    for idx, line in enumerate(lines):
        text = line.text
        if questions and len(text.strip()) <= 60 and END_MCQ_RE.search(text):   # headings are short
            finish()
            break
        m = Q_START_RE.match(text)
        if m:
            n = int(m.group(3))
            zero, paren, punct = m.group(2), m.group(1) or m.group(4), m.group(5)
            rest = text[m.end():]
            marked = bool(zero or paren or punct or not rest.strip())
            could_be_option = cur is not None and not cur.complete() and n == cur.next_opt and not zero
            # e.g. a stray page number "1" started question 1, then the real "01." line arrives
            restart = cur is not None and n == cur.n and not cur.question.strip() and not cur.options
            # A missing number (e.g. question 20 lost in a figure) shouldn't stop the parse.
            skipped = expect < n <= expect + 2 and (zero or n >= 10) and marked
            if ((n == expect or skipped) and marked and not could_be_option) or (restart and marked):
                if restart:
                    cur = None
                finish()
                cur, expect = Q(n=n, line_idx=idx, n_opts=n_opts), n + 1
                text = rest
        if cur is None:
            continue

        # Options can share a line: "1. A   2. B"
        pos, found_marker = 0, False
        for om in opt_re.finditer(text):
            g = om.group(1) or om.group(2)
            val = ROMAN[g] if g in ROMAN else int(g)
            if val != cur.next_opt:
                continue
            cur.add(text[pos:om.start()])
            cur.options.append("")
            cur.extra_lines = 0
            pos, found_marker = om.end(), True
        tail = text[pos:]
        if found_marker:
            cur.add(tail)
        elif tail.strip():
            # A wrapped line continues the current field, but not forever (stops trailing junk).
            cur.extra_lines += 1
            if cur.extra_lines <= 2 or not cur.options:
                cur.add(tail)
    finish()
    return questions if len(questions) >= 5 else []


# ---------------------------------------------------------------- answer keys

def answers_from_pairs(lines, n, max_opt=4):
    """Explicitly numbered answers: "(01) 2", "1. 3", "12 - 4" ... in any order (tables get read across)."""
    text = " ".join(l.text for l in lines)
    answers = {}
    for m in re.finditer(r"(?<![\d.])\(?0?(\d{1,2})\)?\s*[.)\-:–]?\s*\(?([1-%d])\)?(?![\d.])" % max_opt, text):
        q, a = int(m.group(1)), int(m.group(2))
        if 1 <= q <= n:
            answers.setdefault(q, a)
    return answers


def _cluster(toks, key, tol):
    """Group tokens whose `key` coordinate is within `tol` points (same page)."""
    groups, cur = [], []
    for t in sorted(toks, key=lambda t: (t["page"], t[key])):
        if cur and (t["page"] != cur[-1]["page"] or t[key] - cur[-1][key] > tol):
            groups.append(cur)
            cur = []
        cur.append(t)
    if cur:
        groups.append(cur)
    return groups


def answers_from_table(lines, n, max_opt=4):
    """Answer tables in either direction, matched by position on the page:
    across:  1 2 3 ... 10        down:  1 - (2)
             2 1 4 ...  3               2 - (1)
    A header is a run of >= 5 consecutive question numbers; answers sit in the next row below
    (across) or the nearest column to the right (down).
    """
    # Numbers may be written "12", "12." or "( 3 )" (brackets split off as separate tokens).
    toks = []
    for l in lines:
        for t in l.text.split():
            m = re.fullmatch(r"\(?(\d{1,2})[.)]?", t)
            if m:
                toks.append({"page": l.page, "x": (l.bbox[0] + l.bbox[2]) / 2,
                             "y": (l.bbox[1] + l.bbox[3]) / 2, "v": int(m.group(1))})
    answers = {}
    for along, across in (("x", "y"), ("y", "x")):        # rows (sorted by x), then columns (sorted by y)
        groups = [sorted(g, key=lambda t: t[along]) for g in _cluster(toks, across, 4)]
        groups.sort(key=lambda g: (g[0]["page"], sum(t[across] for t in g) / len(g)))
        for i, head in enumerate(groups):
            nums = [t["v"] for t in head]
            if len(nums) < 5 or nums != list(range(nums[0], nums[0] + len(nums))):
                continue
            for other in groups[i + 1:i + 4]:              # the next row below / column to the right
                if other[0]["page"] != head[0]["page"] or len(other) < len(head) // 2:
                    continue
                if not all(1 <= t["v"] <= max_opt for t in other):
                    continue
                for t in other:
                    h = min(head, key=lambda h: abs(h[along] - t[along]))
                    if abs(h[along] - t[along]) <= 6 and 1 <= h["v"] <= n:
                        answers.setdefault(h["v"], t["v"])
                break
    return answers


def answers_from_sequence(lines, n, max_opt=4):
    """A dense block of single digits 1-4 (or 1-5), one per question in order (no question numbers)."""
    best, block = [], []
    for l in lines:
        t = l.text.strip().lower()
        m = re.fullmatch(r"\(?\s*([1-%d])\s*\)?" % max_opt, t)      # "3" or "( 3 )"
        if m:
            block.append(int(m.group(1)))
        elif t in ROMAN and ROMAN[t] <= max_opt:
            block.append(ROMAN[t])
        elif re.fullmatch(r"[-–()\s.]*", t):
            continue                       # dashes/brackets between answers don't break the block
        else:
            best, block = max(best, block, key=len), []
    best = max(best, block, key=len)
    return {i + 1: a for i, a in enumerate(best[:n])} if len(best) >= n else {}


def parse_answers(lines, after_idx, n, max_opt=4):
    """Look after each answer-sheet heading; the first method covering >= 90% of questions wins.

    Explicit numbering is trusted first, a bare digit sequence last, because a sequence can't
    tell a table's question-number row or interleaved columns from real answers.
    """
    tail = lines[after_idx:]
    starts = [i for i, l in enumerate(tail) if len(l.text.strip()) <= 60 and ANSWER_START_RE.search(l.text)]
    starts.append(-1)                      # no heading: the key may follow the last question directly
    for name, fn in (("pairs", answers_from_pairs), ("table", answers_from_table),
                     ("sequence", answers_from_sequence)):
        for start in starts:
            got = fn(tail[start + 1:start + 1 + 5 * n], n, max_opt)
            if len(got) >= 0.9 * n:
                return got, name
    return {}, "none"


# ---------------------------------------------------------------- per PDF

def key_only(pdf_path):
    """Answer key from a PDF that has no questions (answers published as a separate file)."""
    lines, _, _ = read_pdf(pdf_path)
    for n in (50, 40, 35, 30, 25, 20):
        # max_opt=5: the key may belong to a 5-option (A/L) paper; checked against it when paired
        got, how = parse_answers([Line("පිළිතුරු", 0, (0, 0, 0, 0))] + lines, 0, n, max_opt=5)
        if got and max(got) >= 0.9 * n:
            return got
    return {}


def extract(pdf_path, meta):
    lines, fm, other = read_pdf(pdf_path)
    kind = classify(lines, fm, other)
    if kind == "english_medium":
        return kind, [], "none"
    if kind != "sinhala_text":
        # Answer-key files are often Unicode/Latin digits only, so they look "unreadable" here.
        return kind, [], "none"
    qs = parse_mcqs(lines)
    if not qs:
        return "no_mcq_found", [], "none"
    n_opts = len(qs[0]["choices"])
    answers, how = parse_answers(lines, qs[-1]["line_idx"] + 1, max(q["n"] for q in qs), max_opt=n_opts)
    subject = meta.get("subject", "Unknown")
    out = []
    for q in qs:
        out.append({
            "q_no": q["n"],
            "subject": subject,
            "category": CATEGORY.get(subject, "humanities"),
            "question": q["question"],
            "choices": q["choices"],
            "answer": answers.get(q["n"]),
            "metadata": {
                "difficulty": None,
                "grade": int(meta["grade"]) if meta.get("grade") else None,
                "type": "extracted_pdf",
                "term": meta.get("term") or None,
                "year": int(meta["year"]) if meta.get("year") else None,
                "region": meta.get("region") or None,
                "source": meta.get("source_page") or meta.get("pdf_url"),
                "pdf": meta.get("local_path"),
                "n_options": len(q["choices"]),
                "needs_context": bool(NEEDS_IMAGE_RE.search(q["question"])),   # figure/table/passage not included
            },
        })
    return "ok", out, how


def to_four(q):
    """Turn a 5-option question into a 4-option one by removing one wrong option (chosen the same
    way every run). Without a known answer we can't tell which options are wrong, so it is left as is."""
    if len(q["choices"]) != 5 or not q["answer"]:
        return q
    wrong = [i for i in range(5) if i != q["answer"] - 1]
    drop = wrong[int(hashlib.md5(q["question"].encode("utf-8")).hexdigest(), 16) % 4]
    choices = [c for i, c in enumerate(q["choices"]) if i != drop]
    answer = q["answer"] - (1 if drop < q["answer"] - 1 else 0)
    return {**q, "choices": choices, "answer": answer,
            "metadata": {**q["metadata"], "n_options": 4, "original_n_options": 5}}


def qkey(q):
    return norm(q["question"]) + "||" + "|".join(norm(c) for c in q["choices"])


def load_dev_keys(dev_dir):
    keys = set()
    for p in Path(dev_dir).glob("*.json"):
        for q in json.load(open(p, encoding="utf-8")):
            keys.add(qkey(q))
    return keys


def load_papers(papers_dir):
    """PDFs listed in metadata.csv, plus any PDF on disk that isn't listed (e.g. the collector was
    stopped before writing metadata). Unlisted files get subject/grade/term from their folder names:
    grade_10/<Subject>/term_2/<file>.pdf
    """
    papers = []
    meta_file = papers_dir / "metadata.csv"
    if meta_file.exists():
        with open(meta_file, encoding="utf-8-sig") as f:
            papers = [r for r in csv.DictReader(f) if (papers_dir / r["local_path"]).exists()]
    listed = {r["local_path"] for r in papers}
    for pdf in sorted(papers_dir.glob("grade_*/*/*/*.pdf")):
        rel = pdf.relative_to(papers_dir).as_posix()
        if rel in listed:
            continue
        grade_dir, subject, term_dir = pdf.relative_to(papers_dir).parts[:3]
        year = re.findall(r"20[0-3]\d", pdf.stem)
        papers.append({"grade": grade_dir.split("_")[-1], "subject": subject, "local_path": rel,
                       "term": term_dir.split("_")[-1] if term_dir[-1].isdigit() else "",
                       "year": max(year) if year else "", "region": "", "source_page": "", "pdf_url": ""})
    return papers


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--papers", type=Path, default=here / "raw_papers", help="output folder of collect_papers.py")
    ap.add_argument("--out", type=Path, default=here / "extracted")
    ap.add_argument("--dev", type=Path, default=here.parent / "Dev Set", help="drop questions that are in the Dev Set")
    ap.add_argument("--limit", type=int, help="only process the first N PDFs")
    ap.add_argument("--pdf", type=Path, help="process a single PDF and print the questions")
    ap.add_argument("--to-four", action="store_true",
                    help="convert 5-option (A/L) questions to 4 options by removing one wrong option")
    args = ap.parse_args()

    if args.pdf:
        status, qs, how = extract(args.pdf, {"subject": "Unknown"})
        print(f"status={status}  questions={len(qs)}  answers={sum(q['answer'] is not None for q in qs)} (via {how})\n")
        for q in qs:
            print(f"Q{q['q_no']}: {q['question']}")
            for i, c in enumerate(q["choices"], 1):
                mark = " <==" if q["answer"] == i else ""
                print(f"    ({i}) {c}{mark}")
        return

    papers = load_papers(args.papers)
    if not papers:
        raise SystemExit(f"No PDFs found in {args.papers}. Run data/collect_papers.py first "
                         "(it downloads into data/raw_papers/grade_XX/<Subject>/...).")
    print(f"{len(papers)} PDFs in {args.papers}\n")
    if args.limit:
        papers = papers[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    dev_keys = load_dev_keys(args.dev) if args.dev.exists() else set()

    report, needs_ocr, with_ans, no_ans, seen = [], [], defaultdict(list), defaultdict(list), set()
    dropped_dev = dropped_dup = 0
    results, keys_by_page = [], defaultdict(list)
    for i, meta in enumerate(papers, 1):
        pdf = args.papers / meta["local_path"]
        try:
            status, qs, how = extract(pdf, meta)
            if not qs and status != "english_medium":
                key = key_only(pdf)
                if key and meta.get("source_page"):    # pairing needs the page the PDF came from
                    keys_by_page[meta["source_page"]].append(key)
                    status = "answer_key_only"
        except Exception as e:
            status, qs, how = f"error: {type(e).__name__}", [], "none"
        results.append((meta, status, qs, how))
        n_ans = sum(q["answer"] is not None for q in qs)
        print(f"[{i}/{len(papers)}] {status:15s} q={len(qs):3d} answers={n_ans:3d} ({how}) {meta['local_path']}")

    # Question papers without a key: use a separate answer-key PDF from the same web page.
    for meta, status, qs, how in results:
        if qs and all(q["answer"] is None for q in qs):
            nums = {q["q_no"] for q in qs}
            for key in keys_by_page.get(meta.get("source_page"), []):
                if len(nums & set(key)) >= 0.9 * len(nums):
                    for q in qs:
                        a = key.get(q["q_no"])
                        q["answer"] = a if a and a <= len(q["choices"]) else None
                    print(f"  matched separate answer key -> {meta['local_path']}")
                    break

    for meta, status, qs, how in results:
        n_ans = sum(q["answer"] is not None for q in qs)
        report.append({"pdf": meta["local_path"], "subject": meta["subject"], "status": status,
                       "questions": len(qs), "answers": n_ans, "answer_source": how})
        if status in ("needs_ocr", "no_mcq_found"):
            needs_ocr.append({"pdf": meta["local_path"], "subject": meta["subject"], "status": status,
                              "source": meta.get("source_page")})
        for q in qs:
            if args.to_four:
                q = to_four(q)
            k = qkey(q)
            if k in dev_keys:
                dropped_dev += 1
                continue
            if k in seen:                      # same question in another paper
                dropped_dup += 1
                continue
            seen.add(k)
            (with_ans if q["answer"] else no_ans)[q["metadata"]["grade"]].append(q)

    for grade in sorted(set(with_ans) | set(no_ans)):
        json.dump(with_ans[grade], open(args.out / f"questions_grade_{grade}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump(no_ans[grade], open(args.out / f"no_answer_grade_{grade}.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    for name, rows in (("report.csv", report), ("needs_ocr.csv", needs_ocr)):
        if rows:
            with open(args.out / name, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    statuses = Counter(r["status"] for r in report)
    total_ans = sum(len(v) for v in with_ans.values())
    total_no = sum(len(v) for v in no_ans.values())
    print("\n" + "=" * 60)
    print("PDFs:", dict(statuses))
    print(f"Questions with answers: {total_ans}   without answers: {total_no}")
    print(f"Dropped: {dropped_dev} already in the Dev Set, {dropped_dup} repeated across papers")
    by_subj = Counter(q["subject"] for v in with_ans.values() for q in v)
    for s, c in by_subj.most_common():
        print(f"  {c:5d}  {s}")
    print(f"\nOutput: {args.out}")


if __name__ == "__main__":
    main()
