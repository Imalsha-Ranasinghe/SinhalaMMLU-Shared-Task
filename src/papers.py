"""Past exam papers (PDF) -> MCQs in the Dev Set JSON format.

Collection is manual. pastpapers.wiki forbids bots, scrapers and download tools (robots.txt and
terms of use), so team members download papers in a browser and drop them, with their original
file names, into one folder per subject:

    data/raw_pdfs/<Subject>/2023-Grade-09-History-3rd-Term-Test-Paper-with-Answers-Southern-Province.pdf

The folder name is the subject (use the SinhalaMMLU names, e.g. "History", "Eastern Music").
Year, grade (O/L = 11, A/L = 13), term and province are read from the file name. When the answers
are a separate PDF (A/L and O/L marking schemes), save it next to the paper as "<paper name>_answers.pdf".
Optional data/raw_pdfs/sources.csv (columns: file,url) records the page each file came from;
otherwise the file name is the source.

Each page is rendered as horizontal strips (small Sinhala print is unreadable at the vision model's
input size when a whole A4 page is one image) and read by GEN_MODEL (Gemma 4 by default), which
returns the paper's MCQs and its answer key. Only questions with an answer from the paper's own
answer key are kept: the model transcribes, it never decides the answer.

Vision-only reading of Sinhala misreads letters and even swaps words (a test paper came back with
"ප්‍රංශ" replaced by "පෘතුගීසි"). So when the PDF has a text layer (typed papers, including legacy
FM fonts), the model is given that text to copy from, and the question and its
options are replaced by the closest text in the text layer, or dropped when nothing in it is close
(a misread word or a swapped word fails the match). Scanned papers get more, narrower strips and no such check: review
a sample of them (metadata.text_layer = false) before training.

Papers whose file name matches a Dev Set source are skipped before any API call; the remaining
MCQs are de-duplicated and checked against the Dev Set (and any --decontam files).

Papers listed in a downloader manifest (data/pastpapers/grade_06/metadata.csv) are read with --manifest
instead: only term tests with answers in a Dev Set subject are kept (see manifest_papers).

Usage:
    python -m src.papers data/raw_pdfs --out data/papers [--limit 2]
    python -m src.papers --manifest data/pastpapers/grade_06/metadata.csv --out data/papers_g6 [--only REGEX]
"""

import argparse
import base64
import csv
import functools
import hashlib
import json
import re
import time
from datetime import date
from pathlib import Path, PureWindowsPath
from urllib.parse import parse_qs, urlsplit

from .data import load_raw
from .synth import ALL_OR_NONE, SINHALA, ZWJ, Cache, _norm, dedup_and_decontaminate, near_duplicates

PROVINCES = {  # longest first: "North Western" contains "Western"
    "north-western": "NWP", "north-central": "NCP", "western": "WP", "central": "CP",
    "southern": "SP", "northern": "NP", "eastern": "EP", "uva": "UP", "sabaragamuwa": "SGP",
}
TERMS = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3}
# SinhalaMMLU domains (paper, Table 2), in the Dev Set's spelling. The Dev Set files label History,
# Civics and Health as social_science.
CATEGORIES = {
    **dict.fromkeys(["arts", "buddhism", "catholicism", "christianity", "islam", "eastern music", "dancing",
                     "drama and theatre", "buddhist civilization", "oriental music", "history of sri lanka",
                     "dancing indigenous"], "humanities"),
    **dict.fromkeys(["history", "civics", "citizenship education", "health and physical science", "geography",
                     "political science"], "social_science"),
    **dict.fromkeys(["science", "physics", "chemistry", "biology"], "stem"),
    "sinhala language and literature": "language",
    **dict.fromkeys(["business and accounting studies", "entrepreneurship studies", "economics"], "business_studies"),
    **dict.fromkeys(["home economics", "biosystems technology", "communication and media studies",
                     "design and construction technology", "agriculture and food technology"], "other"),
}
ANSWERS_SUFFIX = "_answers"


def difficulty(grade):
    """SinhalaMMLU levels: grades 6-8 easy, 9-11 medium, 12-13 hard."""
    if grade is None:
        return None
    return "easy" if grade <= 8 else "medium" if grade <= 11 else "hard"

PROMPT = """These images are pages {first}-{last} of a Sri Lankan school exam paper ({paper}), Sinhala medium.
Each page is cut into {strips} horizontal strips that overlap slightly, shown top to bottom, page by page.
{text_layer}
Find every multiple-choice question (a question followed by numbered options to choose from, usually
in Part I / පළමු කොටස). Ignore structured, essay, matching and fill-in-the-blank questions.
If the pages include an answer key or marking scheme for the multiple-choice questions
(e.g. "1. (3)  2. (1) ..." or a table of question numbers and answers), read it as well.

Rules:
- Copy the Sinhala text exactly as printed, in Unicode Sinhala. Do not correct, translate or rephrase,
  and never replace a word you cannot read with a different word. {copy_rule}
- Drop the option labels such as (1), (i), 1., A. from the option text.
- A question that appears in two strips (the overlap) is listed once.
- "needs_figure" is true when the question cannot be answered without a picture, map, diagram or table
  on the paper.
- Do not answer the questions yourself. "answer_key" holds only answers printed on these pages, as
  question number -> option number (1 = first option). When the key gives the answer as words instead of
  a number, copy those words as a string. Leave it empty if there are none.

Return only JSON, with no other text:
{{"mcqs": [{{"no": <question number>, "question": "...", "choices": ["...", "...", "...", "..."], "needs_figure": false}}],
  "answer_key": {{"<question number>": <option number or "answer words">}}}}"""

TEXT_LAYER = """
The PDF's text layer for these pages is below. It has the exact characters, but its order and line breaks
may be jumbled. Take every question and option text from it, character for character; use the images to
see which text is a question, which are its options, and what the answer key says.

Text layer:
\"\"\"
{text}
\"\"\"
"""
COPY_RULE_TEXT = "Every question and option must be copied from the text layer."
COPY_RULE_SCAN = "If part of a question is unreadable, leave that question out."


def paper_info(path: Path, subject: str) -> dict:
    """Year, grade, term and province from a pastpapers.wiki style file name."""
    name = re.sub(r"[^a-z0-9]+", "-", path.stem.lower())
    year = re.search(r"(?<!\d)(20\d\d)(?!\d)", name)
    grade = re.search(r"grade-?(\d{1,2})(?!\d)", name)
    national = re.search(r"(?:^|-)(?:(o)-?l|(a)-?l|(o)-level|(a)-level|(o)rdinary-level|(a)dvanced-level)(?:-|$)", name)
    term = re.search(r"(1st|2nd|3rd|first|second|third)-term", name)
    province = next((code for p, code in PROVINCES.items() if f"{p}-province" in name), None)
    zone = re.search(r"([a-z]+)-(?:zonal|education-zone|zone)", name)
    term_n = TERMS[term.group(1)] if term else None
    if national and not grade:
        grade_n = 11 if next(g for g in national.groups() if g) == "o" else 13
    else:
        grade_n = int(grade.group(1)) if grade else None
    if national and not grade and not term and not province:
        kind = "OL" if grade_n == 11 else "AL"   # national exam (Department of Examinations)
    elif province:
        kind = f"P_{province}" + (f"_{term_n}" if term_n else "")
    elif zone:
        kind = f"Z_{zone.group(1).capitalize()}"
    else:
        kind = None
    return {"year": int(year.group(1)) if year else None, "grade": grade_n,
            "term": term_n, "type": kind, "subject": subject}


def page_count(pdf_path) -> int:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        return doc.page_count


def page_images(pdf_path, start, end, strips=2, dpi=150, overlap=0.04):
    """Pages start..end-1, each as `strips` JPEG strips (top to bottom) overlapping by `overlap` of the
    page height. The vision model shrinks every image to a fixed size, so more strips = larger print."""
    import pymupdf

    out = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc.pages(start, end):
            r = page.rect
            step = r.height / strips
            for k in range(strips):
                y0, y1 = max(0, k * step - r.height * overlap), min(r.height, (k + 1) * step + r.height * overlap)
                clip = pymupdf.Rect(r.x0, r.y0 + y0, r.x1, r.y0 + y1)
                out.append(page.get_pixmap(dpi=dpi, clip=clip).tobytes("jpg", jpg_quality=85))
    return out


def page_text(pdf_path, start, end) -> str:
    """Unicode text layer of pages start..end-1 (legacy FM fonts converted), or "" for a scan."""
    import pymupdf

    from .textbooks import page_blocks

    with pymupdf.open(pdf_path) as doc:
        pages = ["\n".join(t for t, _ in page_blocks(page)) for page in doc.pages(start, end)]
    text = "\n\n".join(f"--- page {start + i + 1} ---\n{t}" for i, t in enumerate(pages))
    # A scan has no text, or only a few header words typed over it. A broken font map gives junk
    # codepoints instead of shaped Sinhala (e.g. U+10xx for conjuncts): treat it as a scan too. So does
    # one that gives valid Sinhala letters in the wrong order ("ශ්රීද ස ක් ධ" for "ශරීර ස්කන්ධ"): too few of
    # its words are real words. Typed grade 6 papers score 0.62-0.88 against the Dev Set's words, garbled
    # ones 0.07-0.61.
    sinhala, junk = len(SINHALA.findall(text)), len(JUNK.findall(text))
    if sinhala < 100 * (end - start) or junk > 0.02 * sinhala:
        return ""
    words = [w for w in (w.replace(ZWJ, "") for w in WORD.findall(text)) if len(w) >= 2]
    vocab = dev_vocabulary()
    if vocab and words and sum(w in vocab for w in words) / len(words) < MIN_KNOWN_WORDS:
        return ""
    return text


WORD = re.compile(r"[\u0d80-\u0dff\u200d]+")
MIN_KNOWN_WORDS = 0.65


@functools.cache
def dev_vocabulary(dev_dir=Path(__file__).resolve().parent.parent / "Dev Set") -> frozenset:
    """Sinhala words of the Dev Set's questions and options (ZWJ removed), to tell real text from garbled."""
    words = set()
    for path in Path(dev_dir).glob("*.json"):
        for it in json.loads(path.read_text(encoding="utf-8")):
            for s in [it.get("question") or ""] + list(it.get("choices") or []):
                words.update(w.replace(ZWJ, "") for w in WORD.findall(str(s)))
    return frozenset(words)


JUNK = re.compile(r"[^\x00-\x7f\u0d80-\u0dff\u200b-\u200d\u2018-\u201d\u2022\u2013\u2014\u00a0-\u00ff]")


def parse_reply(raw: str) -> dict:
    """The JSON object in a model reply (tolerates ```json fences and text around it)."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


# "(1)", "1)", "(iv)", "a)", "1. " or "IV.ප්‍රසිද්ධ" (some papers print no space), but not "1.5 kg".
_LABEL = r"(?:[1-5]|i{1,3}|iv|v|[a-e])"
OPTION_LABEL = re.compile(rf"^\s*(?:\(\s*{_LABEL}\s*\)|{_LABEL}\s*\)|{_LABEL}\.(?!\d))\s*", re.I)
# The question's own number, which the text layer can glue to its first word ("16.ජලයෙහි").
QUESTION_NO = re.compile(r"^\s*\(?\d{1,2}\s*[.)](?!\d)\s*")


def qno(no) -> str:
    """Question numbers as the model may write them ("01", 1, "1.") -> "1"."""
    m = re.search(r"\d+", str(no))
    return str(int(m.group(0))) if m else str(no).strip()


def snap(s, layer, lo=0, hi=None):
    """The span of layer[lo:hi] that best matches s, widened to whole words: (score 0-100, start, end)."""
    from rapidfuzz import fuzz

    hi = len(layer) if hi is None else hi
    if not s or lo >= hi:
        return 0, lo, lo
    a = fuzz.partial_ratio_alignment(s, layer[lo:hi])
    if len(s) > hi - lo:  # rapidfuzz aligns the shorter string inside the longer one
        return 0, lo, lo
    start, end = lo + a.dest_start, lo + a.dest_end
    while start > lo and not layer[start - 1].isspace():
        start -= 1
    while end < hi and not layer[end].isspace():
        end += 1
    return fuzz.ratio(s, layer[start:end].strip()), start, end


def ground(q, choices, layer, q_min=90, option_min=85, window=800):
    """Replace the model's transcription with the text layer's own characters. The question is matched
    anywhere in the paper; each option after the previous one (options are printed in order), within
    `window` characters of the question. Returns (q, choices) or None when any part has no close match
    (the model misread or changed a word)."""
    score, q_start, pos = snap(q, layer)
    if score < q_min:
        return None
    q, limit, out = QUESTION_NO.sub("", layer[q_start:pos].strip()), min(len(layer), pos + window), []
    for c in choices:
        score, c_start, pos = snap(c, layer, pos, limit)
        if score < option_min:
            return None
        out.append(OPTION_LABEL.sub("", layer[c_start:pos].strip()).strip())
    return q, out


def answer_by_text(answer, choices, min_score=80, margin=10):
    """Option number (1-based) of the one choice that the answer key's words match, or None when no
    choice is close or two are about equally close."""
    from rapidfuzz import fuzz

    answer = OPTION_LABEL.sub("", answer).strip()
    scores = sorted(((fuzz.ratio(_norm(answer), _norm(c)), i + 1) for i, c in enumerate(choices)), reverse=True)
    if scores[0][0] < min_score or scores[0][0] - scores[1][0] < margin:
        return None
    return scores[0][1]


def check(item, answer, text_layer=""):
    """Return (clean_item, None) or (None, reason). With a text layer, the question and its options are
    replaced by the closest text in it, and dropped if there is none (see ground)."""
    q = str(item.get("question", "")).strip()
    choices = [OPTION_LABEL.sub("", str(c)).strip() for c in item.get("choices") or []]
    if not SINHALA.search(q):
        return None, "question not Sinhala"
    if item.get("needs_figure"):
        return None, "needs a figure"
    if len(choices) not in (4, 5) or any(not c for c in choices):
        return None, "not 4-5 options"
    if len({_norm(c) for c in choices}) < len(choices):
        return None, "duplicate options"
    if answer is None:
        return None, "no answer in the paper's key"
    if isinstance(answer, str):
        answer = answer_by_text(answer, choices)
        if answer is None:
            return None, "answer words match no single option"
    if not 1 <= answer <= len(choices):
        return None, "answer out of range"
    if any(ALL_OR_NONE.search(c) for c in choices):
        return None, "all/none of the above"
    if text_layer:
        grounded = ground(q, choices, re.sub(r"\s+", " ", text_layer.replace(ZWJ, "\0")).replace("\0", ZWJ))
        if not grounded:
            return None, "no close match in the text layer"
        q, choices = grounded
        if len({_norm(c) for c in choices}) < len(choices):
            return None, "duplicate options"
    return {"question": q, "choices": choices, "answer": answer}, None


def dev_source_slugs(dev_dir) -> set:
    """Keys of the papers the Dev Set was built from: slugs of page URLs and file names (query strings such
    as "?swcfpc=1" ignored), and "file-<category>-<id>" for pastpapers.wiki download links
    (admin-ajax.php?...&wpfd_category_id=7811&wpfd_file_id=43297 is the file at /download/7811/<..>/43297/)."""
    keys = set()
    for path in Path(dev_dir).glob("*.json"):
        for it in json.loads(path.read_text(encoding="utf-8")):
            url = urlsplit(((it.get("metadata") or {}).get("source") or "").strip())
            query = parse_qs(url.query)
            if "wpfd_file_id" in query:
                keys.add(f"file-{query.get('wpfd_category_id', ['?'])[0]}-{query['wpfd_file_id'][0]}")
                continue
            name = re.sub(r"%20\(\d+\)|\.pdf$", "", url.path.rstrip("/").split("/")[-1], flags=re.I)
            if name:
                keys.add(slug(name))
    return keys


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# Subject names used by the downloader and in paper titles -> SinhalaMMLU subject keys (see subject_key).
SUBJECT_ALIASES = {
    "art": "arts", "civic": "civics", "civic education": "civics", "citizenship education": "civics",
    "music": "eastern music", "drama": "drama and theatre", "health": "health and physical science",
    "health and physical education": "health and physical science", "catholic": "catholicism",
    "sinhala": "sinhala language and literature", "sinhala language": "sinhala language and literature",
    "sinhala literature": "sinhala language and literature",
}
TITLE_SUBJECT = re.compile(r"grade\s*0?\d{1,2}\W+(.+?)\W+(?:1st|2nd|3rd|first|second|third)\s+term", re.I)
NOT_A_TERM_TEST = re.compile(r"teachers?'?\s*guide|study\s*pack|module|workbook|tamil\s+medium", re.I)
# Answers published as a file of their own ("grade-6-art-answer.pdf"), not a paper "with answers".
ANSWERS_ONLY = re.compile(r"(?:^|(?<!with)-)answers?(?:-\d+)?$")


def subject_key(name: str) -> str:
    """"Eastern_music", "drama and Theatre " or a title's "Sinhala Language" -> one key per subject."""
    key = re.sub(r"[^a-z]+", " ", name.lower()).strip()
    return SUBJECT_ALIASES.get(key, key)


def dev_subjects(dev_dir) -> dict:
    """subject_key -> the Dev Set's most common spelling of the subject, its category and Sinhala name."""
    seen = {}
    for path in sorted(Path(dev_dir).glob("*.json")):
        for it in json.loads(path.read_text(encoding="utf-8")):
            md = it.get("metadata") or {}
            seen.setdefault(subject_key(it["subject"]), []).append(
                (it["subject"].strip(), it["category"].strip(), md.get("subject_original")))
    out = {}
    for key, rows in seen.items():
        most = lambda i: max({r[i] for r in rows}, key=[r[i] for r in rows].count)
        out[key] = {"subject": most(0), "category": most(1), "subject_original": most(2)}
    return out


def folder_papers(raw_dir, dev_slugs):
    """Papers in raw_dir/<Subject>/, as described at the top. Returns (papers, answer_files, sources,
    skipped): [(paper path, info)], {paper path: answers path}, {file name: url}, {file name: reason}."""
    sources = {}
    if (raw_dir / "sources.csv").exists():
        with open(raw_dir / "sources.csv", encoding="utf-8") as f:
            sources = {r["file"].strip(): r["url"].strip() for r in csv.DictReader(f)}
    papers, skipped = [], {}
    pdfs = sorted(raw_dir.rglob("*.pdf"))
    paper_of = lambda a: a.with_name(a.stem[:-len(ANSWERS_SUFFIX)] + ".pdf")
    answer_files = {paper_of(a): a for a in pdfs if a.stem.lower().endswith(ANSWERS_SUFFIX)}
    for p in pdfs:
        if p.stem.lower().endswith(ANSWERS_SUFFIX):
            if paper_of(p) not in pdfs:
                skipped[p.name] = "answers file without a paper of the same name"
            continue
        if p.parent == raw_dir:
            skipped[p.name] = "not in a subject folder"
            continue
        info = paper_info(p, p.relative_to(raw_dir).parts[0].replace("_", " "))
        if slug(re.sub(r" \(\d+\)$", "", p.stem)) in dev_slugs:
            skipped[p.name] = "Dev Set source"
        elif info["grade"] is None:
            skipped[p.name] = "no grade in file name"
        else:
            papers.append((p, info))
    return papers, answer_files, sources, skipped


def manifest_papers(manifest, dev_slugs, subjects):
    """Papers listed in a downloader manifest: a metadata.csv with columns grade, subject, term, year, type
    ("answer" = has answers), title, source_page, pdf_url, local_path, sha256, where local_path runs through
    the manifest's own folder (".../grade_06/Art/term_1/x.pdf" next to grade_06/metadata.csv).
    Keeps term tests with answers in a Dev Set subject (`subjects`, from dev_subjects), taking the subject
    from the title when it has one (the downloader files Christianity, Islam etc. under "Unknown").
    Returns the same (papers, answer_files, sources, skipped) as folder_papers."""
    manifest = Path(manifest)
    with open(manifest, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        parts = PureWindowsPath(r["local_path"]).parts
        if manifest.parent.name in parts:
            r["path"] = manifest.parent.joinpath(*parts[parts.index(manifest.parent.name) + 1:])
        else:
            r["path"] = next(manifest.parent.rglob(parts[-1]), manifest.parent / parts[-1])
        r["file"] = Path(urlsplit(r["pdf_url"]).path).stem or r["path"].stem
        download = re.search(r"//pastpapers\.wiki/download/(\d+)/[^/]+/(\d+)/", r["pdf_url"])
        r["keys"] = {slug(urlsplit(r["source_page"]).path.rstrip("/").split("/")[-1]), slug(r["file"])}
        if download:
            r["keys"].add(f"file-{download.group(1)}-{download.group(2)}")

    # A page offering a paper and its answers as two files: read them as one paper.
    pages = {}
    for r in rows:
        pages.setdefault(r["source_page"] or r["pdf_url"], []).append(r)
    papers, answer_files, sources, skipped, seen = [], {}, {}, {}, set()
    for group in pages.values():
        answers = [r for r in group if ANSWERS_ONLY.search(slug(r["file"]))]
        rest = [r for r in group if r not in answers]
        for a in answers:
            if len(rest) != 1:
                skipped[a["path"].name] = "answers file without its paper"
        in_dev = any(r["keys"] & dev_slugs for r in group)
        for r in rest:
            p, title = r["path"], r["title"]
            m = TITLE_SUBJECT.search(title)
            key = subject_key(m.group(1) if m else r["subject"])
            if not p.exists():
                why = "file missing"
            elif r["sha256"] in seen:
                why = "same file as another row"
            elif r["type"] != "answer" and not answers:
                why = "no answers"
            elif NOT_A_TERM_TEST.search(title) or re.search(r"(?:^|-)tm(?:-|$)", slug(r["file"])):
                why = "not a Sinhala-medium term test"
            elif key not in subjects:
                why = f"not a Dev Set subject ({key})"
            elif in_dev:
                why = "Dev Set source"
            else:
                why = None
            seen.add(r["sha256"])
            if why:
                skipped[p.name] = why
                continue
            info = paper_info(Path(slug(f"{title} {r['file']}") + ".pdf"), subjects[key]["subject"])
            info["grade"] = int(r["grade"]) if r["grade"] else info["grade"]
            info["year"] = int(r["year"]) if r["year"] else info["year"]
            if info["grade"] is None:
                skipped[p.name] = "no grade"
                continue
            papers.append((p, info))
            sources[p.name] = r["source_page"] or r["pdf_url"]
            if answers and len(rest) == 1:
                answer_files[p] = answers[0]["path"]
    return papers, answer_files, sources, skipped


def run(raw_dir, out_dir, backend, model_name, dev_dir, extra_decontam=(), pages_per_call=4,
        max_pages=40, limit=None, batch_size=32, strips=2, scan_strips=4, scan_pages_per_call=2,
        manifest=None, only=None):
    raw_dir, out_dir = Path(raw_dir) if raw_dir else None, Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_dir / "cache.jsonl")
    dev_slugs, subjects = dev_source_slugs(dev_dir), dev_subjects(dev_dir)
    if manifest:
        papers, answer_files, sources, skipped = manifest_papers(manifest, dev_slugs, subjects)
    else:
        papers, answer_files, sources, skipped = folder_papers(raw_dir, dev_slugs)
    if only:
        papers = [(p, info) for p, info in papers if re.search(only, p.name, re.I)]
    papers = papers[:limit]

    # One request per chunk of pages; the cache key carries the file hash and the prompt.
    tasks = []
    # A separate answers PDF is read like more pages of its paper: its chunks merge into the paper.
    for p, info in papers:
        if page_count(p) > max_pages:
            skipped[p.name] = f"{page_count(p)} pages > --max-pages {max_pages}"
            continue
        for f in [p] + ([answer_files[p]] if p in answer_files else []):
            digest = hashlib.sha1(f.read_bytes()).hexdigest()[:12]
            n_pages = min(page_count(f), max_pages)
            start = 0
            while start < n_pages:
                end = min(start + pages_per_call, n_pages)
                text = page_text(f, start, end)
                if not text and end - start > scan_pages_per_call:
                    # A scan is sent as more, narrower strips: 4 pages of them often fail with HTTP 500.
                    end = start + scan_pages_per_call
                    text = page_text(f, start, end)
                n_strips = strips if text else scan_strips
                prompt = PROMPT.format(
                    first=start + 1, last=end, paper=f.stem, strips=n_strips,
                    text_layer=TEXT_LAYER.format(text=text) if text else "",
                    copy_rule=COPY_RULE_TEXT if text else COPY_RULE_SCAN)
                key = f"{digest}-p{start + 1}-{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]}"
                tasks.append((key, p, info, prompt, (f, start, end, n_strips), text))
                start = end
    todo = [t for t in tasks if t[0] not in cache.data]
    print(f"{len(papers)} papers, {len(skipped)} skipped, {len(tasks)} model calls, {len(todo)} not cached yet")

    def message(prompt, pages):
        images = page_images(*pages)  # (file, start, end, strips)
        return [{"role": "user", "content": [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode()}}
            for im in images]}]

    # Render in batches so a few hundred papers' page images are never in memory at once.
    t0, done, failed = time.time(), 0, 0
    for b in range(0, len(todo), batch_size):
        batch = todo[b:b + batch_size]
        for i, reply in backend.generate_iter([message(prompt, pages) for _, _, _, prompt, pages, _ in batch]):
            done += 1
            if reply and parse_reply(reply):
                cache.add(batch[i][0], reply)
            else:
                failed += 1
            print(f"  {done}/{len(todo)} done, {failed} failed, {time.time() - t0:.0f}s  ({batch[i][1].name})")
        if getattr(backend, "daily_limit_hit", False):
            break
    if getattr(backend, "daily_limit_hit", False):
        print("Stopped early at the daily limit: rerun the same command after it resets to continue.")

    # Merge chunks per paper: the answer key is often on the last pages, the questions on the first.
    # Each question keeps the text layer its chunk was read with, and is grounded in that.
    per_paper = {}
    for key, p, info, _, _, text in tasks:
        obj = parse_reply(cache.data.get(key))
        rec = per_paper.setdefault(p, {"info": info, "mcqs": {}, "key": {}, "missing_chunks": 0, "text_layer": False})
        rec["text_layer"] |= bool(text)
        if not obj:
            rec["missing_chunks"] += 1
        for m in obj.get("mcqs") or []:
            if isinstance(m, dict):
                rec["mcqs"].setdefault(qno(m.get("no")), (m, text))
        for no, ans in (obj.get("answer_key") or {}).items():
            try:
                rec["key"][qno(no)] = int(str(ans).strip("() "))
            except ValueError:
                if SINHALA.search(str(ans)):  # answer given as words: matched to an option in check()
                    rec["key"][qno(no)] = str(ans).strip()

    mcqs, reasons, per_file = [], {}, {}
    for p, rec in per_paper.items():
        kept = 0
        for no, (m, text) in rec["mcqs"].items():
            clean, why = check(m, rec["key"].get(no), text)
            if not clean:
                reasons[why] = reasons.get(why, 0) + 1
                continue
            info = rec["info"]
            known = subjects.get(subject_key(info["subject"])) or {}
            mcqs.append({**clean, "subject": known.get("subject", info["subject"]), "metadata": {
                "subject_original": known.get("subject_original"), "difficulty": difficulty(info["grade"]), "grade": info["grade"], "type": info["type"],
                "year": info["year"], "source": sources.get(p.name, p.name), "paper_q_no": no,
                "extracted_by": model_name, "text_layer": bool(text)}})
            kept += 1
        per_file[p.name] = {"answers_file": p in answer_files, "text_layer": rec["text_layer"], "mcqs_found": len(rec["mcqs"]), "answers_in_key": len(rec["key"]),
                            "kept": kept, "missing_chunks": rec["missing_chunks"]}

    full_text = lambda m: m["question"] + " | " + " | ".join(m.get("choices") or [])
    dev = load_raw(dev_dir).to_dict(orient="records")
    for path in extra_decontam:
        dev += (json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else
                [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])
    dev = [r for r in dev if r.get("question")]
    raw_count = len(mcqs)
    mcqs, dropped = dedup_and_decontaminate(mcqs, [r["question"] for r in dev], dup_text=full_text)
    # Misread short questions can slip under the question-only threshold; their options give them away.
    hits, _ = near_duplicates([full_text(m) for m in mcqs], [full_text(r) for r in dev], 0.75)
    mcqs = [m for i, m in enumerate(mcqs) if i not in hits]
    dropped["too_close_to_dev_with_options"] = len(hits)

    # One Dev Set style JSON file per subject, so src.data.load_raw(out_dir / "json") reads them.
    by_subject = {}
    for m in mcqs:
        by_subject.setdefault(m["subject"], []).append(m)
    (out_dir / "json").mkdir(exist_ok=True)
    for old in (out_dir / "json").glob("*_papers.json"):  # a subject left with no MCQs must not keep last run's
        old.unlink()
    for subject, rows in by_subject.items():
        category = (subjects.get(subject_key(subject)) or {}).get("category") or CATEGORIES.get(subject.lower(), "other")
        items = [{"q_no": k + 1, "subject": subject, "category": category, "question": m["question"],
                  "choices": m["choices"], "answer": m["answer"], "metadata": m["metadata"]}
                 for k, m in enumerate(rows)]
        (out_dir / "json" / f"{subject.replace(' ', '_')}_papers.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    grades = {}
    for m in mcqs:
        k = f"{m['subject']} g{m['metadata']['grade']}"
        grades[k] = grades.get(k, 0) + 1
    # Real papers need no length filter (src.synth's is for generated MCQs), but report the shortcut anyway.
    longest = sum(max(range(len(m["choices"])), key=lambda i: len(m["choices"][i])) + 1 == m["answer"] for m in mcqs)
    positions = {str(k): sum(m["answer"] == k for m in mcqs) for k in range(1, 6)}
    report = {
        "date": str(date.today()), "extractor_model": model_name, "papers": len(papers), "skipped": skipped,
        "mcq_parsed": raw_count, "mcq_rejected": reasons, "mcq_dropped": dropped, "mcq_final": len(mcqs),
        "by_subject_grade": dict(sorted(grades.items())), "answer_position": positions,
        "answer_is_longest_option": round(longest / max(len(mcqs), 1), 3), "per_file": per_file,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, ensure_ascii=False, indent=2))
    return mcqs, report


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from .backends import OpenAICompatBackend

    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", help="folder with one sub-folder of PDFs per subject")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="read the papers from a downloader metadata.csv instead of raw_dir (see manifest_papers)")
    ap.add_argument("--only", default=None, help="only papers whose file name matches this regex")
    ap.add_argument("--out", default="data/papers")
    ap.add_argument("--limit", type=int, default=None, help="only the first N papers (cheap trial run)")
    ap.add_argument("--pages-per-call", type=int, default=4)
    ap.add_argument("--max-pages", type=int, default=40, help="skip longer PDFs (modules, books)")
    ap.add_argument("--strips", type=int, default=2, help="image strips per page when the PDF has a text layer")
    ap.add_argument("--scan-strips", type=int, default=4, help="image strips per page for scanned PDFs")
    ap.add_argument("--scan-pages-per-call", type=int, default=2, help="pages per request for scanned PDFs")
    ap.add_argument("--decontam", nargs="*", type=Path, default=[],
                    help="extra .json/.jsonl files with a 'question' field to keep out (e.g. public SinhalaMMLU)")
    ap.add_argument("--rpm", type=int, default=None, help="max requests per minute (default 10 on free tiers)")
    a = ap.parse_args()
    if not (a.raw_dir or a.manifest):
        ap.error("give raw_dir or --manifest")

    load_dotenv()
    model = os.environ.get("GEN_MODEL") or os.environ["MODEL_NAME"]
    base_url = os.environ.get("GEN_BASE_URL") or os.environ["OPENAI_BASE_URL"]
    google = "generativelanguage.googleapis.com" in base_url
    api_key = (os.environ.get("GEN_API_KEY") or (os.environ.get("GEMINI_API_KEY") if google else None)
               or os.environ["OPENAI_API_KEY"])
    free = model.endswith(":free") or google
    # Temperature 0: this is transcription. max_tokens covers Gemma 4's thinking plus ~40 MCQs.
    backend = OpenAICompatBackend(model, base_url, api_key, max_tokens=24000, temperature=0.0,
                                  max_workers=8, max_retries=6, timeout=600,
                                  requests_per_minute=a.rpm or (10 if free else None))
    print(f"extractor: {model} at {base_url}")
    run(a.raw_dir, a.out, backend, model, Path(__file__).resolve().parent.parent / "Dev Set",
        extra_decontam=a.decontam, pages_per_call=a.pages_per_call, max_pages=a.max_pages, limit=a.limit,
        strips=a.strips, scan_strips=a.scan_strips,
        scan_pages_per_call=a.scan_pages_per_call, manifest=a.manifest, only=a.only)
