"""Download Sinhala-medium term test papers (PDF) from pastpapers.wiki for one grade.

Usage (from the repo root):
    python data/collect_papers.py                      # grade 10, all subjects below
    python data/collect_papers.py --dry-run            # only list what would be downloaded
    python data/collect_papers.py --subjects "History,Geography"
    python data/collect_papers.py --grade 11

Output: data/raw_papers/grade_10/<Subject>/term_<n>/<file>.pdf  plus metadata.csv / errors.csv.
Re-running skips files that are already downloaded.

Before saving, each PDF is checked on the server (data/pdf_precheck.py): only papers with
readable Sinhala MCQs (or an answer key) are saved. Scans that would need OCR are skipped and
listed in skipped_unusable.csv. Use --keep-all to download everything.
"""

import argparse
import csv
import hashlib
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://pastpapers.wiki"
REQUEST_DELAY = 1.5       # seconds between requests; be polite to the site
TIMEOUT = 30
MAX_RETRIES = 3

# Your subject name -> the names pastpapers.wiki uses in its URLs ("grade-10-<name>-1st-term-test-paper-...").
# Several subjects are named differently on the site (e.g. Citizenship Education is "civic-education").
# A/L-only subjects (Physics, Economics, ...) are listed too; the summary shows which ones exist for the grade.
SUBJECTS = {
    "History": ["history"],
    "Drama and Theatre": ["drama", "drama-and-theatre"],
    "Dancing": ["dancing", "dance"],
    "Eastern Music": ["eastern-music", "oriental-music"],   # the site uses both names for the same subject
    "Arts": ["art", "arts"],
    "Buddhism": ["buddhism"],
    "Catholicism": ["catholicism", "catholic"],
    "Christianity": ["christianity"],
    "Islam": ["islam", "islamic"],
    "Buddhist Civilization": ["buddhist-civilization", "buddhist-civilisation"],
    "History of Sri Lanka": ["history-of-sri-lanka"],
    "Dancing Indigenous": ["dancing-indigenous", "indigenous-dancing", "indigenous-dance"],
    "Citizenship Education": ["civic-education", "citizenship-education", "civics", "civic"],
    "Health and Physical Science": ["health-and-physical-education", "health-and-physical-science",
                                    "health-physical-education", "health"],
    "Geography": ["geography"],
    "Political Science": ["political-science"],
    "Physics": ["physics"],
    "Chemistry": ["chemistry"],
    "Biology": ["biology"],
    "Science": ["science"],
    "Sinhala Language and Literature": ["sinhala-language", "sinhala-literature",
                                        "sinhala-language-and-literature", "sinhala"],
    "Business and Accounting Studies": ["business-studies", "business-and-accounting-studies",
                                        "business-accounting-studies", "business-and-accounting",
                                        "accounting"],
    "Entrepreneurship Studies": ["entrepreneurship-studies", "entrepreneurship"],
    "Economics": ["economics"],
    "Home Economics": ["home-science", "home-economics"],
    "Biosystems Technology": ["biosystems-technology", "bio-systems-technology"],
    "Communication and Media Studies": ["communication-and-media-studies", "media-studies", "media"],
    "Design and Construction Technology": ["design-and-construction-technology", "construction-technology"],
    "Agriculture and Food Technology": ["agriculture-and-food-technology", "agriculture"],
}
# Extra names accepted by --subjects
SUBJECT_ALIASES = {"oriental music": "Eastern Music"}

# Subjects we never want. Knowing them lets the crawler skip their pages, and stops e.g.
# "art-and-craft" from being read as "art" (the longest matching name wins).
UNWANTED = [
    "mathematics", "maths", "math", "english", "english-literature", "english-language", "tamil",
    "second-language-tamil", "second-language-sinhala", "sinhala-second-language", "tamil-language",
    "ict", "information-and-communication-technology", "art-and-craft", "western-music",
    "design-and-mechanical-technology", "design-electrical-and-electronic-technology",
    "electronic-technology", "electrical-and-electronic-technology", "mechanical-technology",
    "aquatic-bio-resources-technology", "bio-resources-technology", "french", "german", "hindi",
    "japanese", "korean", "chinese", "arabic", "pali", "sanskrit", "hinduism", "saivaneri",
]

EXCLUDED_PATH_PARTS = ["tamil-medium", "english-medium", "textbook", "/wp-admin", "/wp-json",
                       "/feed", "/tag/", "/author/", "/download/", "comment-page", "/amp"]
SKIP_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip", ".rar", ".pdf")
# Only exam papers: lesson notes, unit notes, textbooks etc. are skipped.
PAPER_WORDS = ["paper", "exam", "term-test", "-term-"]
NOT_PAPER_WORDS = ["note", "lesson", "unit-", "chapter", "workbook", "textbook", "syllabus",
                   "teacher", "guide", "activity", "පාඩම", "සටහන"]   # Sinhala: lesson, note

REGIONS = ["north-western", "north-central", "western", "southern", "central", "uva",
           "sabaragamuwa", "eastern", "northern"]

session = requests.Session()
session.headers.update({
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "Chrome/155.0 Safari/537.36 EducationalResearchCollector/1.0")
})


# ---------------------------------------------------------------- helpers

def page_url(url):
    """Normalise a page URL: no query string or fragment (the site adds tracking params)."""
    p = urlparse(url)
    return p._replace(query="", fragment="").geturl()


def is_same_domain(url):
    return urlparse(url).netloc.lower() in ("pastpapers.wiki", "www.pastpapers.wiki")


def clean_filename(text):
    return re.sub(r"\s+", " ", re.sub(r'[<>:"/\\|?*]', "_", text)).strip()[:150]


def get(url, stream=False):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(REQUEST_DELAY)
            r = session.get(url, timeout=TIMEOUT, stream=stream)
            r.raise_for_status()
            return r
        except Exception as e:
            print(f"  [retry {attempt}/{MAX_RETRIES}] {url}: {e}")
            time.sleep(attempt * 2)
    return None


def last_slug(url):
    return unquote(urlparse(url).path.rstrip("/").split("/")[-1]).lower()


def is_not_paper(slug):
    return any(w in slug for w in NOT_PAPER_WORDS)


def is_paper(pdf_slug, page_slug):
    """A PDF counts as an exam paper if it or its page says so, and neither looks like notes."""
    if is_not_paper(pdf_slug) or is_not_paper(page_slug):
        return False
    return any(w in pdf_slug or w in page_slug for w in PAPER_WORDS)


class SubjectMatcher:
    """Finds the subject named anywhere in a URL slug, e.g. both
    'grade-10-history-1st-term-test-paper-...' and 'southern-province-2024-grade-10-history-3rd-term-...'.
    """

    def __init__(self, wanted):
        self.to_subject = {name: s for s in wanted for name in SUBJECTS[s]}
        everything = {name for names in SUBJECTS.values() for name in names} | set(UNWANTED)
        self.names = sorted(everything, key=len, reverse=True)   # longest first

    def site_name(self, slug):
        # "sinhala-medium" is in nearly every URL and is not the Sinhala subject.
        text = "-" + re.sub(r"sinhala-?medium", "-", slug.lower()) + "-"
        for name in self.names:
            if f"-{name}-" in text:
                return name
        return None

    def __call__(self, slug):
        """(subject or None, named): named=True means some subject, wanted or not, is in the slug."""
        name = self.site_name(slug)
        return self.to_subject.get(name), name is not None


def parse_details(slug, site_name):
    rest = slug.replace(site_name, " ") if site_name else slug   # so "eastern-music" is not read as Eastern Province
    t = re.search(r"(1st|2nd|3rd|first|second|third)-term", slug)
    term = {"1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3}
    years = [int(y) for y in re.findall(r"20[0-3]\d", slug)]  # also splits "20232024"
    region = next((r for r in REGIONS if re.search(rf"(^|-){r}(-|$)", rest)), "")
    school = re.search(r"([a-z]+-(?:college|vidyalaya|school))", rest)
    return {
        "term": term[t.group(1)] if t else None,
        "year": max(years) if years else None,
        "region": (region.replace("-", " ").title() + " Province") if region else
                  (school.group(1).replace("-", " ").title() if school else ""),
        "has_answers": "answer" in slug,
    }


# ---------------------------------------------------------------- crawl

def crawl(grade, matcher, max_pages=None):
    start = f"{BASE_URL}/grade-{grade:02d}-sinhala-medium-term-test-papers-past-papers-wiki/"
    grade_re = re.compile(rf"grade-?0?{grade}(?!\d)")
    queue, queued, visited = [start], {start}, set()
    records, seen_pdfs = [], set()

    while queue and (max_pages is None or len(visited) < max_pages):
        url = queue.pop(0)
        visited.add(url)
        print(f"[page {len(visited)}, {len(queue)} queued] {url}")

        r = get(url)
        if r is None:
            log_error(url, "failed to fetch page")
            continue
        if "text/html" not in r.headers.get("Content-Type", "").lower():
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        page_slug = last_slug(url)
        page_subject, _ = matcher(page_slug)

        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"].strip())
            if not urlparse(href).path.lower().endswith(".pdf") or href in seen_pdfs:
                continue
            pdf_slug = unquote(Path(urlparse(href).path).stem).lower()
            pdf_subject, pdf_named = matcher(pdf_slug)
            # Trust the PDF's own name when it names a subject; otherwise use the page's subject.
            subject = pdf_subject if pdf_named else page_subject
            if subject is None or not grade_re.search(pdf_slug + " " + page_slug):
                continue
            if any(x in pdf_slug for x in ("tamil-medium", "english-medium")):
                continue
            if not is_paper(pdf_slug, page_slug):
                continue
            seen_pdfs.add(href)
            details_slug = pdf_slug if pdf_named else page_slug
            records.append({"grade": grade, "subject": subject, "pdf_url": href, "source_page": url,
                            **parse_details(details_slug, matcher.site_name(details_slug))})
            print(f"  [PDF] {subject}: {href}")

        # Follow pages for this grade; skip pages about subjects we don't want.
        for a in soup.find_all("a", href=True):
            link = page_url(urljoin(url, a["href"].strip()))
            path = urlparse(link).path.lower()
            if (not is_same_domain(link) or link in queued or path.endswith(SKIP_EXTENSIONS)
                    or not grade_re.search(path) or any(x in path for x in EXCLUDED_PATH_PARTS)):
                continue
            slug = last_slug(link)
            subject, named = matcher(slug)
            if (named and subject is None) or is_not_paper(slug):
                continue
            queue.append(link)
            queued.add(link)

    return records


# ---------------------------------------------------------------- download

def target_path(rec, out_dir):
    term = f"term_{rec['term']}" if rec["term"] else "term_unknown"
    directory = out_dir / f"grade_{rec['grade']:02d}" / clean_filename(rec["subject"]) / term
    stem = clean_filename(Path(urlparse(rec["pdf_url"]).path).stem)
    old = directory / (stem + ".pdf")
    if old.exists() or len(stem) <= 90:
        return old
    # Windows paths must stay under 260 characters: shorten, with a URL hash to keep names unique.
    return directory / f"{stem[:90]}-{hashlib.sha1(rec['pdf_url'].encode()).hexdigest()[:8]}.pdf"


def download(rec, out_dir):
    path = target_path(rec, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 1000:
        print(f"[skip] {path.name}")
        return path

    r = get(rec["pdf_url"], stream=True)
    if r is None:
        log_error(rec["pdf_url"], "download failed")
        return None
    tmp = path.with_suffix(".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(64 * 1024):
            f.write(chunk)
    with open(tmp, "rb") as f:
        if f.read(5) != b"%PDF-":
            tmp.unlink(missing_ok=True)
            log_error(rec["pdf_url"], "not a valid PDF")
            return None
    tmp.replace(path)
    print(f"[saved] {path}")
    return path


# ---------------------------------------------------------------- csv

METADATA_FIELDS = ["grade", "subject", "term", "year", "region", "has_answers",
                   "source_page", "pdf_url", "local_path", "sha256"]
ERROR_FILE = None


def log_error(url, error):
    new = not ERROR_FILE.exists()
    with open(ERROR_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["url", "error"])
        w.writerow([url, error])


def write_metadata(rows, path):
    """Merge with any existing metadata.csv, one row per PDF URL."""
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            existing = {r["pdf_url"]: r for r in csv.DictReader(f)}
    for r in rows:
        existing[r["pdf_url"]] = r
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        w.writeheader()
        w.writerows(existing.values())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- main

def main():
    global ERROR_FILE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grade", type=int, default=10)
    ap.add_argument("--subjects", help="comma-separated subset of the SUBJECTS names (default: all)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "raw_papers")
    ap.add_argument("--dry-run", action="store_true", help="list PDFs without downloading")
    ap.add_argument("--max-pages", type=int, help="stop crawling after this many pages (for testing)")
    ap.add_argument("--recrawl", action="store_true", help="search the site again instead of reusing the saved list")
    ap.add_argument("--keep-all", action="store_true",
                    help="download every PDF, including scans that need OCR (default: check first, save only readable papers)")
    args = ap.parse_args()

    wanted = list(SUBJECTS)
    if args.subjects:
        lookup = {**{s.lower(): s for s in SUBJECTS}, **SUBJECT_ALIASES}
        wanted = []
        for name in args.subjects.split(","):
            if name.strip().lower() not in lookup:
                raise SystemExit(f"Unknown subject '{name.strip()}'. Choose from: {', '.join(SUBJECTS)}")
            wanted.append(lookup[name.strip().lower()])

    args.out.mkdir(parents=True, exist_ok=True)
    ERROR_FILE = args.out / "errors.csv"
    print(f"Grade {args.grade}, {len(wanted)} subjects -> {args.out}\n")

    # The crawl takes ~25 min, so its result is saved and reused by later runs.
    found_file = args.out / f"found_grade_{args.grade}.csv"
    if found_file.exists() and not args.recrawl and not args.max_pages:
        with open(found_file, encoding="utf-8-sig") as f:
            records = [{**r, "grade": int(r["grade"]), "term": int(r["term"]) if r["term"] else None,
                        "year": int(r["year"]) if r["year"] else None, "has_answers": r["has_answers"] == "True"}
                       for r in csv.DictReader(f) if r["subject"] in wanted]
        print(f"Using the saved list of PDFs in {found_file.name} (add --recrawl to search the site again)")
    else:
        records = crawl(args.grade, SubjectMatcher(wanted), args.max_pages)
        if records and not args.max_pages and not args.subjects:   # only a full crawl is worth reusing
            with open(found_file, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(records[0]))
                w.writeheader()
                w.writerows(records)
    print(f"\nFound {len(records)} PDFs\n")

    rows, duplicates = [], []
    if not args.dry_run:
        # Same paper uploaded under different links: identical bytes -> keep only the first copy.
        known = {}                                   # sha256 -> local_path, including earlier runs
        meta = args.out / "metadata.csv"
        if meta.exists():
            with open(meta, encoding="utf-8-sig") as f:
                known = {r["sha256"]: r["local_path"] for r in csv.DictReader(f)}
        dup_file = args.out / "duplicates.csv"
        if dup_file.exists():
            with open(dup_file, encoding="utf-8") as f:
                duplicates = list(csv.DictReader(f))
        known_dup_urls = {d["pdf_url"] for d in duplicates}
        # PDFs found unreadable (scans etc.) in an earlier run are never downloaded again.
        skip_file = args.out / "skipped_unusable.csv"
        skipped = []
        if skip_file.exists():
            with open(skip_file, encoding="utf-8") as f:
                skipped = list(csv.DictReader(f))
        skipped_urls = {s["pdf_url"] for s in skipped}
        n_skipped_now = 0
        if not args.keep_all:
            from pdf_precheck import precheck
        for rec in records:
            if rec["pdf_url"] in known_dup_urls or rec["pdf_url"] in skipped_urls:
                continue
            target = target_path(rec, args.out)
            if not target.exists() and not args.keep_all:
                # Check the PDF on the server first; only usable papers are saved.
                time.sleep(REQUEST_DELAY)
                keep, reason, fetched, data = precheck(rec["pdf_url"], session)
                name = Path(urlparse(rec["pdf_url"]).path).name[:70]
                if not keep:
                    skipped.append({"pdf_url": rec["pdf_url"], "subject": rec["subject"], "reason": reason})
                    n_skipped_now += 1
                    print(f"[skip]  {name}: {reason} (checked {fetched / 1e3:.0f} KB)")
                    with open(skip_file, "w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=["pdf_url", "subject", "reason"])
                        w.writeheader()
                        w.writerows(skipped)
                    continue
                print(f"[keep]  {name}: {reason}")
                if data is not None:           # already fetched in full during the check
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
            path = download(rec, args.out)
            if not path:
                continue
            rel, digest = path.relative_to(args.out).as_posix(), sha256(path)
            if digest in known and known[digest] != rel:
                path.unlink()
                duplicates.append({"pdf_url": rec["pdf_url"], "duplicate_of": known[digest]})
                print(f"[duplicate] same file as {known[digest]}, removed")
                with open(dup_file, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["pdf_url", "duplicate_of"])
                    w.writeheader()
                    w.writerows(duplicates)
                continue
            known[digest] = rel
            row = {**rec, "local_path": rel, "sha256": digest}
            rows.append(row)
            write_metadata([row], meta)             # saved after every file, so stopping early loses nothing

    found = Counter(r["subject"] for r in records)
    print("\n" + "=" * 60)
    print(f"{'Subject':40s} {'PDFs found':>10s}")
    for s in wanted:
        print(f"{s:40s} {found.get(s, 0):>10d}")
    missing = [s for s in wanted if not found.get(s)]
    if missing:
        print(f"\nNot found for grade {args.grade}: {', '.join(missing)}")
    if not args.dry_run:
        print(f"\nKept {len(rows)} unique PDFs -> {args.out}")
        if n_skipped_now:
            print(f"Skipped {n_skipped_now} PDFs that need OCR or have no readable MCQs (listed in skipped_unusable.csv)")
        if duplicates:
            print(f"Removed {len(duplicates)} duplicate files (listed in duplicates.csv)")
        print(f"Metadata: {args.out / 'metadata.csv'}")


if __name__ == "__main__":
    main()
