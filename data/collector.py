import csv
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://pastpapers.wiki"

GRADE = 6

OUTPUT_DIR = Path("sinhala_term_test_papers")
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
ERROR_FILE = OUTPUT_DIR / "errors.csv"

REQUEST_DELAY = 1.5
TIMEOUT = 30
MAX_RETRIES = 3


# Start specifically from the Grade 6 Sinhala-medium page.
START_URL = (
    "https://pastpapers.wiki/"
    "grade-06-sinhala-medium-term-test-papers-past-papers-wiki/"
)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/155.0 Safari/537.36 "
        "EducationalResearchCollector/1.0"
    )
})


# ============================================================
# HELPERS
# ============================================================

def normalize_url(url):

    parsed = urlparse(url)

    return parsed._replace(
        fragment=""
    ).geturl()


def is_same_domain(url):

    hostname = urlparse(url).netloc.lower()

    return hostname in (
        "pastpapers.wiki",
        "www.pastpapers.wiki"
    )


def clean_filename(text):

    text = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()[:180]


def request_page(url):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(f"[GET] {url}")

            time.sleep(REQUEST_DELAY)

            response = session.get(
                url,
                timeout=TIMEOUT
            )

            response.raise_for_status()

            return response

        except Exception as e:

            print(
                f"[RETRY] {attempt}/{MAX_RETRIES}: {e}"
            )

            time.sleep(attempt * 2)

    return None


# ============================================================
# PDF
# ============================================================

def is_pdf(url):

    return urlparse(url).path.lower().endswith(".pdf")


def find_pdf_links(soup, page_url):

    results = []

    for a in soup.find_all("a", href=True):

        href = a["href"].strip()

        url = normalize_url(
            urljoin(page_url, href)
        )

        if is_pdf(url):

            results.append({
                "url": url,
                "text": a.get_text(
                    " ",
                    strip=True
                )
            })

    return results


# ============================================================
# METADATA
# ============================================================

def extract_title(soup):

    h1 = soup.find("h1")

    if h1:

        return h1.get_text(
            " ",
            strip=True
        )

    if soup.title:

        return soup.title.get_text(
            " ",
            strip=True
        )

    return ""


def extract_grade(text):

    match = re.search(
        r"grade[\s\-]*(0?\d{1,2})",
        text,
        re.IGNORECASE
    )

    if match:

        try:

            return int(
                match.group(1)
            )

        except ValueError:

            pass

    return None


def extract_term(text):

    patterns = [
        (1, r"\b1st\s+term\b"),
        (2, r"\b2nd\s+term\b"),
        (3, r"\b3rd\s+term\b"),
    ]

    for number, pattern in patterns:

        if re.search(
            pattern,
            text,
            re.IGNORECASE
        ):

            return number

    return None


def extract_year(text):

    years = re.findall(
        r"\b20\d{2}\b",
        text
    )

    if not years:

        return None

    return max(
        int(year)
        for year in years
    )


SUBJECTS = [
    "Mathematics",
    "Science",
    "Sinhala",
    "History",
    "English",
    "Buddhism",
    "Geography",
    "Civic",
    "ICT",
    "Health",
    "Art",
    "Music",
    "Dancing",
    "Drama",
]


def extract_subject(text):

    # Every title/URL says "Sinhala Medium", which would otherwise
    # tag every paper as the Sinhala subject.
    lower = re.sub(
        r"sinhala[\s\-_]*medium",
        " ",
        text.lower()
    )

    for subject in SUBJECTS:

        if subject.lower() in lower:

            return subject

    return "Unknown"


# ============================================================
# PAPER / ANSWER DETECTION
# ============================================================

def is_answer(text):

    text = text.lower()

    answer_keywords = [
        "answer",
        "answers",
        "answer sheet",
        "answersheet",
        "marking scheme",
        "marking-scheme",
        "marking_scheme",
    ]

    return any(
        keyword in text
        for keyword in answer_keywords
    )


# ============================================================
# LINK FILTER
# ============================================================

def is_relevant_link(url, text):

    # Every page on the site shares the same nav menu, so generic
    # keywords like "term-test" or "sinhala" match the whole site
    # (A/L, O/L, every grade). Only follow Grade 6 URLs.
    path = urlparse(url).path.lower()

    if not re.search(
        r"grade[-_]?0?6(?!\d)",
        path
    ):

        return False

    excluded = [
        "tamil-medium",
        "english-medium",
        "textbook",
    ]

    return not any(
        keyword in path
        for keyword in excluded
    )


# ============================================================
# CRAWL
# ============================================================

def crawl():

    queue = [START_URL]

    queued = {START_URL}

    visited = set()

    pdf_records = []

    while queue:

        current_url = queue.pop(0)

        current_url = normalize_url(
            current_url
        )

        if current_url in visited:

            continue

        visited.add(
            current_url
        )

        print()
        print(
            f"[CRAWL] {current_url}"
        )

        response = request_page(
            current_url
        )

        if response is None:

            log_error(
                current_url,
                "Failed to retrieve page"
            )

            continue

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # Don't parse images/PDFs as HTML.
        if "text/html" not in content_type:

            continue

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        page_title = extract_title(
            soup
        )

        page_text = soup.get_text(
            " ",
            strip=True
        )

        # ----------------------------------------------------
        # Check PDFs on this page
        # ----------------------------------------------------

        pdf_links = find_pdf_links(
            soup,
            current_url
        )

        for pdf in pdf_links:

            combined_text = " ".join([
                page_title,
                pdf["text"],
                pdf["url"],
            ])

            grade = extract_grade(
                combined_text
            )

            # We only want Grade 6.
            if grade is not None and grade != 6:

                continue

            record = {
                "pdf_url": pdf["url"],
                "source_page": current_url,
                "title": page_title,
                "subject": extract_subject(
                    combined_text
                ),
                "term": extract_term(
                    combined_text
                ),
                "year": extract_year(
                    combined_text
                ),
                "type": (
                    "answer"
                    if is_answer(combined_text)
                    else "paper"
                )
            }

            # Avoid duplicate PDFs.
            if not any(
                r["pdf_url"] == record["pdf_url"]
                for r in pdf_records
            ):

                pdf_records.append(
                    record
                )

                print(
                    f"[PDF FOUND] "
                    f"{record['type']} "
                    f"{pdf['url']}"
                )

        # ----------------------------------------------------
        # Find relevant child pages
        # ----------------------------------------------------

        for a in soup.find_all(
            "a",
            href=True
        ):

            href = a["href"].strip()

            if not href:

                continue

            link_url = normalize_url(
                urljoin(
                    current_url,
                    href
                )
            )

            link_text = a.get_text(
                " ",
                strip=True
            )

            if not is_same_domain(
                link_url
            ):

                continue

            if link_url in visited or link_url in queued:

                continue

            # Never follow obvious non-HTML files.
            if any(
                link_url.lower().endswith(ext)
                for ext in [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".webp",
                    ".zip",
                    ".rar",
                    ".pdf",
                ]
            ):

                continue

            # Only follow relevant paper pages.
            if is_relevant_link(
                link_url,
                link_text
            ):

                queue.append(
                    link_url
                )

                queued.add(
                    link_url
                )

    return pdf_records


# ============================================================
# DOWNLOAD
# ============================================================

def download_pdf(record):

    subject = clean_filename(
        record["subject"]
    )

    term = (
        record["term"]
        if record["term"]
        else "unknown"
    )

    year = (
        record["year"]
        if record["year"]
        else "unknown"
    )

    paper_type = record["type"]

    directory = (
        OUTPUT_DIR
        / "grade_06"
        / subject
        / f"term_{term}"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Include the source file name: many papers share the same
    # year/subject/type (different provinces) and would otherwise
    # overwrite or be [SKIP]ped as duplicates of each other.
    source_name = clean_filename(
        Path(urlparse(record["pdf_url"]).path).stem
    )[:120]

    filename = (
        f"{year}_{subject}_{paper_type}_{source_name}.pdf"
    )

    output_path = (
        directory / filename
    )

    # Already downloaded.
    if (
        output_path.exists()
        and output_path.stat().st_size > 1000
    ):

        print(
            f"[SKIP] {output_path}"
        )

        return output_path

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            print(
                f"[DOWNLOAD] {record['pdf_url']}"
            )

            time.sleep(
                REQUEST_DELAY
            )

            response = session.get(
                record["pdf_url"],
                timeout=TIMEOUT,
                stream=True
            )

            response.raise_for_status()

            temp_path = output_path.with_suffix(
                ".part"
            )

            with open(
                temp_path,
                "wb"
            ) as f:

                for chunk in response.iter_content(
                    64 * 1024
                ):

                    if chunk:

                        f.write(chunk)

            # Verify PDF header.
            with open(
                temp_path,
                "rb"
            ) as f:

                header = f.read(5)

            if header != b"%PDF-":

                temp_path.unlink(
                    missing_ok=True
                )

                raise ValueError(
                    "Downloaded file is not a valid PDF"
                )

            temp_path.replace(
                output_path
            )

            print(
                f"[SAVED] {output_path}"
            )

            return output_path

        except Exception as e:

            print(
                f"[DOWNLOAD ERROR] "
                f"{attempt}/{MAX_RETRIES}: {e}"
            )

            time.sleep(
                attempt * 2
            )

    log_error(
        record["pdf_url"],
        "Download failed"
    )

    return None


# ============================================================
# CSV
# ============================================================

def initialize_csv():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not METADATA_FILE.exists():

        with open(
            METADATA_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "grade",
                    "subject",
                    "term",
                    "year",
                    "type",
                    "title",
                    "source_page",
                    "pdf_url",
                    "local_path",
                    "sha256",
                ]
            )

            writer.writeheader()


def append_metadata(
    record,
    local_path
):

    sha256 = hashlib.sha256()

    with open(
        local_path,
        "rb"
    ) as f:

        while chunk := f.read(
            1024 * 1024
        ):

            sha256.update(
                chunk
            )

    with open(
        METADATA_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            6,
            record["subject"],
            record["term"],
            record["year"],
            record["type"],
            record["title"],
            record["source_page"],
            record["pdf_url"],
            str(local_path),
            sha256.hexdigest(),
        ])


def log_error(url, error):

    exists = ERROR_FILE.exists()

    with open(
        ERROR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "url",
                "error"
            ])

        writer.writerow([
            url,
            error
        ])


# ============================================================
# MAIN
# ============================================================

def main():

    initialize_csv()

    print()
    print("=" * 70)
    print("SINHALA TERM TEST PAPER COLLECTOR")
    print("Grade 6")
    print("=" * 70)
    print()

    records = crawl()

    print()
    print("=" * 70)
    print(
        f"FOUND {len(records)} PDF FILES"
    )
    print("=" * 70)

    downloaded = 0

    for record in records:

        path = download_pdf(
            record
        )

        if path:

            # Don't duplicate metadata for an
            # already-existing file.
            append_metadata(
                record,
                path
            )

            downloaded += 1

    print()
    print("=" * 70)
    print(
        f"FINISHED — {downloaded} files"
    )
    print(
        f"Output: {OUTPUT_DIR.resolve()}"
    )
    print(
        f"Metadata: {METADATA_FILE.resolve()}"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()