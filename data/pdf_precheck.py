"""Decide whether a PDF on the web has extractable MCQs *before* saving it.

Two cases, depending on the server:

* Range requests supported (pastpapers.wiki/download/... links): only the parts of the PDF that
  are needed are fetched - the cross-reference table at the end, then the fonts and text of the
  first pages (images are skipped). Typically 100-450 KB instead of the whole file.
* No range support (links that redirect to cdn.pastpapers.wiki): a PDF can't be judged from its
  first bytes (fonts and the page list are stored at the end), so the file is fetched into memory,
  checked with the full extractor, and handed back only if it is usable. Unusable files are never
  written to disk.

Tested against the full extractor on 42 papers: 41 decisions agree.
"""

import logging
import os
import re
import tempfile
import time
from collections import defaultdict

from pypdf import PdfReader
from pypdf.generic import DictionaryObject, NameObject

from extract_questions import ENGLISH_WORDS, _fm, broken_sinhala_ratio, extract, key_only, looks_fm_encoded

logging.getLogger("pypdf").setLevel(logging.ERROR)

BLOCK = 64 * 1024          # bytes fetched per range request
PAGES_TO_CHECK = 4         # MCQs come first in these papers
RANGE_DELAY = 0.2          # small pause between range requests for the same file
MAX_BYTES = 40 * 1024 * 1024   # bigger files are scans; never hold more than this in memory
TIME_LIMIT = 180           # seconds per paper, so one bad link can't stall the whole run


class CheckFailed(Exception):
    pass


class RemotePDF:
    """A read-only, seekable file over HTTP range requests (what pypdf needs: read/seek/tell)."""

    def __init__(self, url, session, timeout=60):
        self.url, self.session, self.timeout = url, session, timeout
        self.blocks, self.pos, self.whole, self.fetched = {}, 0, None, 0
        self.deadline = time.time() + TIME_LIMIT
        r = session.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout, stream=True)
        r.raise_for_status()
        self.url = r.url                   # after redirects
        if "html" in r.headers.get("Content-Type", "").lower():
            r.close()                      # e.g. the CDN answered with a web page, not the file
            raise CheckFailed("server returned a web page, not a PDF")
        if r.status_code == 206 and "/" in r.headers.get("Content-Range", ""):
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
            r.close()
            return
        # Server ignored the range: this response is the whole file. Read it with limits.
        chunks, total = [], 0
        try:
            for chunk in r.iter_content(256 * 1024):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_BYTES:
                    raise CheckFailed(f"larger than {MAX_BYTES // 2**20} MB (a scan)")
                if time.time() > self.deadline:
                    raise CheckFailed("download too slow")
        finally:
            r.close()
        self.whole = b"".join(chunks)
        self.size = self.fetched = len(self.whole)

    def _block(self, i):
        if time.time() > self.deadline:
            raise CheckFailed("check took too long")
        if i not in self.blocks:
            start = i * BLOCK
            end = min(start + BLOCK, self.size) - 1
            time.sleep(RANGE_DELAY)
            r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=self.timeout,
                                 stream=True)
            r.raise_for_status()
            if r.status_code != 206:       # server stopped honouring ranges: don't pull the whole file
                r.close()
                raise CheckFailed("server stopped answering partial requests")
            self.blocks[i] = r.content
            self.fetched += len(r.content)
        return self.blocks[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if self.whole is not None:
            data = self.whole[self.pos:self.pos + n]
        else:
            out, p, end = [], self.pos, self.pos + n
            while p < end:
                i = p // BLOCK
                chunk = self._block(i)[p - i * BLOCK:p - i * BLOCK + (end - p)]
                if not chunk:
                    break
                out.append(chunk)
                p += len(chunk)
            data = b"".join(out)
        self.pos += len(data)
        return data

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def tell(self):
        return self.pos


def _features(reader):
    """Fonts and text of the first pages, without loading any images."""
    by_font, n_fonts = defaultdict(list), 0
    for page in reader.pages[:PAGES_TO_CHECK]:
        res = page.get("/Resources")
        res = res.get_object() if res is not None else DictionaryObject()
        fonts = res.get("/Font")
        fonts = fonts.get_object() if fonts is not None else {}
        n_fonts += len(fonts)
        if not fonts:
            continue
        # Drop images from the page so reading its text never downloads them.
        page[NameObject("/Resources")] = DictionaryObject({k: v for k, v in res.items() if k != "/XObject"})

        def visit(text, cm, tm, font_dict, font_size):
            if text and font_dict is not None:
                by_font[str(font_dict.get("/BaseFont", "")).lstrip("/").split("+")[-1].lower()].append(text)
        try:
            page.extract_text(visitor_text=visit)
        except Exception:
            pass
    parts = []
    for font, texts in by_font.items():
        t = " ".join(texts)
        parts.append(_fm.convert(t) if font.startswith("fm") or looks_fm_encoded(t) else t)
    text = " ".join(parts)
    words = re.findall(r"[a-z]{2,}", text.lower())
    return {
        "pages": len(reader.pages), "fonts": n_fonts,
        "sinhala": sum("඀" <= c <= "෿" for c in text),
        "nwords": len(words), "eng": sum(w in ENGLISH_WORDS for w in words) / max(len(words), 1),
        "broken": broken_sinhala_ratio(text),
        "pairs": len(re.findall(r"\(?0?\d{1,2}\)?\s*[.)\-:–]?\s*\(?[1-4]\)?(?![\d.])", text)),
    }


def _decide(f):
    """Thresholds chosen on 42 papers checked with the full extractor."""
    if f["fonts"] == 0:
        return False, "scanned (no text on the first pages)"
    if f["nwords"] > 200 and f["eng"] > 0.08 and f["sinhala"] < 2000:
        return False, "English medium"
    if f["broken"] > 0.03:
        return False, "garbled Sinhala text layer"
    if f["pages"] <= 3:
        return (True, "answer key") if f["pairs"] >= 20 else (False, "short file without MCQs")
    if f["sinhala"] >= 300:
        return True, "Sinhala text"
    if f["nwords"] >= 300 and f["eng"] < 0.03 and f["fonts"] >= 6:
        return True, "legacy-font Sinhala text"
    return False, "no readable Sinhala text"


def _check_whole(data):
    """Full extractor on an in-memory PDF (servers without range support)."""
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        status, qs, _ = extract(tmp, {})
        if status == "ok" and qs:
            return True, f"{len(qs)} questions"
        if status != "english_medium" and key_only(tmp):
            return True, "answer key"
        return False, status.replace("_", " ")
    finally:
        os.remove(tmp)


def precheck(url, session):
    """Returns (keep, reason, bytes_fetched, data). `data` is the whole PDF when it had to be fetched
    in full (no range support) and is usable, so the caller can save it without downloading again."""
    try:
        f = RemotePDF(url, session)
    except CheckFailed as e:
        return False, str(e), 0, None
    except Exception as e:
        return False, f"request failed ({type(e).__name__})", 0, None
    if f.whole is not None:
        if f.whole[:5] != b"%PDF-":
            return False, "not a PDF", f.fetched, None
        keep, why = _check_whole(f.whole)
        return keep, why, f.fetched, f.whole if keep else None
    try:
        try:
            reader = PdfReader(f, strict=True)      # strict avoids re-reading the whole file
            feats = _features(reader)
        except Exception:
            reader = PdfReader(f, strict=False)     # damaged cross-reference table
            feats = _features(reader)
    except CheckFailed as e:
        return False, str(e), f.fetched, None
    except Exception as e:
        return False, f"unreadable PDF ({type(e).__name__})", f.fetched, None
    keep, why = _decide(feats)
    return keep, why, f.fetched, None
