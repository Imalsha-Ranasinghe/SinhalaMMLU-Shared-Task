"""Build the training dataset from the extracted past-paper questions, with the Dev Set as test set.

    python data/build_train_data.py

Reads   data/extracted/grade_XX/questions_grade_N.json  (all grades)  and  Dev Set/*.json
Writes  data/train_data/
            train.jsonl        extracted questions for training
            val.jsonl          5% of the extracted questions, held out to pick the best epoch
            test.jsonl         the Dev Set (1,851 questions)
            removed.jsonl      extracted questions removed because they overlap the Dev Set
            summary.json       counts per split / grade / subject
Same columns as data/splits/*.jsonl, so the notebooks can use these files directly.

What is filtered:
  * questions without an answer, or needing a figure/table/passage (needs_context)
  * 5-option (A/L) questions are converted to 4 options by removing one wrong option
  * test leakage: questions from the same exam paper as a Dev Set question, and questions
    that nearly match a Dev Set question (character 4-gram overlap)
  * duplicates across grades
"""

import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from extract_questions import to_four  # noqa: E402
from src.data import load_raw, validate  # noqa: E402

SEED = 42
VAL_FRACTION = 0.05
NEAR_DUP = 0.5          # character 4-gram Jaccard similarity (question + options) that counts as the same question

# Dev Set subject spellings -> the names used in the extracted data
SUBJECT_NAMES = {
    "arts": "Arts", "buddhism": "Buddhism", "catholicism": "Catholicism", "christianity": "Christianity",
    "civics": "Citizenship Education", "eastern music": "Eastern Music", "geography": "Geography",
    "health and physical science": "Health and Physical Science", "history": "History", "islam": "Islam",
    "sinhala language and literature": "Sinhala Language and Literature", "dancing": "Dancing",
    "drama and theatre": "Drama and Theatre", "science": "Science",
}


def canon_subject(s):
    s = re.sub(r"\s+", " ", s.replace("_", " ")).strip()
    return SUBJECT_NAMES.get(s.lower(), s)


def norm_text(t):
    """For comparing questions: NFC, no punctuation/spaces, lower case."""
    t = unicodedata.normalize("NFC", t).lower()
    return re.sub(r"[\W_]+", "", t)


def shingles(t, k=4):
    return {t[i:i + k] for i in range(max(len(t) - k + 1, 1))}


def norm_url(u):
    if not u:
        return ""
    p = urlparse(str(u))
    return (p.netloc.replace("www.", "") + p.path).rstrip("/").lower()


def row(q, split_source, idx):
    m = q.get("metadata", {})
    return {
        "id": idx,
        "subject": canon_subject(q["subject"]),
        "category": q["category"],
        "question": q["question"],
        "choices": q["choices"],
        "answer": int(q["answer"]),
        "grade": m.get("grade"),
        "difficulty": m.get("difficulty"),
        "year": m.get("year"),
        "term": m.get("term"),
        "region": m.get("region"),
        "source": m.get("source"),
        "dataset": split_source,
        "original_n_options": m.get("original_n_options", len(q["choices"])),
    }


def load_extracted():
    out = []
    for path in sorted((HERE / "extracted").glob("grade_*/questions_grade_*.json")):
        for q in json.load(open(path, encoding="utf-8")):
            out.append(q)
    return out


def main():
    rng = random.Random(SEED)
    out_dir = HERE / "train_data"
    out_dir.mkdir(exist_ok=True)

    # ---- test = Dev Set
    dev, _ = validate(load_raw(ROOT / "Dev Set"))
    test = []
    for r in dev.to_dict(orient="records"):
        q = {"subject": r["subject"], "category": r["category"], "question": r["question"],
             "choices": r["choices"], "answer": r["answer"],
             "metadata": {**r["metadata"], "grade": r["grade"], "difficulty": r["difficulty"], "year": r["year"]}}
        test.append(row(q, "dev_set", r["id"]))
    dev_sources = {norm_url(t["source"]) for t in test if t["source"]}

    # Index Dev Set questions by 4-gram shingles for near-duplicate search.
    dev_all = [shingles(norm_text(t["question"] + "".join(t["choices"]))) for t in test]
    dev_q = [shingles(norm_text(t["question"])) for t in test]
    dev_opts = [{norm_text(c) for c in t["choices"]} for t in test]
    index = defaultdict(list)
    for i in range(len(test)):
        for s in dev_q[i] | dev_all[i]:
            index[s].append(i)

    def jaccard(a, b):
        return len(a & b) / max(len(a | b), 1)

    def contained(a, b):
        return len(a & b) / max(min(len(a), len(b)), 1)

    def dev_overlap(q):
        """(Dev Set index, reason) if q is the same question as a Dev Set one, else (None, None).

        Checked on 9,950 questions: these three rules catch reworded copies (spelling variants,
        an instruction added in front, different wrong options) without flagging different
        questions that merely share generic options like "A හා B".
        """
        sa = shingles(norm_text(q["question"] + "".join(q["choices"])))
        sq = shingles(norm_text(q["question"]))
        opts = {norm_text(c) for c in q["choices"]}
        for i, _ in Counter(i for s in sq | sa for i in index.get(s, ())).most_common(8):
            same_opts = len(opts & dev_opts[i])
            if jaccard(sa, dev_all[i]) >= NEAR_DUP:
                return i, f"near-duplicate of a Dev Set question (similarity {jaccard(sa, dev_all[i]):.2f})"
            if jaccard(sq, dev_q[i]) >= 0.8 and same_opts >= 1:
                return i, "same question text as a Dev Set question"
            if contained(sq, dev_q[i]) >= 0.8 and len(sq) >= 15 and same_opts >= 2:
                return i, "Dev Set question with extra words around it"
        return None, None

    # ---- extracted questions -> filtered pool
    stats = Counter()
    pool, removed, seen = [], [], set()
    for q in load_extracted():
        stats["extracted (with answers)"] += 1
        if q.get("answer") in (None, 0) or not q.get("choices"):
            stats["dropped: no answer"] += 1
            continue
        if q.get("metadata", {}).get("needs_context"):
            stats["dropped: needs figure/table/passage"] += 1
            continue
        if len(q["choices"]) == 5:
            q = to_four(q)
            stats["converted 5 -> 4 options"] += 1
        if len(q["choices"]) != 4 or not 1 <= int(q["answer"]) <= 4 or any(not str(c).strip() for c in q["choices"]):
            stats["dropped: malformed"] += 1
            continue
        src = norm_url(q.get("metadata", {}).get("source"))
        if src and src in dev_sources:
            removed.append({**q, "removed_because": "same exam paper as a Dev Set question"})
            stats["dropped: same paper as Dev Set"] += 1
            continue
        dev_i, why = dev_overlap(q)
        if dev_i is not None:
            removed.append({**q, "removed_because": why, "dev_question": test[dev_i]["question"],
                            "dev_choices": test[dev_i]["choices"]})
            stats["dropped: overlaps a Dev Set question"] += 1
            continue
        key = norm_text(q["question"]) + "|" + "|".join(sorted(norm_text(c) for c in q["choices"]))
        if key in seen:
            stats["dropped: duplicate across papers/grades"] += 1
            continue
        seen.add(key)
        m = q.get("metadata", {})
        pool.append(row(q, "extracted", f"g{m.get('grade')}::{Path(str(m.get('pdf', 'x'))).stem[:60]}::{q.get('q_no')}"))

    # unique ids
    counts = Counter()
    for r in pool:
        counts[r["id"]] += 1
        if counts[r["id"]] > 1:
            r["id"] = f"{r['id']}#{counts[r['id']]}"

    # ---- validation: VAL_FRACTION of each subject
    by_subject = defaultdict(list)
    for r in pool:
        by_subject[r["subject"]].append(r)
    train, val = [], []
    for subj, rows in sorted(by_subject.items()):
        rng.shuffle(rows)
        k = max(1, round(len(rows) * VAL_FRACTION)) if len(rows) >= 20 else 0
        val += rows[:k]
        train += rows[k:]
    rng.shuffle(train)

    def write(name, rows):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write("train.jsonl", train)
    write("val.jsonl", val)
    write("test.jsonl", test)
    write("removed.jsonl", removed)

    summary = {
        "splits": {"train": len(train), "val": len(val), "test (Dev Set)": len(test)},
        "filtering": dict(stats),
        "train_by_grade": dict(sorted(Counter(r["grade"] for r in train).items())),
        "train_by_subject": dict(Counter(r["subject"] for r in train).most_common()),
        "test_by_subject": dict(Counter(r["subject"] for r in test).most_common()),
        "subjects_in_test_but_not_train": sorted({r["subject"] for r in test} - {r["subject"] for r in train}),
        "answer_distribution_train": dict(sorted(Counter(r["answer"] for r in train).items())),
    }
    json.dump(summary, open(out_dir / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(json.dumps(summary["splits"], indent=2))
    for k, v in stats.items():
        print(f"  {k:42s} {v:6d}")
    print("\ntrain by grade:", summary["train_by_grade"])
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
