"""Loading, validating, splitting and saving the SinhalaMMLU data."""

import json
import numbers
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(raw_dir) -> pd.DataFrame:
    """Read every *.json file in raw_dir into one flat DataFrame."""
    rows = []
    for path in sorted(Path(raw_dir).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
        for it in items:
            md = it.get("metadata") or {}
            rows.append({
                "id": f"{path.stem}::{it['q_no']}",
                "source_file": path.name,
                "q_no": it["q_no"],
                "subject": it["subject"].strip(),
                "category": it["category"].strip(),
                "question": it["question"],
                "choices": it["choices"],
                "answer": it["answer"],
                "difficulty": md.get("difficulty"),
                "grade": md.get("grade"),
                "year": md.get("year"),
                "metadata": md,
            })
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame):
    """Fix what can be fixed, drop what can't. Returns (clean_df, report)."""
    df = df.copy()
    report = {"repaired": [], "dropped": [], "renamed_subjects": {}}

    # Unify spelling variants like "Eastern_music" / "Eastern music" to the most common form.
    key = df["subject"].str.replace("_", " ").str.strip().str.lower()
    canonical = df.groupby(key)["subject"].agg(lambda s: s.value_counts().index[0])
    new_subject = key.map(canonical)
    for old, new in set(zip(df["subject"], new_subject)):
        if old != new:
            report["renamed_subjects"][old] = new
    df["subject"] = new_subject

    def fix(row):
        ans, choices = row["answer"], [str(c).strip() for c in row["choices"]]
        if isinstance(ans, numbers.Integral) and 1 <= ans <= len(choices):
            return ans
        # Answer given as the choice text instead of its 1-based index.
        if str(ans).strip() in choices:
            new = choices.index(str(ans).strip()) + 1
            report["repaired"].append((row["id"], ans, new))
            return new
        return None

    df["answer"] = df.apply(fix, axis=1)
    bad = (
        df["answer"].isna()
        | df["question"].str.strip().eq("")
        | df["choices"].apply(lambda c: len(c) < 2 or any(not str(x).strip() for x in c))
    )
    report["dropped"] = df.loc[bad, "id"].tolist()
    df = df[~bad].copy()
    df["answer"] = df["answer"].astype(int)

    key = df["question"].str.strip() + "||" + df["choices"].apply(lambda c: "|".join(s.strip() for s in c))
    dups = key.duplicated()
    report["dropped"] += df.loc[dups, "id"].tolist()
    df = df[~dups].reset_index(drop=True)
    return df, report


def split(df: pd.DataFrame, train=0.70, val=0.15, test=0.15, seed=42, stratify_col="subject"):
    """Stratified train/val/test split."""
    assert abs(train + val + test - 1.0) < 1e-6
    strat = df[stratify_col] if stratify_col else None
    train_df, rest = train_test_split(df, test_size=val + test, random_state=seed, stratify=strat)
    strat = rest[stratify_col] if stratify_col else None
    val_df, test_df = train_test_split(rest, test_size=test / (val + test), random_state=seed, stratify=strat)
    return (train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True))


def save_jsonl(df: pd.DataFrame, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def oversample(df: pd.DataFrame, subjects=None, subject_factor=1.0, max_grade=None, grade_factor=1.0, seed=42):
    """Repeat rows to give some data more weight in training.

    subjects/subject_factor: e.g. the subjects that appear in the test set, x1.5.
    max_grade/grade_factor:  e.g. grades <= 9 (closest to the grade 6-8 test set), x2.
    Factors multiply; fractional parts are sampled at random.
    """
    import random
    rng = random.Random(seed)
    out = []
    for r in df.to_dict(orient="records"):
        f = 1.0
        if subjects is not None and r["subject"] in subjects:
            f *= subject_factor
        if max_grade is not None and r.get("grade") is not None and r["grade"] <= max_grade:
            f *= grade_factor
        n = int(f) + (rng.random() < f - int(f))
        for k in range(n):
            out.append({**r, "id": r["id"] if k == 0 else f"{r['id']}#rep{k}"})
    return pd.DataFrame(out).sample(frac=1, random_state=seed).reset_index(drop=True)
