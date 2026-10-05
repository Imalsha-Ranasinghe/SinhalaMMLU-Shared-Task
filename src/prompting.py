"""Prompt construction and answer parsing."""

import random
import re

SYSTEM_PROMPT = (
    "You are an expert at answering multiple-choice questions written in Sinhala. "
    "Read the question and the numbered options, then reply with ONLY the number "
    "of the correct option (for example: 2). Do not add any explanation."
)


def format_question(question: str, choices: list) -> str:
    lines = [f"Question: {question.strip()}", ""]
    lines += [f"{i}. {str(c).strip()}" for i, c in enumerate(choices, 1)]
    lines += ["", "Answer:"]
    return "\n".join(lines)


def build_messages(row: dict, shots: list = ()) -> list:
    """Chat messages for one question; few-shot examples go in as prior turns."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": format_question(s["question"], s["choices"])})
        msgs.append({"role": "assistant", "content": str(s["answer"])})
    msgs.append({"role": "user", "content": format_question(row["question"], row["choices"])})
    return msgs


class ShotSampler:
    """Pick k few-shot examples from the train split, same subject, reproducibly."""

    def __init__(self, train_df, k: int, seed: int = 42):
        self.k, self.seed = k, seed
        self.by_subject = {s: g.to_dict(orient="records") for s, g in train_df.groupby("subject")}

    def __call__(self, row: dict) -> list:
        if self.k <= 0:
            return []
        pool = self.by_subject.get(row["subject"], [])
        rng = random.Random(f"{self.seed}-{row['id']}")
        return rng.sample(pool, min(self.k, len(pool)))


_SINHALA_DIGITS = str.maketrans("෦෧෨෩෪෫෬෭෮෯", "0123456789")


def parse_answer(text: str, choices: list):
    """Extract a 1-based option number from model output, or None."""
    if not text:
        return None
    t = text.strip().translate(_SINHALA_DIGITS)
    n = len(choices)
    m = re.search(rf"(?<!\d)([1-{n}])(?!\d)", t)
    if m:
        return int(m.group(1))
    # Fallback: model echoed an option's text instead of its number.
    for i, c in enumerate(choices, 1):
        if str(c).strip() and str(c).strip() in t:
            return i
    return None
