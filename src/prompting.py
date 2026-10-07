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


def shuffle_augment(df, copies: int, seed: int = 42):
    """Add `copies` versions of each question with the options in a new random order.

    More training data, and the model can't learn shortcuts like "the answer is usually 3".
    """
    import pandas as pd

    if copies <= 0:
        return df
    rng = random.Random(seed)
    extra = []
    for r in df.to_dict(orient="records"):
        # "ඉහත ... සියල්ල" (all of the above) only makes sense in the original order.
        if any("ඉහත" in str(c) for c in r["choices"]):
            continue
        for c in range(copies):
            order = list(range(len(r["choices"])))
            rng.shuffle(order)
            extra.append({**r, "id": f"{r['id']}#shuf{c}",
                          "choices": [r["choices"][i] for i in order],
                          "answer": order.index(r["answer"] - 1) + 1})
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True).sample(frac=1, random_state=seed)


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
