"""Training data generated from textbook passages: paraphrases (for continued pretraining)
and grounded 4-option MCQs (for answer-format fine-tuning).

The generator is any OpenAI-compatible endpoint (see .env.example). Shared-task rule 04:
open models are fine; a closed API (GPT, Claude, Gemini) may be used for training data
only, and must be disclosed. The report written next to the outputs records the model.

Dev Set questions are never shown to the generator. Generated MCQs that come too close to a
Dev Set question are dropped (decontamination), since the Dev Set is used for evaluation.

Usage:
    python -m src.synth data/textbooks/christianity_g8_si.jsonl --subject Christianity --grade 8 \
        --out data/synthetic/christianity_g8 [--limit 5]
"""

import argparse
import hashlib
import json
import random
import time
import re
from datetime import date
from pathlib import Path

from .textbooks import load_pages, passages

ZWJ = "‍"
SINHALA = re.compile(r"[඀-෿]")
ALL_OR_NONE = re.compile(r"සියල්ල|සියල්ලම|කිසිවක් නොවේ|ඉහත කිසිවක්")

PARAPHRASE_STYLES = {
    "explain": "a clear explanatory passage for a student, in your own words and sentence structure",
    "qa": "a list of short question-and-answer pairs that together cover every fact in the passage, "
          "each written as 'ප්‍රශ්නය: ...' on one line and 'පිළිතුර: ...' on the next",
}

PARAPHRASE_PROMPT = """Below is a passage from the Sri Lankan grade {grade} {subject} textbook (Sinhala medium).

Rewrite it in Sinhala as {style}.
Rules:
- Keep every fact, name, number and scripture reference exactly as in the passage.
- Do not add any fact that is not in the passage.
- Write natural, correct Sinhala in Unicode.
- Output only the rewritten text, with no introduction or notes.

Passage:
\"\"\"
{text}
\"\"\""""

MCQ_PROMPT = """Below is a passage from the Sri Lankan grade {grade} {subject} textbook (Sinhala medium).

Write up to {n} multiple-choice questions in Sinhala that test the {subject} knowledge in this passage, like
questions in a school term-test paper.
Rules:
- Test subject knowledge only: teachings, scripture events and people, concepts, values, facts.
  Textbooks often teach through a classroom story (a teacher and named pupils talking). Never ask about that
  story itself: not the pupils' or teacher's names, who said what, or when the lesson happens. Ask about the
  knowledge they discuss. If the passage holds fewer than {n} real facts, write fewer questions.
- Each question must be answerable from the passage alone and have exactly one correct option.
- Exactly 4 options. The 3 wrong options must come from the same subject and be the same kind of thing as
  the answer (other scripture books, other people, other virtues), so that a student who has not studied
  the topic could find them believable. Avoid wrong options that are obviously silly or off-topic.
- Keep all 4 options about the same length and level of detail: the correct option must not be the longest
  or the most complete-sounding one.
- Do not use "all of the above" or "none of the above". Do not refer to "the passage" in the question.
- Mix recall questions with questions that need understanding of an idea, an event or a teaching.
- Each question must test a different fact.
- "evidence" is the exact sentence from the passage that shows the correct answer.

Return only a JSON array, with no other text:
[{{"question": "...", "options": ["...", "...", "...", "..."], "answer": <1-4>, "evidence": "..."}}]

Passage:
\"\"\"
{text}
\"\"\""""


def _norm(t: str) -> str:
    """For matching: no ZWJ, no whitespace or punctuation."""
    return re.sub(r"[\s.,;:!?'\"“”‘’()\-–]+", "", t.replace(ZWJ, ""))


def parse_mcqs(raw: str):
    """JSON array from a model reply (tolerates ```json fences and text around it)."""
    m = re.search(r"\[.*\]", raw or "", re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [x for x in items if isinstance(x, dict)]


def check_mcq(item, passage_text):
    """Return (clean_item, None) or (None, reason)."""
    q = str(item.get("question", "")).strip()
    opts = [str(o).strip() for o in item.get("options") or []]
    ans = item.get("answer")
    if not SINHALA.search(q):
        return None, "question not Sinhala"
    if len(opts) != 4 or any(not o for o in opts):
        return None, "not 4 options"
    if len({_norm(o) for o in opts}) < 4:
        return None, "duplicate options"
    try:
        ans = int(ans)
    except (TypeError, ValueError):
        return None, "bad answer"
    if not 1 <= ans <= 4:
        return None, "bad answer"
    if any(ALL_OR_NONE.search(o) for o in opts):
        return None, "all/none of the above"
    evidence = str(item.get("evidence", "")).strip()
    if not evidence or _norm(evidence) not in _norm(passage_text):
        return None, "evidence not in passage"
    return {"question": q, "choices": opts, "answer": ans, "evidence": evidence}, None


def answer_length_ratio(item):
    """Length of the correct option over the longest wrong option."""
    lens = [len(c) for c in item["choices"]]
    a = lens[item["answer"] - 1]
    return a / max(l for i, l in enumerate(lens) if i != item["answer"] - 1)


def balance_positions(items, seed=42):
    """Shuffle options so the correct answer cycles through positions 1-4 (25% each)."""
    rng = random.Random(seed)
    out = []
    for k, it in enumerate(items):
        correct = it["choices"][it["answer"] - 1]
        wrong = [c for i, c in enumerate(it["choices"]) if i != it["answer"] - 1]
        rng.shuffle(wrong)
        pos = k % 4
        choices = wrong[:pos] + [correct] + wrong[pos:]
        out.append({**it, "choices": choices, "answer": pos + 1})
    rng.shuffle(out)
    return out


def near_duplicates(questions, references, threshold):
    """Indexes of `questions` whose char n-gram cosine similarity to any reference is >= threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if not questions or not references:
        return set(), []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(questions + references)
    sim = cosine_similarity(vec.transform(questions), vec.transform(references))
    best = sim.max(axis=1)
    return {i for i, s in enumerate(best) if s >= threshold}, best


def dedup_and_decontaminate(mcqs, dev_questions, dup_threshold=0.85, dev_threshold=0.7,
                            dup_text=lambda m: m["question"]):
    """Drop near-duplicate MCQs (compared on dup_text), then any whose question is too close to a
    Dev Set question."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    norm = lambda t: re.sub(r"\s+", " ", t.replace(ZWJ, "")).strip()
    keep = []
    if mcqs:
        qs = [norm(dup_text(m)) for m in mcqs]
        sim = cosine_similarity(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(qs))
        kept_idx = []
        for i in range(len(mcqs)):
            if not kept_idx or sim[i, kept_idx].max() < dup_threshold:
                kept_idx.append(i)
        keep = [mcqs[i] for i in kept_idx]
    dropped_dup = len(mcqs) - len(keep)
    hits, _ = near_duplicates([norm(m["question"]) for m in keep], [norm(q) for q in dev_questions], dev_threshold)
    clean = [m for i, m in enumerate(keep) if i not in hits]
    return clean, {"near_duplicate": dropped_dup, "too_close_to_dev": len(hits)}


class Cache:
    """Raw generator replies keyed by task, so an interrupted run resumes without paying twice."""

    def __init__(self, path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.data[r["key"]] = r["reply"]

    def add(self, key, reply):
        self.data[key] = reply
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "reply": reply}, ensure_ascii=False) + "\n")


def run(book_jsonl, subject, grade, out_stem, backend, generator_name, dev_questions,
        n_mcq=5, max_chars=1500, limit=None, batch_size=8, max_answer_length_ratio=1.2):
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    cache = Cache(out_stem.parent / f"{out_stem.name}_cache.jsonl")
    psgs = passages(load_pages(book_jsonl), max_chars=max_chars)[:limit]

    # Cache keys carry a fingerprint of the prompt: editing a prompt regenerates only what it affects.
    key = lambda name, i, prompt: f"{name}-{i}-{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:8]}"
    prompts = {}
    for i, p in enumerate(psgs):
        for style, desc in PARAPHRASE_STYLES.items():
            prompts[("para", style, i)] = PARAPHRASE_PROMPT.format(grade=grade, subject=subject, style=desc, text=p["text"])
        prompts[("mcq", None, i)] = MCQ_PROMPT.format(grade=grade, subject=subject, n=n_mcq, text=p["text"])
    keys = {t: key(f"{t[0]}-{t[1]}" if t[1] else t[0], t[2], pr) for t, pr in prompts.items()}
    tasks = [(keys[t], pr) for t, pr in prompts.items()]
    todo = [(k, prompt) for k, prompt in tasks if k not in cache.data]
    print(f"{len(psgs)} passages, {len(tasks)} generator calls, {len(todo)} not cached yet")
    msgs = [[{"role": "user", "content": prompt}] for _, prompt in todo]
    if hasattr(backend, "generate_iter"):
        # Save each reply as it arrives: an interrupted run loses at most the requests in flight.
        start, done, failed = time.time(), 0, 0
        for i, reply in backend.generate_iter(msgs):
            done += 1
            if reply:
                cache.add(todo[i][0], reply)
            else:
                failed += 1
            if done % 10 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} done, {failed} failed, {time.time() - start:.0f}s elapsed")
    else:
        for b in range(0, len(todo), batch_size):
            replies = backend.generate(msgs[b:b + batch_size])
            for (k, _), reply in zip(todo[b:b + batch_size], replies):
                if reply:
                    cache.add(k, reply)
            print(f"  {min(b + batch_size, len(todo))}/{len(todo)}  ({len(cache.data)} replies cached)")
    if getattr(backend, "daily_limit_hit", False):
        print("Stopped early at the daily limit: rerun the same command after it resets to continue.")

    paraphrases, mcqs, reasons = [], [], {}
    for i, p in enumerate(psgs):
        for style in PARAPHRASE_STYLES:
            text = (cache.data.get(keys[("para", style, i)]) or "").strip()
            if SINHALA.search(text):
                paraphrases.append({"passage": i, "chapter": p["chapter"], "style": style, "text": text})
        for item in parse_mcqs(cache.data.get(keys[("mcq", None, i)])):
            clean, why = check_mcq(item, p["text"])
            if clean:
                mcqs.append({"passage": i, "chapter": p["chapter"], **clean})
            else:
                reasons[why] = reasons.get(why, 0) + 1
    raw_count = len(mcqs)
    mcqs, dropped = dedup_and_decontaminate(mcqs, dev_questions)
    # Generators tend to make the correct option the longest; a model trained on that learns
    # "pick the longest". 1.2 brings the rate close to the Dev Set's (~30%).
    if max_answer_length_ratio:
        kept = [m for m in mcqs if answer_length_ratio(m) <= max_answer_length_ratio]
        dropped["answer_much_longer_than_others"] = len(mcqs) - len(kept)
        mcqs = kept
    mcqs = balance_positions(mcqs)
    for k, m in enumerate(mcqs):
        m.update(id=f"{out_stem.name}-mcq-{k}", subject=subject, category="synthetic", grade=grade)

    for name, rows in [("paraphrases", paraphrases), ("mcq", mcqs)]:
        with open(out_stem.parent / f"{out_stem.name}_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "date": str(date.today()), "generator_model": generator_name, "source": str(book_jsonl),
        "passages": len(psgs), "paraphrases": len(paraphrases),
        "mcq_parsed_valid": raw_count, "mcq_rejected": reasons, "mcq_dropped": dropped, "mcq_final": len(mcqs),
    }
    (out_stem.parent / f"{out_stem.name}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return paraphrases, mcqs, report


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from .backends import OpenAICompatBackend
    from .data import load_raw

    ap = argparse.ArgumentParser()
    ap.add_argument("book_jsonl")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mcq", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only the first N passages (cheap trial run)")
    ap.add_argument("--rpm", type=int, default=None,
                    help="max requests per minute (default: 10 for free tiers - OpenRouter ':free' models "
                         "or a Google AI Studio key - else unlimited)")
    a = ap.parse_args()

    load_dotenv()
    # The generator can use its own endpoint (GEN_*), separate from the evaluation one (OPENAI_*).
    model = os.environ.get("GEN_MODEL") or os.environ["MODEL_NAME"]
    base_url = os.environ.get("GEN_BASE_URL") or os.environ["OPENAI_BASE_URL"]
    google = "generativelanguage.googleapis.com" in base_url
    api_key = (os.environ.get("GEN_API_KEY") or (os.environ.get("GEMINI_API_KEY") if google else None)
               or os.environ["OPENAI_API_KEY"])
    free = model.endswith(":free") or google
    # max_tokens covers the model's thinking too (Gemma 4 thinks before answering).
    backend = OpenAICompatBackend(model, base_url, api_key, max_tokens=8000, temperature=0.7,
                                  max_workers=12 if free else 8, max_retries=6, timeout=300,
                                  requests_per_minute=a.rpm or (10 if free else None))
    print(f"generator: {model} at {base_url}")
    dev = load_raw(Path(__file__).resolve().parent.parent / "Dev Set")["question"].tolist()
    run(a.book_jsonl, a.subject, a.grade, a.out, backend, model, dev, n_mcq=a.n_mcq, limit=a.limit)
