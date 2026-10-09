"""Print extracted questions in a readable form, with the correct answer marked.

    python data/show_questions.py                         # 10 random questions
    python data/show_questions.py --n 30 --subject History
    python data/show_questions.py --pdf "Agriculture"     # questions from PDFs whose path contains this text
    python data/show_questions.py --no-answer             # questions without an answer key
    python data/show_questions.py --all > review.txt      # everything, into a text file
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def main():
    here = Path(__file__).resolve().parent / "extracted"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="how many to show")
    ap.add_argument("--subject", help="only this subject (part of the name is enough)")
    ap.add_argument("--pdf", help="only questions from PDFs whose path contains this text")
    ap.add_argument("--no-answer", action="store_true", help="show questions without an answer key")
    ap.add_argument("--all", action="store_true", help="show every matching question, in order")
    ap.add_argument("--grade", type=int, default=10)
    args = ap.parse_args()

    name = f"{'no_answer' if args.no_answer else 'questions'}_grade_{args.grade}.json"
    path = here / f"grade_{args.grade:02d}" / name
    if not path.exists():
        path = here / name                 # older flat layout
    if not path.exists():
        raise SystemExit(f"{path} not found. Run data/extract_questions.py first.")
    qs = json.load(open(path, encoding="utf-8"))
    if args.subject:
        qs = [q for q in qs if args.subject.lower() in q["subject"].lower()]
    if args.pdf:
        qs = [q for q in qs if args.pdf.lower() in str(q["metadata"].get("pdf", "")).lower()]
    print(f"{len(qs)} questions in {path.name}" + (" (matching filters)" if args.subject or args.pdf else ""))
    if not args.all:
        qs = random.sample(qs, min(args.n, len(qs)))

    for q in qs:
        m = q["metadata"]
        flag = "   [needs figure/table/passage]" if m.get("needs_context") else ""
        print("\n" + "-" * 80)
        print(f"{q['subject']} | Q{q['q_no']} | {m.get('year') or ''} term {m.get('term') or '?'} | {m.get('pdf')}{flag}")
        print(f"\n{q['question']}\n")
        for i, c in enumerate(q["choices"], 1):
            print(f"   ({i}) {c}" + ("   <== answer" if q["answer"] == i else ""))


if __name__ == "__main__":
    main()
