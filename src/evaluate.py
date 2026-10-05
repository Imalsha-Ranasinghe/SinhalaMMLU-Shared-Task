"""Resumable evaluation loop and metrics."""

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .prompting import build_messages, parse_answer


def run_eval(df: pd.DataFrame, backend, shot_sampler, out_path, batch_size=32) -> pd.DataFrame:
    """Predict every row of df, appending to out_path (JSONL) so reruns resume where they stopped."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    todo = [r for r in df.to_dict(orient="records") if r["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go -> {out_path}")

    with open(out_path, "a", encoding="utf-8") as f, tqdm(total=len(todo)) as bar:
        for i in range(0, len(todo), batch_size):
            rows = todo[i:i + batch_size]
            outputs = backend.generate([build_messages(r, shot_sampler(r)) for r in rows])
            for r, out in zip(rows, outputs):
                pred = parse_answer(out, r["choices"])
                rec = {
                    "id": r["id"], "subject": r["subject"], "category": r["category"],
                    "answer": r["answer"], "pred": pred, "raw_output": out,
                    "correct": pred == r["answer"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            bar.update(len(rows))

    preds = pd.read_json(out_path, lines=True)
    return preds[preds["id"].isin(df["id"])].reset_index(drop=True)


def compute_metrics(preds: pd.DataFrame) -> dict:
    def group(col):
        g = preds.groupby(col).agg(n=("correct", "size"), accuracy=("correct", "mean"))
        return g.sort_values("accuracy", ascending=False).round(4)

    return {
        "n": int(len(preds)),
        "accuracy": round(float(preds["correct"].mean()), 4),
        "unparsed_rate": round(float(preds["pred"].isna().mean()), 4),
        "by_subject": group("subject"),
        "by_category": group("category"),
    }


def save_metrics(metrics: dict, path):
    out = {k: (v.reset_index().to_dict(orient="records") if isinstance(v, pd.DataFrame) else v)
           for k, v in metrics.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
