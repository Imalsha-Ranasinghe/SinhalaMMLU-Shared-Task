# SinhalaMMLU-Shared-Task-
An 8B-parameter constrained system for answering Sinhala multiple-choice questions across Easy, Medium, and Hard difficulty levels.

## Project layout

```
Dev Set/                         raw JSON files (one per subject)
data/splits/                     train.jsonl / val.jsonl / test.jsonl  (70/15/15, stratified by subject)
notebooks/
  01_data_split.ipynb            load, clean, split
  02_evaluate_qwen2.5_72b.ipynb  evaluate Qwen2.5-72B-Instruct on val/test
src/
  data.py                        loading, validation, splitting
  prompting.py                   prompt template, few-shot sampling, answer parsing
  backends.py                    OpenAI-compatible API backend and vLLM backend
  evaluate.py                    resumable eval loop and metrics
results/<run_name>/              *_predictions.jsonl, *_metrics.json, summary.csv
```

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env             # then add your API key
jupyter lab
```

1. Run `notebooks/01_data_split.ipynb` to write `data/splits/`.
2. Edit the config cell in `notebooks/02_evaluate_qwen2.5_72b.ipynb` (backend, splits, `N_SHOTS`, `LIMIT`), then run all cells.
   Set `LIMIT = 20` first as a cheap smoke test. If a run is interrupted, re-run and it resumes.
