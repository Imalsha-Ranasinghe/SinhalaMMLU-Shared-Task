# SinhalaMMLU-Shared-Task-
An 8B-parameter constrained system for answering Sinhala multiple-choice questions across Easy, Medium, and Hard difficulty levels.

## Project layout

```
Dev Set/                         raw JSON files (one per subject)
data/splits/                     train.jsonl / val.jsonl / test.jsonl  (70/15/15, stratified by subject)
notebooks/
  01_data_split.ipynb            load, clean, split
  02_evaluate_qwen2.5_72b.ipynb  evaluate Qwen2.5-72B-Instruct on val/test (reference only, over the 8B limit)
  03_llama3.2_3b_lora.ipynb      Llama-3.2-3B-Instruct: baseline, QLoRA fine-tuning, final eval (GPU)
  04_gemma3_4b_lora.ipynb        Gemma-3-4B-it: same pipeline, better Sinhala tokenizer (GPU)
src/
  data.py                        loading, validation, splitting
  prompting.py                   prompt template, few-shot sampling, answer parsing
  backends.py                    OpenAI-compatible API backend and vLLM backend
  evaluate.py                    resumable eval loop and metrics
  hf_model.py                    local HF models: option scoring, LoRA training helpers
outputs/                         trained LoRA adapters (not committed)
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

### Llama-3.2-3B-Instruct (notebook 03, needs a GPU)

1. Accept the licence at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct and create an HF token.
2. Push this repo to GitHub, open `notebooks/03_llama3.2_3b_lora.ipynb` in Colab (File → Open notebook → GitHub)
   or Kaggle, choose a T4 GPU runtime, and add the token as a secret named `HF_TOKEN`.
3. Run all cells. The notebook clones the repo, installs `requirements-hf.txt`, runs the baseline, fine-tunes, and evaluates.
4. Results, checkpoints and the adapter are saved to Google Drive (`MyDrive/SinhalaMMLU-runs/`) on Colab, or `/kaggle/working` on Kaggle.
   If the session is cut off, run the notebook again: finished evaluations are skipped and training resumes from the last epoch.
