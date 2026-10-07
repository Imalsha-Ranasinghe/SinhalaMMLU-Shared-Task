# SinhalaMMLU-Shared-Task-
An 8B-parameter constrained system for answering Sinhala multiple-choice questions across Easy, Medium, and Hard difficulty levels.

## Project layout

```
Dev Set/                         raw JSON files (one per subject)
data/splits/                     train.jsonl / val.jsonl / test.jsonl  (70/15/15, stratified by subject)
notebooks/
  00_dev_set_analysis.ipynb      dataset analysis: composition, provenance, quality audit, recommendations
  01_data_split.ipynb            load, clean, split
  02_evaluate_qwen2.5_72b.ipynb  evaluate Qwen2.5-72B-Instruct on val/test (reference only, over the 8B limit)
  03_llama3.2_3b_lora.ipynb      Llama-3.2-3B-Instruct: baseline, QLoRA fine-tuning, final eval (GPU)
  04_gemma3_textbook_pilot.ipynb Gemma-3-4B: does textbook training help? one-subject pilot (GPU)
src/
  data.py                        loading, validation, splitting
  prompting.py                   prompt template, few-shot sampling, answer parsing
  backends.py                    OpenAI-compatible API backend and vLLM backend
  evaluate.py                    resumable eval loop and metrics
  hf_model.py                    local HF models: option scoring, LoRA training helpers
  textbooks.py                   textbook PDF (legacy FM fonts) -> clean Unicode Sinhala text
  synth.py                       textbook passages -> paraphrases + grounded MCQs (generator LLM)
  papers.py                      past exam paper PDFs -> MCQs in Dev Set format (vision LLM)
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

### Textbook pilot with Gemma 3 4B (notebook 04, needs a GPU)

1. Textbook text (already done for grade 8 Christianity): `python -m src.textbooks data/textbooks/christianity_g8_si.pdf`
2. Generated training data. Set `GEN_MODEL` and an API key in `.env`, try a few passages first, then run all:
   ```bash
   python -m src.synth data/textbooks/christianity_g8_si.jsonl --subject Christianity --grade 8 --out data/synthetic/christianity_g8 --limit 3
   python -m src.synth data/textbooks/christianity_g8_si.jsonl --subject Christianity --grade 8 --out data/synthetic/christianity_g8
   ```
   Read a sample of `data/synthetic/christianity_g8_mcq.jsonl` before training. A closed-API generator must be disclosed (rule 04).
3. Commit and push `data/textbooks/*.jsonl` and `data/synthetic/`, accept the Gemma 3 licence on Hugging Face,
   then run `notebooks/04_gemma3_textbook_pilot.ipynb` on Colab (L4/A100 is fastest; a T4 works in float32).

### More MCQs from past papers (all grades)

pastpapers.wiki forbids bots, scrapers and download tools, so papers are **downloaded by hand in a
browser**. Nothing in this repo fetches from that site. The fast part is extraction, which is automated.
**What to download, and who downloads what: [DATA_COLLECTION.md](DATA_COLLECTION.md).**

1. Download papers, keeping the original file names, into one folder per subject, named as in DATA_COLLECTION.md:
   `data/raw_pdfs/History/2023-Grade-09-History-3rd-Term-Test-Paper-with-Answers-Southern-Province.pdf`.
   Year, grade (O/L = 11, A/L = 13), term and province are read from the file name. A separate marking scheme is saved as `<paper name>_answers.pdf`.
   - Prefer **"with Answers"** papers: questions without an answer from the paper's own key are dropped.
   - Prefer **typed** papers (text can be selected in the PDF viewer) over scans: they are transcribed exactly.
   - Don't bother checking against the Dev Set: papers it came from are skipped automatically.
   - Optional: `data/raw_pdfs/sources.csv` with columns `file,url` records each file's page URL.
2. Try two papers, read `data/papers/json/*.json`, then run everything (resumable; rerun after a quota stop):
   ```bash
   python -m src.papers data/raw_pdfs --limit 2
   python -m src.papers data/raw_pdfs
   ```
3. Check `data/papers/report.json` (per-paper counts, rejection reasons) and read a sample of the
   scanned-paper questions (`metadata.text_layer: false`): vision-only Sinhala reading makes mistakes.
   Load the result with `src.data.load_raw("data/papers/json")`.
