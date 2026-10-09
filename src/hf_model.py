"""Local Hugging Face models: option scoring for evaluation, and LoRA fine-tuning helpers."""

import torch

from .prompting import build_messages

# Fixed so prompts are identical across days (Llama 3.x templates insert today's date otherwise).
TEMPLATE_DATE = "26 Jul 2024"


def load_model_and_tokenizer(model_id, load_in_4bit=True, tokenizer_id=None, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(tokenizer_id or model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
    ) if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant, dtype=dtype, device_map="auto",
    )
    if len(tok) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tok))
    return model, tok


def load_with_dtype_check(model_id, probe_messages, load_in_4bit=True, tokenizer_id=None):
    """Load in bf16 if the GPU supports it. Otherwise try fp16 and fall back to fp32 if it overflows.

    Some models (notably Gemma) produce inf/NaN activations in fp16, which silently turns every
    prediction into option 1. T4 and P100 GPUs have no bf16, so this matters on free Colab/Kaggle.
    """
    import gc

    candidates = [torch.bfloat16] if torch.cuda.is_bf16_supported() else [torch.float16, torch.float32]
    for dtype in candidates:
        model, tok = load_model_and_tokenizer(model_id, load_in_4bit, tokenizer_id, dtype)
        try:
            ChoiceScorer(model, tok).generate(probe_messages)
            print(f"Using {dtype}")
            return model, tok, dtype
        except FloatingPointError as e:
            print(f"{dtype} failed ({e}); trying the next dtype")
            del model
            gc.collect()
            torch.cuda.empty_cache()
    raise RuntimeError("No dtype produced finite logits")


def render_prompt(tok, messages, answer=None):
    """Prompt text (ending where the answer goes); with answer, the full training text."""
    if tok.chat_template:
        if answer is not None:
            messages = messages + [{"role": "assistant", "content": str(answer)}]
        return tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None, date_string=TEMPLATE_DATE,
        )
    # Base models without a chat template: plain few-shot text, "Answer: N" after each shot.
    parts = []
    for m in messages:
        if m["role"] == "assistant":
            parts[-1] += " " + m["content"]
        else:
            parts.append(m["content"])
    text = "\n\n".join(parts)
    return text + (f" {answer}{tok.eos_token}" if answer is not None else "")


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def option_token_ids(tok, n_options=4):
    """Token id of each option number as it appears right after the prompt."""
    probe = render_prompt(tok, build_messages({"question": "q", "choices": ["a"] * n_options}))
    base = _ids(tok, probe)
    out = []
    for d in range(1, n_options + 1):
        full = _ids(tok, render_prompt(tok, build_messages({"question": "q", "choices": ["a"] * n_options}), d))
        assert full[:len(base)] == base, "prompt is not a prefix of prompt+answer"
        out.append(full[len(base)])
    return out


def forward_tail(model, input_ids, attention_mask, k):
    """Logits for the last k positions only (computing all 128k-vocab logits per token wastes GBs).

    Inputs are left-padded, so position ids are derived from the attention mask.
    """
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    args = dict(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
    for kw in ({"logits_to_keep": k}, {"num_logits_to_keep": k}, {}):
        try:
            return model(**args, **kw).logits[:, -k:, :]
        except TypeError:
            continue


class ChoiceScorer:
    """Backend for src.evaluate.run_eval: picks the option number with the highest next-token logit.

    No free-form generation, so there are no unparseable answers.
    """

    def __init__(self, model, tok, n_options=4, max_tokens_per_batch=12000):
        self.model, self.tok = model, tok
        self.option_ids = option_token_ids(tok, n_options)
        self.max_tokens_per_batch = max_tokens_per_batch

    @torch.no_grad()
    def generate(self, batch):
        was_training = self.model.training
        self.model.eval()
        seqs = [_ids(self.tok, render_prompt(self.tok, m)) for m in batch]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        preds = [None] * len(seqs)
        i = 0
        while i < len(order):
            # Grow the micro-batch while padded size stays under the token budget.
            j = i + 1
            while j < len(order) and (j - i + 1) * len(seqs[order[j]]) <= self.max_tokens_per_batch:
                j += 1
            chunk = order[i:j]
            enc = self.tok.pad({"input_ids": [seqs[k] for k in chunk]}, return_tensors="pt").to(self.model.device)
            logits = forward_tail(self.model, enc["input_ids"], enc["attention_mask"], 1)[:, -1, self.option_ids].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("non-finite logits (fp16 overflow?)")
            for k, p in zip(chunk, logits.argmax(-1).tolist()):
                preds[k] = str(p + 1)
            i = j
        if was_training:
            self.model.train()
        return preds


def build_train_examples(tok, df, shot_sampler=None, max_len=1536):
    """Tokenize (prompt, answer) pairs; loss is only on the answer tokens."""
    examples, skipped = [], 0
    for r in df.to_dict(orient="records"):
        msgs = build_messages(r, shot_sampler(r) if shot_sampler else [])
        p = _ids(tok, render_prompt(tok, msgs))
        f = _ids(tok, render_prompt(tok, msgs, r["answer"]))
        if len(f) > max_len:
            skipped += 1
            continue
        examples.append({"input_ids": f, "labels": [-100] * len(p) + f[len(p):]})
    if skipped:
        print(f"Skipped {skipped} examples longer than {max_len} tokens")
    return examples


class Collator:
    """Left-pads, so every example's answer tokens line up at the end of the batch."""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        n = max(len(f["input_ids"]) for f in feats)
        pad = lambda xs, v: [v] * (n - len(xs)) + xs
        return {
            "input_ids": torch.tensor([pad(f["input_ids"], self.pad_id) for f in feats]),
            "attention_mask": torch.tensor([pad([1] * len(f["input_ids"]), 0) for f in feats]),
            "labels": torch.tensor([pad(f["labels"], -100) for f in feats]),
        }


def make_trainer_class():
    """Trainer whose loss only computes logits over the answer tokens at the end of each sequence."""
    import torch.nn.functional as F
    from transformers import Trainer

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            k = int((labels != -100).sum(1).max())
            logits = forward_tail(model, inputs["input_ids"], inputs["attention_mask"], k + 1)
            # logits at position t predict token t+1
            loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                   labels[:, -k:].reshape(-1), ignore_index=-100)
            return (loss, logits) if return_outputs else loss

    return AnswerOnlyTrainer


def make_val_accuracy_callback(scorer, val_df, save_dir, batch_size=64):
    """Trainer callback: after each epoch, measure val accuracy and keep the best adapter.

    Its state lives in save_dir/val_state.json, so a resumed run remembers earlier epochs.
    """
    import json
    from pathlib import Path
    from transformers import TrainerCallback

    state_file = Path(save_dir) / "val_state.json"

    class ValAccuracy(TrainerCallback):
        def __init__(self):
            self.best, self.history = -1.0, []
            if state_file.exists():
                saved = json.loads(state_file.read_text())
                self.best, self.history = saved["best"], saved["history"]

        def on_epoch_end(self, args, state, control, model=None, **kw):
            rows = val_df.to_dict(orient="records")
            correct = 0
            for i in range(0, len(rows), batch_size):
                chunk = rows[i:i + batch_size]
                preds = scorer.generate([build_messages(r) for r in chunk])
                correct += sum(int(p) == r["answer"] for p, r in zip(preds, chunk))
            acc = correct / len(rows)
            self.history.append({"epoch": round(state.epoch, 2), "val_accuracy": round(acc, 4)})
            msg = f"epoch {state.epoch:.2f}: val accuracy = {acc:.2%}"
            if acc > self.best:
                self.best = acc
                model.save_pretrained(save_dir)
                msg += f"  (best so far, saved to {save_dir})"
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps({"best": self.best, "history": self.history}))
            print(msg)

    return ValAccuracy()


LORA_TARGETS = r"^(?!.*vision).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def train_lora(model, tok, train_df, val_df, out_dir, *, ckpt_dir=None, r=16, alpha=None, dropout=0.05,
               targets=LORA_TARGETS, lr=1e-4, epochs=1, batch_size=8, grad_accum=2, max_len=512,
               aug_shuffles=1, seed=42, dtype=None, load_in_4bit=True, scorer_tokens=8000, extra_args=None,
               max_minutes=None):
    """LoRA fine-tuning on (question, answer) pairs, keeping the epoch with the best val accuracy.

    Returns (model with the best adapter active, val history, best val accuracy).
    If out_dir already holds a finished run, it is loaded instead of training again. With ckpt_dir,
    a checkpoint is saved every epoch and an interrupted run resumes from it.
    max_minutes: stop training early when time runs out (e.g. Kaggle's 12-hour limit); the best
    adapter so far is kept, and if no epoch finished, the adapter is scored and saved as it is.
    """
    import json
    import math
    from pathlib import Path

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments

    from .prompting import shuffle_augment

    out_dir = Path(out_dir)
    done_file = out_dir / "training_complete.json"
    if done_file.exists():
        saved = json.loads(done_file.read_text())
        print(f"Already trained: loading the best adapter from {out_dir}")
        return PeftModel.from_pretrained(model, str(out_dir)), saved["history"], saved["best"]

    torch.manual_seed(seed)
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout, target_modules=targets, task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    examples = build_train_examples(tok, shuffle_augment(train_df, aug_shuffles, seed=seed), max_len=max_len)
    steps = math.ceil(len(examples) / (batch_size * grad_accum)) * epochs
    print(f"{len(examples)} training examples, {steps} optimizer steps")

    settings = dict(
        output_dir=str(ckpt_dir or out_dir / "_tmp"),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.05 * steps)),
        logging_steps=10,
        save_strategy="epoch" if ckpt_dir else "no",
        save_total_limit=1,
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=seed,
    )
    args = TrainingArguments(**{**settings, **(extra_args or {})})   # extra_args override the defaults
    scorer = ChoiceScorer(model, tok, max_tokens_per_batch=scorer_tokens)
    val_cb = make_val_accuracy_callback(scorer, val_df, out_dir)
    callbacks = [val_cb]
    if max_minutes:
        import time
        from transformers import TrainerCallback

        deadline = time.time() + max_minutes * 60

        class TimeLimit(TrainerCallback):
            def on_step_end(self, args, state, control, **kw):
                if time.time() > deadline and not control.should_training_stop:
                    print(f"Time budget ({max_minutes:.0f} min) reached at epoch {state.epoch:.2f}: stopping early")
                    control.should_training_stop = True

        callbacks.append(TimeLimit())
    trainer = make_trainer_class()(
        model=model, args=args, train_dataset=examples, data_collator=Collator(tok.pad_token_id), callbacks=callbacks,
    )
    resume = bool(ckpt_dir) and any(Path(ckpt_dir).glob("checkpoint-*"))
    if resume:
        print("Resuming from the last saved epoch in", ckpt_dir)
    trainer.train(resume_from_checkpoint=True if resume else None)
    if val_cb.best < 0:                    # stopped before the first epoch ended: score and keep what we have
        val_cb.on_epoch_end(args, trainer.state, trainer.control, model=model)
    tok.save_pretrained(out_dir)

    # The model in memory is from the last epoch; switch to the best epoch's adapter.
    model.load_adapter(str(out_dir), adapter_name="best")
    model.set_adapter("best")
    model.config.use_cache = True
    done_file.write_text(json.dumps({"history": val_cb.history, "best": val_cb.best}))
    return model, val_cb.history, val_cb.best
