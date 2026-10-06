"""Local Hugging Face models: option scoring for evaluation, and LoRA fine-tuning helpers."""

import torch

from .prompting import build_messages

# Fixed so prompts are identical across days (Llama 3.x templates insert today's date otherwise).
TEMPLATE_DATE = "26 Jul 2024"


def load_model_and_tokenizer(model_id, load_in_4bit=True, tokenizer_id=None, dtype=None):
    """dtype defaults to bf16 where supported, else fp16. Pass torch.float32 for models that
    overflow in fp16 (Gemma 3) on GPUs without bf16, such as the T4."""
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


def build_lm_examples(tok, texts, seq_len=512):
    """Continued-pretraining examples: texts joined with EOS and cut into seq_len chunks, each
    starting with BOS. Loss is on every token, so the answer-only trainer and Collator still apply."""
    ids = []
    for t in texts:
        ids += _ids(tok, t) + [tok.eos_token_id]
    body = seq_len - 1
    examples = []
    for i in range(0, len(ids), body):
        chunk = ids[i:i + body]
        if len(chunk) < body // 4:          # drop a short tail
            break
        examples.append({"input_ids": [tok.bos_token_id] + chunk, "labels": [-100] + chunk})
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
    """Trainer callback: after each epoch, measure val accuracy and keep the best adapter."""
    from transformers import TrainerCallback

    class ValAccuracy(TrainerCallback):
        def __init__(self):
            self.best, self.history = -1.0, []

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
            print(msg)

    return ValAccuracy()
