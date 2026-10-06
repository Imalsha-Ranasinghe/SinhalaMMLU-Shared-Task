"""Inference backends. Both expose generate(list_of_message_lists) -> list[str]."""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_THOUGHT = re.compile(r"<(thought|think)>.*?</\1>", re.S)


def strip_thoughts(text: str) -> str:
    """Remove reasoning blocks some models (Gemma 4, Qwen3, ...) put in the reply text.
    A block that never closes means the token limit ran out mid-thought: no answer, return ""."""
    text = _THOUGHT.sub("", text)
    if re.search(r"<(thought|think)>", text):
        return ""
    return text.strip()


class OpenAICompatBackend:
    """Any OpenAI-compatible endpoint: OpenRouter, Together, DashScope, a vLLM server, etc.

    requests_per_minute spaces out request starts across all threads (OpenRouter's free
    models allow 20/min). A rate-limited request waits at least rate_limit_wait seconds,
    growing with each retry, so the per-minute window can reset.
    """

    def __init__(self, model, base_url, api_key, max_tokens=8, temperature=0.0,
                 max_workers=8, max_retries=6, timeout=120, requests_per_minute=None, rate_limit_wait=20):
        from openai import OpenAI

        # max_retries=0: the client would otherwise silently retry timeouts itself, on top of _one's
        # loop, so one hung request could block a worker for hours.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        self.max_tokens, self.temperature = max_tokens, temperature
        self.max_workers, self.max_retries = max_workers, max_retries
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self.rate_limit_wait = rate_limit_wait
        self._lock, self._next_start = threading.Lock(), 0.0
        self.daily_limit_hit = False

    def _wait_turn(self):
        if not self.min_interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_start)
            self._next_start = start + self.min_interval
        time.sleep(start - now)

    def _one(self, messages):
        import openai

        for attempt in range(self.max_retries):
            if self.daily_limit_hit:
                return ""
            self._wait_turn()
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=self.max_tokens, temperature=self.temperature,
                )
                return strip_thoughts(r.choices[0].message.content or "")
            except (openai.AuthenticationError, openai.PermissionDeniedError,
                    openai.NotFoundError, openai.BadRequestError) as e:
                print(f"[not retrying] {type(e).__name__}: {e}")   # wrong key/model/request: retrying won't help
                return ""
            except openai.RateLimitError as e:
                if "per-day" in str(e):
                    if not self.daily_limit_hit:
                        print("[stopping] daily request limit reached; rerun after it resets "
                              "(finished replies are kept if the caller caches them)")
                    self.daily_limit_hit = True
                    return ""
                if attempt == self.max_retries - 1:
                    print(f"[giving up] RateLimitError: {str(e)[:160]}")
                    return ""
                wait = self.rate_limit_wait * (attempt + 1)
                print(f"[rate limited] waiting {wait}s: {str(e)[:120]}")
                time.sleep(wait)
            except Exception as e:  # timeouts, 5xx
                if attempt == self.max_retries - 1:
                    print(f"[giving up] {type(e).__name__}: {e}")
                    return ""
                print(f"[retry {attempt + 1}] {type(e).__name__}: {str(e)[:100]}")
                time.sleep(min(2 ** attempt, 60))

    def generate(self, batch):
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(self._one, batch))

    def generate_iter(self, batch):
        """Yield (index, reply) as each request finishes, so slow requests don't hold back the rest
        and callers can save every reply as soon as it arrives."""
        from concurrent.futures import as_completed

        with ThreadPoolExecutor(self.max_workers) as ex:
            futures = {ex.submit(self._one, m): i for i, m in enumerate(batch)}
            for f in as_completed(futures):
                yield futures[f], f.result()


class VLLMBackend:
    """Local GPU inference with vLLM (Linux + NVIDIA GPU)."""

    def __init__(self, model, tensor_parallel_size=1, max_model_len=4096,
                 gpu_memory_utilization=0.92, quantization=None, max_tokens=8, temperature=0.0):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model, tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            quantization=quantization,
        )
        self.params = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def generate(self, batch):
        outs = self.llm.chat(batch, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]
