"""Inference backends. Both expose generate(list_of_message_lists) -> list[str]."""

import time
from concurrent.futures import ThreadPoolExecutor


class OpenAICompatBackend:
    """Any OpenAI-compatible endpoint: OpenRouter, Together, DashScope, a vLLM server, etc."""

    def __init__(self, model, base_url, api_key, max_tokens=8, temperature=0.0,
                 max_workers=8, max_retries=6, timeout=120):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens, self.temperature = max_tokens, temperature
        self.max_workers, self.max_retries = max_workers, max_retries

    def _one(self, messages):
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=self.max_tokens, temperature=self.temperature,
                )
                return r.choices[0].message.content or ""
            except Exception as e:  # rate limits, timeouts, 5xx
                if attempt == self.max_retries - 1:
                    print(f"[giving up] {type(e).__name__}: {e}")
                    return ""
                time.sleep(min(2 ** attempt, 60))

    def generate(self, batch):
        with ThreadPoolExecutor(self.max_workers) as ex:
            return list(ex.map(self._one, batch))


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
