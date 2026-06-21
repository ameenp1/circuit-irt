"""Backend-agnostic respondent models for the inference runner.

  vllm : GPU (RunPod) — batched, n-samples-per-item, AWQ/GPTQ/bnb quantization.
  hf   : transformers — works on Mac (CPU/MPS) for smoke tests, or GPU.
  mock : deterministic synthetic completions (no model) — fast runner tests.

Every backend implements `complete(items, n) -> list[list[str]]` ([item][sample]).
load_respondent(spec) dispatches on spec["backend"] (auto-detects if absent).
Reasoning traces (<think>…</think>) are stripped by the shared parser before
scoring, so R1-distill / QwQ outputs flow through unchanged.
"""
from __future__ import annotations

import json
import os
import random
import re

# vLLM engine workers must spawn, not fork (avoids "Cannot re-initialize CUDA in
# forked subprocess"). Set before vllm is imported anywhere.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from circuit_irt.respondent import build_prompt, LocalModel


class HFModel:
    """transformers backend (Mac-friendly). Sequential — for smoke tests."""
    def __init__(self, spec: dict):
        self.lm = LocalModel(spec["id"], max_new_tokens=spec.get("max_new_tokens", 512),
                             temperature=spec.get("temperature", 0.0))

    def complete(self, items, n):
        return [[self.lm.generate(build_prompt(it)) for _ in range(n)] for it in items]


class VLLMModel:
    """vLLM backend (GPU). Batches all prompts; n samples per prompt in one pass."""
    def __init__(self, spec: dict):
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
        kw = dict(model=spec["id"], dtype="auto", trust_remote_code=True,
                  gpu_memory_utilization=spec.get("gpu_mem_util", 0.90))
        if spec.get("quantization"):
            kw["quantization"] = spec["quantization"]      # awq | gptq | bitsandbytes
        if spec.get("max_model_len"):
            kw["max_model_len"] = spec["max_model_len"]
        if spec.get("tp"):
            kw["tensor_parallel_size"] = spec["tp"]        # multi-GPU
        if spec.get("enforce_eager"):
            kw["enforce_eager"] = True                     # skip CUDA-graph compile (fast smoke)
        self.llm = LLM(**kw)
        self.tok = AutoTokenizer.from_pretrained(spec["id"], trust_remote_code=True)
        self.SamplingParams = SamplingParams
        self.spec = spec

    def complete(self, items, n):
        prompts = [self.tok.apply_chat_template(build_prompt(it), tokenize=False,
                                                add_generation_prompt=True) for it in items]
        sp = self.SamplingParams(n=n, temperature=self.spec.get("temperature", 0.7),
                                 top_p=self.spec.get("top_p", 0.95),
                                 max_tokens=self.spec.get("max_new_tokens", 512))
        outs = self.llm.generate(prompts, sp)
        return [[o.text for o in out.outputs] for out in outs]


def _perturb(netlist: str, rng: random.Random) -> str:
    """Scale the first value-like number (not an identifier digit) to break spec."""
    factor = rng.choice([0.2, 0.3, 3.0, 5.0])
    return re.sub(r"(?<![A-Za-z0-9.])(\d+\.?\d*)([kKmMuUnNpP]?)(?![A-Za-z0-9])",
                  lambda m: f"{float(m.group(1)) * factor:g}{m.group(2)}", netlist, count=1)


class MockModel:
    """No model: emits the reference (pass), a perturbed reference (fail), or a
    refusal (parse_failure), mixed by `skill`. Gives a matrix with real variance."""
    def __init__(self, spec: dict):
        self.skill = spec.get("skill", 0.5)
        self.rng = random.Random(spec.get("seed", 0))

    def complete(self, items, n):
        res = []
        for it in items:
            samples = []
            for _ in range(n):
                r = self.rng.random()
                if r < self.skill:
                    samples.append(json.dumps({"netlist": it["reference_netlist"]}))
                elif r < self.skill + 0.7 * (1 - self.skill):
                    samples.append(json.dumps({"netlist": _perturb(it["reference_netlist"], self.rng)}))
                else:
                    samples.append("I'm not able to design this circuit.")
            res.append(samples)
        return res


_BACKENDS = {"hf": HFModel, "vllm": VLLMModel, "mock": MockModel}


def _auto_backend() -> str:
    # Detect a GPU WITHOUT importing torch / initializing CUDA in this parent
    # process (that init is what breaks vLLM's worker startup).
    import importlib.util
    import shutil
    has_gpu = (os.path.exists("/proc/driver/nvidia/version")
               or bool(shutil.which("nvidia-smi")))
    if has_gpu and importlib.util.find_spec("vllm") is not None:
        return "vllm"
    return "hf"


def load_respondent(spec: dict):
    return _BACKENDS[spec.get("backend") or _auto_backend()](spec)
