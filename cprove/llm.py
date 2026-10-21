"""Local LLM backends with schema-constrained JSON output + disk cache.

Backends
  llamacpp       llama-cpp-python, loads a GGUF in-process (fastest on Windows, CUDA wheels)
  ollama         HTTP to a running Ollama server (`format=<json schema>`)
  openai_compat  HTTP to llama.cpp `llama-server` (or LM Studio) with response_format json_schema
  heuristic      no LLM; callers fall back to rule-based code (CPU smoke tests / ablations)

All calls are deterministic (temperature 0) and cached by (model, prompt, schema),
so a paper experiment can be re-run byte-for-byte without the GPU.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .utils import JsonCache, stable_hash, SLOT


class LLM:
    name = "base"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cache = JsonCache(cfg.get("cache_file", "cache/llm_cache.jsonl"))

    # ------------------------------------------------------------------ public
    def json(self, system: str, user: str, schema: dict, max_tokens: Optional[int] = None) -> Optional[dict]:
        key = stable_hash(self.name, self.model_id(), system, user, schema)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        raw = self._generate(system, user, schema, max_tokens or self.cfg.get("max_tokens", 768))
        if raw is None:
            return None
        out = parse_json(raw)
        if out is not None:
            self.cache.put(key, out)
        return out

    def text(self, system: str, user: str, max_tokens: int = 512) -> Optional[str]:
        key = stable_hash(self.name, self.model_id(), system, user, "TEXT")
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        out = self._generate(system, user, None, max_tokens)
        if out is not None:
            self.cache.put(key, out)
        return out

    # ---------------------------------------------------------------- backend
    def model_id(self) -> str:
        return "none"

    def _generate(self, system, user, schema, max_tokens) -> Optional[str]:
        raise NotImplementedError

    @property
    def available(self) -> bool:
        return True


class HeuristicLLM(LLM):
    name = "heuristic"

    def _generate(self, system, user, schema, max_tokens):
        return None

    @property
    def available(self):
        return False


class LlamaCppLLM(LLM):
    name = "llamacpp"

    def model_id(self):
        return Path(self.cfg["gguf_path"]).name

    def _load(self):
        from llama_cpp import Llama  # pip install llama-cpp-python (CUDA wheel, see README)
        return Llama(
            model_path=self.cfg["gguf_path"],
            n_ctx=self.cfg.get("n_ctx", 8192),
            n_gpu_layers=self.cfg.get("n_gpu_layers", -1),
            flash_attn=True,
            verbose=False,
        )

    def _generate(self, system, user, schema, max_tokens):
        llm = SLOT.get("llm:" + self.model_id(), self._load)
        kw = dict(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.cfg.get("temperature", 0.0),
            max_tokens=max_tokens,
            seed=42,
        )
        if schema is not None:
            kw["response_format"] = {"type": "json_object", "schema": schema}
        out = llm.create_chat_completion(**kw)
        return out["choices"][0]["message"]["content"]


class OllamaLLM(LLM):
    name = "ollama"

    def model_id(self):
        return self.cfg.get("ollama_model", "qwen2.5:7b-instruct")

    def _generate(self, system, user, schema, max_tokens):
        import requests
        body = {
            "model": self.model_id(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "options": {"temperature": self.cfg.get("temperature", 0.0), "seed": 42, "num_predict": max_tokens,
                        "num_ctx": self.cfg.get("n_ctx", 8192)},
        }
        if schema is not None:
            body["format"] = schema
        r = requests.post(self.cfg.get("ollama_url", "http://localhost:11434") + "/api/chat", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]


class OpenAICompatLLM(LLM):
    name = "openai_compat"

    def model_id(self):
        return self.cfg.get("openai_model", "local")

    def _generate(self, system, user, schema, max_tokens):
        import requests
        body = {
            "model": self.model_id(),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.cfg.get("temperature", 0.0),
            "max_tokens": max_tokens,
            "seed": 42,
        }
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": True}}
        r = requests.post(self.cfg["openai_compat_url"].rstrip("/") + "/chat/completions", json=body, timeout=600)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def make_llm(cfg: dict, gguf_path: Optional[str] = None) -> LLM:
    c = dict(cfg)
    if gguf_path:
        c["gguf_path"] = gguf_path
    backend = c.get("backend", "heuristic")
    if backend == "llamacpp":
        if not Path(c["gguf_path"]).exists():
            print(f"[cprove] GGUF not found at {c['gguf_path']} -> heuristic fallback")
            return HeuristicLLM(c)
        try:
            import llama_cpp  # noqa: F401
        except ImportError:
            print("[cprove] llama-cpp-python not installed -> heuristic fallback")
            return HeuristicLLM(c)
        return LlamaCppLLM(c)
    return {"ollama": OllamaLLM, "openai_compat": OpenAICompatLLM}.get(backend, HeuristicLLM)(c)


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(raw: str):
    """Robust JSON parse: strips code fences and trailing prose."""
    if raw is None:
        return None
    s = raw.strip()
    m = _FENCE.search(s)
    if m:
        s = m.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    return None
