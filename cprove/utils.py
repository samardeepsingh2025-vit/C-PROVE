"""Paths, config, seeding, caching, and GPU memory management.

Memory policy for an 8 GB laptop GPU + 16 GB RAM:
  * only one large model is resident at a time (ModelSlot.swap)
  * every stage writes its output to cache/ so the next stage can start after
    the previous model is unloaded
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT_MARKERS = ("pyproject.toml", "configs")


def resolve_project_root(start: str | os.PathLike | None = None) -> Path:
    """Walk up until a folder containing pyproject.toml + configs/ is found.
    Mirrors the notebook helper, but works from any sub-folder (notebooks/, scripts/)."""
    env = os.environ.get("CPROVE_ROOT")
    if env:
        return Path(env)
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if all((cand / m).exists() for m in ROOT_MARKERS):
            return cand
    return Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    root = resolve_project_root()
    path = Path(path) if path else root / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        _deep_update(cfg, overrides)
    # make every path absolute relative to root (handles 'D:\\VIT\\2nd year\\meeting  summarisation')
    for k, v in cfg.get("paths", {}).items():
        pv = Path(v)
        cfg["paths"][k] = str(pv if pv.is_absolute() else (root / pv))
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    return cfg


def _deep_update(d: dict, u: dict):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


class ModelSlot:
    """Keeps at most one heavy model resident. `get(name, loader)` unloads the
    previous model if the name differs."""

    def __init__(self):
        self.name = None
        self.obj = None

    def get(self, name: str, loader: Callable[[], Any]):
        if self.name != name:
            self.release()
            self.obj = loader()
            self.name = name
        return self.obj

    def release(self):
        self.obj = None
        self.name = None
        free_gpu()


SLOT = ModelSlot()


def stable_hash(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, ensure_ascii=False, default=str).encode())
    return h.hexdigest()[:16]


class JsonCache:
    """Tiny append-only key->value cache (one JSONL file). Used for LLM calls and
    model scores so re-runs are free and experiments are reproducible."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem: dict[str, Any] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        self.mem[r["k"]] = r["v"]
                    except Exception:
                        continue

    def get(self, key):
        return self.mem.get(key)

    def put(self, key, value):
        self.mem[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")

    def __contains__(self, key):
        return key in self.mem
