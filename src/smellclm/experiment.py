"""Immutable experiment directories, resumable stages, and environment capture (plan section 15)."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import subprocess
import sys

STAGES = (
    "01_environment", "02_dataset_audit", "03_split", "04_embeddings", "05_baselines_zero_shot",
    "06_finetune", "07_validation_selection", "08_locked_test", "09_error_analysis",
    "10_transfer_ablations", "11_export",
)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


class Experiment:
    """``<root>/<UTC timestamp>-<config hash>/``; reopening an existing dir resumes it.

    A stage is complete when ``stages/<name>.done.json`` exists.  ``write`` refuses to
    overwrite files unless ``overwrite=True`` so completed outputs stay immutable.
    """

    def __init__(self, root: str, cfg: dict, resume: str | None = None):
        h = config_hash(cfg)
        if resume:
            self.dir = resume if os.path.isabs(resume) else os.path.join(root, resume)
            saved = json.load(open(os.path.join(self.dir, "config.json")))
            if config_hash(saved) != h:
                raise RuntimeError(f"config changed since {self.dir} was created; start a new experiment")
        else:
            stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.dir = os.path.join(root, f"{stamp}-{h}")
            os.makedirs(os.path.join(self.dir, "stages"), exist_ok=False)
            os.makedirs(os.path.join(self.dir, "figures"), exist_ok=True)
            self.write("config.json", json.dumps(cfg, indent=2, sort_keys=True))
        self.cfg, self.hash = cfg, h

    def path(self, *parts: str) -> str:
        p = os.path.join(self.dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def write(self, name: str, text: str, overwrite: bool = False) -> str:
        p = self.path(name)
        if os.path.exists(p) and not overwrite:
            raise FileExistsError(f"{p} exists; outputs are immutable")
        with open(p, "w") as f:
            f.write(text)
        return p

    def write_json(self, name: str, obj, overwrite: bool = False) -> str:
        return self.write(name, json.dumps(obj, indent=2, sort_keys=True, default=str), overwrite)

    def done(self, stage: str) -> bool:
        return os.path.exists(os.path.join(self.dir, "stages", f"{stage}.done.json"))

    def mark(self, stage: str, info: dict | None = None) -> None:
        assert stage in STAGES, stage
        self.write_json(os.path.join("stages", f"{stage}.done.json"),
                        {"stage": stage, "at": _dt.datetime.now(_dt.timezone.utc).isoformat(), **(info or {})})

    def log(self, msg: str) -> None:
        line = f"{_dt.datetime.now(_dt.timezone.utc).isoformat()} {msg}"
        print(line, flush=True)
        with open(self.path("run.log"), "a") as f:
            f.write(line + "\n")


def environment() -> dict:
    """Python, packages, OS, GPU/CUDA and driver facts for ``environment.json``."""
    env = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    pkgs = {}
    for name in ("numpy", "pandas", "pyarrow", "torch", "transformers", "vllm", "huggingface_hub"):
        try:
            mod = __import__(name)
            if name == "vllm":
                # Top-level vllm is lazy; this import exposes missing CUDA libs.
                from vllm.inputs import TokensPrompt  # noqa: F401
            pkgs[name] = getattr(mod, "__version__", "?")
        except Exception as e:           # noqa: BLE001 - record, never fail
            pkgs[name] = f"unavailable: {type(e).__name__}"
    env["packages"] = pkgs
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = {"name": p.name, "vram_gb": round(p.total_memory / 2**30, 2),
                          "capability": f"{p.major}.{p.minor}", "count": torch.cuda.device_count()}
    except Exception as e:               # noqa: BLE001
        env["cuda_available"] = False
        env["torch_error"] = repr(e)
    try:
        env["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        env["nvidia_smi"] = None
    try:
        env["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        env["pip_freeze"] = None
    return env


def feasibility(env: dict, min_vram_gb: float = 22.0) -> dict:
    """Gate A hardware precheck: Linux + NVIDIA GPU with room for Qwen3-8B bf16 (~16 GB weights + KV)."""
    reasons = []
    if not env.get("platform", "").lower().startswith("linux"):
        reasons.append("official vLLM encoder requires Linux")
    if not env.get("cuda_available"):
        reasons.append("no CUDA GPU visible")
    elif env.get("gpu", {}).get("vram_gb", 0) < min_vram_gb:
        reasons.append(f"GPU has {env['gpu']['vram_gb']} GB; need >= {min_vram_gb} GB for Qwen3-8B bf16 via vLLM")
    if str(env.get("packages", {}).get("vllm", "")).startswith("unavailable"):
        reasons.append("vllm not importable")
    return {"pass": not reasons, "reasons": reasons, "min_vram_gb": min_vram_gb}
