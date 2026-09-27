"""Access to the pinned official CLM release (https://github.com/Contrastive-LM/CLM).

The official code is not vendored or modified.  ``tools/setup_clm.sh`` clones it
at ``CLM_COMMIT`` into ``external/CLM``; ``CLM_ROOT`` overrides the location.
"""
from __future__ import annotations

import os
import subprocess
import sys

CLM_REPO = "https://github.com/Contrastive-LM/CLM"
CLM_COMMIT = "bb42c6c5bf914fd449bed2f6ca65be80602cb1f7"   # checked 2026-09-27
HF_MODEL_REPO = "Contrastive-LM/CLM-v0.1-8B"
HF_HEAD_FILE = "CLM_v0.1-8B.pt"
EMBED_MODEL = "Qwen/Qwen3-8B"

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "external", "CLM")


def root() -> str:
    return os.environ.get("CLM_ROOT", DEFAULT_ROOT)


def available() -> bool:
    return os.path.isfile(os.path.join(root(), "src", "clm", "schema.py"))


def ensure_path() -> str:
    """Put the official ``src`` and ``train`` directories on ``sys.path``; returns the root."""
    r = root()
    if not available():
        raise RuntimeError(f"official CLM not found at {r}; run tools/setup_clm.sh or set CLM_ROOT")
    for sub in ("src", "train", "preprocessing"):
        p = os.path.join(r, sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    return r


def checked_out_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", root(), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def assert_pinned() -> None:
    got = checked_out_commit()
    if got != CLM_COMMIT:
        raise RuntimeError(f"official CLM at {root()} is {got}, expected pinned {CLM_COMMIT}")


def schema():
    ensure_path()
    import clm.schema as s
    return s


def finetune_script() -> str:
    return os.path.join(root(), "train", "finetune.py")


def embed_cache_name(embed_model: str, max_len: int) -> str:
    """File name the official choice trainer uses for its text-embedding cache (``train/finetune.py``)."""
    import re
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", embed_model).strip("_")
    return f"choice_{slug}_{max_len}.npz"
