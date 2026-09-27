"""GPU stages: token audit, frozen-encoder embedding cache, and head scoring.

Embeddings use the official token recipe (``train/embed_utils.Recipe.text_ids``,
tail-kept, no special tokens) and backends (offline vLLM or a ``vllm serve
--runner pooling`` server), and are stored in the official trainer's cache
format (``sha1(text) -> float16`` in ``choice_<model>_<max_len>.npz``).  The
trainer therefore reuses the same vectors instead of re-embedding, and runtime
resets only lose work since the last flush.

Scores follow the official trainer exactly: ``exp(logit_scale).clamp(max=100) *
cos(state_head(s), action_head(c))``.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from . import official


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


class EmbeddingCache:
    """Read/write view of the official trainer's text-embedding cache file."""

    def __init__(self, path: str):
        self.path, self.vecs = path, {}
        if os.path.exists(path):
            z = np.load(path)
            self.vecs = dict(zip(z["keys"].tolist(), z["vecs"]))

    def missing(self, texts) -> list[str]:
        return [t for t in dict.fromkeys(texts) if sha1(t) not in self.vecs]

    def add(self, texts, vecs) -> None:
        self.vecs.update({sha1(t): np.asarray(v, dtype=np.float16) for t, v in zip(texts, vecs)})

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        ks = list(self.vecs)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, keys=np.array(ks), vecs=np.stack([self.vecs[k] for k in ks]).astype(np.float16))
        os.replace(tmp, self.path)

    def get(self, texts) -> np.ndarray:
        return np.stack([self.vecs[sha1(t)] for t in texts]).astype(np.float32)


def token_audit(dec: pd.DataFrame, max_len: int, embed_model: str = official.EMBED_MODEL) -> pd.DataFrame:
    """Token counts per state/candidate text and whether the official recipe would truncate it.

    The recipe keeps the last ``max_len - 1`` tokens, i.e. it silently drops the start of long
    code.  The experiment records this here and excludes truncated decisions (plan section 15).
    """
    official.ensure_path()
    from embed_utils import Recipe
    rec = Recipe(embed_model, max_len)
    n_state = [len(rec._flatten(rec.tok(t, add_special_tokens=False)["input_ids"])) for t in dec["state_text"]]
    n_cand = [max(len(rec._flatten(rec.tok(a, add_special_tokens=False)["input_ids"])),
                  len(rec._flatten(rec.tok(b, add_special_tokens=False)["input_ids"])))
              for a, b in zip(dec["cand_false"], dec["cand_true"])]
    cap = max_len - 1
    return pd.DataFrame({"decision_id": dec["decision_id"], "state_tokens": n_state, "candidate_tokens": n_cand,
                         "truncated": [s > cap or c > cap for s, c in zip(n_state, n_cand)]})


def embed_texts(texts, cache: EmbeddingCache, max_len: int, embed_url: str | None = None,
                served_model_name: str | None = None, gpu_mem: float = 0.85, chunk: int = 2048,
                embed_model: str = official.EMBED_MODEL) -> int:
    """Embed every uncached text with the official recipe/backend; flushes after each chunk."""
    official.ensure_path()
    import embed_utils
    todo = cache.missing(texts)
    if not todo:
        return 0
    rec = embed_utils.Recipe(embed_model, max_len)
    backend = embed_utils.make_backend(embed_url, embed_model, max_len, gpu_mem, served_model_name)
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        cache.add(part, backend.embed([rec.text_ids(t, keep="tail") for t in part]))
        cache.flush()
        print(f"[embed] {min(i + chunk, len(todo))}/{len(todo)}", flush=True)
    return len(todo)


def load_heads(checkpoint: str, device: str = "cpu"):
    """-> (state_head, action_head, scale) from an official checkpoint, via ``clm.heads.make_head``."""
    official.ensure_path()
    import torch
    from clm.heads import HIDDEN, PROJ_DIM, make_head
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    kw = dict(width=cfg["width"], depth=cfg["depth"],
              proj=ck.get("projection_dim", cfg.get("projection_dim", PROJ_DIM)),
              activation=cfg.get("activation", "gelu"), layernorm=cfg.get("layernorm", False),
              residual=cfg.get("residual", False), hidden=cfg.get("hidden_size", HIDDEN))
    sh, ah = make_head(**kw), make_head(**kw)
    sh.load_state_dict(ck["state_head"]); ah.load_state_dict(ck["action_head"])
    scale = float(torch.as_tensor(ck["logit_scale"]).float().exp().clamp(max=100.0))
    return sh.eval().to(device), ah.eval().to(device), scale


def score(dec: pd.DataFrame, cache: EmbeddingCache, checkpoint: str, device: str | None = None,
          batch: int = 4096) -> pd.DataFrame:
    """Logits for the false/true candidates of every decision, plus margin and P(true)."""
    import torch
    import torch.nn.functional as F
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sh, ah, scale = load_heads(checkpoint, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(dec), batch):
            d = dec.iloc[i:i + batch]
            s = torch.from_numpy(cache.get(d["state_text"])).to(device)
            cf = torch.from_numpy(cache.get(d["cand_false"])).to(device)
            ct = torch.from_numpy(cache.get(d["cand_true"])).to(device)
            zs = F.normalize(sh(s), dim=-1)
            lf = scale * (zs * F.normalize(ah(cf), dim=-1)).sum(-1)
            lt = scale * (zs * F.normalize(ah(ct), dim=-1)).sum(-1)
            out.append(np.stack([lf.cpu().numpy(), lt.cpu().numpy()], 1))
    lg = np.concatenate(out) if out else np.zeros((0, 2))
    margin = lg[:, 1] - lg[:, 0]
    return pd.DataFrame({"decision_id": dec["decision_id"].values, "logit_false": lg[:, 0], "logit_true": lg[:, 1],
                         "margin": margin, "p_true": 1.0 / (1.0 + np.exp(-margin))})


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
