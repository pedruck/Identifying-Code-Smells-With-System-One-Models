"""MLCQ (Zenodo 3666840 v1.1, CC BY 4.0) — human-reviewed Java Long Method reference.

Reviews are aggregated per ``sample_id`` by majority vote (``none`` = negative,
any severity = positive); ties are dropped.  Code is not shipped: it is fetched
from GitHub at the pinned commit and sliced to ``start_line..end_line``.  The
source code keeps the license of its origin repository, recorded per sample.
"""
from __future__ import annotations

import os
import urllib.request

import pandas as pd

from ..schema import NEGATIVE, POSITIVE, UNKNOWN, WEAK, Label
from . import make_sample

URL = "https://zenodo.org/api/records/3666840/files/MLCQCodeSmellSamples.csv/content"


def reviews(csv_path: str, smell: str = "long method") -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", dtype=str)
    df = df[(df["smell"] == smell) & (df["type"] == "function")]
    df["pos"] = (df["severity"] != "none").astype(int)
    agg = df.groupby("sample_id").agg(
        n=("pos", "size"), pos=("pos", "sum"), repository=("repository", "first"),
        commit_hash=("commit_hash", "first"), path=("path", "first"), start_line=("start_line", "first"),
        end_line=("end_line", "first"), code_name=("code_name", "first")).reset_index()
    agg = agg[agg["pos"] * 2 != agg["n"]]            # drop ties
    agg["label"] = (agg["pos"] * 2 > agg["n"]).astype(int)
    return agg


def raw_url(repository: str, commit: str, path: str) -> str:
    slug = repository.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return f"https://raw.githubusercontent.com/{slug}/{commit}/{path.lstrip('/')}"


def fetch_code(row, cache_dir: str) -> str | None:
    url = raw_url(row["repository"], row["commit_hash"], row["path"])
    local = os.path.join(cache_dir, url.split("githubusercontent.com/")[1])
    if not os.path.exists(local):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read().decode("utf-8", errors="replace")
        except OSError:
            return None
        os.makedirs(os.path.dirname(local), exist_ok=True)
        open(local, "w", encoding="utf-8").write(data)
    lines = open(local, encoding="utf-8").read().splitlines()
    return "\n".join(lines[int(row["start_line"]) - 1:int(row["end_line"])])


def load(csv_path: str, cache_dir: str, limit: int | None = None) -> tuple[list, list]:
    """-> (samples, unavailable sample ids)."""
    agg = reviews(csv_path)
    if limit:
        agg = agg.head(limit)
    out, missing = [], []
    for _, r in agg.iterrows():
        code = fetch_code(r, cache_dir)
        if not code:
            missing.append(r["sample_id"])
            continue
        out.append(make_sample(
            sample_id=f"mlcq:{r['sample_id']}", dataset="mlcq", repository_url=r["repository"],
            commit_sha=r["commit_hash"], file_path=r["path"], start_line=int(r["start_line"]),
            end_line=int(r["end_line"]), language="java", symbol=r["code_name"], code=code,
            labels={"long_method": Label(POSITIVE if r["label"] else NEGATIVE, WEAK, source="mlcq-majority",
                                         confidence=max(r["pos"], r["n"] - r["pos"]) / r["n"]),
                    "long_parameter_list": Label(UNKNOWN, WEAK), "deep_nesting": Label(UNKNOWN, WEAK)},
            license="code: origin repository license (verify); labels: CC-BY-4.0",
            provenance=f"MLCQ zenodo 3666840 v1.1 sample {r['sample_id']} ({r['n']} reviews)"))
    return out, missing
