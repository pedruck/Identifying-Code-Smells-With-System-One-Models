"""SmellyCode++ (figshare 28519385, CC0) — weak Java Long Method labels for training mechanics.

The CSV (618 MB) stores code on one space-tokenized line, mixes class- and
method-level rows, and carries tool-derived labels.  Loading keeps method-like
rows, deduplicates by normalized hash, and drops duplicate groups whose labels
conflict.  Long Parameter List and Deep Nesting are ``unknown`` (no column), and
``effective_loc`` is not meaningful on one-line code, so these rows must not
feed the rule baseline.
"""
from __future__ import annotations

import re

import pandas as pd

from .. import leakage
from ..schema import NEGATIVE, POSITIVE, UNKNOWN, WEAK, Label
from . import make_sample

URL = "https://ndownloader.figshare.com/files/52714583"
SHA256 = "971a81da3f55326a5a56bae59c3cde34c4ebab1a99615ac0c474ce5ecf5029d2"
CLASS_LIKE = re.compile(r"^\s*(@\w+(\([^)]*\))?\s+)*((public|private|protected|abstract|final|static)\s+)*"
                        r"(class|interface|enum)\b")


def load(csv_path: str, limit: int | None = None) -> list:
    df = pd.read_csv(csv_path, keep_default_na=False, usecols=["File", "Project", "Class", "Code", "Long method"])
    df = df[~df["Code"].str.match(CLASS_LIKE)]
    df["h"] = [leakage.exact_hash(c, "java") for c in df["Code"]]
    conflicting = df.groupby("h")["Long method"].nunique()
    df = df[df["h"].map(conflicting) == 1].drop_duplicates("h")
    if limit:
        df = df.head(limit)
    out = []
    for i, r in df.iterrows():
        out.append(make_sample(
            sample_id=f"scpp:{i}", dataset="smellycodepp", repository_url=f"smellycodepp/{r['Project']}",
            commit_sha="", file_path=r["File"], start_line=None, end_line=None, language="java",
            symbol=r["Class"], code=r["Code"],
            labels={"long_method": Label(POSITIVE if int(r["Long method"]) else NEGATIVE, WEAK, source="smellycode++"),
                    "long_parameter_list": Label(UNKNOWN, WEAK), "deep_nesting": Label(UNKNOWN, WEAK)},
            license="CC0-1.0", provenance=f"{URL} sha256:{SHA256} row {i} (no revision pinned)"))
    return out
