"""Deterministic group-stratified split, frozen manifest, and support gates (plan sections 8-9).

Groups (repository families merged with duplicate clusters, see ``leakage.split_groups``)
are assigned whole.  Assignment is greedy: groups are visited largest first
(ties broken by a seeded hash of the group id) and each goes to the split whose
per-stratum counts fall furthest below target, where strata are
language x smell x label status.  The result depends only on the data and the seed.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

import pandas as pd

from .schema import NEGATIVE, POSITIVE, label_status

SPLIT_VERSION = "split/1"
DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _strata(row, smells) -> Counter:
    c = Counter()
    for s in smells:
        st = label_status(row["labels"], s)
        if st in (POSITIVE, NEGATIVE):
            c[(row["language"], s, st)] += 1
    c[("__rows__",)] += 1
    return c


def assign_splits(df: pd.DataFrame, smells, seed: int, ratios: dict | None = None,
                  group_col: str = "split_group", reserved_test_families: set | None = None) -> pd.Series:
    """``reserved_test_families`` (from ``annotation.write_batches``) pins every group that contains one of
    those repository families to test, and the remaining groups are divided between the other splits.
    The reservation is made before labelling, so the double-reviewed items are exactly the test set."""
    ratios = ratios or DEFAULT_RATIOS
    pinned = {}
    if reserved_test_families:
        pinned = {g: "test" for g, f in zip(df[group_col], df["repository_family_id"]) if f in reserved_test_families}
        rest = {k: v for k, v in ratios.items() if k != "test"}
        ratios = {k: v / sum(rest.values()) for k, v in rest.items()}
    names = list(ratios)
    prof: dict[str, Counter] = defaultdict(Counter)
    for _, row in df.iterrows():
        if row[group_col] not in pinned:
            prof[row[group_col]] += _strata(row, smells)
    total = sum(prof.values(), Counter())

    def order_key(g):
        h = hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()
        return (-prof[g][("__rows__",)], h)

    have = {n: Counter() for n in names}
    where = dict(pinned)
    for g in sorted(prof, key=order_key):
        best, best_score = None, None
        for n in names:
            # deficit of this split relative to target over the strata this group touches
            score = sum((ratios[n] * total[k] - have[n][k]) / max(1, total[k]) * v for k, v in prof[g].items())
            if best_score is None or score > best_score + 1e-12:
                best, best_score = n, score
        where[g] = best
        have[best] += prof[g]
    return df[group_col].map(where).rename("split")


def support_table(df: pd.DataFrame, smells, split: str = "test") -> pd.DataFrame:
    rows = []
    sub = df[df["split"] == split]
    for _, r in sub.iterrows():
        for s in smells:
            st = label_status(r["labels"], s)
            if st in (POSITIVE, NEGATIVE):
                rows.append((r["language"], s, st))
    t = pd.DataFrame(rows, columns=["language", "smell", "status"])
    if t.empty:
        return pd.DataFrame(columns=["language", "smell", "positive", "negative"])
    return (t.groupby(["language", "smell", "status"]).size().unstack("status", fill_value=0)
            .reindex(columns=[POSITIVE, NEGATIVE], fill_value=0).reset_index())


def support_gates(df: pd.DataFrame, smells, languages, split: str = "test", per_smell: int = 50,
                  per_language: int = 50, per_cell: int = 20) -> dict:
    """Plan section 8.  Returns pass/fail per gate plus the list of confirmatory cells."""
    return _gates(support_table(df, smells, split), smells, languages, per_smell, per_language, per_cell)


def decision_support_gates(dec: pd.DataFrame, smells, languages, per_smell: int = 50, per_language: int = 50,
                           per_cell: int = 20) -> dict:
    """Same gates counted on scored decisions (one row per sample x smell, ``label`` 0/1), e.g. the
    test decisions that survive truncation exclusion."""
    if dec.empty:
        t = pd.DataFrame(columns=["language", "smell", POSITIVE, NEGATIVE])
    else:
        t = (dec.assign(status=dec["label"].map({1: POSITIVE, 0: NEGATIVE}))
             .groupby(["language", "smell", "status"]).size().unstack("status", fill_value=0)
             .reindex(columns=[POSITIVE, NEGATIVE], fill_value=0).reset_index())
    return _gates(t, smells, languages, per_smell, per_language, per_cell)


def _gates(t: pd.DataFrame, smells, languages, per_smell, per_language, per_cell) -> dict:
    out = {"per_smell": {}, "per_language": {}, "confirmatory_cells": [], "exploratory_cells": []}
    for s in smells:
        sub = t[t["smell"] == s]
        p, n = int(sub[POSITIVE].sum()), int(sub[NEGATIVE].sum())
        out["per_smell"][s] = {"positive": p, "negative": n, "pass": p >= per_smell and n >= per_smell}
    for lang in languages:
        sub = t[t["language"] == lang]
        p, n = int(sub[POSITIVE].sum()), int(sub[NEGATIVE].sum())
        out["per_language"][lang] = {"positive": p, "negative": n, "pass": p >= per_language and n >= per_language}
    for lang in languages:
        for s in smells:
            sub = t[(t["language"] == lang) & (t["smell"] == s)]
            p, n = int(sub[POSITIVE].sum()), int(sub[NEGATIVE].sum())
            key = "confirmatory_cells" if p >= per_cell and n >= per_cell else "exploratory_cells"
            out[key].append({"language": lang, "smell": s, "positive": p, "negative": n})
    out["pass"] = (all(v["pass"] for v in out["per_smell"].values())
                   and all(v["pass"] for v in out["per_language"].values()))
    out["thresholds"] = {"per_smell": per_smell, "per_language": per_language, "per_cell": per_cell}
    return out


def manifest(df: pd.DataFrame, seed: int, extra: dict | None = None) -> dict:
    """Frozen split manifest; ``checksum`` covers the sorted (sample_id, split, group) triples."""
    triples = sorted(zip(df["sample_id"], df["split"], df["split_group"]))
    checksum = hashlib.sha256(json.dumps(triples).encode()).hexdigest()
    counts = {k: int(v) for k, v in df["split"].value_counts().sort_index().items()}
    return {"split_version": SPLIT_VERSION, "seed": seed, "counts": counts, "checksum": checksum,
            "assignments": {sid: sp for sid, sp, _ in triples}, **(extra or {})}


def verify_manifest(df: pd.DataFrame, m: dict) -> bool:
    triples = sorted(zip(df["sample_id"], df["split"], df["split_group"]))
    return hashlib.sha256(json.dumps(triples).encode()).hexdigest() == m["checksum"]


def leave_one_language_out(df: pd.DataFrame, held_out: str) -> pd.DataFrame:
    """Transfer view (plan section 13): the held-out language keeps only its locked test rows;
    its train/val rows are dropped so it cannot inform training, thresholds or early stopping."""
    keep = (df["language"] != held_out) | (df["split"] == "test")
    return df[keep].copy()
