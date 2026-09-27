"""Leakage control (plan section 9, steps 1-5).

* ``normalize``           comment-free, whitespace-collapsed source (raw code is kept elsewhere)
* ``fingerprint``         identifier/literal-abstracted token stream hash (renaming-insensitive)
* ``near_duplicate_pairs`` MinHash + LSH over token shingles, verified by exact Jaccard
* ``family_id``           canonical repository family from a URL plus an optional fork/mirror map
* ``split_groups``        connected components of repository families and duplicate edges;
                          these components are the unit that the frozen split assigns
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Iterable

import pandas as pd

from .static_metrics import strip_comments_and_strings

LEAKAGE_VERSION = "leakage/1"

_TOKEN = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?|\"\"|\S")
_KEYWORDS = set("""
abstract and as assert async await bool boolean break byte case catch char class const continue def default del
delete do double elif else enum except export extends false final finally float for foreach fun func function go
if implements import in instanceof int interface is lambda let long match namespace new nil none not null or
package pass private protected public raise return self short static struct super switch synchronized this throw
throws true try type typeof using val var void volatile when while with yield
""".split())


def _strip_python_comments(code: str) -> str:
    return "\n".join(re.sub(r"#.*$", "", line) for line in code.splitlines())


def normalize(code: str, language: str) -> str:
    stripped = (_strip_python_comments(code) if language == "python"
                else strip_comments_and_strings(code, language))
    return " ".join(stripped.split())


def tokens(code: str, language: str, abstract: bool = True) -> list[str]:
    out = []
    for t in _TOKEN.findall(normalize(code, language)):
        if abstract and (t[0].isalpha() or t[0] in "_$") and t.lower() not in _KEYWORDS:
            out.append("ID")
        elif abstract and t[0].isdigit():
            out.append("NUM")
        else:
            out.append(t)
    return out


def exact_hash(code: str, language: str) -> str:
    return hashlib.sha256(normalize(code, language).encode()).hexdigest()


MIN_FINGERPRINT_TOKENS = 40


def fingerprint(code: str, language: str) -> str:
    """Hash of the abstracted token stream; empty for snippets too short to be meaningful
    (trivial getters/setters would otherwise collide across unrelated repositories)."""
    toks = tokens(code, language)
    if len(toks) < MIN_FINGERPRINT_TOKENS:
        return ""
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()


def shingles(code: str, language: str, k: int = 5) -> set[str]:
    toks = tokens(code, language, abstract=False)
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _minhash(sh: set[str], num_perm: int) -> list[int]:
    if not sh:
        return [0] * num_perm
    hs = [int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big") for s in sh]
    mask, prime = (1 << 61) - 1, (1 << 61) - 1
    sig = []
    for i in range(num_perm):
        a, b = 6364136223846793005 * (i + 1) & mask, 1442695040888963407 * (i + 7) & mask
        sig.append(min((a * h + b) % prime for h in hs))
    return sig


def near_duplicate_pairs(codes: list[str], languages: list[str], threshold: float = 0.8,
                         num_perm: int = 64, bands: int = 16, k: int = 5,
                         min_tokens: int = MIN_FINGERPRINT_TOKENS) -> list[tuple[int, int, float]]:
    """Index pairs whose token-shingle Jaccard similarity is >= ``threshold``.

    Candidates come from MinHash LSH (``bands`` x ``num_perm // bands`` rows); every
    candidate is verified with the exact Jaccard, so there are no false positives.
    Comparisons are restricted to the same language.
    """
    rows = num_perm // bands
    # trivial snippets (getters, one-line delegations) are excluded: they chain unrelated
    # code into giant components and carry no meaningful leakage
    sh = [shingles(c, l, k) if len(tokens(c, l, abstract=False)) >= min_tokens else set()
          for c, l in zip(codes, languages)]
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        if not s:
            continue
        sig = _minhash(s, num_perm)
        for b in range(bands):
            buckets[(languages[i], b, tuple(sig[b * rows:(b + 1) * rows]))].append(i)
    seen, out = set(), []
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                inter = len(sh[i] & sh[j])
                jac = inter / len(sh[i] | sh[j])
                if jac >= threshold:
                    out.append((i, j, jac))
    return sorted(out)


def family_id(repository_url: str, family_map: dict[str, str] | None = None) -> str:
    """Canonical ``host/owner/name`` (lowercase, no ``.git``); ``family_map`` merges forks and mirrors."""
    u = repository_url.strip().lower()
    u = re.sub(r"^(https?://|git@|ssh://git@)", "", u).replace(":", "/")
    u = re.sub(r"\.git$", "", u).rstrip("/")
    return (family_map or {}).get(u, u)


class _UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedupe_exact(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse exact normalized duplicates before splitting (plan section 9, step 2).

    Each duplicate cluster keeps its lowest ``sample_id`` when all copies agree on every
    label; clusters with conflicting labels are dropped entirely.  Returns (kept rows,
    report of every removed row and why).
    """
    import json as _json
    key = df["labels"].map(lambda l: _json.dumps({k: v["status"] if isinstance(v, dict) else v
                                                  for k, v in sorted(l.items())}))
    report = []
    keep = pd.Series(True, index=df.index)
    for h, idx in df.groupby("content_hash_normalized").groups.items():
        if len(idx) < 2:
            continue
        ids = df.loc[idx, "sample_id"]
        if key.loc[idx].nunique() > 1:
            keep.loc[idx] = False
            report += [{"sample_id": s, "reason": "exact_duplicate_label_conflict", "kept": None} for s in ids]
        else:
            first = ids.min()
            for i, s in zip(idx, ids):
                if s != first:
                    keep.loc[i] = False
                    report.append({"sample_id": s, "reason": "exact_duplicate", "kept": first})
    return df[keep].copy(), pd.DataFrame(report, columns=["sample_id", "reason", "kept"])


def split_groups(df: pd.DataFrame, near_threshold: float = 0.8) -> tuple[pd.Series, pd.DataFrame]:
    """-> (group id per row, duplicate report).

    Rows sharing a repository family, an exact normalized hash, or a verified
    near-duplicate edge end up in one group.  Group ids are the sorted-first
    sample id of the component, so they do not depend on row order.
    """
    n = len(df)
    uf = _UnionFind(n)
    report = []
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, (fam, h, fp) in enumerate(zip(df["repository_family_id"], df["content_hash_normalized"],
                                         df["fingerprint"])):
        by_key[("family", fam)].append(i)
        by_key[("exact", h)].append(i)
        if fp:
            by_key[("renamed", fp)].append(i)
    for (kind, _), idx in by_key.items():
        for j in idx[1:]:
            uf.union(idx[0], j)
            if kind != "family":
                report.append((idx[0], j, kind, 1.0))
    for i, j, jac in near_duplicate_pairs(list(df["code"]), list(df["language"]), threshold=near_threshold):
        uf.union(i, j)
        report.append((i, j, "near", round(jac, 4)))
    ids = list(df["sample_id"])
    comp: dict[int, list[str]] = defaultdict(list)
    for i in range(n):
        comp[uf.find(i)].append(ids[i])
    name = {root: "g:" + min(members) for root, members in comp.items()}
    groups = pd.Series([name[uf.find(i)] for i in range(n)], index=df.index, name="split_group")
    rep = pd.DataFrame([{"sample_a": ids[i], "sample_b": ids[j], "kind": k, "similarity": s,
                         "family_a": df["repository_family_id"].iloc[i],
                         "family_b": df["repository_family_id"].iloc[j],
                         "cross_family": df["repository_family_id"].iloc[i] != df["repository_family_id"].iloc[j]}
                        for i, j, k, s in report],
                       columns=["sample_a", "sample_b", "kind", "similarity", "family_a", "family_b", "cross_family"])
    return groups, rep


def add_hashes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["content_hash_normalized"] = [exact_hash(c, l) for c, l in zip(df["code"], df["language"])]
    df["fingerprint"] = [fingerprint(c, l) for c, l in zip(df["code"], df["language"])]
    return df


def assert_no_leakage(df: pd.DataFrame, split_col: str = "split", pairs: Iterable[tuple[str, str]] = ()) -> list[str]:
    """Violations: a family, exact hash, fingerprint or supplied near-dup pair spanning splits."""
    problems = []
    for col in ("repository_family_id", "content_hash_normalized", "fingerprint"):
        sub = df[df[col].astype(str).str.len() > 0]
        spans = sub.groupby(col)[split_col].nunique()
        bad = spans[spans > 1]
        if len(bad):
            problems.append(f"{len(bad)} {col} values cross splits, e.g. {list(bad.index[:3])}")
    where = dict(zip(df["sample_id"], df[split_col]))
    crossing = [(a, b) for a, b in pairs if a in where and b in where and where[a] != where[b]]
    if crossing:
        problems.append(f"{len(crossing)} near-duplicate pairs cross splits, e.g. {crossing[:3]}")
    return problems
