"""Canonical dataset contract (plan section 7).

One row per code sample.  Labels are explicit per smell: ``positive``,
``negative`` or ``unknown``.  Unknown labels are excluded from that smell's loss
and evaluation and are never converted to negatives.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import pandas as pd

POSITIVE, NEGATIVE, UNKNOWN = "positive", "negative", "unknown"
LABEL_STATUSES = (POSITIVE, NEGATIVE, UNKNOWN)

# Label provenance: only ``gold`` (independently reviewed and adjudicated) labels
# may enter the locked test set; ``weak`` covers tool output and unreviewed
# third-party labels.
GOLD, WEAK = "gold", "weak"
LABEL_QUALITIES = (GOLD, WEAK)

LANGUAGES = ("java", "python", "javascript", "typescript", "cpp", "csharp", "kotlin", "go")
SAMPLE_TYPES = ("method", "class")

SMELLS = {
    "long_method": "Long Method",
    "long_parameter_list": "Long Parameter List",
    "deep_nesting": "Deep Nesting",
}

REQUIRED_COLUMNS = (
    "sample_id", "dataset", "repository_url", "repository_family_id", "commit_sha", "file_path",
    "start_line", "end_line", "language", "sample_type", "symbol", "code", "context", "license",
    "provenance", "labels", "static_metrics", "content_hash", "fingerprint", "extraction_version",
    "rubric_version",
)


@dataclass
class Label:
    status: str
    quality: str = WEAK
    source: str = ""                       # e.g. "enase2026", "reviewer:A+B/adjudicated"
    rubric_version: str = ""
    confidence: float | None = None
    annotators: list[str] = field(default_factory=list)
    adjudicated: bool = False

    def __post_init__(self):
        if self.status not in LABEL_STATUSES:
            raise ValueError(f"label status {self.status!r} not in {LABEL_STATUSES}")
        if self.quality not in LABEL_QUALITIES:
            raise ValueError(f"label quality {self.quality!r} not in {LABEL_QUALITIES}")


@dataclass
class Sample:
    sample_id: str
    dataset: str
    repository_url: str
    repository_family_id: str
    commit_sha: str
    file_path: str
    start_line: int | None
    end_line: int | None
    language: str
    sample_type: str
    symbol: str
    code: str
    labels: dict[str, Label]
    context: str = ""
    license: str = ""
    provenance: str = ""
    static_metrics: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    fingerprint: str = ""
    extraction_version: str = ""
    rubric_version: str = ""

    def __post_init__(self):
        if self.language not in LANGUAGES:
            raise ValueError(f"{self.sample_id}: language {self.language!r} not in {LANGUAGES}")
        if self.sample_type not in SAMPLE_TYPES:
            raise ValueError(f"{self.sample_id}: sample_type {self.sample_type!r} not in {SAMPLE_TYPES}")
        for smell in self.labels:
            if smell not in SMELLS:
                raise ValueError(f"{self.sample_id}: unknown smell {smell!r}")
        if not self.content_hash:
            self.content_hash = sha256_text(self.code)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def label_status(labels: dict, smell: str) -> str:
    """Status of ``smell``; a smell missing from the labels object is unknown."""
    lab = labels.get(smell)
    if lab is None:
        return UNKNOWN
    return lab["status"] if isinstance(lab, dict) else lab.status


def is_clean(labels: dict, smells: Iterable[str]) -> bool:
    """A clean example has an explicit negative for every evaluated smell."""
    return all(label_status(labels, s) == NEGATIVE for s in smells)


def samples_to_frame(samples: Iterable[Sample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        d = asdict(s)
        d["labels"] = json.dumps(d["labels"], sort_keys=True)
        d["static_metrics"] = json.dumps(d["static_metrics"], sort_keys=True)
        rows.append(d)
    return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))


def decode_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the JSON-encoded ``labels`` / ``static_metrics`` columns in place of their strings."""
    df = df.copy()
    for col in ("labels", "static_metrics"):
        df[col] = [json.loads(v) if isinstance(v, str) else v for v in df[col]]
    return df


def validate_frame(df: pd.DataFrame) -> list[str]:
    """Contract violations (empty list when the frame is valid)."""
    problems = [f"missing column {c}" for c in REQUIRED_COLUMNS if c not in df.columns]
    if problems:
        return problems
    if df["sample_id"].duplicated().any():
        problems.append(f"{int(df['sample_id'].duplicated().sum())} duplicated sample_id values")
    for col in ("repository_family_id", "language", "code", "content_hash"):
        empty = int((df[col].astype(str).str.len() == 0).sum())
        if empty:
            problems.append(f"{empty} rows with empty {col}")
    d = decode_frame(df)
    for sid, labels in zip(d["sample_id"], d["labels"]):
        for smell, lab in labels.items():
            if smell not in SMELLS:
                problems.append(f"{sid}: unknown smell {smell}")
            elif lab.get("status") not in LABEL_STATUSES:
                problems.append(f"{sid}: {smell} has invalid status {lab.get('status')!r}")
    return problems


def write_dataset(df: pd.DataFrame, path: str) -> None:
    """Parquet is canonical; a JSONL copy is written next to it for inspection."""
    df.to_parquet(path, index=False)
    df.to_json(path.rsplit(".", 1)[0] + ".jsonl", orient="records", lines=True, force_ascii=False)


def read_dataset(path: str) -> pd.DataFrame:
    return decode_frame(pd.read_parquet(path))
