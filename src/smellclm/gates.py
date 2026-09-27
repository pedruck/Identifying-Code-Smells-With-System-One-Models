"""Primary-test admission: Gate 0, Gate B and gold-only test labels (plan sections 12 and 17).

The locked test set may only be scored as primary evidence when

* every evaluated test label is ``gold``, adjudicated, and carries at least
  ``min_reviewers`` independent annotators (weak labels stay train/validation only);
* the Gate 0 admission record for the corpus says it passed;
* the Gate B evidence reports test-set Cohen's kappa >= ``min_test_kappa`` for every
  evaluated smell under the frozen rubric version;
* support on the *retained* test decisions (after truncation exclusions) meets the
  Gate C thresholds.

Gate 0 and Gate B evidence are small JSON files produced by the audit and the
annotation study; their paths live in the config under ``dataset``:

    gate0_admission.json  {"pass": true, "datasets": [...], "report": "reports/gate0/..."}
    gate_b_evidence.json  {"rubric_version": "rubric/1", "test_kappa": {"long_method": 0.71, ...},
                           "raw_agreement": {...}, "reviewers": ["A", "B"]}
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import rubric
from .schema import GOLD

DEFAULT_MIN_REVIEWERS = 2
DEFAULT_MIN_TEST_KAPPA = 0.60


def is_gold(lab: dict, min_reviewers: int = DEFAULT_MIN_REVIEWERS) -> bool:
    return (lab.get("quality") == GOLD and bool(lab.get("adjudicated"))
            and len(lab.get("annotators") or []) >= min_reviewers)


def test_label_audit(dec: pd.DataFrame) -> dict:
    """``dec`` are the evaluated test decisions (``label_gold`` column from ``clm_adapter.decisions``)."""
    bad = dec[~dec["label_gold"].astype(bool)]
    return {"evaluated": int(len(dec)), "non_gold": int(len(bad)),
            "non_gold_by_quality": bad["label_quality"].fillna("missing").value_counts().to_dict(),
            "non_gold_examples": sorted(bad["decision_id"])[:10],
            "pass": len(dec) > 0 and bad.empty}


def _load(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def gate0_status(path: str | None) -> dict:
    ev = _load(path)
    if ev is None:
        return {"pass": False, "reason": f"no Gate 0 admission record at {path!r}"}
    return {"pass": bool(ev.get("pass")), "record": path, "datasets": ev.get("datasets"),
            **({} if ev.get("pass") else {"reason": "Gate 0 admission record does not pass"})}


def gate_b_status(path: str | None, smells, min_test_kappa: float = DEFAULT_MIN_TEST_KAPPA) -> dict:
    ev = _load(path)
    if ev is None:
        return {"pass": False, "reason": f"no Gate B evidence at {path!r}"}
    reasons = []
    if ev.get("rubric_version") != rubric.RUBRIC_VERSION:
        reasons.append(f"evidence rubric {ev.get('rubric_version')!r} != frozen {rubric.RUBRIC_VERSION!r}")
    if ev.get("machine_reviewers"):
        reasons.append(f"machine reviewer(s) {ev['machine_reviewers']} labelled the test set: LLM-assisted silver "
                       f"labels, not human gold (plan amendment required)")
    kappa = ev.get("test_kappa") or {}
    for s in smells:
        k = kappa.get(s)
        if k is None or k < min_test_kappa:
            reasons.append(f"{s}: test kappa {k} < {min_test_kappa}")
    return {"pass": not reasons, "record": path, "test_kappa": kappa, "min_test_kappa": min_test_kappa,
            "reasons": reasons}


def primary_test_gate(cfg: dict, test_dec: pd.DataFrame, retained_support: dict) -> dict:
    """All conditions for scoring the locked test set as primary evidence."""
    d, q = cfg["dataset"], cfg.get("label_quality", {})
    out = {"labels": test_label_audit(test_dec),
           "gate0": gate0_status(d.get("gate0_admission")),
           "gate_b": gate_b_status(d.get("gate_b_evidence"), cfg["smells"],
                                   q.get("min_test_kappa", DEFAULT_MIN_TEST_KAPPA)),
           "retained_support": {"pass": bool(retained_support["pass"])}}
    out["pass"] = all(v["pass"] for v in out.values())
    return out


def failures(gate: dict) -> list[str]:
    msgs = []
    if not gate["labels"]["pass"]:
        l = gate["labels"]
        msgs.append(f"{l['non_gold']} of {l['evaluated']} test labels are not adjudicated gold "
                    f"({l['non_gold_by_quality']})")
    for k in ("gate0", "gate_b"):
        if not gate[k]["pass"]:
            msgs.append(f"{k}: " + (gate[k].get("reason") or "; ".join(gate[k].get("reasons", []))))
    if not gate["retained_support"]["pass"]:
        msgs.append("support on retained (post-truncation) test decisions is below the Gate C thresholds")
    return msgs
