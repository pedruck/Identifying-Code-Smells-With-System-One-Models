"""Exploratory real-code pilot; never invokes the locked primary test gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from . import evaluate, leakage, pipeline, scoring
from .schema import decode_frame, validate_frame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/pilot/pilot.parquet"
MANIFEST = ROOT / "data/pilot/manifest.json"


def prepare(exp) -> pd.DataFrame:
    """Verify the committed source package before any GPU work."""
    frame = pd.read_parquet(DATA)
    manifest = json.loads(MANIFEST.read_text())
    problems = validate_frame(frame)
    if problems:
        raise RuntimeError("Pilot dataset contract: " + "; ".join(problems[:5]))
    frame = decode_frame(frame)
    if len(frame) != manifest["rows"]:
        raise RuntimeError("Pilot row count differs from frozen manifest")
    counts = {}
    for _, row in frame.iterrows():
        smell, lab = next(iter(row["labels"].items()))
        key = f"{row['split']}/{smell}/{lab['status']}"
        counts[key] = counts.get(key, 0) + 1
        if lab["quality"] != "weak":
            raise RuntimeError("Pilot contains a non-silver label")
    if counts != manifest["counts"]:
        raise RuntimeError("Pilot class counts differ from frozen manifest")
    for smell in exp.cfg["smells"]:
        for part in ("train", "val"):
            p, n = counts.get(f"{part}/{smell}/positive", 0), counts.get(f"{part}/{smell}/negative", 0)
            if p == 0 or n == 0 or p != n:
                raise RuntimeError(f"{part}/{smell} requires equal positive and negative support; got {p}/{n}")
    hashed = leakage.add_hashes(frame)
    problems = leakage.assert_no_leakage(hashed)
    pairs = leakage.near_duplicate_pairs(frame.code.tolist(), frame.language.tolist(), threshold=0.8)
    cross = [(i, j) for i, j, _ in pairs if frame.iloc[i]["split"] != frame.iloc[j]["split"]]
    if problems or cross:
        raise RuntimeError(f"Pilot leakage: {problems[:3]}, {len(cross)} near-duplicate cross-split pairs")
    if set(frame.loc[frame.split == "holdout", "source_kind"]) != {"enase_positive"}:
        raise RuntimeError("Positive holdout must contain only ENASE rows")
    report = {"input": str(DATA), "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
              "rows": len(frame), "counts": counts, "near_duplicate_pairs": len(pairs),
              "cross_split_near_duplicates": len(cross), "source_counts": frame.groupby(
                  ["split", "source_kind", "language"]).size().rename_axis(
                      ["split", "source", "language"]).to_dict(),
              "manifest": manifest, "confirmatory": False}
    report["source_counts"] = {"/".join(k): v for k, v in report["source_counts"].items()}
    exp.write_json("pilot_input_audit.json", report)
    return frame


def finish(exp, decisions: pd.DataFrame, calibration: dict) -> dict:
    """Validation report and positive-only holdout recall, with no test-gate call."""
    use = decisions[~decisions.excluded_truncated]
    hold = use[use.split == "holdout"]
    cache = scoring.EmbeddingCache(pipeline._cache_path(exp))
    out = {"confirmatory": False, "warning": "Silver exploratory result; source-confounded; no gold test.",
           "majority_validation_macro_f1": calibration["val_summary"]["majority"],
           "finetuned_validation_macro_f1": {}, "enase_positive_holdout_recall": {},
           "retained_support": {}}
    for smell in exp.cfg["smells"]:
        d = use[(use.split == "val") & (use.smell == smell)]
        out["retained_support"][smell] = {"positive": int(d.label.sum()),
                                           "negative": int((1 - d.label).sum())}
    for seed, result in calibration["finetuned"].items():
        out["finetuned_validation_macro_f1"][seed] = result["val_macro_f1"]
        scores = hold.merge(scoring.score(hold, cache, result["checkpoint"]), on="decision_id")
        scores["pred"] = evaluate.apply_thresholds(scores, result["thresholds"])
        out["enase_positive_holdout_recall"][seed] = {
            smell: {"recall": float(scores.loc[scores.smell == smell, "pred"].mean()),
                    "support": int((scores.smell == smell).sum())}
            for smell in exp.cfg["smells"]}
    exp.write_json("pilot_report.json", out)
    return out
