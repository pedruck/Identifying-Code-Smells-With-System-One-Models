"""Freeze a small exploratory ENASE-positive / Stage-1-silver-negative pilot.

Run only from the source workspace with --java/--cpp/--python pointing at the
uploaded ENASE archives. The output is the clean-clone input, not gold data.
"""
import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from smellclm import leakage
from smellclm.schema import REQUIRED_COLUMNS, validate_frame

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260927
SMELLS = ("long_method", "long_parameter_list")
N = {"train": 80, "val": 20, "holdout": 20}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def order(value):
    return sha(f"{SEED}:{value}".encode())


def label(status, source, confidence=None):
    return {"status": status, "quality": "weak", "source": source,
            "rubric_version": "rubric/1", "confidence": confidence,
            "annotators": [], "adjudicated": False}


def main():
    ap = argparse.ArgumentParser()
    for language in ("java", "cpp", "python"):
        ap.add_argument(f"--{language}", required=True)
    args = ap.parse_args()
    sheet_dir = ROOT / "data/annotation/stage1"
    selected = pd.read_parquet(sheet_dir / "selected.parquet")
    selected = selected[selected.partition == "trainval"].copy()
    selected["item_id"] = selected.sample_id.map(lambda x: "i" + sha(f"{SEED}:item:{x}".encode())[:10])
    by_item = selected.set_index("item_id")
    candidates = defaultdict(lambda: defaultdict(list))
    sheet_hashes = {}
    for path in sorted(sheet_dir.glob("llm1_batch*.csv")):
        sheet_hashes[path.name] = sha(path.read_bytes())
        with path.open(newline="") as f:
            for rec in csv.DictReader(f):
                if rec["item_id"] not in by_item.index:
                    continue
                base = by_item.loc[rec["item_id"]]
                if len(base["code"]) > 4000:
                    continue
                for smell in SMELLS:
                    if rec[smell] != "negative":
                        continue
                    row = base.to_dict()
                    row["sample_id"] = f"pilot:silver:{base.sample_id}:{smell}"
                    row["labels"] = {smell: label("negative", "stage1-llm1", float(rec[f"{smell}_confidence"]))}
                    row["static_metrics"] = json.loads(row["static_metrics"])
                    row["split_group"] = row["repository_family_id"].split("/")[-1].lower()
                    row["source_kind"] = "stage1_silver"
                    candidates[smell]["negative"].append(row)
    archive_hashes = {}
    names = {"java": "java/java_code_smells.json", "cpp": "c++/c++_code_smells.json",
             "python": "python/python_code_smells.json"}
    aliases = {"longmethod": "long_method", "longparameterlist": "long_parameter_list"}
    for language in names:
        path = Path(getattr(args, language))
        archive_hashes[path.name] = sha(path.read_bytes())
        with zipfile.ZipFile(path) as z:
            records = json.loads(z.read(names[language]))
        for rec in records:
            smell = aliases.get("".join(c for c in rec.get("smell_type", "").lower() if c.isalpha()))
            if smell not in SMELLS or not rec.get("code", "").strip() or len(rec["code"]) > 4000:
                continue
            code = rec["code"]
            project = str(rec.get("project") or "unknown").lower()
            metrics = rec.get("metrics") or {}
            sid = f"pilot:enase:{language}:{rec['id']}:{smell}"
            row = {"sample_id": sid, "dataset": "enase2026", "repository_url": rec.get("url") or "",
                   "repository_family_id": project, "commit_sha": rec.get("commit") or "",
                   "file_path": rec.get("file_path") or "", "start_line": rec.get("line_no"),
                   "end_line": None, "language": language, "sample_type": "method", "symbol": "",
                   "code": code, "context": "", "license": "CC-BY-4.0 (dataset; upstream code license varies)",
                   "provenance": f"ENASE 2026 {path.name} {rec['id']}",
                   "labels": {smell: label("positive", "enase2026")},
                   "static_metrics": {"effective_loc": metrics.get("loc") or metrics.get("LOC"),
                                      "parameter_count": metrics.get("parameter_count")},
                   "content_hash": sha(code.encode()), "fingerprint": "",
                   "extraction_version": "enase2026", "rubric_version": "rubric/1",
                   "split_group": project, "source_kind": "enase_positive"}
            candidates[smell]["positive"].append(row)
    # Assign whole project families. The seed and family name fix the split,
    # regardless of class, source, or later sampling order.
    all_rows = [r for d in candidates.values() for rows in d.values() for r in rows]
    families = sorted({r["split_group"] for r in all_rows}, key=order)
    allocation = {fam: ("val" if i % 5 == 0 else "holdout" if i % 5 == 1 else "train")
                  for i, fam in enumerate(families)}
    # The silver pool has fewer families; ensure a mixed validation class exists.
    silver_families = sorted({r["split_group"] for d in candidates.values() for r in d["negative"]}, key=order)
    for i, fam in enumerate(silver_families):
        allocation[fam] = "val" if i % 4 == 0 else "train"
    python_positive_families = sorted({r["split_group"] for d in candidates.values()
                                       for r in d["positive"] if r["language"] == "python"}, key=order)
    if len(python_positive_families) >= 3:
        allocation[python_positive_families[0]] = "val"
        allocation[python_positive_families[1]] = "holdout"
        allocation[python_positive_families[2]] = "train"
    chosen = []
    seen = set()
    before = {}
    for smell in SMELLS:
        for part in ("train", "val", "holdout"):
            for status in (("positive",) if part == "holdout" else ("positive", "negative")):
                pool = [r for r in candidates[smell][status]
                        if allocation[r["split_group"]] == part]
                before[f"{smell}:{part}:{status}"] = len(pool)
                pool.sort(key=lambda r: order(r["sample_id"]))
                priority = []
                for language in ("java", "python", "cpp"):
                    priority.extend([r for r in pool if r["language"] == language][:5])
                pool = priority + [r for r in pool if r not in priority]
                count = 0
                for row in pool:
                    key = (row["language"], leakage.exact_hash(row["code"], row["language"]))
                    fp = leakage.fingerprint(row["code"], row["language"])
                    near_key = (row["language"], fp) if fp else None
                    if key in seen or (near_key and near_key in seen):
                        continue
                    row["split"] = part
                    row["fingerprint"] = fp
                    row["split_group"] = row["repository_family_id"].split("/")[-1].lower()
                    chosen.append(row)
                    seen.add(key)
                    if near_key:
                        seen.add(near_key)
                    count += 1
                    if count == N[part]:
                        break
                if count < N[part]:
                    raise RuntimeError(f"insufficient {smell} {part} {status}: {count}/{N[part]}")
    frame = pd.DataFrame(chosen)
    problems = validate_frame(frame)
    if problems:
        raise RuntimeError(str(problems[:5]))
    if leakage.assert_no_leakage(leakage.add_hashes(frame)):
        raise RuntimeError("cross-split leakage")
    output = ROOT / "data/pilot"
    output.mkdir(parents=True, exist_ok=True)
    final = frame[list(REQUIRED_COLUMNS) + ["split", "split_group", "source_kind"]].copy()
    final["labels"] = final.labels.map(lambda x: json.dumps(x, sort_keys=True))
    final["static_metrics"] = final.static_metrics.map(lambda x: json.dumps(x, sort_keys=True))
    final.to_parquet(output / "pilot.parquet", index=False)
    counts = Counter((r["split"], next(iter(r["labels"])), next(iter(r["labels"].values()))["status"])
                     for r in chosen)
    manifest = {"seed": SEED, "counts": {"/".join(k): v for k, v in sorted(counts.items())},
                "eligible_before_sampling": before, "rows": len(frame), "families": len(set(frame.split_group)),
                "enase_archive_sha256": archive_hashes, "silver_sheet_sha256": sheet_hashes,
                "license": "ENASE dataset CC BY 4.0; source URL https://figshare.com/s/2c89cce6b2d77c6f324f",
                "limits": "All labels are provisional. ENASE source code may have separate upstream licenses. Validation is source-confounded; holdout has positives only. No gold test."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(frame), "counts": manifest["counts"], "families": manifest["families"]}, indent=2))


if __name__ == "__main__":
    main()
