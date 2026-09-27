"""Research dataset -> official CLM typed decisions (plan sections 10-11).

Every eligible (sample, smell) pair becomes one ``noul`` question.  State and
candidate texts are produced by the official ``clm.schema.build_pairs`` so the
text the heads see is exactly what the official server and trainer would build.

Training data for the unmodified official trainer (``train/finetune.py --task
choice``) is written as ``<dir>/<workflow>/{train,test}-00000.parquet`` rows with
``id`` / ``state`` / ``questions`` / ``gold``.  Two conventions keep it leakage-safe
without touching the trainer:

* ``id`` is the split group (repository family + duplicate cluster), so the
  trainer's own ``--val-frac`` early-stopping carve-out is group-disjoint.
* the trainer's ``test`` file receives the *validation* split; the locked test
  split is never given to the trainer.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from . import gates, official, rubric
from .schema import NEGATIVE, POSITIVE, label_status

WORKFLOW = "smell"


def state_of(row) -> dict:
    state = {"language": rubric.LANGUAGE_NAMES[row["language"]], "scope": row["sample_type"]}
    if row.get("context"):
        state["context"] = row["context"]
    state["code"] = row["code"]
    return state


def questions_of(smells, with_definition: bool = True) -> dict:
    return {s: rubric.question(s, with_definition) for s in smells}


def decisions(df: pd.DataFrame, smells, with_definition: bool = True) -> pd.DataFrame:
    """One row per (sample, smell) with an explicit positive/negative label."""
    s = official.schema()
    qs = questions_of(smells, with_definition)
    rows = []
    for _, r in df.iterrows():
        pairs = s.build_pairs(state_of(r), qs)
        for smell in smells:
            st = label_status(r["labels"], smell)
            if st not in (POSITIVE, NEGATIVE):
                continue
            stext, keys, cands = pairs[smell]
            assert keys == ["false", "true"], keys
            lab = r["labels"][smell]
            rows.append({
                "decision_id": f"{r['sample_id']}::{smell}", "sample_id": r["sample_id"], "smell": smell,
                "language": r["language"], "split": r.get("split"), "split_group": r.get("split_group"),
                "repository_family_id": r["repository_family_id"], "dataset": r["dataset"],
                "label": int(st == POSITIVE), "label_quality": lab.get("quality"),
                "label_confidence": lab.get("confidence"), "label_gold": gates.is_gold(lab),
                "state_text": stext, "cand_false": cands[0], "cand_true": cands[1],
                "effective_loc": r["static_metrics"].get("effective_loc"),
                "parameter_count": r["static_metrics"].get("parameter_count"),
                "max_nesting": r["static_metrics"].get("max_nesting"),
            })
    return pd.DataFrame(rows)


def typed_rows(df: pd.DataFrame, smells, with_definition: bool = True) -> list[dict]:
    """Official typed-decision rows; unknown labels are left out of ``gold`` (the trainer skips them)."""
    qs = questions_of(smells, with_definition)
    out = []
    for _, r in df.iterrows():
        gold = {}
        for smell in smells:
            st = label_status(r["labels"], smell)
            if st in (POSITIVE, NEGATIVE):
                gold[smell] = {"label": "true" if st == POSITIVE else "false"}
        if gold:
            out.append({"id": r["split_group"], "sample_id": r["sample_id"], "workflow": WORKFLOW,
                        "state": json.dumps(state_of(r), ensure_ascii=False),
                        "questions": json.dumps({k: qs[k] for k in gold}, ensure_ascii=False),
                        "gold": json.dumps(gold)})
    return out


def write_trainer_data(df: pd.DataFrame, smells, out_dir: str, with_definition: bool = True,
                       exclude_ids: set | None = None) -> dict:
    """Write train (= our train split) and test (= our val split) parquet for the official trainer."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    exclude_ids = exclude_ids or set()
    wf = os.path.join(out_dir, WORKFLOW)
    os.makedirs(wf, exist_ok=True)
    counts = {}
    for ours, theirs in (("train", "train"), ("val", "test")):
        sub = df[(df["split"] == ours) & (~df["sample_id"].isin(exclude_ids))]
        rows = typed_rows(sub, smells, with_definition)
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(wf, f"{theirs}-00000.parquet"))
        counts[theirs] = {"rows": len(rows), "questions": sum(len(json.loads(r["gold"])) for r in rows),
                          "groups": len({r["id"] for r in rows})}
    return {"data": out_dir, "workflow": WORKFLOW, "counts": counts,
            "note": "trainer 'test' = our validation split; locked test is never written here"}


def finetune_command(data_dir: str, out_dir: str, init_ckpt: str, seed: int, embed_cache: str,
                     max_len: int, embed_url: str | None = None, served_model_name: str | None = None,
                     extra: dict | None = None) -> list[str]:
    """Command line for the unmodified official trainer (plan: hard targets, InfoNCE, warm start)."""
    cmd = ["python", official.finetune_script(), "--task", "choice", "--data", data_dir,
           "--workflow", WORKFLOW, "--out-dir", out_dir, "--init-ckpt", init_ckpt, "--targets", "hard",
           "--loss", "infonce", "--seed", str(seed), "--embed-cache", embed_cache, "--max-len", str(max_len),
           "--embed-model", official.EMBED_MODEL]
    if embed_url:
        cmd += ["--embed-url", embed_url]
    if served_model_name:
        cmd += ["--served-model-name", served_model_name]
    for k, v in (extra or {}).items():
        cmd += [f"--{k.replace('_', '-')}", str(v)]
    return cmd
