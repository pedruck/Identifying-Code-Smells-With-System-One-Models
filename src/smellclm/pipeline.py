"""The eleven resumable stages of the notebook (plan section 15), as plain functions.

Each stage reads its inputs from the experiment directory, writes its outputs
there, and marks itself done; rerunning a completed stage is a no-op.  The
notebook calls these in order, so the same code can run headless:

    python -m smellclm.pipeline --config configs/stage1.json --root runs [--resume DIR] [--until 03_split]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from . import baselines, evaluate, experiment, gates, leakage, official, rubric, split
from .schema import (NEGATIVE, POSITIVE, UNKNOWN, WEAK, decode_frame, label_status, read_dataset,
                     samples_to_frame, validate_frame, write_dataset)

CONDITIONS = ("majority", "rules", "zero_shot", "finetuned")


# --------------------------------------------------------------------------- 01
def stage_environment(exp: experiment.Experiment, allow_cpu_only: bool = False) -> dict:
    if exp.done("01_environment"):
        return json.load(open(exp.path("environment.json")))
    env = experiment.environment()
    env["clm_commit"] = official.checked_out_commit()
    env["clm_pinned"] = exp.cfg["clm"]["repo_commit"]
    feas = experiment.feasibility(env)
    env["feasibility"] = feas
    exp.write_json("environment.json", env)
    if env["clm_commit"] != env["clm_pinned"]:
        raise RuntimeError(f"official CLM checkout {env['clm_commit']} != pinned {env['clm_pinned']}")
    if not feas["pass"] and not allow_cpu_only:
        raise RuntimeError("Gate A hardware precheck failed: " + "; ".join(feas["reasons"]) +
                           ". No substitute encoder or quantized port is used; stopping.")
    exp.mark("01_environment", {"feasibility": feas})
    return env


# --------------------------------------------------------------------------- 02
def _smoke_labels(df: pd.DataFrame, smells) -> pd.DataFrame:
    """SMOKE MODE ONLY: fill unknown labels from the frozen rules so every stage has both classes."""
    df = df.copy()
    new = []
    for labels, m in zip(df["labels"], df["static_metrics"]):
        labels = dict(labels)
        for s in smells:
            if label_status(labels, s) == UNKNOWN:
                labels[s] = {"status": POSITIVE if rubric.rule_predict(s, m) else NEGATIVE, "quality": WEAK,
                             "source": "SMOKE-rule-fill", "rubric_version": rubric.RUBRIC_VERSION,
                             "confidence": None, "annotators": [], "adjudicated": False}
        new.append(labels)
    df["labels"] = new
    return df


def stage_dataset(exp: experiment.Experiment, data_dir: str = "data") -> pd.DataFrame:
    out = exp.path("dataset.parquet")
    if exp.done("02_dataset_audit"):
        return read_dataset(out)
    cfg = exp.cfg
    smells, langs = cfg["smells"], cfg["languages"]
    mode = cfg["dataset"]["mode"]
    if mode == "smoke":
        from .datasets import hri_eu
        root = hri_eu.fetch(os.path.join(data_dir, "raw", "hri_eu"))
        df = decode_frame(samples_to_frame(hri_eu.load(root, smells)))
        df = _smoke_labels(df[df["language"].isin(langs)], smells)
    elif mode == "primary":
        df = read_dataset(cfg["dataset"]["primary_parquet"])
        df = df[df["language"].isin(langs)]
    else:
        raise ValueError(f"dataset mode {mode!r}")
    if df.empty:
        raise RuntimeError(f"no samples for languages {langs}")
    enc = df.copy()
    enc["labels"] = [json.dumps(v, sort_keys=True) for v in enc["labels"]]
    enc["static_metrics"] = [json.dumps(v, sort_keys=True) for v in enc["static_metrics"]]
    problems = validate_frame(enc)
    if problems:
        raise RuntimeError("dataset contract violations: " + "; ".join(problems[:10]))
    write_dataset(enc, out)
    stats = {"mode": mode, "rows": len(df), "by_language": df["language"].value_counts().to_dict(),
             "labels": {s: pd.Series([label_status(l, s) for l in df["labels"]]).value_counts().to_dict()
                        for s in smells},
             "label_quality": pd.Series([l[s].get("quality") for l in df["labels"] for s in smells if s in l])
             .value_counts().to_dict(),
             "families": int(df["repository_family_id"].nunique()),
             "datasets": df["dataset"].value_counts().to_dict()}
    exp.write_json("dataset_statistics.json", stats)
    exp.write("dataset_card.md", _dataset_card(cfg, stats))
    exp.write_json("prompt_and_candidates.json", rubric.prompt_manifest(smells, cfg["prompt"]["with_definition"]))
    exp.mark("02_dataset_audit", {"rows": len(df)})
    return df


def _dataset_card(cfg, stats) -> str:
    warn = ("\n> **SMOKE MODE.** Synthetic HRI-EU fixture with rule-filled labels, split by file. "
            "Every number produced from this run is a pipeline check, not evidence.\n"
            if stats["mode"] == "smoke" else "")
    return (f"# Dataset card\n{warn}\n- mode: {stats['mode']}\n- rows: {stats['rows']}\n"
            f"- repository families: {stats['families']}\n- by language: {stats['by_language']}\n"
            f"- label status per smell: {stats['labels']}\n- label quality: {stats['label_quality']}\n"
            f"- sources: {stats['datasets']}\n- rubric: {rubric.RUBRIC_VERSION}; prompts: {rubric.PROMPT_VERSION}\n")


# --------------------------------------------------------------------------- 03
def stage_split(exp: experiment.Experiment, df: pd.DataFrame) -> pd.DataFrame:
    out = exp.path("dataset_split.parquet")
    if exp.done("03_split"):
        return decode_frame(pd.read_parquet(out))
    cfg = exp.cfg
    df = leakage.add_hashes(df)
    if cfg["dataset"]["mode"] == "smoke":
        # the fixture is one repository family; split by file so every stage can run
        df["repository_family_id"] = df["repository_family_id"] + "#" + df["file_path"]
    df, removed = leakage.dedupe_exact(df)
    removed.to_parquet(exp.path("exact_duplicates_removed.parquet"), index=False)
    groups, rep = leakage.split_groups(df, cfg["split"]["near_duplicate_jaccard"])
    df["split_group"] = groups
    df["split"] = split.assign_splits(df, cfg["smells"], cfg["split"]["seed"], cfg["split"]["ratios"])
    near = [(a, b) for a, b, k in zip(rep["sample_a"], rep["sample_b"], rep["kind"]) if k == "near"]
    problems = leakage.assert_no_leakage(df, pairs=near)
    support = split.support_gates(df, cfg["smells"], cfg["languages"], **cfg["support"])
    gold_support = split.support_gates(_gold_only(df, cfg), cfg["smells"], cfg["languages"], **cfg["support"])
    rep.to_parquet(exp.path("duplicate_report.parquet"), index=False)
    m = split.manifest(df, cfg["split"]["seed"], {"leakage_problems": problems, "support": support,
                                                  "gold_test_support": gold_support,
                                                  "exact_duplicates_removed": len(removed)})
    exp.write_json("split_manifest.json", m)
    exp.write("split_manifest.sha256", m["checksum"] + "\n")
    enc = df.copy()
    enc["labels"] = [json.dumps(v, sort_keys=True) for v in enc["labels"]]
    enc["static_metrics"] = [json.dumps(v, sort_keys=True) for v in enc["static_metrics"]]
    enc.to_parquet(out, index=False)
    if problems:
        raise RuntimeError("Gate C leakage assertions failed: " + "; ".join(problems))
    exp.mark("03_split", {"checksum": m["checksum"], "support_pass": support["pass"],
                          "gold_test_support_pass": gold_support["pass"]})
    if not support["pass"]:
        exp.log("WARNING Gate C support thresholds not met; add repository families and re-split "
                "BEFORE any test inference (plan section 8).")
    if not gold_support["pass"]:
        exp.log("WARNING test support counting adjudicated gold labels only is below the Gate C thresholds; "
                "primary test inference will be refused (plan section 12).")
    return df


def _gold_only(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Test rows keep only adjudicated gold labels; other splits are unchanged."""
    q = cfg.get("label_quality", {}).get("min_reviewers", gates.DEFAULT_MIN_REVIEWERS)
    out = df.copy()
    out["labels"] = [{k: v for k, v in l.items() if sp != "test" or gates.is_gold(v, q)}
                     for l, sp in zip(out["labels"], out["split"])]
    return out


# --------------------------------------------------------------------------- 04
def stage_embeddings(exp: experiment.Experiment, df: pd.DataFrame) -> pd.DataFrame:
    from . import clm_adapter, scoring
    out = exp.path("decisions.parquet")
    if exp.done("04_embeddings"):
        return pd.read_parquet(out)
    c = exp.cfg["clm"]
    dec = clm_adapter.decisions(df, exp.cfg["smells"], exp.cfg["prompt"]["with_definition"])
    audit = scoring.token_audit(dec, c["max_len"], c["embed_model"])
    dec = dec.merge(audit, on="decision_id")
    dec["excluded_truncated"] = dec["truncated"] & (c["truncation_policy"] == "exclude")
    dec.to_parquet(out, index=False)
    kept = dec[~dec["excluded_truncated"]]
    cache = scoring.EmbeddingCache(_cache_path(exp))
    texts = list(kept["state_text"]) + list(kept["cand_false"]) + list(kept["cand_true"])
    n = scoring.embed_texts(texts, cache, c["max_len"], c.get("embed_url"), c.get("served_model_name"),
                            c.get("gpu_mem", 0.85), embed_model=c["embed_model"])
    exp.write_json("embedding_manifest.json", {
        "cache": _cache_path(exp), "embed_model": c["embed_model"], "max_len": c["max_len"],
        "recipe": "official train/embed_utils.Recipe.text_ids(keep='tail'), last-token pooling via vLLM",
        "texts": len(set(texts)), "newly_embedded": n, "decisions": len(dec),
        "truncated": int(dec["truncated"].sum()), "excluded_truncated": int(dec["excluded_truncated"].sum()),
        "truncated_by_split": dec[dec["truncated"]]["split"].value_counts().to_dict()})
    exp.mark("04_embeddings", {"decisions": len(dec), "excluded_truncated": int(dec["excluded_truncated"].sum())})
    return dec


def _cache_path(exp) -> str:
    c = exp.cfg["clm"]
    return exp.path("embeddings", official.embed_cache_name(c["embed_model"], c["max_len"]))


def _head_path(exp) -> str:
    from clm.heads import download  # noqa: F401  (ensure_path is called by the caller)
    return download(exp.cfg["clm"]["hf_model_repo"], exp.cfg["clm"]["head_file"], exp.path("checkpoints"))


# --------------------------------------------------------------------------- 05
def stage_zero_shot(exp: experiment.Experiment, dec: pd.DataFrame) -> None:
    """Gate A parity checks, then validation-only baselines and zero-shot scores. No test rows are scored."""
    from . import scoring
    if exp.done("05_baselines_zero_shot"):
        return
    official.ensure_path()
    head = _head_path(exp)
    cache = scoring.EmbeddingCache(_cache_path(exp))
    use = dec[~dec["excluded_truncated"]]
    val = use[use["split"] == "val"]
    zs = scoring.score(val, cache, head)
    parity = _gate_a_parity(val.head(32), cache, head, zs.head(32))
    exp.write_json("gate_a.json", {"head": head, "head_sha256": scoring.sha256_file(head), **parity})
    if not parity["pass"]:
        raise RuntimeError(f"Gate A score parity failed: {parity}")
    val.merge(zs, on="decision_id").to_parquet(exp.path("val_zero_shot_scores.parquet"), index=False)
    exp.mark("05_baselines_zero_shot", {"head_sha256": scoring.sha256_file(head)})


def _gate_a_parity(dec, cache, head, ours) -> dict:
    """Our scorer vs the official ``HeadPair`` projection path on identical embeddings, plus the
    candidate-order invariance unit test (scores are per candidate, so swapping must not matter)."""
    from clm.heads import HeadPair
    hp = HeadPair("ref", head, "cpu")
    zs, zf = hp.project(cache.get(dec["state_text"]), cache.get(dec["cand_false"]))
    _, zt = hp.project(cache.get(dec["state_text"]), cache.get(dec["cand_true"]))
    ref_margin = hp.scale * ((zs * zt).sum(1) - (zs * zf).sum(1))
    diff = float(np.max(np.abs(ref_margin - ours["margin"].to_numpy()))) if len(dec) else 0.0
    swapped = dec.rename(columns={"cand_false": "cand_true", "cand_true": "cand_false"})
    from . import scoring
    sw = scoring.score(swapped, cache, head, device="cpu")
    order = float(np.max(np.abs(sw["margin"].to_numpy() + ours["margin"].to_numpy()))) if len(dec) else 0.0
    return {"max_margin_diff_vs_official": diff, "order_swap_max_error": order,
            "pass": diff < 1e-3 and order < 1e-3, "n": int(len(dec))}


# --------------------------------------------------------------------------- 06
def stage_finetune(exp: experiment.Experiment, df: pd.DataFrame, dec: pd.DataFrame) -> dict:
    if exp.done("06_finetune"):
        return json.load(open(exp.path("training_history.json")))
    info, runs = _train_heads(exp, df, dec, "trainer_data", "finetune")
    hist = {"trainer_data": info, "runs": runs,
            "selection": "official trainer: best epoch by accuracy on a group-disjoint val_frac carve-out of "
                         "train (row id = split group); our validation split is only reported by the trainer"}
    exp.write_json("training_history.json", hist)
    exp.mark("06_finetune", {"seeds": exp.cfg["training"]["seeds"]})
    return hist


def _train_heads(exp, df: pd.DataFrame, dec: pd.DataFrame, data_name: str, out_name: str) -> tuple[dict, dict]:
    """Write trainer data from ``df`` (train/val splits only) and run the official trainer per seed."""
    from . import clm_adapter
    official.ensure_path()
    c, t = exp.cfg["clm"], exp.cfg["training"]
    excluded = set(dec.loc[dec["excluded_truncated"], "sample_id"])
    info = clm_adapter.write_trainer_data(df, exp.cfg["smells"], exp.path(data_name),
                                          exp.cfg["prompt"]["with_definition"], exclude_ids=excluded)
    head = _head_path(exp)
    runs = {}
    for seed in t["seeds"]:
        out = exp.path(out_name, f"seed{seed}")
        os.makedirs(out, exist_ok=True)
        cmd = clm_adapter.finetune_command(
            exp.path(data_name), out, head, seed, os.path.dirname(_cache_path(exp)), c["max_len"],
            c.get("embed_url"), c.get("served_model_name"),
            {"epochs": t["epochs"], "patience": t["patience"], "val_frac": t["val_frac"]})
        exp.log("RUN " + " ".join(cmd))
        with open(os.path.join(out, "run.log"), "w") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        runs[seed] = {"cmd": cmd, "summary": json.load(open(os.path.join(out, "finetune_summary.json"))),
                      "history": json.load(open(os.path.join(out, "history.json")))}
    return info, runs


# --------------------------------------------------------------------------- 07
def stage_validation(exp: experiment.Experiment, dec: pd.DataFrame) -> dict:
    """Score validation for every condition, pick thresholds, and freeze all decisions."""
    from . import scoring
    if exp.done("07_validation_selection"):
        return json.load(open(exp.path("calibration.json")))
    smells = exp.cfg["smells"]
    use = dec[~dec["excluded_truncated"]]
    train, val = use[use["split"] == "train"], use[use["split"] == "val"]
    cache = scoring.EmbeddingCache(_cache_path(exp))
    cal = {"zero_shot": {}, "finetuned": {}}
    zs = pd.read_parquet(exp.path("val_zero_shot_scores.parquet"))
    cal["zero_shot"] = evaluate.select_thresholds(zs, smells)
    for seed in exp.cfg["training"]["seeds"]:
        ck = exp.path("finetune", f"seed{seed}", "best_head.pt")
        sc = val.merge(scoring.score(val, cache, ck), on="decision_id")
        cal["finetuned"][str(seed)] = {"checkpoint": ck, "sha256": scoring.sha256_file(ck),
                                       "thresholds": evaluate.select_thresholds(sc, smells)}
        sc["pred"] = evaluate.apply_thresholds(sc, cal["finetuned"][str(seed)]["thresholds"])
        cal["finetuned"][str(seed)]["val_macro_f1"] = evaluate.macro_f1(sc, smells)
    maj = baselines.majority(train, val)
    cal["val_summary"] = {"majority": evaluate.summarize(maj, smells)["macro_f1"],
                          "rules": evaluate.summarize(baselines.rules(val), smells)["macro_f1"]}
    cal["locked"] = True
    exp.write_json("calibration.json", cal)
    exp.mark("07_validation_selection", {"locked": True})
    return cal


# --------------------------------------------------------------------------- 08
def stage_test(exp: experiment.Experiment, dec: pd.DataFrame, split_df: pd.DataFrame) -> dict:
    from . import scoring
    if exp.done("08_locked_test"):
        return json.load(open(exp.path("metrics.json")))
    if not exp.done("07_validation_selection"):
        raise RuntimeError("validation decisions must be locked before test inference")
    cfg, smells = exp.cfg, exp.cfg["smells"]
    cal = json.load(open(exp.path("calibration.json")))
    use = dec[~dec["excluded_truncated"]]
    train, test = use[use["split"] == "train"], use[use["split"] == "test"]
    gate = _test_gate(exp, test)
    cache = scoring.EmbeddingCache(_cache_path(exp))
    official.ensure_path()
    preds = {"majority": baselines.majority(train, test), "rules": baselines.rules(test)}
    zs = test.merge(scoring.score(test, cache, _head_path(exp)), on="decision_id")
    zs["pred_raw"] = (zs["margin"] > 0).astype(int)
    zs["pred"] = evaluate.apply_thresholds(zs, cal["zero_shot"])
    preds["zero_shot"] = zs
    ft = {}
    for seed, v in cal["finetuned"].items():
        d = test.merge(scoring.score(test, cache, v["checkpoint"]), on="decision_id")
        d["pred_raw"] = (d["margin"] > 0).astype(int)
        d["pred"] = evaluate.apply_thresholds(d, v["thresholds"])
        d.to_parquet(exp.path(f"finetuned_predictions_seed{seed}.parquet"), index=False)
        ft[int(seed)] = d
    preds["rules"].to_parquet(exp.path("rule_baseline_predictions.parquet"), index=False)
    zs.to_parquet(exp.path("zero_shot_predictions.parquet"), index=False)
    first = ft[min(ft)]
    first.to_parquet(exp.path("finetuned_predictions.parquet"), index=False)
    manifest = json.load(open(exp.path("split_manifest.json")))
    rule = evaluate.success_rule(zs, ft, smells, leakage_ok=not manifest["leakage_problems"],
                                 support_ok=gate["retained_support"]["pass"], rule_baseline=preds["rules"],
                                 min_gain=cfg["evaluation"]["min_gain"],
                                 max_smell_loss=cfg["evaluation"]["max_smell_loss"],
                                 n_boot=cfg["evaluation"]["n_boot"])
    metrics = {"mode": cfg["dataset"]["mode"], "split_checksum": manifest["checksum"],
               "confirmatory": gate["pass"], "test_gate": gate,
               "conditions": {k: evaluate.summarize(v, smells) for k, v in preds.items()},
               "zero_shot_raw_margin": evaluate.summarize(zs.assign(pred=zs["pred_raw"]), smells)["macro_f1"],
               "finetuned_by_seed": {s: evaluate.summarize(d, smells) for s, d in ft.items()},
               "ci95": {k: evaluate.bootstrap_ci(v, smells, cfg["evaluation"]["n_boot"])
                        for k, v in {**preds, **{f"finetuned_seed{s}": d for s, d in ft.items()}}.items()},
               "success_rule": rule, "support": gate["retained_support"],
               "support_pre_truncation": manifest["support"]}
    exp.write_json("metrics.json", metrics)
    _tables(exp, preds, ft, smells)
    exp.mark("08_locked_test", {"pass": rule["pass"]})
    return metrics


def _test_gate(exp, test: pd.DataFrame) -> dict:
    """Admission check run before any test row is scored.  Primary mode refuses to continue unless every
    evaluated label is adjudicated gold, Gate 0 and Gate B evidence pass, and support recomputed on the
    retained (post-truncation) test decisions meets Gate C.  Smoke mode records the failure and carries on,
    and every output is marked non-confirmatory."""
    cfg = exp.cfg
    support = split.decision_support_gates(test, cfg["smells"], cfg["languages"], **cfg["support"])
    gate = gates.primary_test_gate(cfg, test, support)
    gate["retained_support"] = support
    exp.write_json("test_gate.json", gate, overwrite=True)
    if not gate["pass"]:
        msgs = gates.failures(gate)
        if cfg["dataset"]["mode"] == "primary":
            raise RuntimeError("locked test inference refused: " + "; ".join(msgs))
        exp.log("SMOKE: test gate not met, results are non-confirmatory: " + "; ".join(msgs))
    return gate


def _tables(exp, preds, ft, smells) -> None:
    rows_s, rows_l, rows_c = [], [], []
    allp = {**preds, **{f"finetuned_seed{s}": d for s, d in ft.items()}}
    for name, d in allp.items():
        m = evaluate.summarize(d, smells)
        rows_s += [{"condition": name, "smell": s, **v} for s, v in m["per_smell"].items()]
        rows_l += [{"condition": name, "language": l, **v} for l, v in m["per_language"].items()]
        rows_c += [{"condition": name, "cell": c, **v} for c, v in m["language_smell"].items()]
    pd.DataFrame(rows_s).to_csv(exp.path("per_smell_metrics.csv"), index=False)
    pd.DataFrame(rows_l).to_csv(exp.path("per_language_metrics.csv"), index=False)
    pd.DataFrame(rows_c).to_csv(exp.path("language_smell_matrix.csv"), index=False)


# --------------------------------------------------------------------------- 09
def stage_errors(exp: experiment.Experiment, per_stratum: int = 5) -> None:
    """Deterministic error selection: for every smell x language x error type, the most confident errors."""
    if exp.done("09_error_analysis"):
        return
    d = pd.read_parquet(exp.path("finetuned_predictions.parquet"))
    d["error"] = np.where(d["pred"] == d["label"], "correct", np.where(d["pred"] == 1, "false_positive",
                                                                          "false_negative"))
    d["length_bin"] = pd.cut(d["state_tokens"], [0, 256, 512, 1024, 2048, 10**9]).astype(str)
    err = d[d["error"] != "correct"].copy()
    err["abs_margin"] = err["margin"].abs()
    pick = (err.sort_values(["abs_margin", "decision_id"], ascending=[False, True])
            .groupby(["smell", "language", "error"]).head(per_stratum))
    pick.drop(columns=["state_text", "cand_false", "cand_true"]).to_parquet(exp.path("error_analysis.parquet"),
                                                                            index=False)
    strat = d.groupby(["smell", "language", "length_bin", "error"]).size().rename("n").reset_index()
    strat.to_csv(exp.path("error_strata.csv"), index=False)
    exp.mark("09_error_analysis", {"selected": len(pick)})


# --------------------------------------------------------------------------- 10
def stage_transfer(exp: experiment.Experiment, df: pd.DataFrame, dec: pd.DataFrame) -> dict:
    """Leave-one-language-out transfer (plan section 13), run only when Gate F allows it.

    Gate F: the primary result is confirmatory (test gate passed) and the preregistered success rule
    passed.  Otherwise the stage records why it was skipped.  For each held-out language L the heads are
    retrained on the other languages' train split, thresholds are chosen on the other languages' validation
    split, and zero-shot vs fine-tuned are compared on L's locked test decisions with the family bootstrap.
    L never reaches training, early stopping, or threshold selection.

    The secondary ablations of plan section 14 are not implemented here; each is a separate experiment
    (its own config, e.g. ``prompt.with_definition = false``) and is reported as future work.
    """
    from . import scoring
    if exp.done("10_transfer_ablations"):
        return json.load(open(exp.path("transfer_metrics.json")))
    cfg, smells = exp.cfg, exp.cfg["smells"]
    m = json.load(open(exp.path("metrics.json")))
    held = cfg.get("transfer", {}).get("languages", cfg["languages"])
    out = {"gate_f": {"confirmatory": m.get("confirmatory", False), "success_rule": m["success_rule"]["pass"]},
           "ablations": "not implemented; run each ablation as its own experiment config (future work)"}
    out["gate_f"]["pass"] = out["gate_f"]["confirmatory"] and out["gate_f"]["success_rule"]
    if not out["gate_f"]["pass"] or not cfg.get("transfer", {}).get("enabled", False):
        out["skipped"] = ("transfer disabled in config" if out["gate_f"]["pass"]
                          else "Gate F not met: expansion only after a confirmatory primary pass")
        exp.write_json("transfer_metrics.json", out)
        exp.mark("10_transfer_ablations", {"skipped": out["skipped"]})
        return out
    official.ensure_path()
    use = dec[~dec["excluded_truncated"]]
    cache = scoring.EmbeddingCache(_cache_path(exp))
    head = _head_path(exp)
    out["languages"] = {}
    for lang in held:
        view = split.leave_one_language_out(df, lang)
        assert not ((view["language"] == lang) & (view["split"] != "test")).any()
        _, runs = _train_heads(exp, view, dec, os.path.join("transfer", lang, "trainer_data"),
                               os.path.join("transfer", lang, "finetune"))
        val = use[(use["split"] == "val") & (use["language"] != lang)]
        test = use[(use["split"] == "test") & (use["language"] == lang)]
        support = split.decision_support_gates(test, smells, [lang], **cfg["support"])
        zv = val.merge(scoring.score(val, cache, head), on="decision_id")
        zs = test.merge(scoring.score(test, cache, head), on="decision_id")
        zs["pred"] = evaluate.apply_thresholds(zs, evaluate.select_thresholds(zv, smells))
        ft = {}
        for seed in runs:
            ck = exp.path("transfer", lang, "finetune", f"seed{seed}", "best_head.pt")
            fv = val.merge(scoring.score(val, cache, ck), on="decision_id")
            d = test.merge(scoring.score(test, cache, ck), on="decision_id")
            d["pred"] = evaluate.apply_thresholds(d, evaluate.select_thresholds(fv, smells))
            ft[int(seed)] = d
        present = [s for s in smells if (test["smell"] == s).any()]
        rule = evaluate.success_rule(zs, ft, present, leakage_ok=True,
                                     support_ok=support["per_language"][lang]["pass"],
                                     min_gain=cfg["evaluation"]["min_gain"],
                                     max_smell_loss=cfg["evaluation"]["max_smell_loss"],
                                     n_boot=cfg["evaluation"]["n_boot"])
        out["languages"][lang] = {
            "support": support, "zero_shot_macro_f1": rule["zero_shot_macro_f1"],
            "seeds": {s: {k: v[k] for k in ("macro_f1", "delta", "ci95")} for s, v in rule["seeds"].items()},
            "transfer_supported": bool(rule["seeds"]) and support["per_language"][lang]["pass"]
                                  and all(v["ci_ok"] for v in rule["seeds"].values())}
    exp.write_json("transfer_metrics.json", out)
    exp.mark("10_transfer_ablations", {"languages": list(out["languages"])})
    return out


# --------------------------------------------------------------------------- 11
def stage_export(exp: experiment.Experiment) -> str:
    m = json.load(open(exp.path("metrics.json")))
    stats = json.load(open(exp.path("dataset_statistics.json")))
    emb = json.load(open(exp.path("embedding_manifest.json")))
    r = m["success_rule"]
    lines = [f"# Final summary — {exp.cfg['name']}", ""]
    if m["mode"] == "smoke":
        lines += ["> **SMOKE RUN.** Synthetic fixture with rule-filled labels. Not evidence about CLM.", ""]
    if not m.get("confirmatory"):
        lines += ["> **NON-CONFIRMATORY.** Test gate failed: " + "; ".join(gates.failures(m["test_gate"])), ""]
    lines += [
        f"- split checksum: `{m['split_checksum']}`",
        f"- decisions excluded for truncation: {emb['excluded_truncated']} of {emb['decisions']}",
        f"- labels: {stats['labels']}",
        f"- support gates pass (retained test decisions): {m['support']['pass']}",
        "", "## Primary endpoint (test macro-F1, validation-frozen thresholds)", "",
        "| condition | macro-F1 | 95% CI (family bootstrap) |", "|---|---|---|"]
    for k, v in m["conditions"].items():
        lines.append(f"| {k} | {v['macro_f1']:.3f} | {m['ci95'].get(k)} |")
    for s, v in m["finetuned_by_seed"].items():
        lines.append(f"| finetuned seed {s} | {v['macro_f1']:.3f} | {m['ci95'].get(f'finetuned_seed{s}')} |")
    lines += ["", "## Preregistered success rule", "", f"- **pass: {r['pass']}**",
              f"- zero-shot macro-F1 {r['zero_shot_macro_f1']:.3f}; fine-tuned mean {r['finetuned_macro_f1_mean']}"
              f" (sd {r['finetuned_macro_f1_std']})"]
    for s, v in r["seeds"].items():
        lines.append(f"- seed {s}: delta {v['delta']:.3f}, CI {v['ci95']}, per-smell delta {v['per_smell_delta']}")
    if "interpretation" in r:
        lines.append(f"- {r['interpretation']}")
    lines += ["", "Language x smell cells below the support gate are exploratory: "
              f"{[c['language'] + '|' + c['smell'] for c in m['support']['exploratory_cells']]}"]
    if os.path.exists(exp.path("transfer_metrics.json")):
        tr = json.load(open(exp.path("transfer_metrics.json")))
        lines += ["", "## Leave-one-language-out transfer (secondary)", ""]
        if "skipped" in tr:
            lines.append(f"- skipped: {tr['skipped']}")
        for lang, v in tr.get("languages", {}).items():
            lines.append(f"- held out {lang}: zero-shot {v['zero_shot_macro_f1']:.3f}; seeds {v['seeds']}; "
                         f"transfer supported: {v['transfer_supported']}")
        lines.append(f"- ablations: {tr['ablations']}")
    text = "\n".join(lines) + "\n"
    exp.write("final_summary.md", text, overwrite=True)
    exp.write("README.md", f"# {exp.cfg['name']}\n\nResearch question: does fine-tuning the CLM-v0.1-8B "
              f"projection heads improve code-smell verification on unseen repository families?\n\n"
              f"Status: see final_summary.md (mode: {m['mode']}).\n", overwrite=True)
    if not exp.done("11_export"):
        exp.mark("11_export")
    return text


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--until", default="11_export", choices=experiment.STAGES)
    ap.add_argument("--allow-cpu-only", action="store_true", help="run CPU stages (01-03) without a GPU")
    a = ap.parse_args(argv)
    exp = experiment.Experiment(a.root, json.load(open(a.config)), a.resume)
    print(f"experiment: {exp.dir}", flush=True)
    order = list(experiment.STAGES)
    stop = order.index(a.until)
    stage_environment(exp, a.allow_cpu_only)
    df = stage_dataset(exp)
    if stop >= order.index("03_split"):
        df = stage_split(exp, df)
    if stop >= order.index("04_embeddings"):
        dec = stage_embeddings(exp, df)
        if stop >= order.index("05_baselines_zero_shot"):
            stage_zero_shot(exp, dec)
        if stop >= order.index("06_finetune"):
            stage_finetune(exp, df, dec)
        if stop >= order.index("07_validation_selection"):
            stage_validation(exp, dec)
        if stop >= order.index("08_locked_test"):
            stage_test(exp, dec, df)
        if stop >= order.index("09_error_analysis"):
            stage_errors(exp)
        if stop >= order.index("10_transfer_ablations"):
            stage_transfer(exp, df, dec)
        if stop >= order.index("11_export"):
            print(stage_export(exp))


if __name__ == "__main__":
    main(sys.argv[1:])
