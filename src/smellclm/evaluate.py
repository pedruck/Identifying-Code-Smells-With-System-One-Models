"""Metrics, validation-only thresholds, repository-family bootstrap, and the success rule
(plan sections 4 and 12).

A prediction frame has one row per decision with at least ``decision_id``,
``smell``, ``language``, ``repository_family_id``, ``label`` (0/1) and ``pred`` (0/1);
probability metrics also need ``p_true``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- point metrics
def prf(y: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": int(len(y)), "positive": int((y == 1).sum()), "negative": int((y == 0).sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / len(y) if len(y) else 0.0}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise average precision)."""
    if (y == 1).sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y); k = np.arange(1, len(y) + 1)
    return float(((tp / k) * y).sum() / y.sum())


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(y) else float("nan")


def calibration_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    return [{"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int((idx == b).sum()),
             "mean_p": float(p[idx == b].mean()) if (idx == b).any() else None,
             "frac_pos": float(y[idx == b].mean()) if (idx == b).any() else None} for b in range(bins)]


def summarize(df: pd.DataFrame, smells) -> dict:
    """Macro-F1 (primary endpoint), micro-F1, per-smell and per-language metrics with support."""
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    per_smell = {}
    for s in smells:
        d = df[df["smell"] == s]
        m = prf(d["label"].to_numpy(), d["pred"].to_numpy())
        if "p_true" in d:
            m["pr_auc"] = average_precision(d["label"].to_numpy(), d["p_true"].to_numpy())
            m["brier"] = brier(d["label"].to_numpy(), d["p_true"].to_numpy())
        per_smell[s] = m
    per_lang = {}
    for lang, d in df.groupby("language"):
        ms = [prf(x["label"].to_numpy(), x["pred"].to_numpy())["f1"] for _, x in d.groupby("smell")]
        per_lang[lang] = {**prf(d["label"].to_numpy(), d["pred"].to_numpy()), "macro_f1": float(np.mean(ms))}
    cells = {f"{lang}|{s}": prf(d["label"].to_numpy(), d["pred"].to_numpy())
             for (lang, s), d in df.groupby(["language", "smell"])}
    out = {"macro_f1": float(np.mean([per_smell[s]["f1"] for s in smells])) if smells else float("nan"),
           "micro": prf(y, p), "per_smell": per_smell, "per_language": per_lang, "language_smell": cells}
    if "p_true" in df:
        out["brier"] = brier(y, df["p_true"].to_numpy())
    return out


def macro_f1(df: pd.DataFrame, smells) -> float:
    return float(np.mean([prf(df.loc[df["smell"] == s, "label"].to_numpy(),
                              df.loc[df["smell"] == s, "pred"].to_numpy())["f1"] for s in smells]))


# --------------------------------------------------------------------------- thresholds
def select_thresholds(val: pd.DataFrame, smells, score_col: str = "margin") -> dict:
    """Per-smell threshold maximizing validation F1 (ties -> the threshold closest to 0).

    Candidates are midpoints between consecutive distinct validation scores plus 0.
    Chosen once on validation data and then frozen; never refit on test.
    """
    out = {}
    for s in smells:
        d = val[val["smell"] == s]
        y, sc = d["label"].to_numpy(), d[score_col].to_numpy()
        u = np.unique(sc)
        cands = np.concatenate([[0.0], (u[1:] + u[:-1]) / 2, [u.min() - 1e-6] if len(u) else []])
        best = (-1.0, 0.0)
        for t in cands:
            f = prf(y, (sc > t).astype(int))["f1"]
            if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t) < abs(best[1])):
                best = (f, float(t))
        out[s] = {"threshold": best[1], "val_f1": best[0], "n": int(len(d))}
    return out


def apply_thresholds(df: pd.DataFrame, thresholds: dict, score_col: str = "margin") -> pd.Series:
    t = df["smell"].map({s: v["threshold"] for s, v in thresholds.items()})
    return (df[score_col] > t).astype(int)


# --------------------------------------------------------------------------- bootstrap
def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, smells, n_boot: int = 2000, seed: int = 0,
                     cluster_col: str = "repository_family_id") -> dict:
    """Macro-F1 difference ``b - a`` with a 95% interval, resampling repository families.

    ``a`` and ``b`` must hold predictions for identical decisions; each replicate draws
    families with replacement and evaluates both systems on the same draw.
    """
    a = a.set_index("decision_id").sort_index(); b = b.set_index("decision_id").sort_index()
    if not a.index.equals(b.index):
        raise ValueError("paired bootstrap needs identical decision sets")
    fams = a[cluster_col].to_numpy()
    uniq, inv = np.unique(fams, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    ya, pa, pb = a["label"].to_numpy(), a["pred"].to_numpy(), b["pred"].to_numpy()
    sm = a["smell"].to_numpy()
    rng = np.random.default_rng(seed)

    def mf1(idx, pred):
        return np.mean([prf(ya[idx][sm[idx] == s], pred[idx][sm[idx] == s])["f1"] for s in smells])

    base_idx = np.arange(len(a))
    point = mf1(base_idx, pb) - mf1(base_idx, pa)
    diffs, per_smell = [], {s: [] for s in smells}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        diffs.append(mf1(idx, pb) - mf1(idx, pa))
        for s in smells:
            m = idx[sm[idx] == s]
            per_smell[s].append(prf(ya[m], pb[m])["f1"] - prf(ya[m], pa[m])["f1"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_macro_f1": float(point), "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "clusters": int(len(uniq)), "seed": seed,
            "per_smell_ci95": {s: [float(x) for x in np.percentile(v, [2.5, 97.5])] for s, v in per_smell.items()}}


def bootstrap_ci(df: pd.DataFrame, smells, n_boot: int = 2000, seed: int = 0,
                 cluster_col: str = "repository_family_id") -> list[float]:
    fams = df[cluster_col].to_numpy()
    uniq, inv = np.unique(fams, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        vals.append(macro_f1(df.iloc[idx], smells))
    return [float(x) for x in np.percentile(vals, [2.5, 97.5])]


# --------------------------------------------------------------------------- success rule
def success_rule(zero_shot: pd.DataFrame, finetuned_by_seed: dict[int, pd.DataFrame], smells,
                 leakage_ok: bool, support_ok: bool, rule_baseline: pd.DataFrame | None = None,
                 min_gain: float = 0.05, max_smell_loss: float = 0.05, n_boot: int = 2000) -> dict:
    """Preregistered go/no-go (plan section 4) evaluated on every seed.

    The primary gain and interval are computed per seed; the rule requires the gain and
    CI lower bound criteria on every seed ("stable across three seeds").
    """
    zs = summarize(zero_shot, smells)
    seeds = {}
    for seed, ft in sorted(finetuned_by_seed.items()):
        fs = summarize(ft, smells)
        bs = paired_bootstrap(zero_shot, ft, smells, n_boot=n_boot, seed=seed)
        losses = {s: fs["per_smell"][s]["f1"] - zs["per_smell"][s]["f1"] for s in smells}
        seeds[seed] = {"macro_f1": fs["macro_f1"], "delta": bs["delta_macro_f1"], "ci95": bs["ci95"],
                       "per_smell_delta": losses,
                       "gain_ok": bs["delta_macro_f1"] >= min_gain, "ci_ok": bs["ci95"][0] > 0,
                       "no_smell_loss": all(v >= -max_smell_loss for v in losses.values())}
    mf = [v["macro_f1"] for v in seeds.values()]
    passed = (bool(seeds) and leakage_ok and support_ok
              and all(v["gain_ok"] and v["ci_ok"] and v["no_smell_loss"] for v in seeds.values()))
    out = {"pass": passed, "zero_shot_macro_f1": zs["macro_f1"], "seeds": seeds,
           "finetuned_macro_f1_mean": float(np.mean(mf)) if mf else None,
           "finetuned_macro_f1_std": float(np.std(mf, ddof=1)) if len(mf) > 1 else None,
           "leakage_ok": leakage_ok, "support_ok": support_ok,
           "criteria": {"min_gain": min_gain, "max_smell_loss": max_smell_loss, "ci": "95% family bootstrap"}}
    if rule_baseline is not None:
        rb = summarize(rule_baseline, smells)["macro_f1"]
        out["rule_baseline_macro_f1"] = rb
        beats = bool(mf) and all(m > rb for m in mf)
        out["practical_advantage"] = beats
        if passed and not beats:
            out["interpretation"] = "CLM adaptation signal detected, practical advantage not established."
    return out
