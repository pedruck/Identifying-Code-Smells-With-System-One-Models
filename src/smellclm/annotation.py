"""Gold stratum: candidate mining, blind labelling batches, agreement and adjudication (plan sections 6, 7, 12).

Workflow (``python -m smellclm.annotation <command>``):

1. ``resolve``   pin every repository in the pool manifest to a commit SHA and record its SPDX licence.
2. ``mine``      check out each repository at its pinned commit, extract functions (``extract/1``),
                 compute static metrics and write an unlabelled candidate pool.  Writes the Gate 0
                 admission record for the pool (pinned commits + allow-listed licences).
3. ``batches``   reserve whole repository families for the locked test set *before* anyone labels,
                 draw a metric-stratified sample, and write blind reviewer sheets.  Every test item is
                 assigned to both reviewers; ``double_review_frac`` of train/val items also are.
4. ``agreement`` read the filled sheets, report raw agreement and Cohen's kappa per smell, and write the
                 disagreement sheet for adjudication.
5. ``finalize``  merge reviews and adjudications into the primary dataset Parquet and the Gate B evidence.

Reviewers see code, language and static metrics only: no stratum, no repository, no split and no
model output.  Items appear in an independent random order per reviewer.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import urllib.error
import urllib.request
from collections import defaultdict

import pandas as pd

from . import extract, leakage, rubric
from .datasets import make_sample
from .schema import (GOLD, LABEL_STATUSES, NEGATIVE, POSITIVE, UNKNOWN, WEAK, Label, decode_frame,
                     samples_to_frame, write_dataset)

ANNOTATION_VERSION = "annotation/1"
DATASET_NAME = "gold-pool"
LICENSE_ALLOWLIST = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "BSL-1.0", "ISC", "Unlicense",
                     "0BSD", "Zlib", "PSF-2.0", "MPL-2.0", "EPL-2.0", "EPL-1.0", "CC0-1.0"}
EXCLUDED_PATH = re.compile(r"(^|/)(tests?|testing|spec|__tests__|test_?data|fixtures?|examples?|samples?|docs?|"
                           r"benchmarks?|vendor|third[_-]?party|external|extern|deps|node_modules|build|dist|"
                           r"generated|gen|migrations)(/|$)|(_test|Test|Tests|_pb2)\.[a-z]+$|\.min\.|"
                           r"(^|/|[._-])(tests?|spec)\.[a-z]+$|(^|/)test_[^/]*\.py$",
                           re.IGNORECASE)
MIN_LINES = 3

# Metric bands per smell, keyed on the frozen rule metric.  ``borderline`` follows the rubric's
# borderline ranges; ``likely_negative`` excludes trivial one-liners so negatives are not free wins.
BANDS = {
    "long_method": {"likely_positive": (35, None), "borderline": (25, 34), "likely_negative": (8, 24)},
    "long_parameter_list": {"likely_positive": (6, None), "borderline": (5, 5), "likely_negative": (2, 4)},
    "deep_nesting": {"likely_positive": (5, None), "borderline": (3, 4), "likely_negative": (1, 2)},
}
DEFAULT_TARGETS = {"test": {"likely_positive": 35, "borderline": 20, "likely_negative": 35},
                   "trainval": {"likely_positive": 60, "borderline": 30, "likely_negative": 60}}
# Reviewer ids starting with "llm" are machine raters (e.g. ``llm1``).  Their labels are recorded, but an
# LLM-assisted test set is silver, not human gold: Gate B refuses it (see gates.gate_b_status).
MACHINE_REVIEWER = re.compile(r"^llm", re.IGNORECASE)
SHEET_FIELDS = ["item_id", "language", "lines", "effective_loc", "parameter_count", "max_nesting"]


def is_machine(reviewer: str) -> bool:
    return bool(MACHINE_REVIEWER.match(reviewer))


def _h(*parts) -> str:
    return hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def _read_json(path: str):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- 1. resolve
def _slug(url: str) -> str:
    return url.split("github.com/")[-1].removesuffix(".git").strip("/")


def resolve(manifest_path: str) -> dict:
    """Fill ``commit`` (HEAD of the default branch now) and ``license`` (GitHub SPDX id) where missing."""
    m = _read_json(manifest_path)
    for r in m["repositories"]:
        if not r.get("commit"):
            out = subprocess.run(["git", "ls-remote", r["url"], "HEAD"], capture_output=True, text=True, check=True)
            r["commit"] = out.stdout.split()[0]
        if not r.get("license"):
            req = urllib.request.Request(f"https://api.github.com/repos/{_slug(r['url'])}/license")
            if os.environ.get("GITHUB_TOKEN"):  # the anonymous API allows only 60 requests per hour
                req.add_header("Authorization", f"Bearer {os.environ['GITHUB_TOKEN']}")
            try:
                with urllib.request.urlopen(req, timeout=30) as f:
                    r["license"] = (json.load(f).get("license") or {}).get("spdx_id") or "NOASSERTION"
            except urllib.error.HTTPError as e:
                if e.code != 404:  # 404 = GitHub detected no licence file
                    raise
                r["license"] = "NOASSERTION"
    m["resolved_at"] = m.get("resolved_at") or _now()
    _write_json(manifest_path, m)
    return m


# --------------------------------------------------------------------------- 2. mine
def checkout(url: str, commit: str, cache_dir: str) -> str:
    """Shallow checkout of ``commit`` under ``cache_dir`` (reused when already at that commit)."""
    dest = os.path.join(cache_dir, _slug(url).replace("/", "__"))
    git = ["git", "-C", dest]
    if os.path.isdir(os.path.join(dest, ".git")):
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if head == commit:
            return dest
    else:
        os.makedirs(dest, exist_ok=True)
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["remote", "add", "origin", url], check=True)
    subprocess.run(git + ["fetch", "-q", "--depth", "1", "origin", commit], check=True)
    subprocess.run(git + ["checkout", "-q", "--force", "FETCH_HEAD"], check=True)
    return dest


def _license_file(root: str) -> str | None:
    for name in sorted(os.listdir(root)):
        if re.match(r"(LICEN[CS]E|COPYING)(\.|$|-)", name, re.IGNORECASE):
            return name
    return None


def mine_repository(entry: dict, cache_dir: str, max_functions: int | None = None) -> tuple[list, dict]:
    root = checkout(entry["url"], entry["commit"], cache_dir)
    lang = entry["language"]
    exts = {e for e, l in extract.EXTENSIONS.items() if l == lang}
    if lang == "cpp":
        exts |= {".h", ".hpp", ".hh", ".hxx"}
    samples, files = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for fn in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            if os.path.splitext(fn)[1] not in exts or EXCLUDED_PATH.search(rel):
                continue
            try:
                src = open(os.path.join(dirpath, fn), encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            files += 1
            for f in extract.functions(src, lang):
                if f.end_line - f.start_line + 1 < MIN_LINES:
                    continue
                sid = f"gp:{_h(entry['url'], entry['commit'], rel, f.start_line)[:16]}"
                samples.append(make_sample(
                    sample_id=sid, dataset=DATASET_NAME, repository_url=entry["url"], commit_sha=entry["commit"],
                    file_path=rel, start_line=f.start_line, end_line=f.end_line, language=lang, symbol=f.name,
                    code=f.code, labels={}, license=entry["license"],
                    provenance=f"{entry['url']}@{entry['commit']}:{rel}#L{f.start_line}-L{f.end_line}"))
    if max_functions and len(samples) > max_functions:
        samples = sorted(samples, key=lambda s: _h("cap", s.sample_id))[:max_functions]
    lic = _license_file(root)
    audit = {"url": entry["url"], "commit": entry["commit"], "language": lang, "license": entry["license"],
             "license_file": lic, "license_allowed": entry["license"] in LICENSE_ALLOWLIST and lic is not None,
             "files": files, "functions": len(samples)}
    return samples, audit


def mine(manifest_path: str, out_path: str, cache_dir: str, admission_path: str,
         max_functions_per_repo: int | None = 4000) -> pd.DataFrame:
    m = _read_json(manifest_path)
    unresolved = [r["url"] for r in m["repositories"] if not r.get("commit") or not r.get("license")]
    if unresolved:
        raise RuntimeError(f"run `resolve` first; unpinned repositories: {unresolved}")
    samples, audits = [], []
    for r in m["repositories"]:
        s, a = mine_repository(r, cache_dir, max_functions_per_repo)
        print(f"{a['language']:>6} {a['url']}: {a['files']} files, {a['functions']} functions, "
              f"licence {a['license']} ({a['license_file']})", flush=True)
        samples += s
        audits.append(a)
    df = samples_to_frame(samples)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    write_dataset(df, out_path)
    ok = all(a["license_allowed"] and re.fullmatch(r"[0-9a-f]{40}", a["commit"]) for a in audits)
    _write_json(admission_path, {
        "pass": ok, "datasets": [DATASET_NAME], "manifest": manifest_path, "created_at": _now(),
        "criteria": "every repository pinned to a full commit SHA, SPDX licence in the allow-list, licence "
                    "file present at that commit; labels are produced by our own rubric/1 annotation",
        "extraction_version": extract.EXTRACTION_VERSION, "repositories": audits,
        "report": "reports/gate0/dataset_admission_report.md"})
    return decode_frame(df)


# --------------------------------------------------------------------------- 3. batches
def reserve_test_families(families_by_language: dict[str, list[str]], seed: int, frac: float) -> dict[str, list[str]]:
    """Whole families per language, chosen by seeded hash only (never by labels or metrics)."""
    out = {}
    for lang, fams in sorted(families_by_language.items()):
        fams = sorted(set(fams), key=lambda f: _h(seed, "reserve", f))
        k = min(len(fams) - 1, max(2, math.ceil(frac * len(fams)))) if len(fams) > 1 else 0
        out[lang] = sorted(fams[:k])
    return out


def band(smell: str, metrics: dict) -> str | None:
    v = metrics[rubric.RULES[smell]["metric"]]
    for name, (lo, hi) in BANDS[smell].items():
        if v >= lo and (hi is None or v <= hi):
            return name
    return None


def select(pool: pd.DataFrame, smells, seed: int, reserved: dict[str, list[str]], targets: dict | None = None,
           max_family_share: float = 0.35) -> pd.DataFrame:
    """Metric-stratified draw per (partition, language, smell, band) with a per-family cap.

    One row per selected sample with ``partition`` (test/trainval) and the stratum that drew it.
    """
    targets = targets or DEFAULT_TARGETS
    pool = pool.copy()
    test_fams = {f for fams in reserved.values() for f in fams}
    pool["partition"] = ["test" if f in test_fams else "trainval" for f in pool["repository_family_id"]]
    # one representative per normalized-exact duplicate, so reviewers never label the same code twice
    pool["_norm"] = [leakage.exact_hash(c, l) for c, l in zip(pool["code"], pool["language"])]
    pool = pool.sort_values("sample_id").drop_duplicates("_norm")
    pool["_order"] = [_h(seed, "draw", s) for s in pool["sample_id"]]
    pool = pool.sort_values("_order")
    chosen: dict[str, str] = {}
    for (part, lang), sub in pool.groupby(["partition", "language"], sort=True):
        for smell in smells:
            bands = [band(smell, m) for m in sub["static_metrics"]]
            for b, n in targets[part].items():
                cand = sub[[x == b for x in bands]]
                cap = max(1, math.ceil(max_family_share * n)) if cand["repository_family_id"].nunique() > 2 else n
                per_fam: dict[str, int] = defaultdict(int)
                got = sum(1 for sid in cand["sample_id"] if chosen.get(sid) == f"{smell}:{b}")
                for sid, fam in zip(cand["sample_id"], cand["repository_family_id"]):
                    if got >= n:
                        break
                    if sid in chosen or per_fam[fam] >= cap:
                        continue
                    chosen[sid] = f"{smell}:{b}"
                    per_fam[fam] += 1
                    got += 1
    out = pool[pool["sample_id"].isin(chosen)].drop(columns=["_norm", "_order"]).copy()
    out["stratum"] = out["sample_id"].map(chosen)
    return out.sort_values("sample_id").reset_index(drop=True)


def assign_reviewers(sel: pd.DataFrame, reviewers: list[str], seed: int, double_review_frac: float) -> pd.DataFrame:
    """Test items go to every reviewer; train/val items go to the first reviewer, and a seeded
    ``double_review_frac`` of them (per language) to every reviewer as well."""
    rows = []
    for (part, lang), sub in sel.groupby(["partition", "language"], sort=True):
        ids = sorted(sub["sample_id"], key=lambda s: _h(seed, "double", s))
        k = len(ids) if part == "test" else math.ceil(double_review_frac * len(ids))
        for i, sid in enumerate(ids):
            for r in (reviewers if i < k else reviewers[:1]):
                rows.append({"sample_id": sid, "reviewer": r, "double": i < k})
    return pd.DataFrame(rows)


def _item_id(seed, sid) -> str:
    return "i" + _h(seed, "item", sid)[:10]


def _render_item(item_id: str, r) -> str:
    m = r["static_metrics"]
    fence = {"cpp": "cpp", "python": "python"}.get(r["language"], r["language"])
    return (f"### {item_id}\n\n{rubric.LANGUAGE_NAMES[r['language']]} · lines {r['end_line'] - r['start_line'] + 1}"
            f" · effective LOC {m['effective_loc']} · parameters {m['parameter_count']}"
            f" · max nesting {m['max_nesting']}\n\n```{fence}\n{r['code']}\n```\n")


def write_batches(pool_path: str, out_dir: str, smells, reviewers: list[str], seed: int, test_frac: float = 0.25,
                  double_review_frac: float = 0.2, batch_size: int = 40, targets: dict | None = None) -> dict:
    pool = decode_frame(pd.read_parquet(pool_path))
    fams = pool.groupby("language")["repository_family_id"].apply(list).to_dict()
    reserved = reserve_test_families(fams, seed, test_frac)
    sel = select(pool, smells, seed, reserved, targets)
    sel["item_id"] = [_item_id(seed, s) for s in sel["sample_id"]]
    assign = assign_reviewers(sel, reviewers, seed, double_review_frac)
    os.makedirs(out_dir, exist_ok=True)
    by_id = sel.set_index("sample_id")
    batches = []
    for rev in reviewers:
        ids = sorted(assign.loc[assign["reviewer"] == rev, "sample_id"], key=lambda s: _h(seed, "order", rev, s))
        for b in range(0, len(ids), batch_size):
            chunk = ids[b:b + batch_size]
            name = f"{rev}_batch{b // batch_size + 1:02d}"
            with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
                f.write(f"# Labelling batch {name} (reviewer {rev}, {rubric.RUBRIC_VERSION})\n\n"
                        f"Label each item in `{name}.csv` using docs/annotation_rubric.md. Status: positive, "
                        f"negative or unknown. Confidence: 0 to 1. Do not consult other reviewers or models.\n\n")
                for sid in chunk:
                    f.write(_render_item(by_id.loc[sid, "item_id"], by_id.loc[sid]) + "\n")
            with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(SHEET_FIELDS + [c for s in smells for c in (s, f"{s}_confidence")] + ["notes"])
                for sid in chunk:
                    r = by_id.loc[sid]
                    m = r["static_metrics"]
                    w.writerow([r["item_id"], r["language"], r["end_line"] - r["start_line"] + 1, m["effective_loc"],
                                m["parameter_count"], m["max_nesting"]] + [""] * (2 * len(smells)) + [""])
            batches.append({"batch": name, "reviewer": rev, "items": len(chunk)})
    key = sel[["item_id", "sample_id", "partition", "language", "repository_family_id", "stratum"]].merge(
        assign.groupby("sample_id")["double"].first().reset_index(), on="sample_id")
    key.to_csv(os.path.join(out_dir, "KEY_do_not_share.csv"), index=False)
    sel.drop(columns=["item_id"]).assign(labels=[json.dumps(v, sort_keys=True) for v in sel["labels"]],
                                         static_metrics=[json.dumps(v, sort_keys=True) for v in sel["static_metrics"]]
                                         ).to_parquet(os.path.join(out_dir, "selected.parquet"), index=False)
    summary = {
        "annotation_version": ANNOTATION_VERSION, "rubric_version": rubric.RUBRIC_VERSION, "seed": seed,
        "created_at": _now(), "pool": pool_path, "smells": list(smells), "reviewers": reviewers,
        "reserved_test_families": reserved, "test_frac_families": test_frac,
        "double_review_frac_trainval": double_review_frac, "targets": targets or DEFAULT_TARGETS,
        "bands": {s: BANDS[s] for s in smells},
        "selected": {"total": len(sel), "by_partition_language": {
            f"{p}/{l}": int(n) for (p, l), n in sel.groupby(["partition", "language"]).size().items()},
            "by_stratum": sel["stratum"].value_counts().sort_index().to_dict()},
        "assignments": {r: int((assign["reviewer"] == r).sum()) for r in reviewers},
        "double_reviewed_items": int(assign.groupby("sample_id")["double"].first().sum()),
        "batches": batches}
    _write_json(os.path.join(out_dir, "batch_manifest.json"), summary)
    _write_json(os.path.join(out_dir, "split_reservation.json"),
                {"seed": seed, "test_families": sorted(f for fs in reserved.values() for f in fs),
                 "by_language": reserved})
    return summary


# --------------------------------------------------------------------------- 4. agreement
def read_reviews(batch_dir: str, smells) -> pd.DataFrame:
    """Long frame: item_id, reviewer, smell, status, confidence, notes (blank statuses are dropped)."""
    rows, problems = [], []
    for fn in sorted(os.listdir(batch_dir)):
        m = re.fullmatch(r"(.+)_batch\d+\.csv", fn)
        if not m:
            continue
        with open(os.path.join(batch_dir, fn), newline="") as f:
            for rec in csv.DictReader(f):
                for s in smells:
                    st = (rec.get(s) or "").strip().lower()
                    if not st:
                        continue
                    if st not in LABEL_STATUSES:
                        problems.append(f"{fn} {rec['item_id']} {s}: {st!r}")
                        continue
                    c = (rec.get(f"{s}_confidence") or "").strip()
                    rows.append({"item_id": rec["item_id"], "reviewer": m.group(1), "smell": s, "status": st,
                                 "confidence": float(c) if c else None, "notes": rec.get("notes", "")})
    if problems:
        raise ValueError("invalid statuses: " + "; ".join(problems[:20]))
    return pd.DataFrame(rows, columns=["item_id", "reviewer", "smell", "status", "confidence", "notes"])


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    cats = sorted(set(a) | set(b))
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def agreement_stats(batch_dir: str, smells, reviewers: list[str]) -> tuple[dict, pd.DataFrame]:
    """-> (per-partition raw agreement and Cohen's kappa per smell, disagreement rows)."""
    key = pd.read_csv(os.path.join(batch_dir, "KEY_do_not_share.csv"))
    rev = read_reviews(batch_dir, smells).merge(key[["item_id", "partition"]], on="item_id")
    ra, rb = reviewers[:2]
    wide = rev.pivot_table(index=["item_id", "partition", "smell"], columns="reviewer", values="status",
                           aggfunc="first").reset_index()
    both = wide.dropna(subset=[ra, rb]) if {ra, rb} <= set(wide.columns) else wide.iloc[:0].assign(**{ra: [], rb: []})
    out = {"reviewers": [ra, rb], "rubric_version": rubric.RUBRIC_VERSION, "computed_at": _now(), "by_partition": {}}
    for part in ("test", "trainval"):
        out["by_partition"][part] = {}
        for s in smells:
            sub = both[(both["partition"] == part) & (both["smell"] == s)]
            a, b = list(sub[ra]), list(sub[rb])
            out["by_partition"][part][s] = {
                "n": len(sub), "kappa": cohen_kappa(a, b),
                "raw_agreement": round(sum(x == y for x, y in zip(a, b)) / len(sub), 4) if len(sub) else None}
    dis = both[both[ra] != both[rb]][["item_id", "partition", "smell", ra, rb]]
    reviewed = set(rev["item_id"])
    out["disagreements"] = int(len(dis))
    out["items_without_review"] = int((~key["item_id"].isin(reviewed)).sum())
    return out, dis


def agreement(batch_dir: str, smells, reviewers: list[str]) -> dict:
    """Write ``agreement.json`` and ``adjudication.csv``; adjudications already entered are kept."""
    out, dis = agreement_stats(batch_dir, smells, reviewers)
    path = os.path.join(batch_dir, "adjudication.csv")
    cols = ["adjudicated_status", "adjudicator", "rationale"]
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        dis = dis.merge(old[["item_id", "smell"] + cols], on=["item_id", "smell"], how="left").fillna("")
    else:
        dis = dis.assign(**{c: "" for c in cols})
    dis.to_csv(path, index=False)
    _write_json(os.path.join(batch_dir, "agreement.json"), out)
    return out


# --------------------------------------------------------------------------- 5. finalize
def finalize(batch_dir: str, smells, reviewers: list[str], out_parquet: str, gate_b_path: str,
             min_test_kappa: float = 0.60) -> dict:
    """Build the primary dataset.  Doubly reviewed items become ``gold`` once every disagreement is
    adjudicated; singly reviewed train/val items stay ``weak``.  Test items without two reviews and an
    adjudicated outcome keep an ``unknown`` label, so they can never be scored."""
    key = pd.read_csv(os.path.join(batch_dir, "KEY_do_not_share.csv"))
    sel = decode_frame(pd.read_parquet(os.path.join(batch_dir, "selected.parquet")))
    rev = read_reviews(batch_dir, smells)
    agr, _ = agreement_stats(batch_dir, smells, reviewers)
    adj_path = os.path.join(batch_dir, "adjudication.csv")
    adj = pd.read_csv(adj_path, dtype=str, keep_default_na=False) if os.path.exists(adj_path) else pd.DataFrame()
    adj_map = {(r["item_id"], r["smell"]): r for _, r in adj.iterrows()}
    bad = [k for k, r in adj_map.items() if r["adjudicated_status"].strip().lower() not in LABEL_STATUSES]
    item_of = dict(zip(key["sample_id"], key["item_id"]))
    part_of = dict(zip(key["sample_id"], key["partition"]))
    by_item = {k: g for k, g in rev.groupby(["item_id", "smell"])}
    labels, counts = [], defaultdict(int)
    for sid in sel["sample_id"]:
        lab = {}
        for s in smells:
            g = by_item.get((item_of[sid], s))
            if g is None:
                lab[s] = Label(UNKNOWN, WEAK, source="unreviewed", rubric_version=rubric.RUBRIC_VERSION)
                counts["unreviewed"] += 1
                continue
            who = sorted(g["reviewer"].unique())
            statuses = set(g["status"])
            conf = g["confidence"].dropna()
            conf = round(float(conf.mean()), 3) if len(conf) else None
            if len(who) >= 2:
                if len(statuses) == 1:
                    st, adjudicated, src = statuses.pop(), True, "agreed"
                else:
                    a = adj_map.get((item_of[sid], s))
                    ok = a is not None and (item_of[sid], s) not in bad
                    st = a["adjudicated_status"].strip().lower() if ok else UNKNOWN
                    adjudicated, src = ok, f"adjudicated:{a['adjudicator']}" if ok else "disagreement-pending"
                q = GOLD if adjudicated else WEAK
                if part_of[sid] == "test" and not adjudicated:
                    st = UNKNOWN
            else:
                st, q, adjudicated, src = g["status"].iloc[0], WEAK, False, "single-review"
                if part_of[sid] == "test":
                    st = UNKNOWN      # the locked test set is double-reviewed or unlabelled
            lab[s] = Label(st, q, source=f"{ANNOTATION_VERSION}/{src}", rubric_version=rubric.RUBRIC_VERSION,
                           confidence=conf, annotators=who, adjudicated=adjudicated)
            counts[f"{part_of[sid]}:{q}:{st}"] += 1
        labels.append({k: vars(v) for k, v in lab.items()})
    out = sel.copy()
    out["labels"] = labels
    out["rubric_version"] = rubric.RUBRIC_VERSION
    enc = out.drop(columns=["partition", "stratum"]).copy()
    enc["labels"] = [json.dumps(v, sort_keys=True) for v in enc["labels"]]
    enc["static_metrics"] = [json.dumps(v, sort_keys=True) for v in enc["static_metrics"]]
    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    write_dataset(enc, out_parquet)
    test_k = {s: agr["by_partition"]["test"][s]["kappa"] for s in smells}
    tv = key[key["partition"] == "trainval"]
    evidence = {
        "rubric_version": rubric.RUBRIC_VERSION, "annotation_version": ANNOTATION_VERSION, "reviewers": reviewers,
        "machine_reviewers": [r for r in reviewers if is_machine(r)],
        "human_gold": not any(is_machine(r) for r in reviewers),
        "test_kappa": test_k, "raw_agreement": {p: {s: v["raw_agreement"] for s, v in d.items()}
                                                for p, d in agr["by_partition"].items()},
        "trainval_kappa": {s: agr["by_partition"]["trainval"][s]["kappa"] for s in smells},
        "trainval_double_review_fraction": round(float(tv["double"].mean()), 4) if len(tv) else None,
        "unresolved_adjudications": len(bad), "label_counts": dict(sorted(counts.items())),
        "kappa_ok": all(k is not None and k >= min_test_kappa for k in test_k.values()),
        "min_test_kappa": min_test_kappa, "created_at": _now(), "batch_dir": batch_dir}
    _write_json(gate_b_path, evidence)
    return evidence


# --------------------------------------------------------------------------- CLI
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["resolve", "mine", "batches", "agreement", "finalize"])
    ap.add_argument("--config", required=True, help="stage config (smells, languages, dataset paths, seed)")
    ap.add_argument("--repos", default="configs/gold_pool_repos.json")
    ap.add_argument("--pool", default=None, help="default: data/gold_pool/<stage>_pool.parquet")
    ap.add_argument("--batch-dir", default=None, help="default: data/annotation/<stage>")
    ap.add_argument("--cache", default="data/raw/gold_pool")
    ap.add_argument("--reviewers", default="A,B")
    ap.add_argument("--batch-size", type=int, default=40)
    ap.add_argument("--test-frac", type=float, default=0.25, help="share of repository families reserved for test")
    ap.add_argument("--double-review-frac", type=float, default=0.2)
    a = ap.parse_args(argv)
    cfg = _read_json(a.config)
    stage = cfg["stage"]
    pool = a.pool or f"data/gold_pool/{stage}_pool.parquet"
    bdir = a.batch_dir or f"data/annotation/{stage}"
    reviewers = a.reviewers.split(",")
    if a.command == "resolve":
        resolve(a.repos)
    elif a.command == "mine":
        m = _read_json(a.repos)
        sub = {**m, "repositories": [r for r in m["repositories"] if r["language"] in cfg["languages"]]}
        tmp = os.path.join(os.path.dirname(pool) or ".", f"{stage}_repos.json")
        _write_json(tmp, sub)
        mine(tmp, pool, a.cache, cfg["dataset"]["gate0_admission"])
    elif a.command == "batches":
        s = write_batches(pool, bdir, cfg["smells"], reviewers, cfg["split"]["seed"], a.test_frac,
                          a.double_review_frac, a.batch_size, cfg.get("annotation", {}).get("targets"))
        print(json.dumps({k: s[k] for k in ("selected", "assignments", "double_reviewed_items")}, indent=2))
    elif a.command == "agreement":
        print(json.dumps(agreement(bdir, cfg["smells"], reviewers), indent=2))
    elif a.command == "finalize":
        e = finalize(bdir, cfg["smells"], reviewers, cfg["dataset"]["primary_parquet"],
                     cfg["dataset"]["gate_b_evidence"], cfg["label_quality"]["min_test_kappa"])
        print(json.dumps(e, indent=2))


if __name__ == "__main__":
    main()
