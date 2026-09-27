import csv
import json
import os
import subprocess

import pandas as pd
import pytest

from smellclm import annotation, rubric
from smellclm.datasets import make_sample
from smellclm.schema import GOLD, NEGATIVE, POSITIVE, UNKNOWN, WEAK, samples_to_frame, validate_frame, write_dataset

SMELLS = ["long_method", "long_parameter_list"]
TARGETS = {"test": {"likely_positive": 3, "borderline": 2, "likely_negative": 3},
           "trainval": {"likely_positive": 4, "borderline": 3, "likely_negative": 4}}


def _java(name, n_params, n_stmts):
    params = ", ".join(f"int p{i}" for i in range(n_params))
    body = "\n".join(f"        int v{i} = p0 * {i} + {name.count('_')};" for i in range(n_stmts))
    return f"    int {name}({params}) {{\n{body}\n        return 0;\n    }}"


def _python(name, n_params, n_stmts):
    params = ", ".join(f"p{i}" for i in range(n_params))
    body = "".join(f"    v{i} = p0 * {i}\n" for i in range(n_stmts))
    return f"def {name}({params}):\n{body}    return 0\n"


def _pool(n_fams=6, per_fam=36):
    samples = []
    for lang, gen in (("java", _java), ("python", _python)):
        for f in range(n_fams):
            url = f"https://github.com/org{f}/{lang}-repo"
            for k in range(per_fam):
                name = f"f{f}_{k}_{lang}"
                code = gen(name, 2 + k % 6, 6 + (k * 7 + f) % 44)
                samples.append(make_sample(
                    sample_id=f"gp:{lang}{f:02d}{k:03d}", dataset=annotation.DATASET_NAME, repository_url=url,
                    commit_sha="a" * 40, file_path=f"src/m{k}.{lang}", start_line=1,
                    end_line=code.count("\n") + 1, language=lang, symbol=name, code=code, labels={},
                    license="MIT", provenance=url))
    return samples_to_frame(samples)


@pytest.fixture
def batch(tmp_path):
    pool_path = str(tmp_path / "pool.parquet")
    write_dataset(_pool(), pool_path)
    out = str(tmp_path / "batches")
    summary = annotation.write_batches(pool_path, out, SMELLS, ["A", "B"], seed=11, batch_size=10, targets=TARGETS)
    return out, summary


def _sheets(out, reviewer):
    rows = []
    for fn in sorted(os.listdir(out)):
        if fn.startswith(f"{reviewer}_batch") and fn.endswith(".csv"):
            with open(os.path.join(out, fn), newline="") as f:
                rows += [(fn, r) for r in csv.DictReader(f)]
    return rows


def _fill(out, reviewer, flip: set = frozenset()):
    """Label every sheet row from the frozen rule; ``(item_id, smell)`` pairs in ``flip`` get the opposite."""
    by_file = {}
    for fn, r in _sheets(out, reviewer):
        m = {"effective_loc": int(r["effective_loc"]), "parameter_count": int(r["parameter_count"]),
             "max_nesting": int(r["max_nesting"])}
        for s in SMELLS:
            pos = rubric.rule_predict(s, m) == 1
            if (r["item_id"], s) in flip:
                pos = not pos
            r[s], r[f"{s}_confidence"] = (POSITIVE if pos else NEGATIVE), "0.8"
        by_file.setdefault(fn, []).append(r)
    for fn, rows in by_file.items():
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# --------------------------------------------------------------------------- reservation and sampling
def test_reserve_test_families_is_seeded_whole_family_and_order_free():
    fams = {"java": [f"j{i}" for i in range(8)], "python": [f"p{i}" for i in range(3)], "cpp": ["c0"]}
    a = annotation.reserve_test_families(fams, seed=1, frac=0.25)
    b = annotation.reserve_test_families({k: list(reversed(v)) * 2 for k, v in fams.items()}, seed=1, frac=0.25)
    assert a == b
    assert len(a["java"]) == 2 and set(a["java"]) <= set(fams["java"])
    assert len(a["python"]) == 2              # at least two, never all
    assert a["cpp"] == []                     # a single family cannot be held out
    assert annotation.reserve_test_families(fams, seed=2, frac=0.25) != a


def test_band_follows_rule_metric():
    assert annotation.band("long_method", {"effective_loc": 40}) == "likely_positive"
    assert annotation.band("long_method", {"effective_loc": 30}) == "borderline"
    assert annotation.band("long_method", {"effective_loc": 10}) == "likely_negative"
    assert annotation.band("long_method", {"effective_loc": 3}) is None
    assert annotation.band("long_parameter_list", {"parameter_count": 5}) == "borderline"
    assert annotation.band("deep_nesting", {"max_nesting": 0}) is None


@pytest.mark.parametrize("path,excluded", [
    ("src/main/java/Foo.java", False), ("lib/core.py", False), ("src/util.ts", False),
    ("tests/test_x.py", True), ("pkg/test_io.py", True), ("src/foo_test.go", True),
    ("src/FooTest.java", True), ("src/a.spec.ts", True), ("src/a.test.js", True), ("src/fp/add/test.ts", True),
    ("third_party/x.cc", True), ("dist/app.min.js", True), ("proto/x_pb2.py", True), ("src/testing/h.py", True),
])
def test_excluded_paths(path, excluded):
    assert bool(annotation.EXCLUDED_PATH.search(path)) is excluded


def test_select_draws_test_only_from_reserved_families():
    pool = annotation.decode_frame(_pool())
    fams = pool.groupby("language")["repository_family_id"].apply(list).to_dict()
    reserved = annotation.reserve_test_families(fams, seed=3, frac=0.25)
    sel = annotation.select(pool, SMELLS, seed=3, reserved=reserved, targets=TARGETS)
    test_fams = {f for fs in reserved.values() for f in fs}
    assert set(sel.loc[sel["partition"] == "test", "repository_family_id"]) <= test_fams
    assert not set(sel.loc[sel["partition"] == "trainval", "repository_family_id"]) & test_fams
    assert sel["sample_id"].is_unique
    counts = sel.groupby(["partition", "language", "stratum"]).size()
    for (part, _, stratum), n in counts.items():
        assert n <= TARGETS[part][stratum.split(":")[1]]
    again = annotation.select(pool.sample(frac=1, random_state=0), SMELLS, seed=3, reserved=reserved, targets=TARGETS)
    assert list(again["sample_id"]) == list(sel["sample_id"])


def test_assign_reviewers_doubles_all_test_and_a_share_of_trainval():
    sel = pd.DataFrame({"sample_id": [f"s{i}" for i in range(30)], "language": "java",
                        "partition": ["test"] * 10 + ["trainval"] * 20})
    a = annotation.assign_reviewers(sel, ["A", "B"], seed=1, double_review_frac=0.2)
    per = a.groupby("sample_id")["reviewer"].apply(set)
    assert all(per[f"s{i}"] == {"A", "B"} for i in range(10))
    tv = per[[f"s{i}" for i in range(10, 30)]]
    assert sum(v == {"A", "B"} for v in tv) == 4 and all("A" in v for v in tv)


# --------------------------------------------------------------------------- batches, agreement, finalize
def test_batches_are_blind_and_reservation_matches_key(batch):
    out, summary = batch
    key = pd.read_csv(os.path.join(out, "KEY_do_not_share.csv"))
    res = json.load(open(os.path.join(out, "split_reservation.json")))
    assert set(key.loc[key["partition"] == "test", "repository_family_id"]) <= set(res["test_families"])
    assert not set(key.loc[key["partition"] == "trainval", "repository_family_id"]) & set(res["test_families"])
    a_items = {r["item_id"] for _, r in _sheets(out, "A")}
    b_items = {r["item_id"] for _, r in _sheets(out, "B")}
    assert a_items == set(key["item_id"])
    assert set(key.loc[key["partition"] == "test", "item_id"]) <= b_items
    assert b_items == set(key.loc[key["double"], "item_id"])
    assert summary["assignments"] == {"A": len(a_items), "B": len(b_items)}
    for fn in os.listdir(out):
        if fn.startswith(("A_batch", "B_batch")):
            text = open(os.path.join(out, fn)).read()
            assert "gp:" not in text and "github.com" not in text and "likely_" not in text
    header = next(csv.reader(open(os.path.join(out, "A_batch01.csv"))))
    assert header == annotation.SHEET_FIELDS + ["long_method", "long_method_confidence", "long_parameter_list",
                                                "long_parameter_list_confidence", "notes"]


def test_agreement_adjudication_and_finalize(batch, tmp_path):
    out, _ = batch
    key = pd.read_csv(os.path.join(out, "KEY_do_not_share.csv"))
    flipped = key.loc[key["partition"] == "test", "item_id"].iloc[0]
    _fill(out, "A")
    _fill(out, "B", flip={(flipped, "long_method")})

    agr = annotation.agreement(out, SMELLS, ["A", "B"])
    t = agr["by_partition"]["test"]
    assert t["long_parameter_list"]["kappa"] == 1.0 and t["long_method"]["kappa"] < 1.0
    assert t["long_method"]["n"] == int((key["partition"] == "test").sum())
    assert agr["disagreements"] == 1 and agr["items_without_review"] == 0
    adj = pd.read_csv(os.path.join(out, "adjudication.csv"), dtype=str, keep_default_na=False)
    assert list(zip(adj["item_id"], adj["smell"])) == [(flipped, "long_method")]

    parquet, gate_b = str(tmp_path / "primary.parquet"), str(tmp_path / "gate_b.json")
    ev = annotation.finalize(out, SMELLS, ["A", "B"], parquet, gate_b)
    assert ev["unresolved_adjudications"] == 1
    df = annotation.decode_frame(pd.read_parquet(parquet))
    sid = key.set_index("item_id").loc[flipped, "sample_id"]
    lab = df.set_index("sample_id").loc[sid, "labels"]["long_method"]
    assert lab["status"] == UNKNOWN and lab["quality"] == WEAK      # pending test disagreement is never scored

    adj["adjudicated_status"], adj["adjudicator"], adj["rationale"] = POSITIVE, "A+B", "rubric: two jobs"
    adj.to_csv(os.path.join(out, "adjudication.csv"), index=False)
    annotation.agreement(out, SMELLS, ["A", "B"])                  # rerun keeps entered adjudications
    assert pd.read_csv(os.path.join(out, "adjudication.csv"))["adjudicator"].tolist() == ["A+B"]
    ev = annotation.finalize(out, SMELLS, ["A", "B"], parquet, gate_b)
    assert ev["unresolved_adjudications"] == 0 and ev == json.load(open(gate_b))
    assert ev["human_gold"] and ev["machine_reviewers"] == []
    assert annotation.is_machine("llm1") and not annotation.is_machine("A")
    df = annotation.decode_frame(pd.read_parquet(parquet))
    assert validate_frame(pd.read_parquet(parquet)) == []
    assert "partition" not in df.columns and "stratum" not in df.columns
    labels = df.set_index("sample_id")["labels"]
    assert labels[sid]["long_method"]["status"] == POSITIVE and labels[sid]["long_method"]["adjudicated"]
    part = dict(zip(key["sample_id"], key["partition"]))
    double = dict(zip(key["sample_id"], key["double"]))
    for s, lab in labels.items():
        for smell in SMELLS:
            if part[s] == "test" or double[s]:
                assert lab[smell]["quality"] == GOLD and lab[smell]["annotators"] == ["A", "B"]
            else:
                assert lab[smell]["quality"] == WEAK and lab[smell]["annotators"] == ["A"]
                assert lab[smell]["status"] in (POSITIVE, NEGATIVE)


def test_single_review_never_labels_test(batch, tmp_path):
    out, _ = batch
    _fill(out, "A")                                                  # B never hands in
    ev = annotation.finalize(out, SMELLS, ["A", "B"], str(tmp_path / "p.parquet"), str(tmp_path / "g.json"))
    assert ev["test_kappa"] == {s: None for s in SMELLS} and not ev["kappa_ok"]
    key = pd.read_csv(os.path.join(out, "KEY_do_not_share.csv"))
    test_ids = set(key.loc[key["partition"] == "test", "sample_id"])
    df = annotation.decode_frame(pd.read_parquet(str(tmp_path / "p.parquet")))
    for s, lab in zip(df["sample_id"], df["labels"]):
        if s in test_ids:
            assert all(lab[m]["status"] == UNKNOWN for m in SMELLS)


def test_read_reviews_rejects_invalid_status(batch):
    out, _ = batch
    path = os.path.join(out, "A_batch01.csv")
    rows = list(csv.DictReader(open(path, newline="")))
    rows[0]["long_method"] = "maybe"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="maybe"):
        annotation.read_reviews(out, SMELLS)


def test_cohen_kappa():
    assert annotation.cohen_kappa([POSITIVE, POSITIVE, NEGATIVE, NEGATIVE],
                                  [POSITIVE, NEGATIVE, NEGATIVE, NEGATIVE]) == 0.5
    assert annotation.cohen_kappa([NEGATIVE] * 3, [NEGATIVE] * 3) == 1.0
    assert annotation.cohen_kappa([], []) is None


# --------------------------------------------------------------------------- mining (local git, no network)
def test_mine_repository_from_pinned_local_commit(tmp_path):
    repo = tmp_path / "upstream"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "LICENSE").write_text("MIT License\n")
    (repo / "src" / "core.py").write_text(_python("work", 3, 5) + "\n" + _python("tiny", 1, 1))
    (repo / "tests" / "test_core.py").write_text(_python("test_work", 1, 4))
    git = ["git", "-C", str(repo), "-c", "commit.gpgsign=false"]
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-q", "-m", "init"], check=True, env=env)
    sha = subprocess.run(git[:3] + ["rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    entry = {"url": f"file://{repo}", "commit": sha, "language": "python", "license": "MIT"}
    samples, audit = annotation.mine_repository(entry, str(tmp_path / "cache"))
    assert [s.symbol for s in samples] == ["work", "tiny"]
    assert all(s.commit_sha == sha and s.file_path == "src/core.py" and s.labels == {} for s in samples)
    assert audit["license_allowed"] and audit["license_file"] == "LICENSE" and audit["files"] == 1
    again, _ = annotation.mine_repository(entry, str(tmp_path / "cache"))   # cached checkout is reused
    assert [s.sample_id for s in again] == [s.sample_id for s in samples]
