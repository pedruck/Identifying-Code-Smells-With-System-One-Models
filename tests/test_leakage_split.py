import pandas as pd

from smellclm import leakage, split
from smellclm.schema import NEGATIVE, POSITIVE


def _frame(n_repos=40, per_repo=5):
    rows = []
    for r in range(n_repos):
        for k in range(per_repo):
            lang = ["java", "python", "cpp"][r % 3]
            body = " ".join(f"v{r}_{k}_{j} = {j} * {r};" for j in range(8 + 5 * r + k))
            code = f"void m{k}(int a) {{ {body} }}" if lang != "python" else f"def m{k}(a):\n" + "".join(f"    x{j} = a\n" for j in range(3 + 5 * r + k))
            rows.append({"sample_id": f"s{r:03d}_{k}", "repository_family_id": f"github.com/o/r{r}",
                         "language": lang, "code": code,
                         "labels": {"long_method": {"status": POSITIVE if (r + k) % 3 == 0 else NEGATIVE}}})
    return pd.DataFrame(rows)


def test_family_id_normalization_and_fork_map():
    assert leakage.family_id("https://GitHub.com/Owner/Repo.git") == "github.com/owner/repo"
    assert leakage.family_id("git@github.com:owner/repo.git") == "github.com/owner/repo"
    fm = {"github.com/fork/repo": "github.com/owner/repo"}
    assert leakage.family_id("https://github.com/fork/repo", fm) == "github.com/owner/repo"


def test_exact_and_renamed_duplicates_merge_groups():
    df = _frame()
    # copy one method into another repository (exact clone) and a renamed clone into a third
    src = df.iloc[0]["code"]
    df.loc[len(df)] = {"sample_id": "dup", "repository_family_id": "github.com/x/copy", "language": "java",
                       "code": "// copied\n" + src, "labels": {"long_method": {"status": NEGATIVE}}}
    df.loc[len(df)] = {"sample_id": "ren", "repository_family_id": "github.com/y/other", "language": "java",
                       "code": src.replace("v0_0_", "renamed_"), "labels": {"long_method": {"status": NEGATIVE}}}
    df = leakage.add_hashes(df)
    groups, rep = leakage.split_groups(df)
    g = dict(zip(df["sample_id"], groups))
    assert g["dup"] == g["s000_0"] == g["ren"]
    assert set(rep["kind"]) >= {"exact"}
    df["split_group"] = groups
    df["split"] = split.assign_splits(df, ["long_method"], seed=7)
    assert leakage.assert_no_leakage(df) == []


def test_split_is_deterministic_grouped_and_near_target():
    df = leakage.add_hashes(_frame())
    df["split_group"], _ = leakage.split_groups(df)
    a = split.assign_splits(df, ["long_method"], seed=3)
    b = split.assign_splits(df.sample(frac=1, random_state=1).sort_index(), ["long_method"], seed=3)
    assert (a == b).all()
    df["split"] = a
    assert df.groupby("split_group")["split"].nunique().max() == 1
    frac = df["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.08 and abs(frac["test"] - 0.15) < 0.08
    m = split.manifest(df, seed=3)
    assert split.verify_manifest(df, m)
    df.loc[df.index[0], "split"] = "test" if df["split"].iloc[0] != "test" else "train"
    assert not split.verify_manifest(df, m)


def test_near_duplicates_detected():
    base = "int f(int a) { int s = 0; for (int i = 0; i < a; i++) { s += i * 2; s -= 1; } return s + a * 3 - 7; }"
    near = base.replace("s + a * 3 - 7", "s + a * 3 - 8")
    far = "void g() { System.out.println(\"hello world\"); return; }"
    pairs = leakage.near_duplicate_pairs([base, near, far], ["java"] * 3, threshold=0.7)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_support_gates():
    df = _frame()
    df["split"] = "test"
    g = split.support_gates(df, ["long_method"], ["java", "python", "cpp"], per_smell=10, per_language=5, per_cell=5)
    assert g["per_smell"]["long_method"]["pass"]
    assert g["pass"]


def test_dedupe_exact_keeps_first_and_drops_conflicts():
    df = pd.DataFrame([
        {"sample_id": "b", "content_hash_normalized": "h1", "labels": {"long_method": {"status": POSITIVE}}},
        {"sample_id": "a", "content_hash_normalized": "h1", "labels": {"long_method": {"status": POSITIVE}}},
        {"sample_id": "c", "content_hash_normalized": "h2", "labels": {"long_method": {"status": POSITIVE}}},
        {"sample_id": "d", "content_hash_normalized": "h2", "labels": {"long_method": {"status": NEGATIVE}}},
        {"sample_id": "e", "content_hash_normalized": "h3", "labels": {"long_method": {"status": NEGATIVE}}},
    ])
    kept, rep = leakage.dedupe_exact(df)
    assert sorted(kept["sample_id"]) == ["a", "e"]
    assert dict(zip(rep["sample_id"], rep["reason"])) == {
        "b": "exact_duplicate", "c": "exact_duplicate_label_conflict", "d": "exact_duplicate_label_conflict"}
