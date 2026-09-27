import numpy as np
import pandas as pd

from smellclm import evaluate


def _preds(seed, flip, n_fam=30, per=10):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_fam):
        for k in range(per):
            y = int(rng.random() < 0.4)
            pred = y if rng.random() > flip else 1 - y
            rows.append({"decision_id": f"{f}_{k}", "smell": ["long_method", "long_parameter_list"][k % 2],
                         "language": ["java", "python"][f % 2], "repository_family_id": f"fam{f}",
                         "label": y, "pred": pred})
    return pd.DataFrame(rows)


def test_prf_and_ap():
    m = evaluate.prf(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert evaluate.average_precision(np.array([1, 0, 1]), np.array([0.9, 0.8, 0.1])) == (1 + 2 / 3) / 2


def test_thresholds_are_chosen_on_validation():
    val = pd.DataFrame({"smell": ["long_method"] * 4, "label": [0, 0, 1, 1], "margin": [-2.0, 1.0, 2.0, 3.0]})
    t = evaluate.select_thresholds(val, ["long_method"])
    assert 1.0 <= t["long_method"]["threshold"] < 2.0 and t["long_method"]["val_f1"] == 1.0
    assert list(evaluate.apply_thresholds(val, t)) == [0, 0, 1, 1]


def test_paired_bootstrap_detects_improvement():
    a = _preds(0, flip=0.40)
    b = a.copy()
    rng = np.random.default_rng(1)
    b["pred"] = np.where(rng.random(len(b)) < 0.9, b["label"], 1 - b["label"])
    r = evaluate.paired_bootstrap(a, b, ["long_method", "long_parameter_list"], n_boot=300)
    assert r["delta_macro_f1"] > 0.2 and r["ci95"][0] > 0
    assert r["clusters"] == 30


def test_success_rule_requires_every_seed():
    smells = ["long_method", "long_parameter_list"]
    zs = _preds(0, flip=0.45)
    good = zs.copy(); good["pred"] = good["label"]
    bad = zs.copy()
    ok = evaluate.success_rule(zs, {1: good, 2: good, 3: good}, smells, True, True, n_boot=200)
    assert ok["pass"]
    mixed = evaluate.success_rule(zs, {1: good, 2: good, 3: bad}, smells, True, True, n_boot=200)
    assert not mixed["pass"]
    leak = evaluate.success_rule(zs, {1: good}, smells, False, True, n_boot=100)
    assert not leak["pass"]
