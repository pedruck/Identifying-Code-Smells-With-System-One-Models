import json

import pandas as pd

from smellclm import gates, rubric, split


def _dec(n_pos=60, n_neg=60, gold=True, lang="java", smell="long_method"):
    rows = [{"decision_id": f"{lang}{i}::{smell}", "language": lang, "smell": smell, "label": int(i < n_pos),
             "label_quality": "gold" if gold else "weak", "label_gold": gold} for i in range(n_pos + n_neg)]
    return pd.DataFrame(rows)


def _cfg(tmp_path, kappa=0.7, gate0=True, smells=("long_method",)):
    g0, gb = tmp_path / "g0.json", tmp_path / "gb.json"
    g0.write_text(json.dumps({"pass": gate0, "datasets": ["x"]}))
    gb.write_text(json.dumps({"rubric_version": rubric.RUBRIC_VERSION, "test_kappa": {s: kappa for s in smells}}))
    return {"smells": list(smells), "languages": ["java"],
            "dataset": {"gate0_admission": str(g0), "gate_b_evidence": str(gb)},
            "label_quality": {"min_reviewers": 2, "min_test_kappa": 0.6}}


def test_is_gold_needs_adjudication_and_two_reviewers():
    assert gates.is_gold({"quality": "gold", "adjudicated": True, "annotators": ["A", "B"]})
    assert not gates.is_gold({"quality": "gold", "adjudicated": False, "annotators": ["A", "B"]})
    assert not gates.is_gold({"quality": "gold", "adjudicated": True, "annotators": ["A"]})
    assert not gates.is_gold({"quality": "weak", "adjudicated": True, "annotators": ["A", "B"]})


def test_primary_gate_passes_only_with_all_evidence(tmp_path):
    dec = _dec()
    sup = split.decision_support_gates(dec, ["long_method"], ["java"])
    assert gates.primary_test_gate(_cfg(tmp_path), dec, sup)["pass"]
    weak = pd.concat([dec, _dec(1, 0, gold=False).assign(decision_id="w::long_method")])
    g = gates.primary_test_gate(_cfg(tmp_path), weak, sup)
    assert not g["pass"] and g["labels"]["non_gold"] == 1
    assert not gates.primary_test_gate(_cfg(tmp_path, kappa=0.5), dec, sup)["pass"]
    assert not gates.primary_test_gate(_cfg(tmp_path, gate0=False), dec, sup)["pass"]
    cfg = _cfg(tmp_path)
    ev = json.load(open(cfg["dataset"]["gate_b_evidence"]))
    json.dump({**ev, "machine_reviewers": ["llm1"]}, open(cfg["dataset"]["gate_b_evidence"], "w"))
    assert not gates.primary_test_gate(cfg, dec, sup)["pass"]         # LLM-assisted test labels are silver
    cfg["dataset"]["gate_b_evidence"] = str(tmp_path / "missing.json")
    assert gates.failures(gates.primary_test_gate(cfg, dec, sup))


def test_retained_support_drops_below_gate_after_truncation():
    dec = _dec(55, 55)
    assert split.decision_support_gates(dec, ["long_method"], ["java"])["pass"]
    kept = dec.iloc[10:]             # e.g. 10 positives excluded for truncation
    assert not split.decision_support_gates(kept, ["long_method"], ["java"])["pass"]


def test_stage_test_gate_refuses_primary_and_marks_smoke(tmp_path):
    import pytest
    from smellclm import experiment, pipeline
    cfg = {**_cfg(tmp_path), "support": {"per_smell": 50, "per_language": 50, "per_cell": 20}}
    weak = _dec(gold=False)
    cfg["dataset"]["mode"] = "primary"
    with pytest.raises(RuntimeError, match="not adjudicated gold"):
        pipeline._test_gate(experiment.Experiment(str(tmp_path / "p"), cfg), weak)
    cfg["dataset"]["mode"] = "smoke"
    assert not pipeline._test_gate(experiment.Experiment(str(tmp_path / "s"), cfg), weak)["pass"]
    cfg["dataset"]["mode"] = "primary"
    assert pipeline._test_gate(experiment.Experiment(str(tmp_path / "g"), cfg), _dec())["pass"]
