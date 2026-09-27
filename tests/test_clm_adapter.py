import json

import pandas as pd
import pytest

from smellclm import official

pytestmark = pytest.mark.skipif(not official.available(), reason="official CLM not checked out (tools/setup_clm.sh)")


def _df():
    return pd.DataFrame([
        {"sample_id": "a", "dataset": "t", "repository_family_id": "f1", "split_group": "g:a", "split": "train",
         "language": "java", "sample_type": "method", "code": "void a() {}", "context": "",
         "labels": {"long_method": {"status": "positive", "quality": "gold"},
                    "long_parameter_list": {"status": "unknown", "quality": "weak"}},
         "static_metrics": {"effective_loc": 1, "parameter_count": 0, "max_nesting": 0}},
        {"sample_id": "b", "dataset": "t", "repository_family_id": "f2", "split_group": "g:b", "split": "val",
         "language": "python", "sample_type": "method", "code": "def b(x):\n    return x", "context": "",
         "labels": {"long_method": {"status": "negative", "quality": "gold"},
                    "long_parameter_list": {"status": "negative", "quality": "gold"}},
         "static_metrics": {"effective_loc": 2, "parameter_count": 1, "max_nesting": 0}},
    ])


def test_decisions_skip_unknown_and_use_official_text():
    from smellclm.clm_adapter import decisions
    d = decisions(_df(), ["long_method", "long_parameter_list"])
    assert list(d["decision_id"]) == ["a::long_method", "b::long_method", "b::long_parameter_list"]
    row = d.iloc[0]
    assert row["state_text"].startswith("language: Java")
    assert row["state_text"].endswith("Does this code have the Long Method code smell?")
    assert row["cand_true"].startswith("true: Yes.") and row["cand_false"].startswith("false: No.")


def test_trainer_rows_roundtrip_through_official_adapter(tmp_path):
    official.ensure_path()
    import adapters
    from smellclm.clm_adapter import write_trainer_data
    info = write_trainer_data(_df(), ["long_method", "long_parameter_list"], str(tmp_path))
    assert info["counts"]["train"]["questions"] == 1 and info["counts"]["test"]["questions"] == 2
    import pyarrow.parquet as pq
    rows = pq.read_table(str(tmp_path / "smell" / "train-00000.parquet")).to_pylist()
    ex = list(adapters.typed_decision_examples(rows))
    assert len(ex) == 1 and ex[0].keys == ["false", "true"] and ex[0].label == 1
    assert ex[0].group == "g:a"
    assert json.loads(rows[0]["questions"]).keys() == {"long_method"}
