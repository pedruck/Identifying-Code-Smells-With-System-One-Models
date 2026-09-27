import json

import pytest

from smellclm import experiment, pipeline


def test_environment_retry_preserves_original_hardware_error(tmp_path, monkeypatch):
    cfg = {"clm": {"repo_commit": "expected"}}
    exp = experiment.Experiment(str(tmp_path), cfg)
    monkeypatch.setattr(experiment, "environment", lambda: {"gpu": {"name": "A100"}})
    monkeypatch.setattr(experiment, "feasibility", lambda env: {
        "pass": False, "reasons": ["vllm not importable"]})
    monkeypatch.setattr(pipeline.official, "checked_out_commit", lambda: "expected")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="vllm not importable"):
            pipeline.stage_environment(exp)
    assert not exp.done("01_environment")
    assert json.load(open(exp.path("environment.json")))["gpu"]["name"] == "A100"


def test_environment_retry_finishes_interrupted_stage(tmp_path, monkeypatch):
    cfg = {"clm": {"repo_commit": "expected"}}
    exp = experiment.Experiment(str(tmp_path), cfg)
    exp.write_json("environment.json", {
        "clm_commit": "expected", "clm_pinned": "expected",
        "feasibility": {"pass": True, "reasons": []}, "gpu": {"name": "A100"}})
    monkeypatch.setattr(experiment, "environment", lambda: pytest.fail("unexpected recapture"))

    assert pipeline.stage_environment(exp)["gpu"]["name"] == "A100"
    assert exp.done("01_environment")
