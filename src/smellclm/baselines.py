"""Non-CLM baselines (plan section 11, conditions 1-2)."""
from __future__ import annotations

import pandas as pd

from . import rubric


def majority(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Per-smell training-prevalence majority class."""
    maj = train.groupby("smell")["label"].mean().ge(0.5).astype(int).to_dict()
    out = target.copy()
    out["pred"] = out["smell"].map(maj).fillna(0).astype(int)
    out["p_true"] = out["smell"].map(train.groupby("smell")["label"].mean()).fillna(0.0)
    return out


def rules(target: pd.DataFrame) -> pd.DataFrame:
    """Frozen deterministic static-metric thresholds from ``rubric.RULES``."""
    out = target.copy()
    out["pred"] = [rubric.rule_predict(s, {"effective_loc": loc, "parameter_count": pc, "max_nesting": mn})
                   for s, loc, pc, mn in zip(out["smell"], out["effective_loc"], out["parameter_count"],
                                             out["max_nesting"])]
    return out
