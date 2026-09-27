"""Frozen smell definitions: rule thresholds, question text and candidate text.

Everything here is preregistered.  Changing a threshold, question or candidate
wording requires bumping ``RUBRIC_VERSION`` / ``PROMPT_VERSION`` and must never
happen after locked-test inference (plan sections 6, 10 and 12).

Candidates are symmetric: the true and false descriptions use the same
operational definition, the same length class, and no rationale that only one
side gets.
"""
from __future__ import annotations

RUBRIC_VERSION = "rubric/1"
PROMPT_VERSION = "prompt/1"

# Rule-baseline thresholds (positive when metric >= threshold).  These are the
# reviewers' evidence, not the hidden ground truth; see docs/annotation_rubric.md.
RULES = {
    "long_method": {"metric": "effective_loc", "threshold": 30},
    "long_parameter_list": {"metric": "parameter_count", "threshold": 5},
    "deep_nesting": {"metric": "max_nesting", "threshold": 4},
}

DEFINITIONS = {
    "long_method": ("a Long Method: the function does too much and is too long to understand at a glance "
                    "(roughly 30 or more non-blank, non-comment lines of code)"),
    "long_parameter_list": ("a Long Parameter List: the function declares too many parameters "
                            "(roughly 5 or more, not counting self, cls or this)"),
    "deep_nesting": ("Deep Nesting: control-flow blocks (if, loops, switch, try) are nested too deeply "
                     "inside the function body (roughly 4 or more levels)"),
}

NAMES = {
    "long_method": "Long Method",
    "long_parameter_list": "Long Parameter List",
    "deep_nesting": "Deep Nesting",
}

LANGUAGE_NAMES = {"java": "Java", "python": "Python", "javascript": "JavaScript", "typescript": "TypeScript",
                  "cpp": "C++", "csharp": "C#", "kotlin": "Kotlin", "go": "Go"}


def question(smell: str, with_definition: bool = True) -> dict:
    """Official CLM ``noul`` question for one smell (binary; option keys ``false``/``true``)."""
    what = DEFINITIONS[smell] if with_definition else f"the {NAMES[smell]} code smell"
    return {
        "type": "noul",
        "instructions": f"Does this code have the {NAMES[smell]} code smell?",
        "criteria": {
            "true": f"Yes. The code has {what}.",
            "false": f"No. The code does not have {what}.",
        },
    }


def rule_predict(smell: str, metrics: dict) -> int:
    r = RULES[smell]
    return int(metrics[r["metric"]] >= r["threshold"])


def prompt_manifest(smells, with_definition: bool = True) -> dict:
    """Serializable record of every question/candidate pair (``prompt_and_candidates.json``)."""
    return {"rubric_version": RUBRIC_VERSION, "prompt_version": PROMPT_VERSION,
            "with_definition": with_definition, "rules": {s: RULES[s] for s in smells},
            "questions": {s: question(s, with_definition) for s in smells}}
