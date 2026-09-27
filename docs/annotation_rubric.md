# Annotation rubric — `rubric/1`

Frozen before sampling or splitting (plan section 6). Any change bumps the version and must happen before
locked-test inference. The machine-readable thresholds live in `src/smellclm/rubric.py`.

Every targeted smell gets exactly one status per sample: **positive**, **negative**, or **unknown**.
Unknown is never converted to negative; it is excluded from that smell's loss and evaluation.

Unit of analysis for all three pilot smells: one method/function, including its signature and body, with no
enclosing class context (`extract/1`).

## Long Method (`long_method`)

- **Positive:** the function does more than one job or is too long to understand at a glance. Evidence:
  effective lines of code (non-blank, non-comment) ≥ 30.
- **Negative:** a single coherent responsibility, readable without scrolling. Typically < 25 effective lines.
- **Borderline (25–35 lines):** decide on responsibility, not the count. Long but flat data tables, generated
  switch statements, and test fixtures that are just setup are negative unless the logic is also tangled.
- **Unknown:** the snippet is truncated, generated or minified, or the extraction cut the function.
- **Language notes:** Python docstrings count as lines. C++ header-only declarations are not samples.

## Long Parameter List (`long_parameter_list`)

- **Positive:** 5 or more declared parameters, excluding `self`/`cls` (Python), the `this:` parameter
  (TypeScript), and the Go method receiver. `*args`/`**kwargs` and varargs count as one each.
- **Negative:** 4 or fewer parameters.
- **Borderline:** 5 parameters where the function is a constructor that only assigns fields, or an overridden
  framework callback whose signature is imposed. Label positive but record confidence ≤ 0.6.
- **Unknown:** the signature is not fully visible.

## Deep Nesting (`deep_nesting`, Stage 2)

- **Positive:** control-flow blocks (`if`, loops, `switch`/`match`, `try`, `with`) nested 4 or more levels deep
  inside the body. `else`/`elif`/`catch` branches are siblings, not deeper levels. Nested function or lambda
  bodies do not add depth.
- **Negative:** maximum depth ≤ 2.
- **Borderline (depth 3):** positive only if the nesting hurts readability, e.g. no guard clauses and
  interleaved state changes.
- **Unknown:** braceless one-line bodies make the depth ambiguous, or the code is truncated.

## Process

1. Two reviewers label each gold test example independently, plus at least 20% of train/validation examples,
   without seeing model output.
2. Record reviewer id or role, timestamp, rubric version, confidence (0–1), and the adjudication outcome.
3. Adjudicate disagreements. Report raw agreement and Cohen's kappa per smell. If test kappa is below 0.60,
   revise this rubric and relabel **before** any training.
4. Static metrics are shown to reviewers as evidence. Tool output alone is a *weak* label and never enters the
   locked test set.

### Worked examples

Five positive, five negative and several borderline examples per smell are drawn from the admitted corpus at
Gate B. They are stored with the labelling batch rather than here, so this rubric stays language- and
corpus-neutral.
