# Labelling guide — gold stratum (`annotation/1`)

How the two reviewers (**A** = pedruck, **B** = the second reviewer) produce the gold labels for the locked
test set and the double-reviewed share of train/validation. What counts as positive, negative or unknown is
defined only in [`annotation_rubric.md`](annotation_rubric.md) (`rubric/1`); this guide covers the process.

All commands run from the repository root. Replace `stage1` with `stage2` for the Stage 2 languages and smells.

## 0. Roles and ground rules

- **Operator** (A): runs the commands below and holds `KEY_do_not_share.csv`.
- **Reviewers** (A and B): label independently. Do not discuss items, compare sheets, or look at model output,
  rule-baseline predictions, repository names or the key until both reviewers have handed in every batch.
- The operator being a reviewer is acceptable because sheets are blind: they carry an opaque `item_id`, the
  language and static metrics only. The operator must not open the key or `selected.parquet` while labelling.
- Rubric changes are allowed only before training and before any locked-test inference, and bump
  `RUBRIC_VERSION` (see step 4).

## 1. Pin the candidate pool

```bash
python -m smellclm.annotation resolve --config configs/stage1.json   # once; pins commits + SPDX licences
git add configs/gold_pool_repos.json && git commit -m "Pin gold-pool repositories"
```

`resolve` fills `commit` (current default-branch HEAD) and `license` for every entry in
`configs/gold_pool_repos.json`. Check the licences: an entry outside the allow-list in
`annotation.LICENSE_ALLOWLIST` fails Gate 0; replace it with another repository of the same language before
mining. Commit the pinned manifest. It must not change after batches are drawn.

## 2. Mine candidates

```bash
python -m smellclm.annotation mine --config configs/stage1.json
```

This checks out each repository of the stage's languages at its pinned commit under `data/raw/gold_pool/`. It
skips tests, examples, vendored and generated code, extracts functions with `extract/1`, and writes
`data/gold_pool/stage1_pool.parquet` (no labels) and the Gate 0 admission record
`reports/gate0/stage1_admission.json`. Each repository contributes at most 4000 functions.

## 3. Draw the batches (test reservation happens here)

```bash
python -m smellclm.annotation batches --config configs/stage1.json --reviewers A,B
```

1. **Test families are reserved first**, by seeded hash only, before anyone sees a label: about 25% of the
   repository families per language (`--test-frac`), at least two and never all of them. They are written to
   `data/annotation/stage1/split_reservation.json`. `pipeline.stage_split` reads that file in `primary` mode
   and puts every row from those families in test. The other rows go to train/val.
2. **Metric-stratified draw.** For each partition, language and smell, candidates are drawn from three bands of
   the rule metric: `likely_positive`, `borderline` and `likely_negative`. The targets come from
   `annotation.targets` in the stage config. No family may supply more than 35% of one band. Exact normalized
   duplicates are drawn once.
3. **Assignment.** Every test item goes to both reviewers. Train/val items go to A, and a seeded 20% of them
   per language (`--double-review-frac`) go to B as well. Each reviewer gets an independent random order,
   split into batches of 40 (`--batch-size`).

Output in `data/annotation/stage1/`:

| File | Who opens it |
|---|---|
| `A_batchNN.md`, `A_batchNN.csv` | reviewer A |
| `B_batchNN.md`, `B_batchNN.csv` | reviewer B |
| `batch_manifest.json` | anyone. It holds counts, bands, targets and reserved families, but no item mapping |
| `KEY_do_not_share.csv`, `selected.parquet` | nobody until step 4 |
| `split_reservation.json` | pipeline only |

Commit `split_reservation.json` and `batch_manifest.json` right away. They show that the test families were
fixed before labelling. `data/annotation/` holds third-party code, so share the sheets with B privately rather
than in a public repository.

`batch_manifest.json` → `assignments` gives each reviewer's workload. With the default targets, a pool that
fills every band gives B about half as many items as A, because B skips most of train/val.

## 4. Label

Open `X_batchNN.md` to read the code and fill in the matching `X_batchNN.csv`. Do not rename or reorder files
or columns, and do not edit `item_id`.

| Column | Value |
|---|---|
| `<smell>` | `positive`, `negative` or `unknown` (lower case). Every smell of the stage, for every row |
| `<smell>_confidence` | `0` to `1`. Use ≤ 0.6 for rubric borderline cases |
| `notes` | optional: the deciding reason, especially for borderline or unknown |

- Judge against the rubric definition, not the metric. The metrics on the sheet are evidence. A 40-line flat
  lookup table can be a negative Long Method.
- Use **unknown** only for the rubric's unknown cases: truncated, generated or minified code, or a broken
  extraction. Do not use it as "not sure". Unknown is never turned into negative, and an unknown test label is
  never scored.
- If an item looks like the same code as another item, label both, and mention it in `notes`.
- Suggested pace: one batch per sitting, with a break every 40 items. Do the test batches in the same weeks for
  both reviewers, so that drift in how the rubric is applied shows up in agreement.

When both reviewers have handed in all batches:

```bash
python -m smellclm.annotation agreement --config configs/stage1.json --reviewers A,B
```

This writes `agreement.json`, with n, raw agreement and Cohen's kappa per partition and smell, and
`adjudication.csv`, with one row per item and smell where A and B disagree. An invalid status stops the command
and names the file and item. Fix the sheet and rerun.

**Kappa gate.** If any smell has **test kappa < 0.60**:

1. Read the disagreements together and find the part of the rubric that caused them.
2. Revise `docs/annotation_rubric.md` and `rubric.py`, and bump `RUBRIC_VERSION`.
3. Relabel that smell for all double-reviewed items, from blank sheets, then rerun `agreement`.

Do not fix a low kappa by adjudicating it away.

## 5. Adjudicate

With two reviewers, adjudication is a joint session between A and B. For each row of `adjudication.csv`, open
the item, discuss it against the rubric, and fill in:

- `adjudicated_status`: `positive`, `negative` or `unknown`
- `adjudicator`: `A+B`
- `rationale`: one sentence that cites the rubric clause

Rerunning `agreement` keeps adjudications already entered. A blank or invalid `adjudicated_status` counts as
unresolved.

## 6. Finalize

```bash
python -m smellclm.annotation finalize --config configs/stage1.json --reviewers A,B
```

This writes the primary dataset (`dataset.primary_parquet` in the config) and the Gate B evidence
(`dataset.gate_b_evidence`).

| Item | Resulting label |
|---|---|
| Double-reviewed, agreed | `gold`, the agreed status |
| Double-reviewed, adjudicated | `gold`, the adjudicated status |
| Double-reviewed, not adjudicated | test: `unknown`. Train/val: `weak`, `unknown` |
| Single-reviewed (train/val only) | `weak`, the reviewer's status |
| Test item missing a second review | `unknown`, never scored |

Gate B needs `kappa_ok: true` and `unresolved_adjudications: 0`. Then set `dataset.mode` to `primary` in the
stage config. The split stage will read `split_reservation.json`, and its manifest records the reserved
families.

## Checklist

- [ ] `configs/gold_pool_repos.json` pinned and committed; every licence allowed
- [ ] `split_reservation.json` and `batch_manifest.json` committed before the first label
- [ ] Both reviewers finished every batch without talking about items
- [ ] Test kappa ≥ 0.60 for every smell (otherwise revise the rubric and relabel)
- [ ] Every row of `adjudication.csv` has a valid status, `A+B` and a rationale
- [ ] `finalize` run; Gate B evidence shows `kappa_ok: true` and no unresolved adjudications
