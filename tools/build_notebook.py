"""Generate notebooks/clm_code_smell_pilot.ipynb from the cell list below (keeps the notebook diffable)."""
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md("""
# CLM-v0.1-8B code-smell verification — staged pilot

Implements the approved plan (PED-2, revision 4). Stages are resumable: every stage writes to an immutable
experiment directory and is skipped when already complete, so a Colab reset only costs the stage in progress.

| Stage | Gate |
|---|---|
| 1 Environment and pinned versions | A (hardware / pin) |
| 2 Dataset and annotation audit | 0 / B |
| 3 Leakage detection and frozen split | C |
| 4 Frozen-encoder embedding cache | |
| 5 Baselines and zero-shot (validation only) + score parity | A |
| 6 Head fine-tuning (official trainer, 3 seeds) | D |
| 7 Validation-only selection and calibration (locks decisions) | D |
| 8 Locked test evaluation + success rule | E |
| 9 Error analysis | |
| 10 Leave-one-language-out transfer (ablations: future work) | F |
| 11 Export and summary | |

**Current data status:** Gate 0 did not admit a primary corpus (see `reports/gate0/dataset_admission_report.md`).
`dataset.mode = "smoke"` runs the whole pipeline on the synthetic HRI-EU fixture; those numbers are pipeline
checks, not evidence. Set `mode = "primary"` and point `primary_parquet` at an admitted, adjudicated corpus
for the real experiment.

**Hardware:** Qwen3-8B via vLLM needs a Linux NVIDIA GPU with at least 22 GB VRAM (A100/L4 24 GB or better).
Stage 1 stops with a diagnostic otherwise. No substitute encoder or quantized port is used.
""")

code("""
# Setup: this repository, the pinned official CLM release, and dependencies
import os, subprocess, sys
REPO_URL = "https://github.com/pedruck/Identifying-Code-Smells-With-System-One-Models"
WORKDIR = "/content/smellclm" if os.path.isdir("/content") else os.getcwd()
if not os.path.isfile(os.path.join(WORKDIR, "pyproject.toml")):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, WORKDIR], check=True)
os.chdir(WORKDIR)
subprocess.run(["bash", "tools/setup_clm.sh", "--install"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[gpu]"], check=True)
sys.path.insert(0, os.path.join(WORKDIR, "src"))
""")

code("""
# Optional: keep experiment outputs and embedding caches on Google Drive so they survive resets
USE_DRIVE = False
RUNS_ROOT = "runs"
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS_ROOT = "/content/drive/MyDrive/smellclm_runs"
""")

code("""
# Configuration (edit the JSON file, not this cell: the config hash names the experiment directory)
import json
from smellclm import experiment, pipeline
CONFIG = "configs/stage1.json"
RESUME = None          # e.g. "20260927T183447Z-29857364d711" to continue an experiment
cfg = json.load(open(CONFIG))
exp = experiment.Experiment(RUNS_ROOT, cfg, RESUME)
print("experiment dir:", exp.dir)
print("dataset mode:", cfg["dataset"]["mode"])
""")

md("## Stage 1 — environment and Gate A hardware precheck")
code("""
env = pipeline.stage_environment(exp)
print(json.dumps({k: env.get(k) for k in ("platform", "gpu", "cuda", "clm_commit", "feasibility")}, indent=1))
""")

md("## Stage 2 — dataset and annotation audit (Gate 0 / B)")
code("""
df = pipeline.stage_dataset(exp)
print(open(exp.path("dataset_card.md")).read())
""")

md("## Stage 3 — leakage control and frozen split (Gate C)")
code("""
df = pipeline.stage_split(exp, df)
m = json.load(open(exp.path("split_manifest.json")))
print("split counts:", m["counts"], "| checksum:", m["checksum"])
print("leakage problems:", m["leakage_problems"] or "none")
print("support gates pass:", m["support"]["pass"])
print(json.dumps(m["support"]["per_smell"], indent=1))
if not m["support"]["pass"] and cfg["dataset"]["mode"] == "primary":
    raise SystemExit("Gate C: support thresholds not met. Add repository families and re-split before any test inference.")
""")

md("## Stage 4 — token audit and frozen-encoder embedding cache")
code("""
dec = pipeline.stage_embeddings(exp, df)
print(json.dumps(json.load(open(exp.path("embedding_manifest.json"))), indent=1))
""")

md("""
## Stage 5 — Gate A parity and zero-shot scores on validation

Downloads the released head, records its SHA-256, checks that our scorer matches the official `HeadPair`
projection on identical embeddings, and runs the candidate-order invariance test. Only validation rows are scored.
""")
code("""
pipeline.stage_zero_shot(exp, dec)
print(json.dumps(json.load(open(exp.path("gate_a.json"))), indent=1))
""")

md("""
## Stage 6 — fine-tune the projection heads (official trainer, unmodified)

`train/finetune.py --task choice --targets hard --loss infonce`, warm-started from the released head, one run per
seed. The trainer's `test` file is our **validation** split; the locked test split is never given to it. Row ids are
split groups, so its early-stopping carve-out is repository-family-disjoint.
""")
code("""
hist = pipeline.stage_finetune(exp, df, dec)
for seed, r in hist["runs"].items():
    s = r["summary"]
    print(f"seed {seed}: best epoch {s['best_epoch']}, trainer val acc {s['val_acc']:.4f}")
""")

md("## Stage 7 — validation-only thresholds (locks every decision before test)")
code("""
cal = pipeline.stage_validation(exp, dec)
print(json.dumps({"zero_shot": cal["zero_shot"], "val_summary": cal["val_summary"],
                  "finetuned_val_macro_f1": {s: v["val_macro_f1"] for s, v in cal["finetuned"].items()}}, indent=1))
""")

md("""
## Stage 8 — locked test evaluation and the preregistered success rule

Before any test row is scored, the test gate checks that every evaluated test label is adjudicated gold with two
reviewers, that the Gate 0 admission record and Gate B kappa evidence (paths in `config["dataset"]`) pass, and that
support recomputed on the test decisions kept after truncation still meets Gate C. In primary mode a failed gate
stops here. In smoke mode it is recorded and every output is marked non-confirmatory.
""")
code("""
metrics = pipeline.stage_test(exp, dec, df)
print("confirmatory:", metrics["confirmatory"])
print(json.dumps(metrics["success_rule"], indent=1))
""")

md("## Stage 9 — error analysis (deterministic selection)")
code("""
pipeline.stage_errors(exp)
import pandas as pd
pd.read_csv(exp.path("error_strata.csv")).head(30)
""")

md("""
## Stage 10 — leave-one-language-out transfer (Gate F only)

Runs only when the primary result is confirmatory (the stage 8 test gate passed) **and** the preregistered success
rule passed, and `transfer.enabled` is true in the config. Otherwise it records why it was skipped. For each held-out
language the heads are retrained on the other languages, thresholds are chosen on their validation split, and the
held-out language's locked test repositories are scored. The secondary ablations (plan section 14) are **not**
implemented in this notebook: each is its own experiment config and is future work.
""")
code("""
transfer = pipeline.stage_transfer(exp, df, dec)
print(json.dumps({k: v for k, v in transfer.items() if k != "languages"}, indent=1))
""")

md("## Stage 11 — export")
code("""
print(pipeline.stage_export(exp))
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}})
out = os.path.join(HERE, "notebooks", "clm_code_smell_pilot.ipynb")
nbf.write(nb, out)
print(out)
