"""Generate the default clean-clone Colab pilot notebook."""
from pathlib import Path
import nbformat as nbf
root = Path(__file__).resolve().parents[1]
cells = []
def md(value): cells.append(nbf.v4.new_markdown_cell(value.strip()))
def code(value): cells.append(nbf.v4.new_code_cell(value.strip()))
md("""
# CLM-v0.1-8B real-code exploratory pilot

Run every cell on a Colab A100 or other NVIDIA GPU with at least 22 GB VRAM.
The repository includes a frozen 440-row real-code pilot input: ENASE positive
examples and Stage 1 machine-labelled negatives, balanced 1:1 for each smell in
training and mixed validation. The positive-only ENASE holdout measures recall.
All labels are provisional. Source differences may drive the mixed score. This
notebook does not run the locked gold test or make a validated accuracy claim.

The official frozen Qwen3-8B encoder and unmodified CLM choice trainer train
only the projection heads for one seed and two epochs. Model downloads and GPU
embedding can take time; runs/ contains the audit, cache and report.
""")
code("""
import os, subprocess, sys
REPO_URL = "https://github.com/pedruck/Identifying-Code-Smells-With-System-One-Models"
WORKDIR = "/content/smellclm" if os.path.isdir("/content") else os.getcwd()
if not os.path.isfile(os.path.join(WORKDIR, "pyproject.toml")):
    subprocess.run(["git", "clone", "--quiet", REPO_URL, WORKDIR], check=True)
elif os.path.isdir(os.path.join(WORKDIR, ".git")):
    subprocess.run(["git", "-C", WORKDIR, "pull", "--ff-only"], check=True)
os.chdir(WORKDIR)
subprocess.run(["bash", "tools/setup_clm.sh", "--install"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[gpu]"], check=True)
subprocess.run([sys.executable, "tools/check_gpu_stack.py"], check=True)
sys.path.insert(0, os.path.join(WORKDIR, "src"))
""")
code("""
import json
from smellclm import experiment, pipeline, pilot
cfg = json.load(open("configs/pilot.json"))
exp = experiment.Experiment("runs", cfg)
print("run directory:", exp.dir)
df = pilot.prepare(exp)
print(json.dumps(json.load(open(exp.path("pilot_input_audit.json")))["counts"], indent=2))
""")
code("""
# The official CLM pin and >=22 GB CUDA check fail with an actionable diagnostic.
env = pipeline.stage_environment(exp)
print("GPU:", env.get("gpu"))
""")
code("""
# Embed real code with the frozen Qwen3-8B encoder.
decisions = pipeline.stage_embeddings(exp, df)
print(json.dumps(json.load(open(exp.path("embedding_manifest.json"))), indent=2))
""")
code("""
# Official released head parity and validation-only zero-shot baseline.
pipeline.stage_zero_shot(exp, decisions)
print(json.dumps(json.load(open(exp.path("gate_a.json"))), indent=2))
""")
code("""
# One seed, two epochs, official unmodified choice trainer.
history = pipeline.stage_finetune(exp, df, decisions)
print(json.dumps({s: run["summary"] for s, run in history["runs"].items()}, indent=2))
""")
code("""
# Exploratory mixed silver validation and separate ENASE positive recall.
# No locked test scoring is performed.
calibration = pipeline.stage_validation(exp, decisions)
report = pilot.finish(exp, decisions, calibration)
print(json.dumps(report, indent=2))
print("Report:", exp.path("pilot_report.json"))
""")
for index, cell in enumerate(cells):
    cell["id"] = f"pilot-{index:02d}"
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"}})
out = root / "notebooks/clm_code_smell_pilot.ipynb"
nbf.write(nb, out)
print(out)
