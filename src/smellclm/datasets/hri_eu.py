"""HRI-EU/SmellyCodeDataset — synthetic multilingual smoke-test fixture (plan section 19).

* pinned commit ``70d987a032355d25651b62a7df8fa56e5ea09e85``, MIT license
* source is read only from ``<Lang>/SmellyUnannotated/``; the MIT banner is removed
* label-revealing method names (``longMethod``, ``orderWithUnnecessaryDetails`` ...) are
  replaced with neutral ``fn_<hash>`` identifiers throughout the snippet
* ground-truth rows give weak *positives*; every other pair is ``unknown`` (the table
  is not exhaustive), and Deep Nesting is always ``unknown``
* every sample shares one repository family, so the fixture can never be split
"""
from __future__ import annotations

import hashlib
import os
import re
import subprocess

import pandas as pd

from ..extract import EXTENSIONS, functions
from ..schema import POSITIVE, UNKNOWN, WEAK, Label
from . import make_sample

REPO = "https://github.com/HRI-EU/SmellyCodeDataset"
COMMIT = "70d987a032355d25651b62a7df8fa56e5ea09e85"
LANG_DIRS = {"Java": "java", "Python": "python", "JavaScript": "javascript", "C++": "cpp"}
SMELL_MAP = {"Long Method": "long_method", "Long Parameter List": "long_parameter_list"}
LEAKY_NAME = re.compile(r"long|smell|unnecessar|parameter|complex|nest", re.I)


def fetch(dest: str) -> str:
    if not os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "clone", "--quiet", REPO, dest], check=True)
    subprocess.run(["git", "-C", dest, "checkout", "--quiet", COMMIT], check=True)
    return dest


def ground_truth(root: str) -> pd.DataFrame:
    gt = pd.read_csv(os.path.join(root, "Analysis", "GroundTruthLevel", "GroundTruth.csv"), dtype=str)
    gt = gt[gt["Language"] != "Language"]          # repeated header rows inside the raw file
    return gt


def _strip_banner(code: str) -> str:
    lines = code.splitlines()
    return "\n".join(l for l in lines if "smelly code for research" not in l.lower())


def _neutral(name: str) -> str:
    return "fn_" + hashlib.sha256(name.encode()).hexdigest()[:6]


def load(root: str, smells=("long_method", "long_parameter_list", "deep_nesting")) -> list:
    gt = ground_truth(root)
    pos = {(LANG_DIRS[r["Language"]], r["Class"], r["Method"], SMELL_MAP[r["Code Smell"]])
           for _, r in gt.iterrows() if r["Code Smell"] in SMELL_MAP and r["Language"] in LANG_DIRS}
    out = []
    for lang_dir, lang in LANG_DIRS.items():
        base = os.path.join(root, lang_dir, "SmellyUnannotated")
        for fn in sorted(os.listdir(base)):
            ext = os.path.splitext(fn)[1]
            if EXTENSIONS.get(ext) != lang:
                continue
            cls = os.path.splitext(fn)[0]
            src = open(os.path.join(base, fn), encoding="utf-8", errors="replace").read()
            for f in functions(src, lang):
                code = _strip_banner(f.code)
                symbol = f.name
                if LEAKY_NAME.search(symbol):
                    code = re.sub(rf"\b{re.escape(symbol)}\b", _neutral(symbol), code)
                labels = {}
                for s in smells:
                    hit = (lang, cls, f.name, s) in pos or (lang, "Main" if cls.lower() == "main" else cls,
                                                            f.name, s) in pos
                    labels[s] = Label(POSITIVE if hit else UNKNOWN, WEAK, source="hri-eu-groundtruth")
                out.append(make_sample(
                    sample_id=f"hri:{lang}:{cls}:{f.name}:{f.start_line}", dataset="hri_eu", repository_url=REPO,
                    commit_sha=COMMIT, file_path=f"{lang_dir}/SmellyUnannotated/{fn}", start_line=f.start_line,
                    end_line=f.end_line, language=lang, symbol=_neutral(symbol) if LEAKY_NAME.search(symbol)
                    else symbol, code=code, labels=labels, license="MIT",
                    provenance=f"{REPO}@{COMMIT} (synthetic; auxiliary smoke test only)"))
    return out
