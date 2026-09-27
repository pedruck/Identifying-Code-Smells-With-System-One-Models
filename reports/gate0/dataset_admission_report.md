# Gate 0: Dataset Admission Report (CLM-v0.1-8B code-smell verification pilot)

Retrieval / audit date: 2026-09-27. Auditor: automated agent. Working files: `reports/gate0/`.
Stage 1 scope: Java, Python, C++; Long Method (LM) and Long Parameter List (LPL), method scope.

Legend: **V** = verified directly from downloaded bytes; **M** = taken from metadata/paper only (files not inspected); **X** = could not be verified.

---

## 1. ENASE 2026 cross-language release (Moldovan et al.)

Planned role: primary Stage 1 multi-language benchmark.

| # | Checklist item | Result | Basis |
|---|---|---|---|
| 1 | Files, sizes, hashes, URL/version | Figshare item **31066354**, private link `https://figshare.com/s/2c89cce6b2d77c6f324f`, title "Multi-language Code Smells Instances - Python, Java & C++ Dataset", version 0 (unpublished/private, no DOI), modifiedDate 2026-09-17T13:08:44Z, 9 files, total 16,532,821 bytes. Files listed on the share page: `Prompts.pdf` (id 61013938, 14.65 kB), `python.zip` (61039969, 538.36 kB), `c++.zip` (61039972, 8.01 MB), `java.zip` (61039975, 7.21 MB), and folder `RQ4-RQ5_Results/` (ids 68796865, 68796868, 68796871, 68796874, 68796877; names not shown). **No file bytes were retrieved, so there are no SHA-256 hashes.** | M (share-page state rendered by headless Chromium) |
| 2 | Explicit dataset license | **CC BY 4.0** is attached to the figshare item's license field. The paper itself is CC BY-NC-ND 4.0, but that is not what governs the dataset. | M (item metadata) |
| 3 | Mapping to lang/smell/label/repo/revision/span/scope | The item description says each entry has "project, file path, smell type, line number, code snippet, and smell-specific size and complexity metrics". Column names are unknown. No revision/commit field is mentioned. The paper says C++ was taken from "the latest version" of 50 GitHub repos and Java from Qualitas Corpus, so the pinned revision is probably missing for C++ and Python. | M / X |
| 4 | Explicit negatives | Probably none. The description lists smell *instances* only, and the paper frames the task as multi-label over positive instances. Absence from the list counts as unknown, not negative. | M / X |
| 5 | Repo provenance / duplicates | A "project" field is claimed. Duplicate checks were not possible. The paper also says identifiers were normalized (renamed classes, functions, and variables), which weakens exact-hash leakage checks across sources. | X |
| 6 | Counts per cell | The paper caps at "500 samples per code smell and programming language, except for God Class in C++". Actual counts could not be verified. | M |
| 7 | Human 30+30 review | **PENDING**. No sample list was produced because the data is unavailable. | X |
| 8 | Split manifest + leakage | The paper describes a 70/10/20 split "balanced by code smell" and mentions project-level splitting in related work. It is unknown whether a manifest is shipped. | X |

**Download status (precise):**
- `https://api.figshare.com/v2/articles/private/2c89cce6b2d77c6f324f` returns 404 (nginx "resource could not be found"). The variants `/articles/private_link/<key>`, `/private_links/<key>`, `/private/articles/<key>` and `/collections/private/<key>` also return 404. `https://api.figshare.com/v2/articles/31066354` returns 404 `{"message": "Entity not found: ArticleVersion"}` because the item is private.
- `https://figshare.com/s/<key>` and every `https://figshare.com/ndownloader/files/<id>?private_link=<key>` and `.../ndownloader/articles/31066354?private_link=<key>` URL, when fetched with curl, returns **HTTP 202, 0 bytes, `x-amzn-waf-action: challenge`** (AWS WAF JavaScript challenge).
- Headless Chromium passed the challenge for the HTML share page, which is where the metadata above comes from. A programmatic GET of the ndownloader URLs from that browser context returned **HTTP 403** (nginx). Driving the browser further (click-to-download) was denied by the sandbox permission policy, and I did not pursue it.
- **A human with a normal browser should be able to download** `python.zip`, `c++.zip` and `java.zip` from the share link. After that, the audit steps 1 and 3–8 can be rerun on those files.

**Verdict: FAIL (not admissible at Gate 0 as of 2026-09-27)** for the primary-benchmark role. The reason is that the data bytes could not be obtained or inspected, not a licence problem (CC BY 4.0 is attached). Also, based on the metadata, the release looks like a positives-only instance list with identifier-normalized code and likely no pinned revisions. That would make it ADMIT-AS-WEAK at best even once downloaded. Re-audit is warranted after a manual download.

---

## 2. SmellyCode++ (Alomari, Alazba, Aljamaan, Alshayeb 2025)

Planned role: Java mechanics / auxiliary.

| # | Checklist item | Result | Basis |
|---|---|---|---|
| 1 | Files/hash/URL | `multi-smell-dataset-v1_2.csv`, 618,128,672 B, SHA-256 `971a81da3f55326a5a56bae59c3cde34c4ebab1a99615ac0c474ce5ecf5029d2`, MD5 902d4af8d1a52ef94666ecf620d7479d (matches figshare supplied/computed MD5). URL `https://ndownloader.figshare.com/files/52714583`. Figshare article 28519385 v1, DOI 10.6084/m9.figshare.28519385.v1, published 2025-03-02. The article lists the identical file twice (ids 52714583 and 52714586, same MD5). | V |
| 2 | License | **CC0** (figshare license field). | V |
| 3 | Mapping | Language: Java only (implicit). Smell labels: `Long method`, `God class`, `Feature envy`, `Data class` (0/1 int columns). **There is no Long Parameter List column.** Repository: `Project` (89 Apache-style project names). There is **no revision or version**, **no line span**, and no file path (only a dotted `File` id such as `activemq-amq-store.src.main.org...amqreader`). Scope is not explicit: rows mix whole classes and single methods, and scope must be inferred from the code text. | V |
| 4 | Explicit negatives | Yes, as explicit 0 labels in each smell column. These are tool/heuristic labels, not human-confirmed. There is visible noise, e.g. a 4-logical-line `tearDown()` labelled Long method = 1. | V |
| 5 | Provenance / duplicates | `Project` gives the repository family. Exact duplicates after whitespace normalization: 107,554 rows collapse to 94,235 unique code hashes. 4,367 duplicate groups hold 17,686 rows. **391 groups have conflicting labels** and 289 groups span more than one project. For method-like rows only: 3,475 dup groups, **288 with conflicting Long method labels**, 212 cross-project. The 1,566 LM positives contain only 768 unique code hashes. | V |
| 6 | Counts | See the counts table below. | V |
| 7 | Human review | **PENDING**. Sample list produced: `review_sample_smellycodepp_java_long_method.csv` (30 pos + 30 neg). | V |
| 8 | Split manifest | None published. | V |

**Verdict: AUXILIARY-ONLY (ADMIT-AS-WEAK for Java Long Method mechanics / loader smoke tests).** It has no LPL, no C++/Python, no revision pinning, heavy duplication with label conflicts, and tool-derived labels. Deduplicate by normalized hash and drop conflicting groups before any use.

---

## 3. MLCQ (Madeyski & Lewowski, EASE 2020)

Planned role: auxiliary human-labelled Java data.

| # | Checklist item | Result | Basis |
|---|---|---|---|
| 1 | Files/hash/URL | Zenodo record 3666840, v1.1, DOI 10.5281/zenodo.3666840. `MLCQCodeSmellSamples.csv` 7,513,770 B sha256 `8e05db55…3551b`; `MLCQCodeSmellSamples.xlsx` 1,913,137 B `18cee0ce…bdfdea`; `MLCQCodeSmellDevelopersSurvey.csv` 46,184 B `a19c80b4…cbdd`; `MLCQCodeSmellDevelopersSurvey.xlsx` 23,788 B `5bc9c151…e7d6`; `MadeyskiLewowskiMLCQAppendix.pdf` 212,591 B `01d628a8…37eb`. Full hashes are in `sha256_manifest.txt`. | V |
| 2 | License | **CC BY 4.0** (Zenodo license `cc-by-4.0`). | V |
| 3 | Mapping | Java only. Smells: `blob`, `data class`, `feature envy`, `long method`. **No LPL.** Label from `severity` ∈ {none, minor, major, critical}. Repository is `repository` (git URL), revision is `commit_hash` (immutable SHA), plus `path`, `start_line`, `end_line`, and scope `type` ∈ {class, function}. **The code text is not included** and has to be fetched from GitHub at the pinned commit. | V |
| 4 | Explicit negatives | Yes: `severity=none` is an explicit human judgement. | V |
| 5 | Provenance / duplicates | 522 repositories, 522 commits, 26 reviewers, 4,770 samples, and 14,739 review rows (multiple reviewers per sample). Code-hash dedup is not possible without fetching code. | V |
| 6 | Counts | See the counts table (LM: per-sample majority vote). | V |
| 7 | Human review | **PENDING**. Sample list produced: `review_sample_mlcq_java_long_method.csv`. | V |
| 8 | Split manifest | None published. | V |

**Verdict: AUXILIARY-ONLY (ADMIT-AS-WEAK for Java Long Method with human labels).** The provenance is the best of the four sources (repo + commit + span) and it has real human negatives. However, it is Java only, has no LPL, and code must be retrieved (some repos may have disappeared, which I did not check). It is 2020 data and likely present in model pretraining corpora.

---

## 4. HRI-EU/SmellyCodeDataset

Planned role: smoke tests.

| # | Checklist item | Result | Basis |
|---|---|---|---|
| 1 | Files/hash/URL | `https://github.com/HRI-EU/SmellyCodeDataset` at commit `70d987a032355d25651b62a7df8fa56e5ea09e85` (2025-02-24, "initial commit"), 245 tracked files. `Analysis/GroundTruthLevel/GroundTruth.csv` sha256 `5626b8ce…55f4`; `Cleaned_GroundTruth.csv` `e0c9538e…4c64`; per-source-file hashes are in `formats.md`. | V |
| 2 | License | **MIT** (`LICENSE`, Copyright 2025 Ahmed R. Sadik, Honda Research Institute Europe GmbH; also in each file header). | V |
| 3 | Mapping | `Language`, `Class`, `Code Smell`, `Method` in the ground-truth CSV. Source is 7 synthetic files per language (a pizza-shop program). The "repository" is a single synthetic project, and the revision is the pinned git commit. The span has to be located by method name, since no line numbers are given. | V |
| 4 | Explicit negatives | **No.** The ground truth lists positives only. It is also demonstrably non-exhaustive: `Java/SmellyUnannotated/Customer.java` has a 6-parameter `orderWithUnnecessaryDetails(...)`, but Java LPL is listed only for `Pizza`. | V |
| 5 | Provenance / duplicates | One synthetic program re-implemented in 4 languages, so there is extreme near-duplication across languages. | V |
| 6 | Counts | See the counts table. | V |
| 7 | Human review | Not applicable (smoke-test role). | – |
| 8 | Split manifest | None. | V |

**Label leakage:** the method names themselves (`longMethod`, `long_method`, `longComplaintMethod`, `orderWithUnnecessaryDetails`) reveal the label. The `SmellyAnnotated/` trees also contain smell-name comments (e.g. `//PrimitiveObsession`), so use `SmellyUnannotated/` only. There is no Deep Nesting label.

**Verdict: AUXILIARY-ONLY (smoke tests / pipeline sanity only; never for metrics).**

---

## Counts table (language × smell), verified sources

Positives / explicit negatives. "n/a" = smell or language not present.

| Source | Lang | Long Method | Long Param List | God Class / Large Class / Blob | Other |
|---|---|---|---|---|---|
| ENASE 2026 | Py / Java / C++ | ≤500 each, positives only (M, **unverified**) | ≤500 each (M) | ≤500; C++ fewer (M) | Long Ternary ≤500 (M) |
| SmellyCode++ (all rows) | Java | 1,566 / 105,988 | n/a | God class 4,333 / 103,221 | Feature envy 1,996; Data class 3,284 |
| SmellyCode++ (method-like rows†) | Java | 1,566 / 45,376 (35,032 unique label-consistent hashes after dedup) | n/a | – | – |
| MLCQ (review rows) | Java | 806 non-none / 2,556 none (3,362 reviews) | n/a | blob 974 / 3,045 | data class 1,057/2,964; feature envy 454/2,883 |
| MLCQ (per-sample majority) | Java | 276 pos / 2,148 neg (6 ties) over 2,430 samples, 426 repos | n/a | – | – |
| HRI-EU (GroundTruth.csv, positives only) | Python | 5 / – | 5 / – | Large Class 5 | no Deep Nesting |
| | Java | 5 / – | 1 / – | 6 | |
| | C++ | 6 / – | 3 / – | 6 | |
| | JavaScript | 5 / – | 2 / – | 5 | |

† "method-like" = code does not start with a class/interface/enum declaration (regex heuristic). All LM positives fall in this subset.

**Stage 1 coverage gap:** no verified source provides **Long Parameter List with explicit negatives** in any language, and **no verified source covers Python or C++** beyond the 7-file HRI synthetic program.

---

## Gate 0 decision

- **ENASE does NOT pass Gate 0** as of 2026-09-27. The licence is present (CC BY 4.0), but the data could not be retrieved or inspected. The figshare private link sits behind an AWS WAF challenge and ndownloader returns 403 to automated clients. The metadata also suggests positives-only, identifier-normalized snippets with no pinned revisions. A manual browser download followed by a re-audit could change the verdict, at best to ADMIT-AS-WEAK.
- **The plan's fallback therefore applies:**
  1. **HRI-EU**: smoke tests only (loader, prompt, and parsing sanity), using `SmellyUnannotated/` trees at the pinned commit. No metrics.
  2. **SmellyCode++**: Java Long Method mechanics only, after dedup and conflict removal. Weak tool labels. Not usable for LPL.
  3. **Build a gold stratum**: human-labelled, revision-pinned samples for Java/Python/C++ × LM/LPL with explicit negatives. MLCQ (Java LM, human, commit-pinned) is the most useful seed or auxiliary source for the Java LM cell. Python/C++ and all LPL cells need new annotation.
- **Human stratified review (30+30 per cell): PENDING** for every cell. Sample lists were generated (seed 20260927) for SmellyCode++ Java LM and MLCQ Java LM only.

## Artifacts
- `dataset_admission_report.md` (this file), `formats.md`
- `review_sample_smellycodepp_java_long_method.csv`, `review_sample_mlcq_java_long_method.csv`
- `sha256_manifest.txt`; analysis scripts `analyze_scpp.py`, `analyze_scpp2.py`, `analyze_mlcq.py`, `make_samples.py`; `scpp_index.parquet` (row index + labels + normalized-code SHA-256)
- Raw downloads in `dl/` (ENASE: only `enase_paper.pdf` and the rendered share page `enase_share_dom.html`)
