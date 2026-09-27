# Loader format notes (Gate 0, retrieved 2026-09-27)

## 1. ENASE 2026 cross-language release: NOT DOWNLOADED (format unverified)

- Share link: `https://figshare.com/s/2c89cce6b2d77c6f324f` → figshare item id `31066354` (private, version 0, no DOI). License field: CC BY 4.0.
- File download URLs (need a real browser; curl gets an AWS WAF challenge (HTTP 202, empty body, `x-amzn-waf-action: challenge`); scripted fetches from headless Chromium got 403):
  - `https://figshare.com/ndownloader/files/61039969?private_link=2c89cce6b2d77c6f324f` → `python.zip` (538.36 kB)
  - `https://figshare.com/ndownloader/files/61039972?private_link=2c89cce6b2d77c6f324f` → `c++.zip` (8.01 MB)
  - `https://figshare.com/ndownloader/files/61039975?private_link=2c89cce6b2d77c6f324f` → `java.zip` (7.21 MB)
  - `https://figshare.com/ndownloader/files/61013938?private_link=2c89cce6b2d77c6f324f` → `Prompts.pdf` (14.65 kB)
  - Folder `RQ4-RQ5_Results` (5 files, ids 68796865/68796868/68796871/68796874/68796877): `https://figshare.com/ndownloader/articles/31066354?folder_path=RQ4-RQ5_Results&private_link=2c89cce6b2d77c6f324f`
  - Whole item zip: `https://figshare.com/ndownloader/articles/31066354?private_link=2c89cce6b2d77c6f324f`
- The public API does not resolve private links (`/v2/articles/private/<key>` returns 404; `/v2/articles/31066354` returns 404 "Entity not found: ArticleVersion").
- Expected content (from the item description, unverified): per-entry `project`, `file path`, `smell type`, `line number`, `code snippet`, and smell-specific metrics. There are 4 smells (Long Method, God Class, Long Parameter List, Long Ternary Conditional Expression) in 3 languages.
- Gotchas from the paper: code is **identifier-normalized** (classes, functions and variables renamed). There are ≤500 samples per smell×language. Sample ids look like `py 593290`. Python was built from Moldovan et al. 2024 via the GitHub API, Java from Qualitas Corpus + Designite, and C++ from the "latest version" of 50 GitHub repos + CLEAN++. Long Ternary is derived as the intersection of Long Statement and Complex Conditional plus a ternary check. Positives only are likely.

## 2. SmellyCode++ (figshare 28519385, CC0)

- URL: `https://ndownloader.figshare.com/files/52714583` (duplicate copy: `/files/52714586`, same MD5 `902d4af8d1a52ef94666ecf620d7479d`).
- File: `multi-smell-dataset-v1_2.csv`, 618,128,672 B, SHA-256 `971a81da3f55326a5a56bae59c3cde34c4ebab1a99615ac0c474ce5ecf5029d2`. 107,554 data rows × 22 columns.
- Format: comma-separated, header row, standard `"` quoting with `""` escapes, ASCII, LF. Some `Code` cells are huge (max 5.7 M chars). pandas `read_csv` default engine works fine. Set `keep_default_na=False` so code strings like `null` are not turned into NaN.
- Columns (example values from row 0):
  - `File` = `.mvn.wrapper.mavenwrapperdownloader`: dotted path-like id, lowercased, no extension, not a real path.
  - `Project` = `plc4x` (89 distinct; Apache projects such as jena, hive, hadoop, hbase, …). This is the repository-family key. **There is no version or commit.**
  - Metrics (int/float): `Logical Lines`=39, `Distinct Operators`=6, `Distinct Operands`=75, `Total Operators`=45, `Total Operands`=220, `Vocabulary`=81, `Length`=265, `Calculated Length`=482.67, `Volume`=1680.06, `Difficulty`=8.8, `Effort`=14784.5, `Time Required`=821.36, `Bugs`=0.56, `Cyclomatic Complexity`=9.
  - `Class` = `mavenwrapperdownloader` (lowercased class name).
  - `Code` = `public class MavenWrapperDownloader { private static final String WRAPPER_VERSION = "0.5.2" ; ...`
  - Labels (int 0/1): `Long method`, `God class`, `Feature envy`, `Data class`. **There is no Long Parameter List.**
- Code storage: **one line, space-tokenized** (every token separated by a single space, no newlines or indentation). Comments are mostly stripped (4,294 rows still contain `//` or `/*`). Line-based metrics cannot be recomputed from the text. For an LLM you may want to re-pretty-print it, e.g. with a Java formatter.
- Scope: mixed. 60,612 rows start with a class/interface/enum declaration (class-level). 46,942 rows are method-level (heuristic regex `^\s*((public|private|protected|abstract|final|static)\s+)*(class|interface|enum)\b`; annotated classes may be misclassified). All `Long method`=1 rows are method-like.
- Language: Java only (implicit, there is no language column).
- Gotchas: 13,319 exact duplicate rows after whitespace normalization. 391 dup groups have conflicting labels (288 for Long method within method rows). There are cross-project duplicates. LM positives: 1,566 rows but only 768 unique. Labels are noisy (for example `tearDown()` with 4 logical lines is labelled LM=1). Dedupe on the normalized-code SHA-256 and drop conflicting groups. A precomputed index is in `scpp_index.parquet` (`row` = 0-based data-row index, `code_sha256` = SHA-256 of whitespace-collapsed, stripped code, `is_class_like`).

## 3. MLCQ (Zenodo 3666840 v1.1, CC BY 4.0)

- URLs: `https://zenodo.org/api/records/3666840/files/<name>/content` for `MLCQCodeSmellSamples.csv` (7,513,770 B, md5 9019beae88089838d2d13dc12e69241b), `MLCQCodeSmellSamples.xlsx`, `MLCQCodeSmellDevelopersSurvey.csv/.xlsx`, `MadeyskiLewowskiMLCQAppendix.pdf`.
- Format: **semicolon-separated**, CRLF line endings, ASCII, header row. 14,739 rows (one row per reviewer×sample×smell review).
- Columns (example row):
  `id`=527; `reviewer_id`=6; `sample_id`=5771277; `smell`=`long method`; `severity`=`none`; `review_timestamp`=`2019-03-27 10:34:53.042443`; `type`=`function`; `code_name`=`org.apache.syncope.client.ui.commons.ConnIdSpecialName.ConnIdSpecialName`; `repository`=`git@github.com:apache/syncope.git`; `commit_hash`=`114c412afbfba24ffb4fbc804e5308a823a16a78`; `path`=`/client/idrepo/ui/src/main/java/.../ConnIdSpecialName.java`; `start_line`=35; `end_line`=37; `link`=`https://github.com/apache/syncope/blob/<sha>/<path>/#L35-L37`; `is_from_industry_relevant_project`=1.
- Smell names (lowercase): `blob`, `data class`, `feature envy`, `long method`. `type` is `class` or `function`.
- Labels: `severity` ∈ `none|minor|major|critical`. Binary label: `none` = negative, anything else = positive. Aggregate per `sample_id` (1–5 reviews; majority vote gives 276 pos / 2,148 neg / 6 ties for LM).
- Code: **not included**. Fetch with `https://raw.githubusercontent.com/<owner>/<repo>/<commit_hash><path>` and slice lines `start_line..end_line` (1-based, inclusive). `path` has a leading `/`. `link` has a stray `/` before `#L`. Convert `repository` from `git@github.com:owner/repo.git` form. Repo availability was not checked.
- `code_name` for functions includes the parameter types after a space, e.g. `...SocketServer#... NetworkConfig|SSLConfig|MetricRegistry|ArrayList<Port>`.

## 4. HRI-EU/SmellyCodeDataset (MIT)

- Clone `https://github.com/HRI-EU/SmellyCodeDataset.git`, checkout `70d987a032355d25651b62a7df8fa56e5ea09e85`.
- Use only the unannotated trees (the `SmellyAnnotated/` siblings add smell-name comments such as `//PrimitiveObsession`, `//DataClumps`):
  ```
  C++/SmellyUnannotated/        Cashier.{cpp,h} Chef.{cpp,h} Customer.{cpp,h} Drink.{cpp,h} Pizza.{cpp,h} Shop.{cpp,h} main.cpp Makefile  (+ committed *.o and `main` binary; ignore)
  Java/SmellyUnannotated/       Cashier.java Chef.java Customer.java Drink.java Pizza.java Shop.java main.java  (+ committed *.class; ignore)
  JavaScript/SmellyUnannotated/ Cashier.js Chef.js Customer.js Drink.js Pizza.js Shop.js main.js
  Python/SmellyUnannotated/     Cashier.py Chef.py Customer.py Drink.py Pizza.py Shop.py Main.py  (+ __pycache__; ignore)
  ```
  There is no TypeScript.
- Ground truth: `Analysis/GroundTruthLevel/GroundTruth.csv` (sha256 `5626b8ce300ac55f3d292fbae830623d299709b050430028cf34f8097c3155f4`), comma-separated with quoted free text.
  - Columns: `Language` (`Python|Java|JavaScript|C++`), `Class` (e.g. `Shop`, matching the file stem), `Code Smell #` (int), `Category` (e.g. `Bloaters`), `Code Smell` (e.g. `Long Method`, `Long Parameter List`, `Large Class`, … 20 names), `Method` (e.g. `long_method`, `orderWithUnnecessaryDetails`, or `Entire Class`), `Type` (free-text rationale).
  - **Gotcha:** the raw file contains 23 repeated header lines inside the data (rows where `Language == "Language"`); filter them out.
  - `Cleaned_GroundTruth.csv` (same dir; identical copies in every `Analysis/*Level/` dir, sha256 `e0c9538ee0193105fb618be99a8187e0a591c47a18a7bb647962529d55330c64`) has 478 rows and the same columns plus `Identifier` = `<Language>_<Class>_<Method>_<Code Smell>`, e.g. `Python_Shop___init___Primitive Obsession`.
- No line numbers are given. Locate spans by method name within `<Language>/SmellyUnannotated/<Class>.<ext>`. For C++, look in both `.h` (declaration) and `.cpp` (`Class::method` definition). Java `main.java` / Python `Main.py` carry the class name `Main`.
- Gotchas: **label-revealing identifiers** (`longMethod`, `long_method`, `longComplaintMethod`, `orderWithUnnecessaryDetails`). Rename them before any model-facing use. Every file has an MIT header saying "This dataset contains smelly code for research and refactoring purposes"; strip it. The ground truth is positives-only and **not exhaustive** (e.g. the 6-parameter `Customer.orderWithUnnecessaryDetails` in Java is not listed as LPL). There is no Deep Nesting label.
- Unannotated source SHA-256 (selected): Java `Shop.java` 0596c8e8…c689c7, Python `Shop.py` 0ccba55c…5e8c6, C++ `Shop.cpp` 5738bcc7…74ab. Full list is reproducible with `git ls-files */SmellyUnannotated | xargs sha256sum` at the pinned commit.

## Review sample CSVs (seed 20260927)
- `review_sample_smellycodepp_java_long_method.csv`: 30 LM=1 + 30 LM=0, drawn from method-like, label-consistent, hash-deduplicated rows. `sample_id` = `scpp_row<0-based data row>`. Columns: source, language, smell, label, sample_id, csv_data_row_0based, project, file, code_sha256.
- `review_sample_mlcq_java_long_method.csv`: 30 majority-positive + 30 majority-negative MLCQ samples (ties excluded). `sample_id` = `mlcq_<sample_id>`, with repo/commit/path/lines/link for code retrieval.
- Sampling method: `pandas.DataFrame.sample(n=30, random_state=20260927)` per label stratum.
