"""Source-specific loaders.  Each returns ``schema.Sample`` objects with explicit labels.

Roles follow the Gate 0 report (``reports/gate0/dataset_admission_report.md``):

* ``hri_eu``        synthetic smoke-test fixture only (one repository family, positives only)
* ``smellycodepp``  weak Java Long Method labels for training mechanics only
* ``mlcq``          human-reviewed Java Long Method reference (weak until our rubric review)

None of them may supply the locked primary test set.
"""
from __future__ import annotations

from .. import leakage, static_metrics
from ..extract import EXTRACTION_VERSION
from ..rubric import RUBRIC_VERSION
from ..schema import Label, Sample, sha256_text


def make_sample(*, sample_id, dataset, repository_url, commit_sha, file_path, start_line, end_line, language,
                symbol, code, labels: dict[str, Label], license, provenance, family_map=None,
                sample_type="method") -> Sample:
    return Sample(
        sample_id=sample_id, dataset=dataset, repository_url=repository_url,
        repository_family_id=leakage.family_id(repository_url, family_map), commit_sha=commit_sha,
        file_path=file_path, start_line=start_line, end_line=end_line, language=language,
        sample_type=sample_type, symbol=symbol, code=code, labels=labels, license=license,
        provenance=provenance, static_metrics=static_metrics.compute(code, language),
        content_hash=sha256_text(code), fingerprint=leakage.fingerprint(code, language),
        extraction_version=EXTRACTION_VERSION, rubric_version=RUBRIC_VERSION)
