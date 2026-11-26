# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The re-sealed grader releases the #12329 review flagged, bound at their new pins.

The re-seal worker re-converts the seven stale-grader publications at main and
seals new releases; each landing re-pins one source here from its registry
receipt chain. The rlvr_ifeval entry is the first (commit ba7fcbe426 on the
atlas lanes, registry chain tip b6b19498, license union 20261126_b_v1): the
re-measurement shows 0 false passes, 0 false fails and 0 raises, and SAMPLE QC
is good — the acceptance criteria the review comment set.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from finestore.rl.curation import (
    ActorCensusInput,
    DatasetRelease,
    LexicalContaminationEvidence,
    OversightEvidence,
    OversightLevel,
)
from finestore.rl.licenses import LicenseReceipt
from huggingface_hub import hf_hub_download
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.retained_release import RetainedCohort, import_retained
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

REGISTRY = "open-athena/marin-rl-data-atlas-registry"
REGISTRY_TIP = "b6b19498d91e1e70fa7aace332a232c5ebdc4460"
REEVAL_RECEIPT_PATH = "experiments/post_training/task_curation/datasets/retained_release_receipts/part-000.json"
LEXICAL_RECEIPT_PATH = "experiments/post_training/task_curation/datasets/lexical_measurement_receipts/part-000.json"

SOURCE_ID = "AllenAI:rlvr_ifeval"
RELEASE_REPO = "open-athena/marinskyrl-rlvr-ifeval-00227508-retained-20261126-v1"
RELEASE_REVISION = "d95028e2c46a1c6dfa4543a08a3335f8044a3393"
TASKS_PARQUET_SHA256 = "c3a871e151f23f8cd184c909a1d82b20265ed15ce44c28a9d5af2043532f0a26"
EXPECTED_ROWS = 14352
UPSTREAM = "allenai/RLVR-IFeval"
UPSTREAM_REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
RETENTION_EVIDENCE = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/private-retention/2026.11.26.2/00227508d00904a922c7/result.json"
)
GRADING_EVIDENCE = "s3://marin-us-east-02a/marin/rl-data/atlas/retained-grading/2026.11.26.2/00227508d00904a922c7.json"
RAW_ARCHIVE_REPO = "open-athena/atlas-raw-marinskyrl-rlvr-ifeval-00227508-20261126-v1"
RAW_ARCHIVE_REVISION = "a0ca33edc1f000cfb17b5cb8b80ee63ef99dc149"
REESEAL_COMMIT = "ba7fcbe426"


def _read_registry(path: str) -> dict:
    downloaded = hf_hub_download(REGISTRY, repo_type="dataset", revision=REGISTRY_TIP, filename=path)
    return json.loads(Path(downloaded).read_text())


def resealed_source(sealed: RlDataSource) -> RlDataSource[RetainedCohort]:
    """Re-pin the sealed retained cohort to the re-sealed release.

    The receipt chain pins every field: the retention publication, the actor
    census, the lexical measurement and the license union. The sealed info's
    identity metadata carries; the pin-dependent fields are replaced.
    """
    receipt = _read_registry(REEVAL_RECEIPT_PATH)[SOURCE_ID]
    lexical_measurement = _read_registry(LEXICAL_RECEIPT_PATH)[SOURCE_ID]["measurement"]
    license_receipt = _read_license_receipt()
    publication = receipt["publication"]
    actor_input = receipt["actor"]["input"]
    plan = publication["plan"]
    assert (publication["repo"], publication["revision"]) == (RELEASE_REPO, RELEASE_REVISION)
    assert publication["tasks_parquet_sha256"] == TASKS_PARQUET_SHA256
    assert actor_input["measured_tasks"] == EXPECTED_ROWS
    assert plan["retained_count"] == EXPECTED_ROWS

    release_url = f"https://huggingface.co/datasets/{RELEASE_REPO}/tree/{RELEASE_REVISION}"
    info = replace(
        sealed.info,
        count=EXPECTED_ROWS,
        current_release=DatasetRelease(repo=RELEASE_REPO, revision=RELEASE_REVISION, url=release_url),
        dataset=SourceReference(RELEASE_REPO, RELEASE_REVISION, release_url),
        license_receipt=license_receipt,
        lexical_contamination=LexicalContaminationEvidence.model_validate(
            {**lexical_measurement, "receipt": _lexical_receipt(plan)}
        ),
        actor_census_input=ActorCensusInput.model_validate(actor_input),
        oversight=OversightEvidence(
            campaign="rlvr_ifeval grader re-seal",
            recipe_revision=f"datasets/skyrl/ifeval.py v3 @ {REESEAL_COMMIT}",
            verification=(
                "The re-seal re-measures the stored graders at the new pin: 0 false passes, "
                "0 false fails and 0 raises; SAMPLE QC verdict good (coordinator relay "
                "2026-11-26, the acceptance criteria the #12329 review comment set). "
                "Grading-boundary census: 14,352/14,352 matched. Evidence: " + GRADING_EVIDENCE
            ),
            level=OversightLevel.SAMPLE,
            reviewed=None,
            admitted=None,
            receipt_uri=RETENTION_EVIDENCE,
        ),
        notes=(
            "Re-sealed release (the #12329 review comment 7190184244 flagged the stored "
            "grader as predating main's IFEval-checker fixes). The re-converted cohort "
            f"carries 14,352 tasks graded by the verifyit IFEval checks at converter "
            f"version 3 ({UPSTREAM}@{UPSTREAM_REVISION[:12]} upstream); the raw archive "
            f"{RAW_ARCHIVE_REPO}@{RAW_ARCHIVE_REVISION[:12]} preserves the pinned upstream. "
            "The re-measurement shows 0 false passes, 0 false fails and 0 raises; SAMPLE "
            "QC good. Registry chain tip " + REGISTRY_TIP[:12] + "."
        ),
    )
    cohort = RetainedCohort(
        name="rlvr_ifeval-retained",
        version="2026.11.13.1",
        release=info.current_release,
        original_dataset=SourceReference(
            UPSTREAM,
            UPSTREAM_REVISION,
            f"https://huggingface.co/datasets/{UPSTREAM}/tree/{UPSTREAM_REVISION}",
        ),
        expected_rows=EXPECTED_ROWS,
        tasks_parquet_sha256=TASKS_PARQUET_SHA256,
        intended_use=IntendedUse.TRAIN,
        evidence_url=RETENTION_EVIDENCE,
    )
    return RlDataSource(info=info, config=cohort, pipeline=import_retained)


def _lexical_receipt(plan: dict) -> dict:
    """The decontamination receipt the private-release gate reads."""
    return {
        "use": "train",
        "status": "remediated",
        "retained_tasks": EXPECTED_ROWS,
        "dropped_tasks": plan["combined_drop_count"],
        "duplicate_dropped_tasks": plan["within_source_duplicates"],
        "release": {
            "private": True,
            "format": "taskcompendium",
            "dataset_id": RELEASE_REPO,
            "dataset_revision": RELEASE_REVISION,
            "drop_manifest_uri": plan["combined_drop_manifest_uri"],
            "original_private_dataset_id": RAW_ARCHIVE_REPO,
            "original_private_dataset_revision": RAW_ARCHIVE_REVISION,
            "duplicate_manifest_uri": plan["drop_manifest_uri"],
            "dedup_census_uri": plan["census_evidence_url"],
            "original_dataset_id": UPSTREAM,
            "original_dataset_revision": UPSTREAM_REVISION,
        },
    }


def _read_license_receipt() -> LicenseReceipt:
    """The re-pinned license union row for the new pin (index 20261126_b_v1).

    The committed index pins the private receipts revision and every shard's
    bytes; the read verifies each shard before decoding.
    """
    index = json.loads((Path(__file__).parent / "license_receipts" / "index.json").read_text())
    for reference in index["references"]:
        downloaded = hf_hub_download(
            reference["repo"], repo_type="dataset", revision=reference["revision"], filename=reference["path"]
        )
        data = Path(downloaded).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError(f"License receipt bytes changed: {reference['repo']}:{reference['path']}")
        document = json.loads(data)
        if SOURCE_ID in document:
            receipt = LicenseReceipt.model_validate(document[SOURCE_ID])
            assert (receipt.dataset_id, receipt.revision) == (RELEASE_REPO, RELEASE_REVISION)
            return receipt
    raise ValueError(f"The re-pinned license index lacks {SOURCE_ID}")

__all__ = ["SOURCE_ID", "resealed_source"]
