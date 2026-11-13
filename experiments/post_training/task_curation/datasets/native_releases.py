# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executed QUICK releases recovered from the retired source ledger."""

import json
from pathlib import Path

from finestore.rl.curation import (
    ConvertedExamples,
    DatasetRelease,
    DaytonaMeasurement,
    GradingEnvironmentEvidence,
    IndependentQC,
    OracleValidation,
    SourceOrigin,
    SourceQualityResult,
    Verdict,
)
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.native_release import import_native_quick
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat
from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_REASON, CROSSCODE_EVAL_SOURCE_IDS

LEDGER = Path(__file__).parents[1] / "retired-inventory-only.json"
RETIRED_ORIGINS = {
    "Harbor Hub": SourceOrigin.HARBOR_HUB,
    "Task Trove": SourceOrigin.TASKTROVE,
    "Snowball Smartie RL": SourceOrigin.TASKTROVE,
}
PYTHON_QC_PIN = (
    "open-athena/task-trove-laion-exp-rpt-crosscodeeval-python-v2-dc0ebee3-taskcompendium-20261112",
    "a953f93bc830704525ff2cce0b296dd262009662",
)
PYTHON_QC_ARTIFACT = (
    "s3://marin-us-east-02a/marin/rl-data/atlas/full-qc/shard-reviews/2026.11.13.3/"
    "b5d2abe06afc8a793318/attempts/e700d267866a43558fb771fddf3a8a53"
)


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    result = []
    for entry in json.loads(LEDGER.read_text())["entries"]:
        if entry["disposition"] != "quick_converted":
            continue
        release = entry["disposition_release"]
        if "daytona_measurement" not in release:
            continue
        measurement = DaytonaMeasurement.model_validate(release["daytona_measurement"])
        if (measurement.dataset_id, measurement.dataset_revision) != (release["repo"], release["revision"]):
            raise ValueError(f"Native actor measurement does not cover the private source pin: {entry['id']}")
        evidence = f"https://huggingface.co/datasets/{release['repo']}/blob/{release['revision']}/manifest.json"
        config = PrebuiltCohort(
            intended_use=IntendedUse.EVAL if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS else IntendedUse.TRAIN,
            name=entry["name"],
            version="2026.11.12.1",
            repo=release["repo"],
            revision=release["revision"],
            expected_rows=release["converted_rows"],
        )
        tags = tuple(tag for tag in entry["tags"] if tag != "excluded")
        notes = entry["notes"].split(" Admission blocked:")[0]
        if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS:
            tags = (*tags, "eval")
            notes += " " + CROSSCODE_EVAL_REASON
        qc = IndependentQC.model_validate(entry["independent_qc"]) if entry["independent_qc"] else None
        quality = None
        if (config.repo, config.revision) == PYTHON_QC_PIN:
            qc = IndependentQC(
                verdict=Verdict.GOOD,
                method=(
                    "FULL strict content assembly, coverage_complete=true: all 500 exact TaskSpec/runtime "
                    "contracts; execution qualification pending"
                ),
                reviewed=500,
                evidence_url=PYTHON_QC_ARTIFACT,
            )
            quality = SourceQualityResult(
                reviewed=500,
                issue_count=8,
                issues=[],
                repair_recommendation="Inspect the pinned task-level content evidence and renew QA for repairs.",
                unavailable_count=0,
                uncertain_count=0,
            )
        origin = RETIRED_ORIGINS[entry["origin"]]
        result.append(
            RlDataSource(
                config=config,
                pipeline=import_native_quick,
                info=SourceInfo(
                    id=entry["id"],
                    origin=origin,
                    upstream_dataset=entry["dataset_id"] if origin is SourceOrigin.HARBOR_HUB else entry["name"],
                    upstream_name=entry["display_name"],
                    upstream_url=entry["url"],
                    build_path="Native Harbor QUICK conversion",
                    current_release=DatasetRelease(
                        repo=release["repo"],
                        revision=release["revision"],
                        url=f"https://huggingface.co/datasets/{release['repo']}/tree/{release['revision']}",
                    ),
                    family=entry["family"],
                    tags=tags,
                    task_format=TaskFormat.HARBOR,
                    count=release["converted_rows"],
                    notes=(
                        notes + " Native Harbor package bytes are preserved in a private TaskCompendium QUICK release. "
                        "QUICK supplies no oracle or admission claim. "
                        "Native Dockerfile base tags and build downloads may remain mutable. "
                        "Source defects remain recorded."
                    ),
                    independent_qc=qc,
                    quality_result=quality,
                    oracle_validation=OracleValidation.SKIPPED,
                    daytona_snapshot_count=measurement.daytona_snapshot_count,
                    daytona_machine_slots=measurement.daytona_machine_slots,
                    daytona_execution_failures=measurement.daytona_execution_failures,
                    daytona_count_basis=measurement.daytona_count_basis,
                    daytona_roles=measurement.daytona_roles,
                    daytona_unsafe_reasons=tuple(measurement.daytona_unsafe_reasons),
                    daytona_unsafe_details=measurement.daytona_unsafe_details,
                    daytona_unsafe_counts=measurement.daytona_unsafe_counts,
                    grading_environment=GradingEnvironmentEvidence(
                        **release["grading_environment"], evidence_url=evidence
                    ),
                    converted_examples=ConvertedExamples(
                        count=release["converted_rows"],
                        input_rows=release["input_rows"],
                        rejections=release["rejections"],
                        admission_rejections={},
                        source_dataset_id=entry["dataset_id"],
                        source_revision=entry["dataset_revision"],
                        converter_revision=release["converter_identity"],
                        converter_version="2026.11.12.1",
                        evidence_url=evidence,
                    ),
                ),
            )
        )
    return result
