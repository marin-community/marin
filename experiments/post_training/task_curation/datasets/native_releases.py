# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executed QUICK releases recovered from the retired source ledger."""

from pathlib import Path

from finestore.rl.curation import (
    ConvertedExamples,
    DatasetRelease,
    DaytonaMeasurement,
    GradingEnvironmentEvidence,
    IndependentQC,
    OracleValidation,
    SourceOrigin,
)
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.native_release import import_native_quick
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat
from experiments.post_training.task_curation.source_policy import CROSSCODE_EVAL_REASON, CROSSCODE_EVAL_SOURCE_IDS

LEDGER = Path(__file__).parents[1] / "retired-inventory-only.json"
RETIRED_ORIGINS = {
    "Harbor Hub": SourceOrigin.HARBOR_HUB,
    "Task Trove": SourceOrigin.TASKTROVE,
    "Snowball Smartie RL": SourceOrigin.TASKTROVE,
}


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    result = []
    for entry in read_registry_json(LEDGER)["entries"]:
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
