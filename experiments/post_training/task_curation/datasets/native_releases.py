# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executed QUICK releases recovered from the retired source ledger."""

from finestore.rl.curation import (
    ConvertedExamples,
    DatasetRelease,
    DaytonaMeasurement,
    GradingEnvironmentEvidence,
    IndependentQC,
    OracleValidation,
    SourceOrigin,
)
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation import retirement
from experiments.post_training.task_curation.difficulty_config import JUDGE_ROUTE_REPAIRS_BY_SOURCE
from experiments.post_training.task_curation.native_release import import_native_quick
from experiments.post_training.task_curation.prebuilt import PrebuiltCohort
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, TaskFormat
from experiments.post_training.task_curation.source_policy import (
    CROSSCODE_EVAL_REASON,
    CROSSCODE_EVAL_SOURCE_IDS,
    SUPERSEDED_REPLACEMENTS,
)
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

RETIRED_ORIGINS = {
    "Harbor Hub": SourceOrigin.HARBOR_HUB,
    "Task Trove": SourceOrigin.TASKTROVE,
    "Snowball Smartie RL": SourceOrigin.TASKTROVE,
}


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    result = []
    # The retired-inventory ledger belongs to the retirement module; reading it
    # through one path keeps the fixture boundary and the pin in one place.
    for entry in read_registry_json(retirement.RETIREMENTS_PATH)["entries"]:
        if entry["disposition"] != "quick_converted":
            continue
        release = entry["disposition_release"]
        if "daytona_measurement" not in release:
            continue
        measurement = DaytonaMeasurement.model_validate(release["daytona_measurement"])
        if (measurement.dataset_id, measurement.dataset_revision) != (release["repo"], release["revision"]):
            raise ValueError(f"Native actor measurement does not cover the private source pin: {entry['id']}")
        conversion_version = release.get("conversion_version") or "2026.11.12.1"
        evidence = f"https://huggingface.co/datasets/{release['repo']}/blob/{release['revision']}/manifest.json"
        config = PrebuiltCohort(
            intended_use=IntendedUse.EVAL if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS else IntendedUse.TRAIN,
            name=entry["name"],
            version=conversion_version,
            repo=release["repo"],
            revision=release["revision"],
            expected_rows=release["converted_rows"],
        )
        tags = tuple(tag for tag in entry["tags"] if tag != "excluded")
        notes = entry["notes"].split(" Admission blocked:")[0]
        if entry["id"] in CROSSCODE_EVAL_SOURCE_IDS:
            tags = (*tags, "eval")
            notes += " " + CROSSCODE_EVAL_REASON
        # A versioned judge-route repair re-declares the source's own judge on the
        # served route; its protocol difference is recorded on this declaration.
        repair = JUDGE_ROUTE_REPAIRS_BY_SOURCE.get(entry["id"])
        if repair is not None:
            notes = f"{notes} {repair.note}".strip()
        qc = IndependentQC.model_validate(entry["independent_qc"]) if entry["independent_qc"] else None
        origin = RETIRED_ORIGINS[entry["origin"]]
        result.append(
            RlDataSource(
                config=config,
                pipeline=import_native_quick,
                info=SourceInfo(
                    data_restriction=DataRestriction.OPEN,
                    id=entry["id"],
                    origin=origin,
                    upstream_dataset=entry["dataset_id"] if origin is SourceOrigin.HARBOR_HUB else entry["name"],
                    upstream_name=entry["display_name"],
                    upstream_url=entry["url"],
                    build_path="Native Harbor QUICK conversion",
                    replacement_source_id=SUPERSEDED_REPLACEMENTS.get(canonical_source_id(entry["id"])),
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
                        **{
                            name: value
                            for name, value in release["grading_environment"].items()
                            if name != "evidence_url"
                        },
                        evidence_url=release["grading_environment"].get("evidence_url") or evidence,
                    ),
                    converted_examples=ConvertedExamples(
                        count=release["converted_rows"],
                        input_rows=release["input_rows"],
                        rejections=release["rejections"],
                        admission_rejections={},
                        source_dataset_id=entry["dataset_id"],
                        source_revision=entry["dataset_revision"],
                        converter_revision=release["converter_identity"],
                        converter_version=conversion_version,
                        evidence_url=evidence,
                    ),
                ),
            )
        )
    return result
