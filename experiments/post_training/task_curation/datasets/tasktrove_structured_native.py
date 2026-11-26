# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the native TaskTrove structured-output population with its exact-pin admission evidence."""

from pathlib import Path

from pydantic import TypeAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["info"])
    cohort = TypeAdapter(PrebuiltCohort).validate_python(record["cohort"])
    release, actor = info.current_release, info.actor_census_input
    if (
        info.id != record["source_id"]
        or release is None
        or actor is None
        or cohort.intended_use is not IntendedUse.TRAIN
        or (cohort.repo, cohort.revision, cohort.version, cohort.expected_rows)
        != (release.repo, release.revision, actor.converter_version, actor.measured_tasks)
        or (actor.dataset_id, actor.dataset_revision) != (release.repo, release.revision)
        or info.converted_examples is None
        or info.converted_examples.evidence_url != record["taskspec_conversion_receipt"]["uri"]
    ):
        raise ValueError("TaskTrove structured-output qualification differs from its prepared private population")
    return [RlDataSource(info=info, config=cohort, pipeline=import_taskcompendium_prebuilt)]
