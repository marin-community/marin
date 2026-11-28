# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import Workplace's prepared image and exact-pin native consumer qualification."""

from functools import lru_cache
from pathlib import Path

from finestore.rl.curation import ConsumerNeed
from pydantic import TypeAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")
CONSUMER_RUNTIME = "e4e00aaa24c46ddc9f63db11e06e55655df73e2c"


@lru_cache(maxsize=1)
def qualification() -> dict:
    """The pinned qualification record, read once per process at first use."""
    return read_registry_json(QUALIFICATION)


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    record = qualification()
    info = TypeAdapter(SourceInfo).validate_python(record["info"])
    cohort = TypeAdapter(PrebuiltCohort).validate_python(record["cohort"])
    release, actor, consumer = info.current_release, info.actor_census_input, info.consumer_receipt
    if (
        info.id != record["source_id"]
        or release is None
        or actor is None
        or consumer is None
        or cohort.intended_use is not IntendedUse.TRAIN
        or (cohort.repo, cohort.revision, cohort.version, cohort.expected_rows)
        != (release.repo, release.revision, actor.converter_version, actor.measured_tasks)
        or (actor.dataset_id, actor.dataset_revision) != (release.repo, release.revision)
        or info.consumer_runtime_commit != CONSUMER_RUNTIME
        or (
            consumer.source_id,
            consumer.dataset_id,
            consumer.dataset_revision,
            consumer.task_specs_sha256,
            consumer.runtime_commit,
            consumer.checked_tasks,
            consumer.failed_tasks,
        )
        != (
            info.id,
            actor.dataset_id,
            actor.dataset_revision,
            actor.task_specs_sha256,
            info.consumer_runtime_commit,
            actor.measured_tasks,
            0,
        )
        or set(info.consumer_requirements) != set(consumer.requirements)
        or consumer.requirements.get(ConsumerNeed.TOOL_PROVIDER) != cohort.expected_rows
    ):
        raise ValueError("Workplace qualification differs from its selected runtime or private population")
    return [RlDataSource(info=info, config=cohort, pipeline=import_taskcompendium_prebuilt)]
