# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select the unchanged native NeMo conversion before combined retention."""

from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
    config = TypeAdapter(PrebuiltCohort).validate_python(record["config"])
    plan = record["retained"]["publication"]["plan"]
    if (
        info.id != record["source_id"]
        or (config.repo, config.revision, config.version)
        != (plan["source_dataset_id"], plan["source_revision"], plan["converter_version"])
        or config.expected_rows != plan["raw_count"]
    ):
        raise ValueError("NeMo retention differs from its original native conversion")
    return [RlDataSource(info=info, config=config, pipeline=import_taskcompendium_prebuilt)]
