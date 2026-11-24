# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select TerminalWorld's measured execution scope before current-pin oracle admission."""

from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_name("qualification.json")


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
    config = TypeAdapter(PrebuiltCohort).validate_python(record["config"])
    original = record["admission"]["filtration"]
    if (
        info.id != record["source_id"]
        or info.current_release is None
        or (config.repo, config.revision) != (info.current_release.repo, info.current_release.revision)
        or config.expected_rows != original["original_count"]
    ):
        raise ValueError("TerminalWorld admission differs from its original execution population")
    return [RlDataSource(info=info, config=config, pipeline=import_taskcompendium_prebuilt)]
