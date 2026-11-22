# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the privately pinned MiMo code population with static no-oracle evidence."""

from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.retained_release import RetainedCohort, import_retained
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_name("qualification.json")


def sources() -> list[RlDataSource[RetainedCohort]]:
    """Preserve the qualified release's TaskSpec bytes and native LOCAL routing."""
    qualification = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(qualification["info"])
    cohort = TypeAdapter(RetainedCohort).validate_python(qualification["cohort"])
    if info.id != qualification["source_id"] or info.current_release != cohort.release:
        raise ValueError("MiMo qualification belongs to another source or private pin")
    return [RlDataSource(info=info, config=cohort, pipeline=import_retained)]
