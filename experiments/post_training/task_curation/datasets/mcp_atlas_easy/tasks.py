# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import the complete, measured MCP-Atlas Easy evaluation population."""

from pathlib import Path

from pydantic import TypeAdapter
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.prebuilt import PrebuiltCohort, import_taskcompendium_prebuilt
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_name("qualification.json")


def sources() -> list[RlDataSource[PrebuiltCohort]]:
    qualification = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(qualification["info"])
    cohort = TypeAdapter(PrebuiltCohort).validate_python(qualification["cohort"])
    release = info.current_release
    if (
        info.id != qualification["source_id"]
        or release is None
        or (cohort.repo, cohort.revision) != (release.repo, release.revision)
        or cohort.intended_use is not IntendedUse.EVAL
    ):
        raise ValueError("MCP-Atlas Easy qualification differs from its evaluation release")
    return [RlDataSource(info=info, config=cohort, pipeline=import_taskcompendium_prebuilt)]
