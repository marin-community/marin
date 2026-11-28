# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the measured source conversion before selecting its retained release."""

from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.datasets import qa_formatting
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")

_FAMILY, _REVISION, _FILENAME, _COUNT = qa_formatting.SOURCES["freeform"]
_REPO = "nvidia/" + _FAMILY
_VERSION = qa_formatting.VERSION


def sources() -> list[RlDataSource[CurationRecipe]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
    source = qa_formatting.source("freeform")
    actor = info.actor_census_input
    if (
        info.id != source.info.id
        or actor is None
        or (actor.dataset_id, actor.dataset_revision, actor.converter_version) != (_REPO, _REVISION, _VERSION)
        or actor.measured_tasks != record["original_converted_tasks"]
        or actor.task_specs_sha256 != record["original_task_specs_sha256"]
    ):
        raise ValueError("Free-Form Formatting intake differs from its measured source conversion")
    return [replace(source, info=info)]
