# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the measured source conversion before selecting its retained release."""

from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.datasets import dolci_precise_if
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")


def sources() -> list[RlDataSource[CurationRecipe]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
    source = dolci_precise_if.source(tuple(record["source_files"]))
    actor = info.actor_census_input
    if (
        info.id != source.info.id
        or actor is None
        or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
        != (dolci_precise_if.REPO, dolci_precise_if.REVISION, dolci_precise_if.VERSION)
        or actor.measured_tasks != record["original_converted_tasks"]
        or actor.task_specs_sha256 != record["original_task_specs_sha256"]
    ):
        raise ValueError("Dolci Precise IF intake differs from its measured source conversion")
    return [replace(source, info=info)]
