# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the measured hardmath@rhs conversion before its version-2 admission.

The qualification record (the registry-pinned hardmath_rhs_admitted.json) carries the measured
SourceInfo snapshot at recipe version 2 — the census, within-source dedup, lexical and SAMPLE QC
bindings at the landed converter fix 513d18175b — plus the BAD verdict documented as the source's
weakness (the coordinator's ruling 32d3f992: QC never blocks admission). The binding validates the
identity and pins against the declaration and serves the measured source.
"""

from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.datasets.skyrl.math import SOURCES, math_source
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).with_suffix(".json")


def sources() -> list[RlDataSource[CurationRecipe]]:
    record = read_registry_json(QUALIFICATION)
    info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
    source = math_source(next(item for item in SOURCES if item.name == "hardmath_rhs"))
    dataset = source.dataset
    if (
        info.id != source.info.id
        or dataset is None
        or (
            info.within_source_dedup.source_dataset_id,
            info.within_source_dedup.source_revision,
            info.within_source_dedup.converter_version,
        )
        != (dataset.name, dataset.revision, "2")
        or info.actor_census_input.measured_tasks != record["original_converted_tasks"]
        or info.actor_census_input.task_specs_sha256 != record["original_task_specs_sha256"]
    ):
        raise ValueError("The hardmath@rhs intake differs from its measured source conversion")
    return [replace(source, info=info)]
