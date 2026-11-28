# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the measured Nemotron rename conversions before selecting their retained releases.

Each admitted rename row (the five Nemotron-Ultra lean/rdkit R41 intake rows) carries a
registry-pinned qualification record: the retained release receipt's original_info (the
original-pin SourceInfo with its actor census binding), the converted count and the
original TaskSpec digest. The module validates the record against the declared component
recipe and returns the measured binding; the retained registry receipt selects the
private release at serving time.
"""

from dataclasses import replace
from pathlib import Path

from pydantic import TypeAdapter

from experiments.post_training.task_curation.datasets.nemotron_ultra import components
from experiments.post_training.task_curation.pipeline import CurationRecipe
from experiments.post_training.task_curation.registry import read_registry_json
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo

QUALIFICATION = Path(__file__).parent

_ROWS = {
    "lean": "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_lean",
    "rdkit": "Nemotron:nemotron_ultra_mopd/ultra_sft_step3200_rdkit",
    "rlvr1-lean": "Nemotron:nemotron_ultra_rlvr1/ultra_sft_step3200_lean",
    "rlvr2-lean": "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_lean",
    "rlvr2-rdkit": "Nemotron:nemotron_ultra_rlvr2/ultra_sft_step3200_rdkit",
}


def sources() -> list[RlDataSource[CurationRecipe]]:
    base = {
        source.info.id: source
        for source in components.sources()
        if source.info.id.startswith("Nemotron:nemotron_ultra_")
    }
    out = []
    for row, source_id in _ROWS.items():
        record = read_registry_json(QUALIFICATION / f"nemotron_ultra_renames_admitted_{row}.json")
        info = TypeAdapter(SourceInfo).validate_python(record["original_info"])
        source = base[source_id]
        actor = info.actor_census_input
        if (
            info.id != source.info.id
            or actor is None
            or source.dataset is None
            or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
            != (source.dataset.name, source.dataset.revision, source.config.version)
            or actor.measured_tasks != record["original_converted_tasks"]
            or actor.task_specs_sha256 != record["original_task_specs_sha256"]
        ):
            raise ValueError(f"{info.id} intake differs from its measured source conversion")
        out.append(replace(source, info=info))
    return out
