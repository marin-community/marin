# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Ultra components, their source files, and their reward contracts."""

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from zephyr.input_file import InputFileSpec
from zephyr.readers import load_parquet

from taskcompendium.pipeline.datasets.nemotron_ultra.normalization import DATASET, REVISION, normalize
from taskcompendium.pipeline.inputs import HubDownload, RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    NormalizedTask,
    RawRow,
    ReviewRubric,
)

SWE_GYM_REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
SWE_GYM_FILE = "swe-gym-membership/data/train-00000-of-00001.parquet"
PLACEHOLDER_PINS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
    "Skywork/Skywork-OR1-RL-Data": "1cdedc52e0e2db85fdf252f9be682e63a5a38c33",
}
PLACEHOLDER_FILES = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "placeholder-dapo/data/dapo-math-17k.parquet",
    "Skywork/Skywork-OR1-RL-Data": "placeholder-skywork/data/math-00000-of-00001.parquet",
}
PLACEHOLDER_SPLITS = {
    "BytedTsinghua-SIA/DAPO-Math-17k": "train",
    "Skywork/Skywork-OR1-RL-Data": "math",
}


@lru_cache(maxsize=1)
def _swe_gym_ids(path: StoragePath) -> frozenset[str]:
    return frozenset(row["instance_id"] for row in load_parquet(str(path)))


@dataclass(frozen=True)
class UltraComponentSelector:
    selector: str
    component: str
    family: str

    def __call__(self, row: dict[str, Any], staged_root: StoragePath) -> bool:
        identity = row.get("dataset") or "agent:" + row["agent_ref"]["name"]
        if identity != self.selector:
            return False
        if self.family != "swe-repo":
            return True
        gym_member = row["metadata"]["instance_id"] in _swe_gym_ids(staged_root / SWE_GYM_FILE)
        return gym_member == self.component.endswith("/SWE-Gym/SWE-Gym")


@lru_cache(maxsize=1024)
def _placeholder_source(root: StoragePath, dataset: str, split: str, index: int) -> dict[str, Any]:
    pin = PLACEHOLDER_PINS[dataset]
    if split != PLACEHOLDER_SPLITS[dataset]:
        raise ValueError(f"Unsupported placeholder split {dataset}/{split}")
    path = root / PLACEHOLDER_FILES[dataset]
    records = list(load_parquet(InputFileSpec(path=str(path), row_start=index, row_end=index + 1)))
    if len(records) != 1:
        raise ValueError(f"Pinned placeholder file omitted {dataset}/{split}/{index}")
    record = records[0]
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {
        "dataset": dataset,
        "revision": pin,
        "split": split,
        "row_index": index,
        "record_sha256": digest,
        "record": record,
    }


def resolve_ultra_placeholder(row: dict[str, Any], staged_root: StoragePath) -> dict[str, Any]:
    """Retain pinned upstream evidence needed to reconstruct Ultra math placeholders."""
    placeholder = row.get("_hf_question_placeholder")
    if placeholder is None:
        return row
    source = _placeholder_source(staged_root, placeholder["dataset"], placeholder["split"], int(placeholder["row"]))
    return {**row, "placeholder_source": source}


ACTION_COMPARISON_CRITERION = (
    "Only when agent_ref.name is single_step_tool_use_with_argument_comparison_agent, "
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent, or "
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent at verifier revision "
    "d8b6e8c163def3660e9d3072c1c174226a1709fa, expected_action.type=message accepts any nonempty "
    "assistant text with no tool calls; the stored message is not a literal answer key. For "
    "expected_action.type=function_call, the scorer requires one call with the expected name and "
    "recursively matching argument keys and values, allowing floating-point tolerance 1e-6. These "
    "documented rules do not certify a bound runtime."
)


@dataclass(frozen=True)
class NemotronSource:
    name: str
    blend: str
    selector: str
    component: str
    family: str
    atlas_id: str
    upstream: str
    rubric: ReviewRubric
    family_module: str


def recipe_for_source(
    source: NemotronSource,
) -> DatasetRecipe:
    """Bind a selected component to its pinned source and reward contract."""

    def normalize_row(row: RawRow) -> NormalizedTask | ImportRejection:
        return normalize(row, source.selector, source.family)

    blend_file = f"{source.blend}.jsonl"
    downloads = [HubDownload(DATASET, REVISION, (blend_file,))]
    if source.family == "swe-repo":
        downloads.append(
            HubDownload(
                "SWE-Gym/SWE-Gym", SWE_GYM_REVISION, ("data/train-00000-of-00001.parquet",), "swe-gym-membership"
            )
        )
    elif source.family == "math-answer":
        for dataset, revision in PLACEHOLDER_PINS.items():
            subdirectory, path = PLACEHOLDER_FILES[dataset].split("/", 1)
            downloads.append(HubDownload(dataset, revision, (path,), subdirectory))
    return DatasetRecipe(
        name=source.name,
        version=source.name + "-v1",
        source=HFSource(DATASET, REVISION, f"{source.blend}/{source.component}", "train"),
        inputs=RecipeInputs(
            files=SourceFiles(
                (blend_file,),
                SourceFormat.JSONL,
                selector=UltraComponentSelector(source.selector, source.component, source.family),
                decoder=resolve_ultra_placeholder if source.family == "math-answer" else None,
            ),
            downloads=tuple(downloads),
        ),
        normalize=normalize_row,
        rubric=source.rubric,
        intended_use=IntendedUse.TRAIN,
    )


def quality_source(
    blend: str,
    selector: str,
    upstream: str,
    family: str,
    criteria: tuple[str, ...],
    family_module: str,
    *,
    component: str | None = None,
) -> NemotronSource:
    """Describe a blend selection with its family and provenance review criteria."""
    component = selector if component is None else component
    name = "nemotron_ultra_" + blend + "_" + re.sub(r"[^a-z0-9]+", "_", component.lower()).strip("_")
    provenance = (
        f"This is the {component} selection from the {blend} training blend, originating at {upstream}. "
        "Judge its actual retained source fields and agent reward contract; do not infer byte "
        "equivalence with another blend."
    )
    return NemotronSource(
        name=name,
        blend=blend,
        selector=selector,
        component=component,
        family=family,
        atlas_id=f"MarinSkyRL:nemotron_ultra_{blend}/{component}",
        upstream=upstream,
        rubric=ReviewRubric(id=name + "-quality", version="1", criteria=(*criteria, provenance)),
        family_module=family_module,
    )


def preference_source(
    blend: str, selector: str, upstream: str, criteria: tuple[str, ...], family_module: str
) -> NemotronSource:
    """Describe the generation-based GenRM contract without inventing preference pairs."""
    name = f"nemotron_ultra_{blend}_{selector}"
    return NemotronSource(
        name=name,
        blend=blend,
        selector=selector,
        component=selector,
        family="preference",
        atlas_id=f"MarinSkyRL:nemotron_ultra_{blend}/{selector}",
        upstream=upstream,
        rubric=ReviewRubric(id=name + "-answerability", version="1", criteria=criteria),
        family_module=family_module,
    )
