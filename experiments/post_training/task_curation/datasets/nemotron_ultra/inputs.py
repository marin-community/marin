# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned Nemotron acquisition and reference-file bindings."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.datasets.nemotron_ultra.source import placeholder_record, swe_gym_ids
from taskcompendium.pipeline.inputs import HubDownload, RecipeInputs, SourceFiles, SourceFormat

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


@dataclass(frozen=True)
class UltraComponentSelector:
    selector: str

    def __call__(self, row: dict[str, Any], _staged_root: StoragePath) -> bool:
        identity = row.get("dataset") or "agent:" + row["agent_ref"]["name"]
        return identity == self.selector


@dataclass(frozen=True)
class SWEComponentSelector:
    selector: str
    component: str
    membership_path: str | None = None

    def __call__(self, row: dict[str, Any], staged_root: StoragePath) -> bool:
        if not UltraComponentSelector(self.selector)(row, staged_root):
            return False
        membership = StoragePath(self.membership_path) if self.membership_path else staged_root / SWE_GYM_FILE
        gym_member = row["metadata"]["instance_id"] in swe_gym_ids(membership)
        return gym_member == self.component.endswith("/SWE-Gym/SWE-Gym")


def resolve_ultra_placeholder(
    row: dict[str, Any], staged_root: StoragePath, *, reference_paths: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Attach pinned upstream evidence needed to reconstruct Ultra math placeholders."""
    placeholder = row.get("_hf_question_placeholder")
    if placeholder is None:
        return row
    dataset = placeholder["dataset"]
    split = placeholder["split"]
    index = int(placeholder["row"])
    if split != PLACEHOLDER_SPLITS[dataset]:
        raise ValueError(f"Unsupported placeholder split {dataset}/{split}")
    reference = (
        StoragePath(reference_paths[dataset])
        if reference_paths is not None
        else staged_root / PLACEHOLDER_FILES[dataset]
    )
    record = placeholder_record(reference, index)
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    source = {
        "dataset": dataset,
        "revision": PLACEHOLDER_PINS[dataset],
        "split": split,
        "row_index": index,
        "record_sha256": digest,
        "record": record,
    }
    return {**row, "placeholder_source": source}


@dataclass(frozen=True)
class UltraPlaceholderDecoder:
    reference_paths: Mapping[str, str]

    def __call__(self, row: dict[str, Any], staged_root: StoragePath) -> dict[str, Any]:
        return resolve_ultra_placeholder(row, staged_root, reference_paths=self.reference_paths)


def bind_reference_paths(files: SourceFiles, roots: Mapping[str, str]) -> SourceFiles:
    """Resolve separately acquired auxiliary files without copying the blend input."""
    if isinstance(files.selector, SWEComponentSelector):
        relative_file = SWE_GYM_FILE.split("/", 1)[1]
        selector = replace(files.selector, membership_path=str(StoragePath(roots["swe-gym-membership"]) / relative_file))
        return replace(files, selector=selector)
    references = {
        dataset: str(StoragePath(roots[subdirectory]) / relative_file)
        for dataset, path in PLACEHOLDER_FILES.items()
        for subdirectory, relative_file in (path.split("/", 1),)
    }
    return replace(files, decoder=UltraPlaceholderDecoder(references))


def blend_inputs(blend: str, selector: str, hf_id: str, revision: str) -> RecipeInputs:
    blend_file = f"{blend}.jsonl"
    return RecipeInputs(
        files=SourceFiles((blend_file,), SourceFormat.JSONL, selector=UltraComponentSelector(selector)),
        downloads=(HubDownload(hf_id, revision, (blend_file,)),),
    )


def math_inputs(blend: str, selector: str, hf_id: str, revision: str) -> RecipeInputs:
    base = blend_inputs(blend, selector, hf_id=hf_id, revision=revision)
    downloads = list(base.downloads)
    for dataset, revision in PLACEHOLDER_PINS.items():
        subdirectory, path = PLACEHOLDER_FILES[dataset].split("/", 1)
        downloads.append(HubDownload(dataset, revision, (path,), subdirectory))
    return RecipeInputs(
        files=SourceFiles(
            base.files.patterns, base.files.format, selector=base.files.selector, decoder=resolve_ultra_placeholder
        ),
        downloads=tuple(downloads),
    )


def swe_inputs(blend: str, selector: str, component: str, hf_id: str, revision: str) -> RecipeInputs:
    base = blend_inputs(blend, selector, hf_id=hf_id, revision=revision)
    return RecipeInputs(
        files=SourceFiles(base.files.patterns, base.files.format, selector=SWEComponentSelector(selector, component)),
        downloads=(
            *base.downloads,
            HubDownload(
                "SWE-Gym/SWE-Gym", SWE_GYM_REVISION, ("data/train-00000-of-00001.parquet",), "swe-gym-membership"
            ),
        ),
    )
