# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize oracle-verified curriculum ablation rows for SFT."""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.inference.openai_batch import jsonl_text
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.curriculum_sft.ablation.matrix import AblationCell, generated_payloads_to_rows

DEFAULT_GENERATION_URI = "s3://marin-us-east-02a/marin/users/power/documents/curriculum-sft/ablation/2026.09.21.4"
GENERATION_FILENAME = "generation.json"
TRAIN_FILENAME = "train/examples.jsonl.gz"
MANIFEST_FILENAME = "manifest.json"


class AblationDataset(Artifact):
    """Canonical messages for one curriculum x generation-specification cell."""


@dataclass(frozen=True)
class MaterializeDatasetConfig:
    generation_root: str
    output_path: str
    cell: AblationCell


def materialize_dataset(config: MaterializeDatasetConfig) -> AblationDataset:
    """Filter one GLM cell and write the exact canonical messages consumed by SFT."""

    generation_path = StoragePath(config.generation_root) / GENERATION_FILENAME
    ledger = json.loads(generation_path.read_text())
    expected_name = config.cell.name
    matches = [entry for entry in ledger["cells"] if entry["cell"] == expected_name]
    if len(matches) != 1:
        raise ValueError(f"expected one generation ledger entry for {expected_name}, found {len(matches)}")
    entry = matches[0]
    rows = generated_payloads_to_rows(config.cell, entry["tasks"])

    output = StoragePath(config.output_path)
    output.mkdirs()
    training_path = output / TRAIN_FILENAME
    training_path.parent.mkdirs()
    with training_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(jsonl_text(rows).encode())
    manifest = {
        "cell": config.cell.name,
        "generation_cell": expected_name,
        "generation_root": config.generation_root,
        "accepted_examples": len(rows),
        "generation_quality": {
            key: entry[key]
            for key in (
                "requested",
                "accepted",
                "unique_accepted",
                "format_rate",
                "arithmetic_rate",
                "evidence_rate",
                "replicates",
            )
        },
        "generation_batch_id": ledger["batch_id"],
        "training_file": TRAIN_FILENAME,
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return AblationDataset(path=config.output_path)


def dataset_step(
    generation: ArtifactStep[Artifact],
    cell: AblationCell,
    *,
    version: str,
) -> ArtifactStep[AblationDataset]:
    """Materialize one oracle-verified training arm from the shared GLM ledger."""

    condition = f"{cell.curriculum}__{cell.generation_spec}"

    def build_config(ctx: StepContext) -> MaterializeDatasetConfig:
        return MaterializeDatasetConfig(
            generation_root=ctx.artifact_path(generation),
            output_path=ctx.output_path,
            cell=cell,
        )

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/ablation/{condition}"),
        version=version,
        artifact_type=AblationDataset,
        run=materialize_dataset,
        build_config=build_config,
        deps=(generation,),
    )
