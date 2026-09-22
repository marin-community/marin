# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize oracle-verified curriculum ablation rows through Datakit."""

from __future__ import annotations

import json
from dataclasses import dataclass

from levanter.data.text.formats import TextLmDatasetFormat
from levanter.tokenizers import TokenizerBackend
from marin.datakit.chat_normalize import CHAT_SCHEMA, normalize_chat_to_parquet
from marin.datakit.chat_render import render_chat_to_parquet
from marin.datakit.download.rollout_transforms import openai_chat_document
from marin.datakit.normalize import NormalizedData, normalize_to_parquet
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from marin.processing.tokenize.attributes import TokenizeAttributesConfig, tokenize_attributes
from marin.processing.tokenize.store_builder import (
    BuildLevanterStoreConfig,
    build_levanter_store,
)
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr.writers import write_parquet_file

from experiments.post_training.curriculum_sft.ablation.matrix import AblationCell, generated_payloads_to_rows

DEFAULT_GENERATION_URI = "s3://marin-us-east-02a/marin/users/power/documents/curriculum-sft/ablation/2026.09.21.4"
GENERATION_FILENAME = "generation.json"
RAW_CHAT_FILENAME = "chat/part-00000-of-00001.parquet"
MANIFEST_FILENAME = "manifest.json"
DATA_SOURCE = "curriculum-sft"
NORMALIZED_MAIN_RELATIVE_PATH = "normalized/outputs/main"
STORE_RELATIVE_PATH = "store"


class AblationDataset(Artifact):
    """Normalized rendered-text Parquet for one curriculum ablation cell."""

    main_output_dir: str


class AblationStore(Artifact):
    """Datakit token store consumed by one curriculum SFT arm."""

    cache_path: str
    total_tokens: int


@dataclass(frozen=True)
class MaterializeDatasetConfig:
    generation_root: str
    output_path: str
    cell: AblationCell


@dataclass(frozen=True)
class BuildStoreConfig:
    normalized_path: str
    output_path: str
    tokenizer: str


def materialize_dataset(config: MaterializeDatasetConfig) -> AblationDataset:
    """Filter one GLM cell and produce Datakit-normalized rendered text."""

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
    chat_path = output / RAW_CHAT_FILENAME
    chat_documents = [openai_chat_document(row["messages"], DATA_SOURCE, source_id=row["id"]) for row in rows]
    write_parquet_file(chat_documents, str(chat_path), schema=CHAT_SCHEMA)

    normalized_chat = normalize_chat_to_parquet(
        input_path=str(chat_path.parent),
        output_path=str(output / "normalized-chat"),
        file_extensions=(".parquet",),
        max_workers=1,
    )
    rendered_path = output / "rendered"
    render_chat_to_parquet(
        input_path=normalized_chat.main_output_dir,
        output_path=str(rendered_path),
        max_workers=1,
    )
    normalized = normalize_to_parquet(
        input_path=str(rendered_path),
        output_path=str(output / "normalized"),
        file_extensions=(".parquet",),
        max_workers=1,
        bare=True,
    )
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
        "training_data": f"{NORMALIZED_MAIN_RELATIVE_PATH}/*.parquet",
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return AblationDataset(path=config.output_path, main_output_dir=normalized.main_output_dir)


def build_store(config: BuildStoreConfig) -> AblationStore:
    """Tokenize normalized Parquet and build the Levanter store used for training."""

    normalized = NormalizedData(
        main_output_dir=config.normalized_path,
        dup_output_dir=prefix_join(config.output_path, "unused-dups"),
        counters={},
    )
    tokenized = tokenize_attributes(
        TokenizeAttributesConfig(
            train_source=normalized,
            output_path=prefix_join(config.output_path, "tokenized"),
            tokenizer=config.tokenizer,
            tokenizer_backend=TokenizerBackend.HF,
            format=TextLmDatasetFormat(),
            max_workers=1,
        )
    )
    store = build_levanter_store(
        BuildLevanterStoreConfig(
            sources=[tokenized],
            cache_path=prefix_join(config.output_path, STORE_RELATIVE_PATH),
            max_workers=1,
        )
    )
    train = store.splits.get("train")
    if train is None or train.total_tokens <= 0:
        raise ValueError("curriculum SFT Datakit store produced no training tokens")
    return AblationStore(path=config.output_path, cache_path=store.cache_path, total_tokens=train.total_tokens)


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


def store_step(
    dataset: ArtifactStep[AblationDataset],
    tokenizer: ArtifactStep[Artifact],
    cell: AblationCell,
    *,
    version: str,
) -> ArtifactStep[AblationStore]:
    """Build a Datakit token store for one materialized ablation cell."""

    condition = f"{cell.curriculum}__{cell.generation_spec}"

    def build_config(ctx: StepContext) -> BuildStoreConfig:
        return BuildStoreConfig(
            normalized_path=prefix_join(ctx.artifact_path(dataset), NORMALIZED_MAIN_RELATIVE_PATH),
            output_path=ctx.output_path,
            tokenizer=ctx.artifact_path(tokenizer),
        )

    return ArtifactStep(
        name=user_owned_name(f"tokenized/curriculum-sft/ablation/{condition}"),
        version=version,
        artifact_type=AblationStore,
        run=build_store,
        build_config=build_config,
        deps=(dataset, tokenizer),
    )
