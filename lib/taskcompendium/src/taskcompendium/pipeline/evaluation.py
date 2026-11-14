# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish complete evaluation populations from the shared conversion pipeline."""

import json
from collections.abc import Iterator, Mapping
from functools import partial

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import write_parquet_file

from taskcompendium.evaluation.corpus import (
    MANIFEST_FILENAME,
    PROTOCOL_FILENAME,
    CorpusFile,
    EvaluationManifest,
    EvaluationShard,
    ItemBuilder,
    canonical_task,
    file_sha256,
    parquet_rows,
    source_order,
    validate_item,
    validate_task_item,
)
from taskcompendium.evaluation.spec import EvaluationProtocol
from taskcompendium.pipeline.audit_schema import NORMALIZED_SCHEMA
from taskcompendium.pipeline.inputs import SourceFileOverride
from taskcompendium.pipeline.models import IntendedUse, SourceRecipe, SourceStatus
from taskcompendium.pipeline.source_processing import ConversionResult, SourceProcessingMode, run_source_pipeline

ITEM_SCHEMA = pa.schema([("task_id", pa.string()), ("source_row", pa.string()), ("evaluation_json", pa.string())])


def _source_order_key(row: dict) -> str:
    file, index = source_order(row["source_row"])
    return f"{file}:{index:020d}"


def _publish_shard(
    path: str,
    *,
    output_path: str,
    protocol: EvaluationProtocol,
    item_builder: ItemBuilder,
    source_dataset: str,
    source_revision: str,
    importer_revision: str,
) -> dict:
    source = StoragePath(path)
    destination = StoragePath(output_path) / "evaluation" / source.name
    count = 0
    first = last = ""

    def rows() -> Iterator[dict[str, str]]:
        nonlocal count, first, last
        for row in parquet_rows(source):
            task = canonical_task(
                row, source_dataset=source_dataset, source_revision=source_revision, importer_revision=importer_revision
            )
            item = item_builder(task)
            validate_item(item, protocol)
            validate_task_item(task, item)
            if (item.task_id, item.source_row) != (task.id, task.source.row):
                raise ValueError(f"Evaluation sidecar changed task provenance: {task.id}")
            if count and source_order(item.source_row) <= source_order(last):
                raise ValueError("Evaluation source rows must be in original source order")
            first = first or item.source_row
            last = item.source_row
            count += 1
            yield {"task_id": task.id, "source_row": task.source.row, "evaluation_json": item.model_dump_json()}

    write_parquet_file(rows(), str(destination), schema=ITEM_SCHEMA)
    return EvaluationShard(
        tasks=CorpusFile(path="evaluation-tasks/" + source.name, sha256=file_sha256(source)),
        items=CorpusFile(path="evaluation/" + source.name, sha256=file_sha256(destination)),
        rows=count,
        first_source_row=first,
        last_source_row=last,
    ).model_dump(mode="json")


def publish_evaluation_corpus(
    recipe: SourceRecipe,
    protocol: EvaluationProtocol,
    item_builder: ItemBuilder,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    *,
    source_overrides: Mapping[str, SourceFileOverride] | None = None,
    curation_identity: str | None = None,
    protocol_name: str | None = None,
) -> EvaluationManifest:
    """Convert all selected rows and publish a manifest only after population and sidecar checks pass.

    Conversion drops exact duplicates and preserves retained source order. The protocol pins the
    raw input population; the manifest records unique tasks. Controls and semantic comparison
    qualify grader behavior independently. ``curation_identity`` binds the semantic recipe
    independently of storage placement for comparisons across regional builds.
    """
    if recipe.intended_use is not IntendedUse.EVAL:
        raise ValueError("An evaluation corpus requires an evaluation source recipe")
    output = StoragePath(output_path)
    if (output / MANIFEST_FILENAME).exists():
        raise FileExistsError(f"Evaluation corpus already published: {output_path}")
    if any(
        (output / name).exists()
        for name in (
            "normalize",
            "evaluation",
            "evaluation-tasks",
            "within-source-dedup",
            "manifest.json",
            PROTOCOL_FILENAME,
        )
    ):
        raise FileExistsError(f"Incomplete evaluation output requires a new artifact version: {output_path}")
    result = run_source_pipeline(
        recipe,
        context,
        source_input,
        output_path,
        mode=SourceProcessingMode.QUICK,
        canonical_source=recipe.name,
        source_overrides=source_overrides,
    )
    assert isinstance(result, ConversionResult)
    if (
        result.rejections
        or result.input_rows != protocol.expected_population
        or result.converted_rows + result.within_source_dedup["within_source_duplicates"] != result.input_rows
    ):
        raise ValueError(
            f"Evaluation population mismatch: expected {protocol.expected_population}, "
            f"read {result.input_rows}, converted {result.converted_rows}, rejections={result.rejections}"
        )
    paths = sorted(str(path) for path in (StoragePath(result.normalized_path) / "*.parquet").glob())
    ordered = context.execute(
        Dataset.from_list(paths)
        .flat_map(lambda path: parquet_rows(StoragePath(path)))
        .group_by(lambda _row: "source", reducer=lambda _key, rows: rows, sort_by=_source_order_key, num_output_shards=1)
        .write_parquet(str(output / "evaluation-tasks/part-{shard:05d}.parquet"), schema=NORMALIZED_SCHEMA),
        name="evaluation-source-order",
    )
    paths = [str(path) for path in ordered.results]
    published = context.execute(
        Dataset.from_list(paths).map(
            partial(
                _publish_shard,
                output_path=output_path,
                protocol=protocol,
                item_builder=item_builder,
                source_dataset=recipe.source.dataset,
                source_revision=recipe.source.revision,
                importer_revision=recipe.version,
            )
        ),
        name="evaluation-sidecars",
    )
    shards = sorted(
        (shard for row in published.results if (shard := EvaluationShard.model_validate(row)).rows),
        key=lambda shard: source_order(shard.first_source_row),
    )
    conversion_path = output / "manifest.json"
    conversion = json.loads(conversion_path.read_text())
    conversion.update(
        status=SourceStatus.COMPLETED.value,
        mode=SourceProcessingMode.QUICK.value,
        source=recipe.name,
        source_dataset=recipe.source.dataset,
        source_revision=recipe.source.revision,
        recipe_revision=recipe.version,
        intended_use=IntendedUse.EVAL.value,
        raw_population_count=result.input_rows,
        raw_sample_count=result.input_rows,
        raw_population_census=True,
        full_expansion=True,
        processed_rows=result.converted_rows,
        unprocessed_rows=0,
    )
    conversion_path.write_text(json.dumps(conversion, indent=2))
    manifest = EvaluationManifest(
        source_dataset=recipe.source.dataset,
        source_revision=recipe.source.revision,
        importer_revision=recipe.version,
        protocol_hash=protocol.fingerprint,
        conversion_manifest_sha256=file_sha256(conversion_path),
        curation_identity=curation_identity,
        protocol_name=protocol_name,
        rows=result.converted_rows,
        shards=tuple(shards),
    )
    (output / PROTOCOL_FILENAME).write_text(protocol.model_dump_json(indent=2))
    (output / MANIFEST_FILENAME).write_text(manifest.model_dump_json(indent=2))
    return manifest
