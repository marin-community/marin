# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mechanical source conversion on an entered Zephyr pool, without review or grading."""

import json
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from functools import partial
from typing import Any

import pyarrow as pa
from rigging.filesystem.storage_path import StoragePath
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset, ShardInfo, format_shard_path
from zephyr.readers import compute_parquet_splits
from zephyr.writers import write_parquet_file

from taskcompendium.pipeline.audit_schema import TASK_SCHEMA
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from taskcompendium.pipeline.models import ImportRejection, RawRow, SourceRecipe
from taskcompendium.pipeline.sources import SourceShard, conversion_context, source_shards, staged_file_rows
from taskcompendium.pipeline.transforms import convert_row, row_source, row_task_id

CONVERSION_COLUMNS = (
    "task_id",
    "source_dataset",
    "source_revision",
    "source_row",
    "task_json",
    "normalization_kind",
    "normalization_reason",
    "normalization_detail",
    "normalization_changes",
)
CONVERSION_SCHEMA = pa.schema(
    [*(TASK_SCHEMA.field(name) for name in CONVERSION_COLUMNS), ("original_path", pa.string())]
)
PARQUET_SHARD_BYTES = 128 * 1024 * 1024


def conversion_shards(
    source_input: str, spec: SourceFiles, *, parquet_shard_bytes: int = PARQUET_SHARD_BYTES
) -> tuple[SourceShard, ...]:
    """Split native Parquet reads at row-group boundaries for mechanical conversion.

    Row groups remain intact, so a single large group can exceed the byte target.
    Custom readers and explicitly partitioned sources retain their own shard definitions.
    """
    shards = source_shards(source_input, spec)
    if spec.format != SourceFormat.PARQUET or spec.read is not None or spec.parts is not None:
        return shards
    parts = []
    for shard in shards:
        ranges = compute_parquet_splits(str(StoragePath(source_input) / shard.file), parquet_shard_bytes)
        parts.extend(
            replace(shard, part=index, parts=len(ranges), row_start=start, row_end=end)
            for index, (start, end) in enumerate(ranges)
        )
    return tuple(parts)


@dataclass(frozen=True)
class ConversionResult:
    normalized_path: str
    manifest_path: str
    input_rows: int
    converted_rows: int
    rejections: dict[str, int]
    elapsed_seconds: float


def _conversion_row(record: dict[str, Any], *, recipe: SourceRecipe) -> dict[str, Any]:
    source = row_source(recipe, record["locator"])
    task_id = row_task_id(recipe, source)
    result = convert_row(RawRow(task_id, source, record["data"]), recipe)
    rejection = result if isinstance(result, ImportRejection) else None
    return {
        "task_id": task_id,
        "source_dataset": source.dataset,
        "source_revision": source.revision,
        "source_row": source.row,
        "original_path": record["data"].get("path"),
        "task_json": None if isinstance(result, ImportRejection) else result.task.model_dump_json(),
        "normalization_kind": rejection.kind.value if rejection else None,
        "normalization_reason": rejection.reason if rejection else None,
        "normalization_detail": rejection.detail if rejection else None,
        "normalization_changes": (
            [] if isinstance(result, ImportRejection) else [change.model_dump(mode="json") for change in result.changes]
        ),
    }


def _write_conversion(rows: Iterator[dict[str, Any]], shard: ShardInfo, *, output_path: str) -> Iterator[Counter[str]]:
    counts: Counter[str] = Counter()

    def counted() -> Iterator[dict[str, Any]]:
        for row in rows:
            counts["input_rows"] += 1
            if row["task_json"] is not None:
                counts["converted_rows"] += 1
            else:
                counts[f"rejection:{row['normalization_kind']}:{row['normalization_reason']}"] += 1
            yield row

    path = format_shard_path(
        str(StoragePath(output_path) / "part-{shard:05d}.parquet"), shard.shard_idx, shard.total_shards
    )
    write_parquet_file(counted(), path, schema=CONVERSION_SCHEMA)
    yield counts


def run_conversion(
    recipe: SourceRecipe,
    context: ZephyrContext,
    source_input: str,
    output_path: str,
    *,
    parquet_shard_bytes: int = PARQUET_SHARD_BYTES,
) -> ConversionResult:
    """Convert every selected staged row, retaining tasks and typed conversion rejections.

    Model review, resource budgets, deduplication, mechanical checks and grader controls do not run.
    The output is unreviewed and has no production admission or ``final/`` view.
    Existing outputs are rejected so a failed rerun cannot mix old and new shards.
    """
    output = StoragePath(output_path)
    if output.exists():
        raise FileExistsError(f"Conversion output already exists: {output_path}")
    started = time.monotonic()
    dataset = (
        Dataset.from_list(list(conversion_shards(source_input, recipe.source, parquet_shard_bytes=parquet_shard_bytes)))
        .flat_map(partial(staged_file_rows, source_input, spec=recipe.source, context=conversion_context(recipe)))
        .map(partial(_conversion_row, recipe=recipe))
        .map_shard(partial(_write_conversion, output_path=str(output / "normalize")))
    )
    counts: Counter[str] = Counter()
    for shard_counts in context.execute(dataset).results:
        counts.update(shard_counts)
    result = ConversionResult(
        normalized_path=str(output / "normalize"),
        manifest_path=str(output / "manifest.json"),
        input_rows=counts["input_rows"],
        converted_rows=counts["converted_rows"],
        rejections={
            key.removeprefix("rejection:"): count for key, count in counts.items() if key.startswith("rejection:")
        },
        elapsed_seconds=time.monotonic() - started,
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                **asdict(result),
                "mode": "quick",
                "source": recipe.name,
                "source_dataset": recipe.source.dataset,
                "source_revision": recipe.source.revision,
                "recipe_revision": recipe.version,
                "reviewed": False,
                "verified": False,
            },
            indent=2,
        )
    )
    return result
