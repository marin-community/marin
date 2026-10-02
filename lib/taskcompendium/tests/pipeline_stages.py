# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local fixtures that exercise the supported curation stages."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from taskcompendium.pipeline.models import DatasetRecipe, FilterPolicy
from taskcompendium.pipeline.review import BatchReviewer
from taskcompendium.pipeline.sources import SourceFiles, SourceFormat
from taskcompendium.pipeline.zephyr import AuditExecution, ReviewConfig, audit_source, filter_source


def source_files() -> SourceFiles:
    return SourceFiles(("source.jsonl",), SourceFormat.JSONL)


def review_config(reviewer: BatchReviewer) -> ReviewConfig:
    return ReviewConfig(
        reviewer.model,
        reviewer.model_revision,
        reviewer.max_prompt_characters,
        reviewer.max_tokens,
        reviewer.max_attempts,
        reviewer.retry_max_tokens,
        reviewer.retry_max_prompt_characters,
    )


def run_stages(
    recipe: DatasetRecipe,
    rows: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
    limit: int,
    reviewer: BatchReviewer,
    policy: FilterPolicy = FilterPolicy(),
) -> dict[str, Any]:
    staged, audited, filtered = (output_path / name for name in ("staged", "audited", "filtered"))
    source = staged / "source.jsonl"
    if not source.exists():
        staged.mkdir(parents=True)
        source.write_text("".join(json.dumps(dict(row)) + "\n" for row in rows))
    if not (audited / "manifest.json").exists():
        audit_source(
            str(staged),
            str(audited),
            recipe,
            review_config(reviewer),
            AuditExecution(reviewer=reviewer),
            source_files(),
            limit,
        )
    return filter_source(str(audited), str(filtered), policy)


def stage_table(output_path: Path, view: str = "audit") -> pa.Table:
    files = sorted((output_path / "filtered" / view).glob("*.parquet"))
    table = pa.concat_tables([pq.read_table(file) for file in files])
    rows = sorted(
        table.to_pylist(),
        key=lambda row: (row["source_row"].rsplit(":", 1)[0], int(row["source_row"].rsplit(":", 1)[1])),
    )
    return pa.Table.from_pylist(rows, schema=table.schema)
