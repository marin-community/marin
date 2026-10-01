# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded local exercises of the same Zephyr stages used by artifact workflows."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from itertools import islice
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.pipeline.fingerprints import code_digest
from taskcompendium.pipeline.models import DatasetRecipe, FilterPolicy, SnapshotSource
from taskcompendium.pipeline.records import write_jsonl
from taskcompendium.pipeline.review import BatchReviewer, Reviewer
from taskcompendium.pipeline.zephyr import (
    AuditExecution,
    ReviewConfig,
    SourceAcquisition,
    acquire_source,
    audit_source,
    filter_source,
)


def _export_sample(filtered_path: Path, audited_path: Path, output_path: Path) -> None:
    """Write convenient flat ledgers for bounded pilot and rewrite exercises."""
    audit = pa.concat_tables([pq.read_table(path) for path in sorted((filtered_path / "audit").glob("*.parquet"))])
    records = sorted(audit.to_pylist(), key=lambda row: int(row["source_row"].rsplit(":", 1)[1]))
    audit = pa.Table.from_pylist(records, schema=audit.schema)
    pq.write_table(audit, output_path / "audit.parquet", compression="zstd")
    pq.write_table(
        audit.filter(pa.array([row["filter_status"] == "keep" for row in records])),
        output_path / "accepted.parquet",
        compression="zstd",
    )
    write_jsonl(output_path / "raw.jsonl", (json.loads(row["raw_json"]) for row in records))
    write_jsonl(output_path / "normalized.jsonl", (json.loads(row["task_json"]) for row in records if row["task_json"]))
    write_jsonl(
        output_path / "decisions.jsonl",
        (
            {
                "task_id": row["task_id"],
                "disposition": row["filter_status"],
                "reasons": row["filter_reasons"],
                "duplicate_of": row["duplicate_of"],
            }
            for row in records
        ),
    )
    checks, reviews, rollouts = {}, {}, []
    for directory in sorted((audited_path / "evidence").glob("*")):
        checks.update(json.loads((directory / "checks.json").read_text()))
        reviews.update((row["task_id"], row) for row in json.loads((directory / "reviews.json").read_text()))
    write_jsonl(output_path / "reviews.jsonl", (reviews[row["task_id"]] for row in records if row["task_id"] in reviews))
    check_rows = []
    for row in records:
        if row["task_id"] not in checks:
            continue
        task = json.loads(row["task_json"])
        task.pop("id")
        task.pop("source")
        evidence = checks[row["task_id"]]
        check_rows.append({"task_id": row["task_id"], "task_sha256": canonical_sha256(task), **evidence})
        rollouts.extend(evidence["rollouts"])
    write_jsonl(output_path / "checks.jsonl", check_rows)
    write_jsonl(output_path / "rollouts.jsonl", rollouts)


def run_pipeline(
    recipe: DatasetRecipe,
    rows: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
    limit: int,
    reviewer: Reviewer,
    policy: FilterPolicy = FilterPolicy(),
) -> dict[str, Any]:
    """Exercise a bounded sample locally through acquisition, audit, and filter stages.

    The artifact workflow owns production caching and source composition. This
    helper retains flat local ledgers needed by the instruction-rewrite exercises.
    Refiltering reuses the audited observations without invoking the reviewer.
    """
    if limit <= 0:
        raise ValueError("A positive sample limit is required")
    output_path.mkdir(parents=True, exist_ok=True)
    identity = json.loads(
        json.dumps(
            {
                "recipe": recipe.name,
                "recipe_version": recipe.version,
                "source": asdict(recipe.source),
                "rubric": asdict(recipe.rubric),
                "intended_use": recipe.intended_use.value,
                "limit": limit,
                "reviewer": reviewer.identity,
                "checks": (
                    {
                        "id": recipe.check_suite.id,
                        "revision": recipe.check_suite.revision,
                        "parameters": dict(recipe.check_suite.parameters),
                    }
                    if recipe.check_suite
                    else {"id": "answer-controls", "revision": "1"}
                ),
                "code_sha256": code_digest(recipe),
            }
        )
    )
    identity_path = output_path / "run-config.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Output directory belongs to a different curation run")
    identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    acquired, audited, filtered = (output_path / name for name in ("acquired", "audited", "filtered"))
    if not (acquired / "manifest.json").exists():
        snapshot = output_path / "input.jsonl"
        write_jsonl(snapshot, (dict(row) for row in islice(rows, limit)))
        source = recipe.source
        acquisition = SourceAcquisition(
            SnapshotSource(source.dataset, source.revision, source.config, source.split, str(snapshot)), limit
        )
        acquire_source(acquisition, str(acquired))
    if not (audited / "manifest.json").exists():
        if isinstance(reviewer, BatchReviewer):
            review = ReviewConfig(
                reviewer.model,
                reviewer.model_revision,
                reviewer.max_prompt_characters,
                reviewer.max_tokens,
                reviewer.max_attempts,
                reviewer.retry_max_tokens,
                reviewer.retry_max_prompt_characters,
            )
        else:
            review = ReviewConfig("injected", canonical_sha256(reviewer.identity))
        audit_source(str(acquired), str(audited), recipe, review, AuditExecution(reviewer=reviewer))
    manifest = filter_source(str(audited), str(filtered), policy)
    _export_sample(filtered, audited, output_path)
    manifest = {
        **identity,
        **manifest,
        "raw_sample_sha256": json.loads((acquired / "manifest.json").read_text())["raw_sample_sha256"],
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
