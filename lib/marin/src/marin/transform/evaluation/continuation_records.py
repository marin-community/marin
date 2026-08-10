# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering and staging for few-shot continuation PPL evals.

Both ``code_interpretation`` and ``prompt_format_sensitivity`` build the same
kind of supervised, target-only records: a prompt of finished support examples
plus one unfinished held-out query, scored only on the continuation the template
would append. The task/template/example shapes differ per eval, so the helpers
here are generic over the example type ``E`` and duck-type the small structural
surface they touch via the protocols below.

Both evals also stage a single task/template slice the same way — one JSONL file
per slice plus an ingestion metadata sidecar — so :func:`stage_continuation_slice`
owns that write path and the per-eval modules supply only records and metadata.
"""

import json
import posixpath
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    JsonValue,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from rigging.filesystem import StoragePath, atomic_rename, open_url

E = TypeVar("E")

DEFAULT_CONTINUATION_OUTPUT_FILENAME = "staged.jsonl.gz"


class ContinuationTask(Protocol[E]):
    key: str
    title: str
    description: str
    support_examples: tuple[E, ...]


class ContinuationTemplate(Protocol[E]):
    key: str
    description: str
    renderer: Callable[[E, bool], str]


@dataclass(frozen=True)
class ContinuationStagingConfig:
    """Configuration for staging one task/template slice of a continuation eval."""

    output_path: str
    task_key: str
    template_key: str
    output_filename: str = DEFAULT_CONTINUATION_OUTPUT_FILENAME
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


def render_support_and_query(
    *,
    task: ContinuationTask[E],
    template: ContinuationTemplate[E],
    heldout: E,
    num_fewshot: int,
) -> str:
    """Render ``num_fewshot`` finished support examples followed by one unfinished held-out query."""
    if len(task.support_examples) != num_fewshot:
        raise ValueError(f"{task.key} must have exactly {num_fewshot} support examples")
    header = f"Task: {task.title}\nInstruction: {task.description}\nFormat: {template.description}"
    blocks = [header, *(template.renderer(example, True) for example in task.support_examples)]
    blocks.append(template.renderer(heldout, False))
    return "\n\n".join(blocks)


def render_continuation_target(*, template: ContinuationTemplate[E], heldout: E) -> str:
    """Return the suffix by which the finished render extends the unfinished held-out query.

    Scoring the target on this suffix trains only the continuation tokens.
    """
    unfinished = template.renderer(heldout, False)
    finished = template.renderer(heldout, True)
    if not finished.startswith(unfinished):
        raise ValueError(f"{template.key} renderer must extend its unfinished held-out query")
    return finished[len(unfinished) :]


def stage_continuation_slice(
    cfg: ContinuationStagingConfig,
    *,
    records: Sequence[dict[str, Any]],
    source_id: str,
    metadata: dict[str, JsonValue],
) -> dict[str, Any]:
    """Write ``records`` as one JSONL slice and, when configured, its metadata sidecar.

    Args:
        cfg: Output location and the source manifest to validate against.
        records: Staged records, written one JSON object per line. Gzipped when
            ``cfg.output_filename`` ends in ``.gz``.
        source_id: Static identifier for the generator, recorded as the
            materialized output's input path.
        metadata: Slice-specific provenance for the metadata sidecar.

    Returns the staged slice summary: record count, bytes written, output file,
    and the metadata file path when a source manifest is configured.
    """
    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    StoragePath(cfg.output_path).mkdirs(exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for record in records:
                json.dump(record, outfile)
                outfile.write("\n")

    bytes_written = StoragePath(out_file).size()
    result: dict[str, Any] = {
        "record_count": len(records),
        "bytes_written": bytes_written,
        "output_file": out_file,
    }
    if cfg.source_manifest is not None:
        result["metadata_file"] = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path=source_id,
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=len(records),
                bytes_written=bytes_written,
                metadata=metadata,
            ),
        )
    return result
