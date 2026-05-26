# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr pipeline assembly for the distributed inference library.

Each Zephyr shard processes exactly one input file. The file's position in
the deterministically-sorted input list is the content shard ID — the same
file always maps to the same content shard regardless of which region picks
it up. Per-region rotation reorders the processing order to avoid
inter-region collisions on the first shards.

The map_shard callable writes its output **directly** to the canonical
``shard-NNNNNNNN.jsonl.gz`` path in the results region. We do not use
Zephyr's `write_jsonl` because Zephyr's per-shard output filename pattern
depends on its positional shard index, which is per-region and therefore not
content-stable. Doing our own write lets the output filename encode the
content shard ID instead.

Cross-region race semantics: `skip_existing` (checked before inference) skips
shards already published by another region. Two regions racing the same
shard both compute the work; whichever's write lands last overwrites the
first, but since records are deterministic per content shard, the final file
content is identical.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any

import fsspec
from zephyr.dataset import Dataset

from . import vllm_worker
from .config import ModelSpec, SamplingParams
from .input import load_jsonl_records
from .output import ResponseRecord, shard_output_path

logger = logging.getLogger(__name__)


def assign_shard_ids(input_files: Sequence[str]) -> list[tuple[int, str]]:
    """Pair each input file with its content shard ID.

    The shard ID is the file's position in the *sorted* input list, so the
    same file always maps to the same shard regardless of region. Callers
    pass an already-sorted list (e.g. from `input.list_input_files`); we
    sort defensively.
    """
    sorted_files = sorted(input_files)
    return [(idx, path) for idx, path in enumerate(sorted_files)]


def rotate_for_region(items: Sequence[tuple[int, str]], region: str) -> list[tuple[int, str]]:
    """Cyclically rotate the work list so each region starts at a different shard.

    Determinism: the rotation offset is ``hash(region) % len(items)``, so
    each (region, items) pair produces a stable order.
    """
    if not items:
        return []
    offset = hash(region) % len(items)
    return list(items[offset:]) + list(items[:offset])


def build_dataset(
    work_items: Sequence[tuple[int, str]],
    *,
    model_spec: ModelSpec,
    sampling: SamplingParams,
    region: str,
    results_uri: str,
) -> Dataset[tuple[int, str]]:
    """Build the Zephyr Dataset for one region's inference job.

    ``work_items`` is the per-region rotated list of ``(shard_id, file_path)``
    tuples. The returned Dataset uses `reshard(len(work_items))` to put each
    tuple in its own Zephyr shard, then `map_shard` invokes the file-level
    inference step.
    """
    if not work_items:
        raise ValueError("build_dataset requires at least one work item.")
    process = _make_process_shard(model_spec=model_spec, sampling=sampling, region=region, results_uri=results_uri)
    return Dataset.from_iterable(list(work_items)).reshard(len(work_items)).map_shard(process)


def _make_process_shard(
    *,
    model_spec: ModelSpec,
    sampling: SamplingParams,
    region: str,
    results_uri: str,
):
    """Build the per-shard processor closure."""

    def process_shard(items: Iterator[tuple[int, str]], shard_info: Any) -> Iterator[tuple[int, str]]:
        item_list = list(items)
        # ``reshard(len(work_items))`` should leave exactly one tuple per
        # Zephyr shard. If Zephyr ever groups more than one, the loop below
        # still does the right thing — we just process each in sequence.
        for content_shard_id, file_path in item_list:
            output_path = shard_output_path(results_uri, content_shard_id)
            if _output_exists(output_path):
                logger.info(
                    "Shard %d: output already exists at %s — skipping inference.",
                    content_shard_id,
                    output_path,
                )
                yield (content_shard_id, output_path)
                continue

            input_records = load_jsonl_records(file_path)
            responses = vllm_worker.infer_records(
                input_records,
                model_spec=model_spec,
                sampling=sampling,
                region=region,
                shard_idx=content_shard_id,
            )
            _write_shard_output(output_path, responses)
            logger.info(
                "Shard %d: wrote %d responses to %s (zephyr shard %d)",
                content_shard_id,
                len(responses),
                output_path,
                shard_info.shard_idx,
            )
            yield (content_shard_id, output_path)

    return process_shard


def _output_exists(output_path: str) -> bool:
    """Return True if the shard output file already exists."""
    fs, path = fsspec.core.url_to_fs(output_path)
    return fs.exists(path)


def _write_shard_output(output_path: str, responses: Sequence[ResponseRecord]) -> None:
    """Write a shard's responses to ``output_path`` as gzipped JSONL.

    Serializes to an in-memory buffer first so the GCS write is a single
    upload (atomic at the object level). Concurrent writers from different
    regions overwrite each other's results, but since responses are
    deterministic per content shard the final content is identical.
    """
    buffer = io.BytesIO()
    with gzip.open(buffer, "wt", encoding="utf-8") as gz:
        for response in responses:
            gz.write(json.dumps(response.to_dict(), ensure_ascii=False) + "\n")
    data = buffer.getvalue()
    with fsspec.open(output_path, "wb") as f:
        f.write(data)
