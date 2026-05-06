# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize Hugging Face dataset rows as raw-text eval shards."""

from __future__ import annotations

import json
import logging
import posixpath
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from datasets import load_dataset
from fray import LocalClient
from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.transform.huggingface.dataset_to_eval import get_nested_item
from marin.utils import fsspec_mkdirs
from rigging.filesystem import open_url, url_to_fs
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)

LOCAL_ZEPHYR_MAX_THREADS = 8


class HfRawTextRenderMode(StrEnum):
    """How to render one Hugging Face row into a raw text document."""

    STRING_FIELD = "string_field"
    JOIN_LIST_FIELD = "join_list_field"
    JSON_FIELDS = "json_fields"


@dataclass(frozen=True)
class HfRawTextSurfaceConfig:
    """One raw-text surface to materialize from a Hugging Face dataset split."""

    name: str
    dataset_id: str
    revision: str
    config_name: str
    split: str
    input_glob: str
    output_filename: str
    render_mode: HfRawTextRenderMode
    field: str = ""
    fields: tuple[str, ...] = ()
    max_rows: int = 2_000
    join_separator: str = "\n"
    source_url: str = ""
    license_note: str = ""
    access_note: str = "Public Hugging Face dataset; downloaded from a pinned revision before staging."


def render_hf_raw_text(row: dict[str, Any], surface: HfRawTextSurfaceConfig) -> str:
    """Render a Hugging Face row as one raw-text document."""

    if surface.render_mode == HfRawTextRenderMode.STRING_FIELD:
        value = get_nested_item(row, surface.field)
        if not isinstance(value, str):
            raise ValueError(f"Field {surface.field!r} is not a string.")
        return value

    if surface.render_mode == HfRawTextRenderMode.JOIN_LIST_FIELD:
        value = get_nested_item(row, surface.field)
        if not isinstance(value, list):
            raise ValueError(f"Field {surface.field!r} is not a list.")
        return surface.join_separator.join(str(item) for item in value)

    if surface.render_mode == HfRawTextRenderMode.JSON_FIELDS:
        rendered = {field: get_nested_item(row, field) for field in surface.fields}
        return json.dumps(rendered, ensure_ascii=False, sort_keys=True)

    raise ValueError(f"Unsupported render mode {surface.render_mode!r}.")


def _output_file_path(output_path: str, surface: HfRawTextSurfaceConfig) -> str:
    return posixpath.join(str(output_path), surface.output_filename)


def _existing_record_count(output_file: str) -> int:
    with open_url(output_file, "rt", encoding="utf-8", compression="gzip") as infile:
        return sum(1 for line in infile if line.strip())


def _fsspec_url(fs: Any, path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    if protocol in (None, "file"):
        return path
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


def _surface_data_files(input_path: str, surface: HfRawTextSurfaceConfig) -> list[str]:
    fs, root = url_to_fs(input_path)
    pattern = posixpath.join(root, surface.input_glob)
    matches = sorted(path for path in fs.glob(pattern) if path.endswith(".parquet"))
    if not matches:
        raise FileNotFoundError(
            f"No parquet files matched {surface.input_glob!r} for surface {surface.name!r} under {input_path}"
        )
    return [_fsspec_url(fs, path) for path in matches]


def _load_surface_rows(input_path: str, surface: HfRawTextSurfaceConfig) -> Any:
    data_files = _surface_data_files(input_path, surface)
    return load_dataset("parquet", data_files={"data": data_files}, split="data", streaming=True)


def _surface_metadata_filename(surface: HfRawTextSurfaceConfig) -> str:
    filename = posixpath.basename(surface.output_filename)
    for suffix in (".jsonl.gz", ".jsonl"):
        if filename.endswith(suffix):
            filename = filename[: -len(suffix)]
            break
    return f"{filename}.metadata.json"


def _validate_manifest(
    surface: HfRawTextSurfaceConfig,
    source_manifests: dict[str, IngestionSourceManifest],
    content_fingerprints: dict[str, str],
) -> IngestionSourceManifest:
    manifest = source_manifests.get(surface.name)
    if manifest is None:
        raise ValueError(f"Missing IngestionSourceManifest for raw-text surface {surface.name!r}")
    expected = manifest.fingerprint()
    actual = content_fingerprints.get(surface.name)
    if actual != expected:
        raise ValueError(
            f"content_fingerprint mismatch for {surface.name!r}: config has {actual}, manifest has {expected}"
        )
    return manifest


def _write_ingestion_metadata(
    *,
    surface: HfRawTextSurfaceConfig,
    manifest: IngestionSourceManifest,
    input_path: str,
    output_file: str,
    record_count: int,
    metadata: dict[str, Any],
) -> str:
    fs, _ = url_to_fs(output_file)
    bytes_written = int(fs.info(output_file)["size"])
    return write_ingestion_metadata_json(
        manifest=manifest,
        materialized_output=MaterializedOutputMetadata(
            input_path=input_path,
            output_path=posixpath.dirname(output_file),
            output_file=output_file,
            record_count=record_count,
            bytes_written=bytes_written,
            metadata=metadata,
        ),
        metadata_filename=_surface_metadata_filename(surface),
    )


def _write_surface(
    input_path: str,
    output_path: str,
    surface: HfRawTextSurfaceConfig,
    *,
    source_manifests: dict[str, IngestionSourceManifest],
    content_fingerprints: dict[str, str],
    skip_existing: bool,
) -> dict[str, Any]:
    manifest = _validate_manifest(surface, source_manifests, content_fingerprints)
    output_file = _output_file_path(output_path, surface)
    fsspec_mkdirs(posixpath.dirname(output_file), exist_ok=True)

    if skip_existing:
        try:
            record_count = _existing_record_count(output_file)
            logger.info("Skipping existing raw-text shard %s with %s records", output_file, record_count)
            metadata_file = _write_ingestion_metadata(
                surface=surface,
                manifest=manifest,
                input_path=input_path,
                output_file=output_file,
                record_count=record_count,
                metadata={"skipped": True, "underfilled": record_count < surface.max_rows},
            )
            return {
                "name": surface.name,
                "records": record_count,
                "output_file": output_file,
                "metadata_file": metadata_file,
                "skipped": True,
                "underfilled": record_count < surface.max_rows,
            }
        except FileNotFoundError:
            pass

    record_count = 0
    with atomic_rename(output_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as outfile:
            for row_index, row in enumerate(_load_surface_rows(input_path, surface)):
                if not isinstance(row, dict):
                    raise ValueError(f"Parquet row for {surface.name!r} did not contain a row object.")
                text = render_hf_raw_text(row, surface)
                if not text:
                    continue
                record = {
                    "id": f"{surface.name}:{row_index}",
                    "text": text,
                    "source": surface.dataset_id,
                    "metadata": {
                        "config": surface.config_name,
                        "split": surface.split,
                        "row_idx": row_index,
                        "surface": surface.name,
                        "hf_revision": surface.revision,
                        "input_glob": surface.input_glob,
                    },
                }
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")
                record_count += 1
                if record_count >= surface.max_rows:
                    break

    underfilled = record_count < surface.max_rows
    if underfilled:
        logger.warning(
            "Surface %s wrote %s records, below requested cap %s", surface.name, record_count, surface.max_rows
        )
    metadata_file = _write_ingestion_metadata(
        surface=surface,
        manifest=manifest,
        input_path=input_path,
        output_file=output_file,
        record_count=record_count,
        metadata={"skipped": False, "underfilled": underfilled},
    )
    logger.info("Wrote %s records to %s", record_count, output_file)
    return {
        "name": surface.name,
        "records": record_count,
        "output_file": output_file,
        "metadata_file": metadata_file,
        "skipped": False,
        "underfilled": underfilled,
    }


def materialize_hf_raw_text(
    *,
    input_paths: dict[str, str],
    output_path: str,
    surfaces: tuple[HfRawTextSurfaceConfig, ...],
    source_manifests: dict[str, IngestionSourceManifest],
    content_fingerprints: dict[str, str],
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Materialize configured Hugging Face raw-text surfaces as JSONL.GZ shards."""

    fsspec_mkdirs(str(output_path), exist_ok=True)

    def _run_surface(surface: HfRawTextSurfaceConfig) -> dict[str, Any]:
        input_path = input_paths.get(surface.dataset_id)
        if input_path is None:
            raise ValueError(f"Missing input path for dataset {surface.dataset_id!r}")
        return _write_surface(
            input_path,
            output_path,
            surface,
            source_manifests=source_manifests,
            content_fingerprints=content_fingerprints,
            skip_existing=skip_existing,
        )

    pipeline = Dataset.from_list(list(surfaces)).map(_run_surface)
    max_workers = max(1, min(len(surfaces), LOCAL_ZEPHYR_MAX_THREADS))
    ctx = ZephyrContext(client=LocalClient(max_threads=max_workers), name="hf-raw-text", max_workers=max_workers)
    results_by_name = {result["name"]: result for result in ctx.execute(pipeline).results}

    results = []
    for surface in surfaces:
        result = results_by_name[surface.name]
        results.append(result)

    return {"surfaces": results}
