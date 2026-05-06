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

import requests
from fray import LocalClient
from marin.transform.huggingface.dataset_to_eval import get_nested_item
from marin.utils import fsspec_mkdirs
from requests.adapters import HTTPAdapter
from rigging.filesystem import open_url
from urllib3.util import Retry
from zephyr import Dataset, ZephyrContext
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)

HF_DATASETS_SERVER_URL = "https://datasets-server.huggingface.co"
DEFAULT_PAGE_LENGTH = 100
DEFAULT_REQUEST_TIMEOUT = 120
DEFAULT_METADATA_FILENAME = "metadata.json"
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
    config_name: str
    split: str
    output_filename: str
    render_mode: HfRawTextRenderMode
    field: str = ""
    fields: tuple[str, ...] = ()
    max_rows: int = 2_000
    page_length: int = DEFAULT_PAGE_LENGTH
    join_separator: str = "\n"
    source_url: str = ""
    license_note: str = ""
    access_note: str = "Public Hugging Face dataset; sampled through Dataset Viewer rows API."


def _requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _rows_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/rows"


def _fetch_rows_page(
    session: requests.Session,
    surface: HfRawTextSurfaceConfig,
    *,
    datasets_server_url: str,
    offset: int,
    length: int,
    request_timeout: int,
) -> list[dict[str, Any]]:
    response = session.get(
        _rows_url(datasets_server_url),
        params={
            "dataset": surface.dataset_id,
            "config": surface.config_name,
            "split": surface.split,
            "offset": offset,
            "length": length,
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Dataset Viewer response for {surface.name!r} did not contain a rows list.")
    return rows


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


def _write_surface(
    session: requests.Session,
    output_path: str,
    surface: HfRawTextSurfaceConfig,
    *,
    datasets_server_url: str,
    skip_existing: bool,
    request_timeout: int,
) -> dict[str, Any]:
    output_file = _output_file_path(output_path, surface)
    fsspec_mkdirs(posixpath.dirname(output_file), exist_ok=True)

    if skip_existing:
        try:
            record_count = _existing_record_count(output_file)
            logger.info("Skipping existing raw-text shard %s with %s records", output_file, record_count)
            return {"name": surface.name, "records": record_count, "output_file": output_file, "skipped": True}
        except FileNotFoundError:
            pass

    record_count = 0
    offset = 0
    page_length = min(surface.page_length, DEFAULT_PAGE_LENGTH)
    with atomic_rename(output_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression="gzip") as outfile:
            while record_count < surface.max_rows:
                remaining = surface.max_rows - record_count
                rows = _fetch_rows_page(
                    session,
                    surface,
                    datasets_server_url=datasets_server_url,
                    offset=offset,
                    length=min(page_length, remaining),
                    request_timeout=request_timeout,
                )
                if not rows:
                    break
                for wrapper in rows:
                    row = wrapper.get("row")
                    row_index = wrapper.get("row_idx", offset)
                    if not isinstance(row, dict):
                        raise ValueError(f"Dataset Viewer row for {surface.name!r} did not contain a row object.")
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
                        },
                    }
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    record_count += 1
                    if record_count >= surface.max_rows:
                        break
                offset += len(rows)
                if len(rows) < min(page_length, remaining):
                    break

    logger.info("Wrote %s records to %s", record_count, output_file)
    return {"name": surface.name, "records": record_count, "output_file": output_file, "skipped": False}


def _metadata_record(surface: HfRawTextSurfaceConfig, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": surface.name,
        "dataset_id": surface.dataset_id,
        "config": surface.config_name,
        "split": surface.split,
        "source_url": surface.source_url or f"https://huggingface.co/datasets/{surface.dataset_id}",
        "license": surface.license_note,
        "access": surface.access_note,
        "render_mode": surface.render_mode.value,
        "field": surface.field,
        "fields": list(surface.fields),
        "max_rows": surface.max_rows,
        "sampling_plan": f"First {surface.max_rows} non-empty rendered rows from {surface.split}.",
        "output_file": result["output_file"],
        "records": result["records"],
        "skipped": result["skipped"],
    }


def materialize_hf_raw_text(
    *,
    output_path: str,
    surfaces: tuple[HfRawTextSurfaceConfig, ...],
    datasets_server_url: str = HF_DATASETS_SERVER_URL,
    metadata_filename: str = DEFAULT_METADATA_FILENAME,
    skip_existing: bool = True,
    request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Materialize configured Hugging Face raw-text surfaces as JSONL.GZ shards."""

    fsspec_mkdirs(str(output_path), exist_ok=True)

    def _run_surface(surface: HfRawTextSurfaceConfig) -> dict[str, Any]:
        return _write_surface(
            _requests_session(),
            output_path,
            surface,
            datasets_server_url=datasets_server_url,
            skip_existing=skip_existing,
            request_timeout=request_timeout,
        )

    pipeline = Dataset.from_list(list(surfaces)).map(_run_surface)
    max_workers = max(1, min(len(surfaces), LOCAL_ZEPHYR_MAX_THREADS))
    ctx = ZephyrContext(client=LocalClient(max_threads=max_workers), name="hf-raw-text", max_workers=max_workers)
    results_by_name = {result["name"]: result for result in ctx.execute(pipeline).results}

    results = []
    metadata: list[dict[str, Any]] = []
    for surface in surfaces:
        result = results_by_name[surface.name]
        results.append(result)
        metadata.append(_metadata_record(surface, result))

    metadata_path = posixpath.join(str(output_path), metadata_filename)
    with open_url(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)
    logger.info("Wrote metadata to %s", metadata_path)

    return {"surfaces": results, "metadata": metadata_path}
