# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Acquire bounded, pinned direct-source probes without materializing whole corpora."""

import argparse
import csv
import hashlib
import io
import json
import random
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests
from taskcompendium.pipeline.datasets import (
    aime_1983_2024,
    apps,
    asdiv,
    dapo_math,
    eurus2_code,
    gretel_text_to_sql,
    gsm8k,
    math500,
    numina_math,
    openscience,
    rlvr_math,
    verifiable_code,
)

from experiments.post_training.task_curation_partitions import assign_partitions
from experiments.post_training.task_curation_prefix_sampling import (
    READ_BLOCK_BYTES,
    BudgetedHfFileSystem,
    TransferBudget,
)


class SourceFormat(StrEnum):
    PARQUET = "parquet"
    JSONL = "jsonl"
    CSV = "csv"
    XML = "xml"


@dataclass(frozen=True)
class DirectSource:
    name: str
    dataset: str
    revision: str
    config: str
    split: str
    path: str
    format: SourceFormat


def xml_rows(data: bytes) -> list[dict[str, Any]]:
    """Retain original ASDiv fields and problem attributes from the pinned XML."""
    root = ET.fromstring(data)
    return [{**problem.attrib, **{child.tag: child.text or "" for child in problem}} for problem in root.iter("Problem")]


def sample_direct(
    source: DirectSource,
    output: Path,
    *,
    count: int,
    seed: int,
    max_transfer_bytes: int,
    select: Callable[[dict[str, Any]], bool] | None = None,
    selector_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Record an ordered prefix of selected records with bounded transfer evidence."""
    directory = output / source.name
    directory.mkdir(parents=True, exist_ok=True)
    budget = TransferBudget(max_transfer_bytes)
    rows = []
    scanned = 0
    source_rows = None
    groups = []
    try:
        if source.format == SourceFormat.XML:
            chunks = []
            with requests.get(source.path, stream=True, timeout=60) as response:
                response.raise_for_status()
                for chunk in response.iter_content(READ_BLOCK_BYTES):
                    budget.transferred += len(chunk)
                    if budget.transferred > budget.maximum:
                        raise OSError("XML source exceeded bounded transfer budget")
                    chunks.append(chunk)
            candidates = xml_rows(b"".join(chunks))
            source_rows = len(candidates)
            for index, row in enumerate(candidates):
                scanned += 1
                if select is not None and not select(row):
                    continue
                rows.append({**row, "sample_index": index, "sample_file": source.path})
                if len(rows) == count:
                    break
        else:
            filesystem = BudgetedHfFileSystem(budget=budget)
            path = f"datasets/{source.dataset}@{source.revision}/{source.path}"
            with filesystem.open(path, block_size=READ_BLOCK_BYTES, cache_type="none") as stream:
                if source.format == SourceFormat.PARQUET:
                    parquet = pq.ParquetFile(stream)
                    source_rows = parquet.metadata.num_rows
                    offset = 0
                    for group in range(parquet.num_row_groups):
                        if selector_columns:
                            if select is None:
                                raise ValueError("selector_columns require a selector")
                            constants = {}
                            metadata = parquet.metadata.row_group(group)
                            for column_index in range(metadata.num_columns):
                                column = metadata.column(column_index)
                                statistics = column.statistics
                                if (
                                    column.path_in_schema in selector_columns
                                    and statistics is not None
                                    and statistics.has_min_max
                                    and statistics.min == statistics.max
                                ):
                                    constants[column.path_in_schema] = statistics.min
                            if len(constants) == len(selector_columns) and not select(constants):
                                scanned += metadata.num_rows
                                offset += metadata.num_rows
                                continue
                            selection = parquet.read_row_group(group, columns=list(selector_columns)).to_pylist()
                            if not any(select(candidate) for candidate in selection):
                                scanned += len(selection)
                                offset += len(selection)
                                continue
                        candidates = parquet.read_row_group(group).to_pylist()
                        groups.append(group)
                        for index, row in enumerate(candidates):
                            scanned += 1
                            if select is not None and not select(row):
                                continue
                            rows.append(
                                {
                                    **row,
                                    "sample_index": offset + index,
                                    "sample_row_group": group,
                                    "sample_file": source.path,
                                }
                            )
                            if len(rows) == count:
                                break
                        offset += len(candidates)
                        if len(rows) == count:
                            break
                elif source.format == SourceFormat.JSONL:
                    index = 0
                    while len(rows) < count:
                        line = stream.readline()
                        if not line:
                            break
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        scanned += 1
                        if select is None or select(row):
                            rows.append({**row, "sample_index": index, "sample_file": source.path})
                        index += 1
                else:
                    data = stream.read(min(stream.size, max_transfer_bytes + 1))
                    if len(data) > max_transfer_bytes:
                        raise OSError("CSV source exceeded bounded transfer budget")
                    candidates = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
                    source_rows = len(candidates)
                    for index, row in enumerate(candidates):
                        scanned += 1
                        if select is not None and not select(row):
                            continue
                        rows.append({**row, "sample_index": index, "sample_file": source.path})
                        if len(rows) == count:
                            break
        if len(rows) != count:
            raise ValueError(f"Selected {len(rows)} records after scanning {scanned}; expected {count}")
    except Exception as error:
        failure = {
            "source": source.name,
            "dataset": source.dataset,
            "revision": source.revision,
            "path": source.path,
            "transferred_bytes": budget.transferred,
            "transfer_budget_bytes": budget.maximum,
            "error": repr(error),
            "scanned_rows": scanned,
        }
        (directory / "acquisition-failure.json").write_text(json.dumps(failure, indent=2))
        raise
    for row in rows:
        serialized = json.dumps(
            {key: value for key, value in row.items() if not key.startswith("sample_")},
            sort_keys=True,
            ensure_ascii=False,
        )
        row["sample_record_sha256"] = hashlib.sha256(serialized.encode()).hexdigest()
        row["sample_group"] = row["sample_record_sha256"]
    assign_partitions(rows, random.Random(f"{seed}:{source.name}"), "sample_index")
    snapshot = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "sample.jsonl").write_text(snapshot)
    manifest = {
        "name": source.name,
        "dataset": source.dataset,
        "revision": source.revision,
        "config": source.config,
        "split": source.split,
        "path": source.path,
        "format": source.format.value,
        "sample_rows": len(rows),
        "source_rows": source_rows,
        "scanned_rows": scanned,
        "seed": seed,
        "row_groups": groups,
        "selector_columns": selector_columns,
        "transfer_budget_bytes": budget.maximum,
        "transferred_bytes": budget.transferred,
        "http_ranges": budget.ranges,
        "sampling": "Ordered prefix of selected source records; not population-uniform",
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def sample_viewer(
    source: DirectSource, output: Path, *, count: int, seed: int, max_transfer_bytes: int, offset: int = 0
) -> dict[str, Any]:
    """Accept viewer rows only when their response identifies the exact source commit."""
    endpoint = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": source.dataset,
        "config": source.config,
        "split": source.split,
        "offset": offset,
        "length": count,
    }
    with requests.get(endpoint, params=params, stream=True, timeout=60) as response:
        response.raise_for_status()
        if response.headers.get("x-revision") != source.revision:
            raise ValueError("Viewer response revision differs from the requested source pin")
        chunks = []
        transferred = 0
        for chunk in response.iter_content(READ_BLOCK_BYTES):
            transferred += len(chunk)
            if transferred > max_transfer_bytes:
                raise OSError("Viewer response exceeded bounded transfer budget")
            chunks.append(chunk)
        payload = json.loads(b"".join(chunks))
        headers, url = dict(response.headers), response.url
    records = payload["rows"]
    if len(records) != count or any(record["truncated_cells"] for record in records):
        raise ValueError("Viewer returned fewer records or truncated cells")
    rows = []
    for record in records:
        row = record["row"]
        digest = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        rows.append({**row, "sample_index": record["row_idx"], "sample_record_sha256": digest, "sample_group": digest})
    assign_partitions(rows, random.Random(f"{seed}:{source.name}"), "sample_index")
    snapshot = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    directory = output / source.name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sample.jsonl").write_text(snapshot)
    manifest = {
        "name": source.name,
        "dataset": source.dataset,
        "revision": source.revision,
        "config": source.config,
        "split": source.split,
        "sample_rows": count,
        "source_rows": payload["num_rows_total"],
        "seed": seed,
        "url": url,
        "response_headers": headers,
        "no_truncated_cells": True,
        "transferred_bytes": transferred,
        "transfer_budget_bytes": max_transfer_bytes,
        "sampling": "Ordered viewer prefix; not population-uniform",
        "snapshot_sha256": hashlib.sha256(snapshot.encode()).hexdigest(),
    }
    (directory / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


SOURCES = {
    "aime_1983_2024": aime_1983_2024,
    "apps": apps,
    "asdiv": asdiv,
    "dapo_math": dapo_math,
    "eurus2_code": eurus2_code,
    "gretel_text_to_sql": gretel_text_to_sql,
    "gsm8k": gsm8k,
    "math500": math500,
    "numina_math": numina_math,
    "openscience": openscience,
    "rlvr_math": rlvr_math,
    "verifiable_code": verifiable_code,
}


def sample_source(name: str, output: Path, count: int, seed: int, max_transfer_bytes: int) -> dict[str, Any]:
    """Acquire a named pinned leaf using its declared file or validated viewer strategy."""
    module = SOURCES[name]
    source = DirectSource(
        name,
        module.DATASET,
        module.REVISION,
        module.CONFIG,
        module.SPLIT,
        module.SOURCE_FILE,
        SourceFormat(module.SOURCE_FORMAT),
    )
    if module.ACQUISITION == "viewer":
        manifest = sample_viewer(
            source, output, count=count, seed=seed, max_transfer_bytes=max_transfer_bytes, offset=module.VIEWER_OFFSET
        )
        if name == "eurus2_code":
            rows = [json.loads(line) for line in (output / name / "sample.jsonl").read_text().splitlines()]
            if any(row["ability"] != "code" for row in rows):
                raise ValueError("Pinned Eurus viewer offset returned a non-code row")
            manifest["selector"] = {
                "ability": "code",
                "first_row": module.VIEWER_OFFSET,
                "basis": "Pinned Parquet ability column, row group455",
            }
            (output / name / "sample-manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest
    if module.ACQUISITION == "file":
        return sample_direct(source, output, count=count, seed=seed, max_transfer_bytes=max_transfer_bytes)
    raise ValueError(f"Unknown acquisition strategy: {module.ACQUISITION}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", choices=sorted(SOURCES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-transfer-bytes", type=int, required=True)
    args = parser.parse_args()
    for name in args.sources:
        sample_source(name, args.output, args.count, args.seed, args.max_transfer_bytes)


if __name__ == "__main__":
    main()
