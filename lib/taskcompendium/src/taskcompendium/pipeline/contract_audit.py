# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for every pinned archive row before a faithful converter exists."""

import gzip
import hashlib
import json
import tarfile
import zlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from functools import partial
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath
from zephyr import counters
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.writers import write_parquet_file

from taskcompendium.datasets.source_definitions import tasktrove_files
from taskcompendium.importers.nemo_predicted_action import canonical_sha256
from taskcompendium.importers.tasktrove.convert import MAX_ARCHIVE_BYTES, MAX_ARCHIVE_MEMBERS, archive_files
from taskcompendium.pipeline.audit_schema import TASK_SCHEMA, audit_columns
from taskcompendium.pipeline.inputs import RecipeInputs, SourceFiles, SourceFormat
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportFailureKind,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    TaskAudit,
    TaskPolicy,
)
from taskcompendium.pipeline.sources import staged_file_rows, staged_files
from taskcompendium.pipeline.transforms import normalize_row

CONTRACT_AUDIT_REVISION = "1"
CONTRACT_FREQUENCIES_FILE = "contract-frequencies.json"
AUDIT_BUFFER_BYTES = 4 * 1024 * 1024


def _archive_rows(path: StoragePath) -> Iterator[dict[str, Any]]:
    # Task blobs can make a single Arrow row group much larger than worker RAM.
    # Decode sequentially; avoid the ordinary row-group-to-table reader here.
    with path.open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        for batch in parquet.iter_batches(batch_size=1, use_threads=False):
            yield from batch.to_pylist()


def _contract_record(row: dict[str, Any], _root: StoragePath) -> dict[str, Any]:
    blob = row["task_binary"]
    record: dict[str, Any] = {
        "path": row["path"],
        "archive_sha256": hashlib.sha256(blob).hexdigest() if isinstance(blob, bytes) else None,
        "archive_bytes": len(blob) if isinstance(blob, bytes) else None,
        "decode_status": "complete",
        "decode_detail": None,
        "files": [],
        "contract_sha256": None,
    }
    try:
        if not isinstance(blob, bytes):
            raise ValueError("Task binary is not archived bytes")
        files = archive_files(blob)
    except (ValueError, tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error) as error:
        # The immutable row locator remains sufficient to recover bytes that
        # exceed our inspection limits; this is not evidence of a source defect.
        record.update(decode_status="unavailable", decode_detail=f"{type(error).__name__}: {error}")
        return record
    record["files"] = [
        {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for path, data in sorted(files.items())
    ]
    contract = [
        item
        for item in record["files"]
        if item["path"] == "task.toml" or item["path"].startswith(("environment/", "tests/"))
    ]
    record["contract_sha256"] = canonical_sha256({"files": contract})
    record["solution_files"] = sum(path.startswith("solution/") for path in files)
    record["expanded_bytes"] = sum(len(data) for data in files.values())
    return record


def _unbound_contract(row: RawRow) -> ImportRejection:
    unavailable = row.data["decode_status"] != "complete"
    return ImportRejection(
        kind=ImportFailureKind.UNSUPPORTED,
        reason="archive_decode_unavailable" if unavailable else "unbound_source_contract",
        detail=(
            row.data["decode_detail"] if unavailable else "Archive inspected; no faithful TaskSpec converter is bound"
        ),
    )


def _audit_file(
    relative_file: str, *, source_path: str, output_path: str, recipe: DatasetRecipe, files: SourceFiles
) -> dict[str, Any]:
    source_file = StoragePath(source_path) / relative_file
    with source_file.open("rb") as stream:
        expected = pq.ParquetFile(stream).metadata.num_rows
    contracts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    counts: Counter[str] = Counter()

    def records() -> Iterator[dict[str, Any]]:
        for record in staged_file_rows(source_path, relative_file, files):
            data = record["data"]
            data["archive_locator"] = {"parquet": str(source_file), "row": record["index"]}
            if data["decode_status"] == "complete":
                contracts[data["contract_sha256"]] += 1
                counts["decoded_rows"] += 1
            else:
                failures[data["decode_detail"]] += 1
                counts["decode_unavailable_rows"] += 1
            counts["archive_bytes"] += data["archive_bytes"] or 0
            counters.pipeline.update_counter(f"contract_audit/{data['decode_status']}", 1)
            audit = normalize_row(record, recipe)["audit"]
            yield audit_columns(TaskAudit.model_validate(audit))

    shard = hashlib.sha256(relative_file.encode()).hexdigest()
    result = write_parquet_file(
        records(),
        str(StoragePath(output_path) / "audit" / f"part-{shard}.parquet"),
        schema=TASK_SCHEMA,
        target_buffer_bytes=AUDIT_BUFFER_BYTES,
    )
    if result["count"] != expected:
        raise ValueError(f"Archive audit lost rows in {relative_file}: {result['count']} != {expected}")
    return {
        "file": relative_file,
        "input_rows": expected,
        **counts,
        "contracts": dict(contracts),
        "decode_unavailable_reasons": dict(failures),
    }


def audit_tasktrove_contracts(
    source_path: str,
    output_path: str,
    source: HFSource,
    *,
    max_workers: int,
    worker_resources: ResourceConfig | None = None,
) -> dict[str, Any]:
    """Persist a deferred audit for every archive, retaining locators and file hashes.

    Conversion and model evaluation never run. Archives beyond the shared reader
    limits retain their provenance and count as unavailable contract inspections.
    """
    files = SourceFiles(
        tasktrove_files(source.config).patterns, SourceFormat.PARQUET, decoder=_contract_record, reader=_archive_rows
    )
    recipe = DatasetRecipe(
        name=f"raw-contract-{source.config}",
        version=CONTRACT_AUDIT_REVISION,
        source=source,
        policy=TaskPolicy(_unbound_contract, ReviewRubric("raw-contract-audit", CONTRACT_AUDIT_REVISION, ())),
        intended_use=IntendedUse.TRAIN,
        inputs=RecipeInputs(files, ()),
    )
    selected = staged_files(source_path, files)
    dataset = Dataset.from_list(list(selected)).map(
        partial(_audit_file, source_path=source_path, output_path=output_path, recipe=recipe, files=files)
    )
    with ZephyrContext(max_workers=max_workers, resources=worker_resources, name=f"contract-{source.config}") as context:
        results = context.execute(dataset).results
    contracts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    for result in results:
        contracts.update(result["contracts"])
        failures.update(result["decode_unavailable_reasons"])
    total = sum(result["input_rows"] for result in results)
    decoded = sum(result.get("decoded_rows", 0) for result in results)
    unavailable = sum(result.get("decode_unavailable_rows", 0) for result in results)
    if decoded + unavailable != total:
        raise ValueError("Archive audit decode accounting does not match input rows")
    output = StoragePath(output_path)
    with (output / CONTRACT_FREQUENCIES_FILE).open("wt", auto_mkdir=True) as stream:
        json.dump(dict(sorted(contracts.items())), stream, indent=2)
    manifest = {
        "stage": "raw_contract_audit",
        "revision": CONTRACT_AUDIT_REVISION,
        "source": asdict(source),
        "source_path": source_path,
        "input_rows": total,
        "audit_rows": total,
        "normalized_rows": 0,
        "reviewed_rows": 0,
        "dispositions": {"defer": total},
        "decoded_rows": decoded,
        "decode_unavailable_rows": unavailable,
        "decode_unavailable_reasons": dict(failures),
        "complete_row_accounting": True,
        "complete_contract_coverage": unavailable == 0,
        "contract_count": len(contracts),
        "contract_fingerprint_scope": "Exact task.toml, environment/ and tests/ file names, sizes and hashes",
        "contract_frequency_interpretation": "Byte identity only; not a grading-quality or homogeneity certification",
        "contract_frequencies": str(output / CONTRACT_FREQUENCIES_FILE),
        "archive_bytes": sum(result.get("archive_bytes", 0) for result in results),
        "limits": {"archive_bytes": MAX_ARCHIVE_BYTES, "archive_members": MAX_ARCHIVE_MEMBERS},
        "files": [
            {k: v for k, v in result.items() if k not in {"contracts", "decode_unavailable_reasons"}}
            for result in results
        ],
    }
    with (output / "manifest.json").open("wt", auto_mkdir=True) as stream:
        json.dump(manifest, stream, indent=2)
    return manifest
