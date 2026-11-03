# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Locate selected Common Corpus documents in pinned original provider Parquet files."""

import argparse
import dataclasses
import hashlib
import json
import logging
import tempfile
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from marin.datakit.download.common_corpus import OPEN_TYPES
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.scripts.metadata_cluster_selection import _digest, _read_json, _write_json
from experiments.datakit.store.provider_reconstruction import (
    ProviderTransform,
    ReconstructionRecipe,
    provider_normalizer,
    reconstruction_code_digests,
)

logger = logging.getLogger(__name__)
CANDIDATE_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("identifier", pa.string()),
        ("text", pa.string()),
        ("provider_path", pa.string()),
        ("file_row_offset", pa.int64()),
        ("row_group", pa.int32()),
        ("row_in_group", pa.int64()),
    ]
)


def common_corpus_recipe() -> ReconstructionRecipe:
    normalization, transform = reconstruction_code_digests(ProviderTransform.DIRECT)
    return ReconstructionRecipe(
        transform=ProviderTransform.DIRECT,
        text_field="text",
        id_field="identifier",
        max_whitespace_run_chars=128,
        normalization_sha256=normalization,
        transform_sha256=transform,
        drop_fields=("__index_level_0__", "curator"),
    )


def _targets(selected_rows: list[dict]) -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for row in selected_rows:
        identifier = json.loads(row["source_id"])
        if not isinstance(identifier, str):
            raise ValueError("Common Corpus selected source IDs must contain provider identifier strings")
        targets.setdefault(identifier, set()).add(row["id"])
    return targets


def _file_pin(info: dict) -> dict:
    return {"size": info["size"], "etag": info.get("ETag")}


def scan_common_corpus_file(
    provider_path: str, selected_rows: list[dict], output: str, *, batch_size: int = 4096
) -> dict:
    """Scan identifiers, then replay only matching row groups from an original file."""
    targets = _targets(selected_rows)
    normalize = provider_normalizer(common_corpus_recipe())
    fs, resolved = filesystem_for(provider_path)
    pin = _file_pin(fs.info(resolved))
    matches = []
    projected_bytes = 0
    with tempfile.TemporaryDirectory(prefix="common-corpus-replay-") as temporary:
        candidate_file = Path(temporary) / "candidates.parquet"
        with (
            fs.open(resolved, "rb", block_size=65536, cache_type="readahead") as stream,
            pq.ParquetWriter(candidate_file, CANDIDATE_SCHEMA) as writer,
        ):
            parquet = pq.ParquetFile(stream, pre_buffer=False)
            identifiers = pa.array(sorted(targets), type=parquet.schema_arrow.field("identifier").type)
            row_group_rows = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
            file_base = 0
            for group_index, group_rows in enumerate(row_group_rows):
                group = parquet.metadata.row_group(group_index)
                projected_bytes += sum(
                    group.column(i).total_compressed_size
                    for i in range(group.num_columns)
                    if group.column(i).path_in_schema == "identifier"
                )
                positions: set[int] = set()
                offset = 0
                for batch in parquet.iter_batches(
                    columns=["identifier"], row_groups=[group_index], batch_size=batch_size
                ):
                    mask = pc.is_in(batch.column(0), value_set=identifiers)
                    positions.update(offset + i for i in pc.indices_nonzero(mask).to_pylist())
                    offset += batch.num_rows
                if offset != group_rows:
                    raise ValueError(f"Incomplete identifier scan in {provider_path} row group {group_index}")
                if positions:
                    columns = ["identifier", "text", "language", "open_type"]
                    projected_bytes += sum(
                        group.column(i).total_compressed_size
                        for i in range(group.num_columns)
                        if group.column(i).path_in_schema in {"text", "language", "open_type"}
                    )
                    offset = 0
                    for batch in parquet.iter_batches(columns=columns, row_groups=[group_index], batch_size=batch_size):
                        local = [
                            position - offset
                            for position in sorted(positions)
                            if offset <= position < offset + batch.num_rows
                        ]
                        for local_row, row in zip(
                            local, batch.take(pa.array(local, type=pa.int64())).to_pylist(), strict=True
                        ):
                            if row["language"] != "English" or row["open_type"] not in OPEN_TYPES:
                                continue
                            for document in normalize([row]):
                                if document["id"] not in targets[row["identifier"]]:
                                    continue
                                locator = {
                                    "provider_path": provider_path,
                                    "file_row_offset": file_base + offset + local_row,
                                    "row_group": group_index,
                                    "row_in_group": offset + local_row,
                                }
                                candidate = {
                                    "id": document["id"],
                                    "identifier": row["identifier"],
                                    "text": document["text"],
                                    **locator,
                                }
                                writer.write_table(pa.Table.from_pylist([candidate], schema=CANDIDATE_SCHEMA))
                                matches.append({key: value for key, value in candidate.items() if key != "text"})
                        offset += batch.num_rows
                file_base += group_rows
            if file_base != parquet.metadata.num_rows:
                raise ValueError(f"Incomplete original provider scan: {provider_path}")
        fs.invalidate_cache(resolved)
        if _file_pin(fs.info(resolved)) != pin:
            raise ValueError(f"Original provider file changed during replay: {provider_path}")
        if matches:
            output_fs, destination = filesystem_for(output)
            output_fs.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
            with candidate_file.open("rb") as source, output_fs.open(destination, "wb") as target:
                while block := source.read(8 * 1024 * 1024):
                    target.write(block)
    return {
        "provider_path": provider_path,
        "object_pin": pin,
        "rows": file_base,
        "row_group_rows": row_group_rows,
        "projected_compressed_bytes": projected_bytes,
        "matches": matches,
        "candidates": output if matches else None,
    }


def _scan_config(
    provider_manifest: dict, selected_rows: list[dict], task_index: int, task_count: int, max_files: int | None
) -> dict:
    return {
        "provider_manifest_sha256": _digest(provider_manifest),
        "selected_rows_sha256": _digest(selected_rows),
        "recipe": json.loads(json.dumps(dataclasses.asdict(common_corpus_recipe()))),
        "filter": {"language": "English", "open_types": OPEN_TYPES},
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_index": task_index,
        "task_count": task_count,
        "max_files": max_files,
    }


def run_common_corpus_scan(
    provider_manifest: dict,
    selected_rows: list[dict],
    output_prefix: str,
    *,
    task_index: int,
    task_count: int,
    workers: int,
    batch_size: int = 4096,
    max_files: int | None = None,
) -> dict:
    """Checkpoint a deterministic partition of the original provider file inventory."""
    files = provider_manifest["files"]
    if len(set(files)) != len(files):
        raise ValueError("Provider inventory contains repeated file paths")
    if not 0 <= task_index < task_count or workers <= 0 or (max_files is not None and max_files <= 0):
        raise ValueError("Invalid task partition, worker count, or file limit")
    _targets(selected_rows)
    assigned = list(enumerate(files))[task_index::task_count]
    if max_files is not None:
        assigned = assigned[:max_files]
    config = _scan_config(provider_manifest, selected_rows, task_index, task_count, max_files)
    checkpoint = prefix_join(output_prefix, f"task-{task_index:05d}-of-{task_count:05d}.json")
    fs, path = filesystem_for(checkpoint)
    report: dict = _read_json(checkpoint) if fs.exists(path) else {"config": config, "completed": [], "complete": False}
    if report["config"] != config:
        raise ValueError("Original-provider scan checkpoint differs from the requested pinned inputs")
    expected = dict(assigned)
    completed = {item["file_index"] for item in report["completed"]}
    if len(completed) != len(report["completed"]) or any(
        item["file_index"] not in expected or item["provider_path"] != expected[item["file_index"]]
        for item in report["completed"]
    ):
        raise ValueError("Original-provider checkpoint has repeated or unexpected files")
    attempt = uuid.uuid4().hex

    def scan(item: tuple[int, str]) -> dict:
        index, provider_path = item
        output = prefix_join(
            output_prefix, f"task={task_index:05d}/attempt={attempt}/file={index:05d}/candidates.parquet"
        )
        return {
            "file_index": index,
            **scan_common_corpus_file(provider_path, selected_rows, output, batch_size=batch_size),
        }

    remaining = iter(item for item in assigned if item[0] not in completed)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while batch := list(islice(remaining, workers)):
            futures = [executor.submit(scan, item) for item in batch]
            try:
                for future in futures:
                    report["completed"].append(future.result())
            except Exception:
                _write_json(checkpoint, report)
                raise
            logger.info(
                "Common Corpus task %d: %d/%d original files", task_index, len(report["completed"]), len(assigned)
            )
            _write_json(checkpoint, report)
    report["complete"] = len(report["completed"]) == len(assigned)
    report["sampled"] = max_files is not None
    report["matched_selected_ids"] = sorted({match["id"] for item in report["completed"] for match in item["matches"]})
    report["validation"] = "original file partition scanned; full-source coverage requires all task reports"
    _write_json(checkpoint, report)
    return report


def aggregate_common_corpus_scan(provider_manifest: dict, selected_rows: list[dict], reports: list[dict]) -> dict:
    """Require complete original-file coverage before reporting source-level content recovery."""
    if not reports:
        raise ValueError("No original-provider scan reports supplied")
    files = provider_manifest["files"]
    task_count = reports[0]["config"]["task_count"]
    if sorted(report["config"]["task_index"] for report in reports) != list(range(task_count)):
        raise ValueError("Original-provider reports do not cover every task exactly once")
    targets = {(row["id"], json.loads(row["source_id"])) for row in selected_rows}
    completed = {}
    found: Counter[tuple[str, str]] = Counter()
    for report in reports:
        config = report["config"]
        expected_config = _scan_config(provider_manifest, selected_rows, config["task_index"], task_count, None)
        if config != expected_config or not report["complete"] or report["sampled"]:
            raise ValueError("Incomplete or sampled original-provider report cannot establish full-file coverage")
        expected = dict(list(enumerate(files))[config["task_index"] :: task_count])
        items = report["completed"]
        if len(items) != len(expected) or {item["file_index"] for item in items} != set(expected):
            raise ValueError("Original-provider report has missing, repeated, or unexpected files")
        for item in items:
            index = item["file_index"]
            if item["provider_path"] != expected[index] or sum(item["row_group_rows"]) != item["rows"]:
                raise ValueError("Original provider file identity or row counts do not match")
            positions = set()
            for match in item["matches"]:
                key = (match["id"], match["identifier"])
                position = match["file_row_offset"]
                group = match["row_group"]
                if (
                    key not in targets
                    or match["provider_path"] != item["provider_path"]
                    or not 0 <= group < len(item["row_group_rows"])
                    or not 0 <= match["row_in_group"] < item["row_group_rows"][group]
                    or position != sum(item["row_group_rows"][:group]) + match["row_in_group"]
                    or position in positions
                ):
                    raise ValueError("Original-provider matching-row locator is inconsistent or repeated")
                positions.add(position)
                found[key] += 1
            completed[index] = item
    missing = sorted(targets - found.keys())
    return {
        "provider_manifest_sha256": _digest(provider_manifest),
        "selected_rows_sha256": _digest(selected_rows),
        "complete_original_file_coverage": True,
        "original_files": len(completed),
        "original_rows": sum(item["rows"] for item in completed.values()),
        "content_complete_for_supplied_selection": not missing,
        "selected_occurrences": len(selected_rows),
        "matched_selected_occurrences": sum((row["id"], json.loads(row["source_id"])) in found for row in selected_rows),
        "missing_targets": [{"id": doc_id, "identifier": identifier} for doc_id, identifier in missing],
        "ambiguous_provider_targets": sum(count > 1 for count in found.values()),
        "validation": (
            "supplied Common Corpus selection only; matching content does not prove original historical occurrence"
        ),
    }


def reconstruct_common_corpus_selection(
    provider_manifest: dict, selected_rows: list[dict], reports: list[dict], output_prefix: str
) -> dict:
    """Emit exactly the supplied selected occurrences after full original-file scan and hash verification."""
    summary = aggregate_common_corpus_scan(provider_manifest, selected_rows, reports)
    if not summary["content_complete_for_supplied_selection"]:
        raise ValueError("Original-provider scan did not recover every supplied selected document")
    expected_matches = {
        (match["provider_path"], match["file_row_offset"]): match
        for report in reports
        for item in report["completed"]
        for match in item["matches"]
    }
    candidates = {}
    for report in reports:
        for item in report["completed"]:
            if not item["matches"]:
                continue
            fs, path = filesystem_for(item["candidates"])
            observed = set()
            with fs.open(path, "rb") as stream:
                parquet = pq.ParquetFile(stream)
                if not parquet.schema_arrow.equals(CANDIDATE_SCHEMA):
                    raise ValueError("Original-provider candidate schema differs from the expected replay schema")
                for batch in parquet.iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        locator = (row["provider_path"], row["file_row_offset"])
                        match = {key: value for key, value in row.items() if key != "text"}
                        if (
                            locator in observed
                            or match != expected_matches.get(locator)
                            or generate_id(row["text"]) != row["id"]
                        ):
                            raise ValueError(
                                "Original-provider candidate content or locator differs from its completed scan"
                            )
                        observed.add(locator)
                        key = (row["id"], row["identifier"])
                        prior = candidates.get(key)
                        if prior is not None and prior["text"] != row["text"]:
                            raise ValueError("Conflicting original text for a selected normalized ID")
                        if prior is None or locator < (prior["provider_path"], prior["file_row_offset"]):
                            candidates[key] = row
            if len(observed) != len(item["matches"]):
                raise ValueError("Original-provider candidate file is incomplete")
    rows = []
    multiplicity = Counter((match["id"], match["identifier"]) for match in expected_matches.values())
    for selected in selected_rows:
        key = (selected["id"], json.loads(selected["source_id"]))
        candidate = candidates[key]
        rows.append(
            {
                **selected,
                "text": candidate["text"],
                "provider_locator": json.dumps(
                    {key: value for key, value in candidate.items() if key not in {"id", "identifier", "text"}},
                    sort_keys=True,
                ),
                "matching_provider_rows": multiplicity[key],
            }
        )
    output = prefix_join(output_prefix, "reconstructed.parquet")
    fs, path = filesystem_for(output)
    if fs.exists(path):
        raise FileExistsError(output)
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(path, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows), stream, compression="zstd")
    summary.update(output=output, reconstructed_occurrences=len(rows))
    _write_json(prefix_join(output_prefix, "_complete.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-manifest", required=True)
    parser.add_argument("--selected-rows", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--task-count", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    info = get_job_info()
    task_index = args.task_index if args.task_index is not None else info.task_index if info is not None else None
    task_count = args.task_count if args.task_count is not None else info.num_tasks if info is not None else None
    if task_index is None or task_count is None:
        parser.error("Supply task index/count outside Iris")
    report = run_common_corpus_scan(
        _read_json(args.provider_manifest),
        _read_json(args.selected_rows),
        args.output_prefix,
        task_index=task_index,
        task_count=task_count,
        workers=args.workers,
        batch_size=args.batch_size,
        max_files=args.max_files,
    )
    logger.info(
        "Completed original-provider scan: %d files, %d selected IDs found",
        len(report["completed"]),
        len(report["matched_selected_ids"]),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
