# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay bounded real normalization inputs against frozen production partitions.

Run near the object store. Inputs are explicit frozen recipe parameters and file
paths, never the current source registry. A staged download is a normalization
replay, not proof that the upstream download/transform can be reproduced.
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

import pyarrow.parquet as pq
from marin.datakit import partition_filename
from marin.datakit.normalize import _make_normalize_fn, _make_whitespace_compactor, _text_from_value
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from zephyr.readers import SUPPORTED_EXTENSIONS, iter_parquet_row_groups
from zephyr.shard_keys import deterministic_hash


@dataclass(frozen=True)
class SourceReplay:
    source: str
    input_paths: list[str]
    normalized_output: str
    num_partitions: int
    normalization_parameters: dict[str, Any]
    input_provenance: str
    recipe_commit: str


def recipe_from_inventory(entry: dict, recipe_commit: str) -> SourceReplay:
    """Resolve one real input file and production shard count near the store."""
    input_paths = []
    extensions = tuple(entry["normalization_parameters"].get("file_extensions") or SUPPORTED_EXTENSIONS)
    for root in entry["normalization_input_paths"]:
        fs, resolved = filesystem_for(root)
        if fs.isfile(resolved):
            input_paths.append(root)
            break
        for directory, _dirs, files in fs.walk(resolved):
            candidates = sorted(name for name in files if not name.startswith(".") and name.endswith(extensions))
            if candidates:
                input_paths.append(fs.unstrip_protocol(prefix_join(directory, candidates[0])))
                break
        if input_paths:
            break
    if not input_paths:
        raise FileNotFoundError(f"No normalization input data file: {entry['marin_source']}")
    output = entry["normalized_artifact"]
    fs, resolved = filesystem_for(prefix_join(output, "outputs/main/part-00000-of-*.parquet"))
    partitions = fs.glob(resolved)
    if len(partitions) != 1:
        raise ValueError(f"Expected one partition-zero file for {output}, found {len(partitions)}")
    match = re.search(r"part-00000-of-([0-9]+)\.parquet$", partitions[0])
    assert match is not None
    return SourceReplay(
        source=entry["marin_source"],
        input_paths=input_paths,
        normalized_output=output,
        num_partitions=int(match.group(1)),
        normalization_parameters=entry["normalization_parameters"],
        input_provenance="frozen normalization input artifact; may already contain upstream transforms",
        recipe_commit=recipe_commit,
    )


@dataclass
class ScanBudget:
    max_projected_bytes: int
    projected_bytes: int = 0
    exhausted_paths: list[str] = field(default_factory=list)


def _input_rows(path: str, text_field: str, id_field: str, max_rows: int):
    fs, resolved = filesystem_for(path)
    if path.endswith(".parquet"):
        with fs.open(resolved, "rb") as stream:
            parquet = pq.ParquetFile(stream)
            columns = [column for column in (text_field, id_field) if column in parquet.schema_arrow.names]
            for table in iter_parquet_row_groups(parquet, columns=columns, row_end=max_rows):
                yield from table.to_pylist()
        return
    with fs.open(resolved, "rt", compression="infer") as stream:
        rows = (json.loads(line) for line in stream if line.strip())
        yield from islice(rows, max_rows)


def sample_normalization_inputs(recipe: SourceReplay, max_rows: int) -> tuple[list[dict], int]:
    """Recompute canonical IDs and retain coordinates without retaining text."""
    attrs = recipe.normalization_parameters
    text_field = attrs["text_field"]
    id_field = attrs["id_field"]
    normalize = _make_normalize_fn(
        text_field, id_field, bare=attrs.get("bare", False), drop_fields=tuple(attrs.get("drop_fields", ()))
    )
    compact = _make_whitespace_compactor(attrs["max_whitespace_run_chars"])
    samples = []
    examined = 0
    for path in recipe.input_paths:
        records = _input_rows(path, text_field, id_field, max_rows - examined)
        try:
            for offset, record in enumerate(islice(records, max_rows - examined)):
                examined += 1
                value = record.get(text_field)
                if value is None or not _text_from_value(value).strip():
                    continue
                normalized = compact(normalize(record))
                samples.append(
                    {
                        "input_path": path,
                        "input_row": offset,
                        "id": normalized["id"],
                        "has_source_id": "source_id" in normalized,
                    }
                )
        finally:
            records.close()
        if examined >= max_rows:
            break
    return samples, examined


def find_normalized_rows(path: str, ids: set[str], budget: ScanBudget) -> dict[str, int]:
    """Project ID row groups selected by Parquet statistics under a byte budget.

    The budget measures compressed projected column chunks, excluding footer
    reads and filesystem read-ahead. No normalized text column is downloaded.
    """
    found: dict[str, int] = {}
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        column_index = parquet.schema_arrow.get_field_index("id")
        if column_index < 0:
            raise ValueError(f"Normalized partition lacks id: {path}")
        row_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            column = next(group.column(i) for i in range(group.num_columns) if group.column(i).path_in_schema == "id")
            group_start = row_start
            row_start += group.num_rows
            remaining = ids - found.keys()
            if not remaining:
                break
            statistics = column.statistics
            if statistics is not None and statistics.has_min_max:
                if not any(statistics.min <= doc_id <= statistics.max for doc_id in remaining):
                    continue
            size = column.total_compressed_size
            if budget.projected_bytes + size > budget.max_projected_bytes:
                budget.exhausted_paths.append(path)
                break
            budget.projected_bytes += size
            for batch in parquet.iter_batches(batch_size=8192, row_groups=[group_index], columns=["id"]):
                for offset, doc_id in enumerate(batch.column(0).to_pylist()):
                    if doc_id in remaining and doc_id not in found:
                        found[doc_id] = group_start + offset
                group_start += batch.num_rows
    return found


def match_normalized_samples(recipe: SourceReplay, samples: list[dict], budget: ScanBudget) -> set[str]:
    """Group sample IDs by production partition and annotate matching physical rows."""
    by_partition: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        partition = deterministic_hash(sample["id"]) % recipe.num_partitions
        sample["normalized_partition"] = partition
        by_partition[partition].add(sample["id"])
    found: set[str] = set()
    for partition, ids in sorted(by_partition.items()):
        filename = partition_filename(partition, recipe.num_partitions)
        path = prefix_join(recipe.normalized_output, prefix_join("outputs/main", filename))
        matched_rows = find_normalized_rows(path, ids, budget)
        found.update(matched_rows)
        for sample in samples:
            if sample["normalized_partition"] == partition and sample["id"] in matched_rows:
                sample["normalized_path"] = path
                sample["normalized_row"] = matched_rows[sample["id"]]
    return found


def run_source_replay(recipe: SourceReplay, *, max_rows: int, max_projected_bytes: int) -> dict:
    """Check real source-derived hashes in their original normalized partitions."""
    if max_rows < 1 or max_projected_bytes < 1 or recipe.num_partitions < 1:
        raise ValueError("Replay limits and partition count must be positive")
    samples, examined = sample_normalization_inputs(recipe, max_rows)
    budget = ScanBudget(max_projected_bytes)
    found = match_normalized_samples(recipe, samples, budget)
    missing = {sample["id"] for sample in samples} - found
    return {
        "source": recipe.source,
        "status": "passed" if samples and not missing else "incomplete",
        "trial_kind": "normalization_input_replay",
        "input_provenance": recipe.input_provenance,
        "recipe_commit": recipe.recipe_commit,
        "normalizer_implementation": "current checkout shared normalizer with frozen parameters",
        "upstream_transform_replay": "not_run",
        "input_rows_examined": examined,
        "nonempty_rows": len(samples),
        "distinct_ids": len({sample["id"] for sample in samples}),
        "matched_ids": len(found),
        "missing_ids": sorted(missing),
        "projected_normalized_bytes": budget.projected_bytes,
        "budget_exhausted_paths": budget.exhausted_paths,
        "samples": samples,
        "limitations": [
            "Input files may already contain source-specific transforms; upstream transform replay is separate.",
            "recipe_commit records expected provenance; this probe does not load code from that commit.",
            "Membership validates normalized text hashes, not the retained provider ID for duplicate text.",
            "Production selection, scores, tokenization and final store order are separate checks.",
            "Input sampling reads the first bounded rows; unusually large records or Parquet row groups may be large.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recipe-commit", help="Required when reading a frozen artifact inventory")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--max-projected-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    manifest_fs, manifest_path = filesystem_for(args.manifest)
    with manifest_fs.open(manifest_path, "r") as stream:
        manifest = json.load(stream)
    reports = []
    for entry in manifest["sources"]:
        source = entry.get("marin_source", entry.get("source"))
        try:
            if "marin_source" in entry:
                if not args.recipe_commit:
                    raise ValueError("Pin --recipe-commit for an artifact inventory")
                recipe = recipe_from_inventory(entry, args.recipe_commit)
            else:
                recipe = SourceReplay(**entry)
            report = run_source_replay(recipe, max_rows=args.max_rows, max_projected_bytes=args.max_projected_bytes)
        except Exception as error:
            # Preserve per-source failure evidence while completing coverage of the frozen inventory.
            report = {"source": source, "status": "failed", "error": str(error)}
        reports.append(report)
        print(json.dumps({key: report[key] for key in ("source", "status")}), flush=True)
    complete = bool(reports) and all(report["status"] == "passed" for report in reports)
    output_fs, output_path = filesystem_for(args.output)
    with output_fs.open(output_path, "w") as stream:
        json.dump({"status": "passed" if complete else "incomplete", "sources": reports}, stream, indent=2)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
