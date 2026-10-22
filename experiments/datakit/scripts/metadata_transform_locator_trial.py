# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay pinned raw source transforms while retaining physical row locators.

Run near the raw artifacts. The manifest must point at provider-format files,
not the processed normalization inputs. This script never submits a job.
Biodiversity requires an exhaustive raw split manifest and a complete bounded
scan: a truncated page prefix is never treated as a complete book.
"""

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pyarrow.parquet as pq
from marin.datakit import partition_filename
from marin.datakit.normalize import _make_normalize_fn, _make_whitespace_compactor, _text_from_value
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from zephyr.shard_keys import deterministic_hash

from experiments.datakit.scripts.metadata_source_replay import ScanBudget, SourceReplay, find_normalized_rows

# Project only fields consumed by the production transforms.
RAW_COLUMNS = {
    "finetranslations": ["id", "og_full_text", "translated_text"],
    "institutional_books": ["text_by_page_gen", "text_by_page_src"],
    "swe-zero-12m": ["messages"],
    "swe-rebench-contree": ["text"],
    "stack-v3": ["repo_path", "repo_id", "commit_id", "github_metadata", "num_files", "files"],
    "cp/biodiversity": ["item_id", "page_num", "text"],
}


@dataclass(frozen=True)
class TransformReplay:
    normalization: SourceReplay
    raw_paths: list[str]
    transform_module_path: str
    transform_module_sha256: str
    # A caller must obtain every raw split file from the pinned provider snapshot.
    raw_split_manifest_complete: bool = False
    transform_provenance: str = "SHA-verified transform candidate; exact historical source equivalence unproven"


@dataclass
class RawBudget:
    max_rows: int
    max_projected_bytes: int
    rows: int = 0
    projected_bytes: int = 0
    exhausted: bool = False


def raw_rows(path: str, columns: list[str], budget: RawBudget):
    """Yield physical zero-based row coordinates with projected read limits."""
    if not path.endswith(".parquet"):
        fs, resolved = filesystem_for(path)
        with fs.open(resolved, "rt", compression="infer") as stream:
            row_offset = 0
            while True:
                # This branch budgets decoded input bytes, since compressed column projection is unavailable.
                remaining = budget.max_projected_bytes - budget.projected_bytes
                line = stream.readline(remaining + 1)
                if not line:
                    return
                size = len(line.encode("utf-8"))
                if size > remaining or budget.rows >= budget.max_rows:
                    budget.exhausted = True
                    return
                budget.projected_bytes += size
                if not line.strip():
                    continue
                row = json.loads(line)
                budget.rows += 1
                yield {"input_path": path, "input_row": row_offset}, {
                    column: row[column] for column in columns if column in row
                }
                row_offset += 1
        return
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        available = set(parquet.schema_arrow.names)
        selected = [column for column in columns if column in available]
        row_offset = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            size = sum(
                group.column(i).total_compressed_size
                for i in range(group.num_columns)
                if group.column(i).path_in_schema.split(".")[0] in selected
            )
            if budget.rows >= budget.max_rows or budget.projected_bytes + size > budget.max_projected_bytes:
                budget.exhausted = True
                return
            budget.projected_bytes += size
            for batch in parquet.iter_batches(row_groups=[group_index], columns=selected, batch_size=8):
                for row in batch.to_pylist():
                    if budget.rows >= budget.max_rows:
                        budget.exhausted = True
                        return
                    budget.rows += 1
                    yield {"input_path": path, "input_row": row_offset}, row
                    row_offset += 1


def _transform_module(recipe: TransformReplay):
    path = recipe.transform_module_path
    if "://" in path:
        raise ValueError("Stage the pinned transform module locally before replay")
    blob = StoragePath(path).read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != recipe.transform_module_sha256:
        raise ValueError(f"Transform source digest mismatch: {path}")
    # A pinned local source file keeps production functions and their local helpers together.
    spec = importlib.util.spec_from_file_location(f"_metadata_transform_{digest}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_transformed_rows(recipe: TransformReplay, budget: RawBudget, *, max_documents: int) -> list[dict]:
    """Run production transforms and normalization, dropping text from evidence."""
    source = recipe.normalization.source
    columns = RAW_COLUMNS[source]
    module = _transform_module(recipe)
    attrs = recipe.normalization.normalization_parameters
    normalize = _make_normalize_fn(
        attrs["text_field"],
        attrs["id_field"],
        bare=attrs.get("bare", False),
        drop_fields=tuple(attrs.get("drop_fields", ())),
    )
    compact = _make_whitespace_compactor(attrs["max_whitespace_run_chars"])
    samples = []

    def record_sample(document: dict, locators: list[dict]) -> None:
        value = document.get(attrs["text_field"])
        if value is None or not _text_from_value(value).strip():
            return
        normalized = compact(normalize(document))
        samples.append({"id": normalized["id"], "input_locators": locators, "has_source_id": "source_id" in normalized})

    if source == "cp/biodiversity":
        if not recipe.raw_split_manifest_complete:
            raise ValueError("Biodiversity requires every raw file in the pinned split, not a sampled file list")
        groups: dict[str, list[tuple[dict, dict]]] = {}
        # Bound selected groups, but scan all raw rows to collect every page of each selected item.
        for path in recipe.raw_paths:
            for locator, row in raw_rows(path, columns, budget):
                key = row["item_id"]
                if key not in groups and len(groups) < max_documents:
                    groups[key] = []
                if key in groups:
                    groups[key].append((locator, row))
        if budget.exhausted:
            return []  # Never emit a partially reconstructed item.
        for key, pages in groups.items():
            pages.sort(key=lambda pair: pair[1]["page_num"])
            # Equal page_num keys across input partitions have no explicit tie-break in production.
            numbers = [row["page_num"] for _, row in pages]
            if len(numbers) != len(set(numbers)):
                raise ValueError(f"Biodiversity item {key!r} has ambiguous duplicate page_num ordering")
            for document in module.stitch_pages(key, iter(row for _, row in pages)):
                record_sample(document, [locator for locator, _ in pages])
        return samples

    transform = module.fold_parallel_text if source == "finetranslations" else module.row_to_doc
    for path in recipe.raw_paths:
        for locator, row in raw_rows(path, columns, budget):
            documents = [transform(row)] if source == "stack-v3" else transform(row)
            for document in documents:
                record_sample(document, [locator])
                if len(samples) >= max_documents:
                    return samples
        if budget.exhausted:
            break
    return samples


def run_transform_replay(
    recipe: TransformReplay,
    *,
    max_raw_rows: int,
    max_documents: int,
    max_projected_bytes: int,
) -> dict[str, Any]:
    """Match raw-derived IDs against frozen production normalized partitions."""
    if min(max_raw_rows, max_documents, max_projected_bytes) < 1:
        raise ValueError("Trial budgets must be positive")
    budget = RawBudget(max_raw_rows, max_projected_bytes)
    samples = sample_transformed_rows(recipe, budget, max_documents=max_documents)
    normalized = recipe.normalization
    by_partition: dict[int, set[str]] = defaultdict(set)
    for sample in samples:
        partition = deterministic_hash(sample["id"]) % normalized.num_partitions
        sample["normalized_partition"] = partition
        by_partition[partition].add(sample["id"])
    match_budget = ScanBudget(max_projected_bytes)
    found = set()
    for partition, ids in sorted(by_partition.items()):
        filename = partition_filename(partition, normalized.num_partitions)
        path = f"{normalized.normalized_output.rstrip('/')}/outputs/main/{filename}"
        matches = find_normalized_rows(path, ids, match_budget)
        found.update(matches)
        for sample in samples:
            if sample["normalized_partition"] == partition and sample["id"] in matches:
                sample.update(normalized_path=path, normalized_row=matches[sample["id"]])
    missing = {sample["id"] for sample in samples} - found
    return {
        "source": normalized.source,
        "status": "passed" if samples and not missing else "incomplete",
        "trial_kind": "raw_provider_transform_locator_replay",
        "recipe_commit": normalized.recipe_commit,
        "transform_module_path": recipe.transform_module_path,
        "transform_module_sha256": recipe.transform_module_sha256,
        "transform_provenance": recipe.transform_provenance,
        "transform_compatibility": "sampled_normalized_hash_match" if samples and not missing else "unproven",
        "raw_rows_examined": budget.rows,
        "raw_projected_bytes": budget.projected_bytes,
        "raw_budget_exhausted": budget.exhausted,
        "normalized_projected_bytes": match_budget.projected_bytes,
        "normalized_budget_exhausted_paths": match_budget.exhausted_paths,
        "matched_ids": len(found),
        "missing_ids": sorted(missing),
        "samples": samples,
        "limitations": [
            "Input paths and offsets name provider-format physical files; split-global row mapping is separate.",
            "The caller must pin the original raw snapshot; a processed input file is not provider replay.",
            "Biodiversity completeness requires an exhaustive caller-provided split manifest and full bounded scan.",
            "Matching hashes prove sampled normalized membership, not all-source coverage or final selection.",
            "Parquet budgets compressed projected chunks; JSON budgets decoded lines. Read-ahead bytes are excluded.",
        ],
    }


def _normalized_source_id(path: str, ordinal: int, budget: ScanBudget) -> tuple[bool, Any]:
    fs, resolved = filesystem_for(path)
    with fs.open(resolved, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        if "source_id" not in parquet.schema_arrow.names:
            return False, None
        start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(group_index)
            if not start <= ordinal < start + group.num_rows:
                start += group.num_rows
                continue
            size = sum(
                group.column(i).total_compressed_size
                for i in range(group.num_columns)
                if group.column(i).path_in_schema == "source_id"
            )
            if budget.projected_bytes + size > budget.max_projected_bytes:
                budget.exhausted_paths.append(path)
                return False, None
            budget.projected_bytes += size
            for batch in parquet.iter_batches(row_groups=[group_index], columns=["source_id"], batch_size=8192):
                if ordinal < start + batch.num_rows:
                    return True, batch.column(0)[ordinal - start].as_py()
                start += batch.num_rows
    return False, None


def run_biodiversity_candidate_replay(
    recipe: TransformReplay,
    *,
    max_raw_rows: int,
    max_documents: int,
    max_projected_bytes: int,
) -> dict[str, Any]:
    """Verify individual book reconstructions from a bounded raw page prefix.

    A matching normalized hash AND retained item identity prove reconstructed
    normalized text. They do not establish that every original page was read:
    empty pages and whitespace compaction can erase differences in input lineage.
    """
    if recipe.normalization.source != "cp/biodiversity":
        raise ValueError("Candidate item proof is only supported for Biodiversity")
    if min(max_raw_rows, max_documents, max_projected_bytes) < 1:
        raise ValueError("Trial budgets must be positive")
    module = _transform_module(recipe)
    budget = RawBudget(max_raw_rows, max_projected_bytes)
    groups: dict[str, list[tuple[dict, dict]]] = {}
    for path in recipe.raw_paths:
        for locator, row in raw_rows(path, RAW_COLUMNS["cp/biodiversity"], budget):
            key = row["item_id"]
            if key not in groups and len(groups) < max_documents:
                groups[key] = []
            if key in groups:
                groups[key].append((locator, row))
        if budget.exhausted:
            break
    attrs = recipe.normalization.normalization_parameters
    normalize = _make_normalize_fn(
        attrs["text_field"],
        attrs["id_field"],
        bare=attrs.get("bare", False),
        drop_fields=tuple(attrs.get("drop_fields", ())),
    )
    compact = _make_whitespace_compactor(attrs["max_whitespace_run_chars"])
    candidates = []
    rejected = []
    for item_id, pages in groups.items():
        pages.sort(key=lambda pair: pair[1]["page_num"])
        numbers = [row["page_num"] for _, row in pages]
        if len(numbers) != len(set(numbers)):
            rejected.append({"item_id": item_id, "reason": "ambiguous_page_order"})
            continue
        for document in module.stitch_pages(item_id, iter(row for _, row in pages)):
            normalized = compact(normalize(document))
            candidates.append(
                {"item_id": item_id, "id": normalized["id"], "input_locators": [locator for locator, _ in pages]}
            )
    matched = []
    match_budget = ScanBudget(max_projected_bytes)
    for candidate in candidates:
        partition = deterministic_hash(candidate["id"]) % recipe.normalization.num_partitions
        path = (
            f"{recipe.normalization.normalized_output.rstrip('/')}/outputs/main/"
            f"{partition_filename(partition, recipe.normalization.num_partitions)}"
        )
        matches = find_normalized_rows(path, {candidate["id"]}, match_budget)
        ordinal = matches.get(candidate["id"])
        if ordinal is None:
            rejected.append({"item_id": candidate["item_id"], "reason": "normalized_hash_not_found_or_budget_exhausted"})
            continue
        found, source_id = _normalized_source_id(path, ordinal, match_budget)
        if not found or source_id != candidate["item_id"]:
            rejected.append({"item_id": candidate["item_id"], "reason": "normalized_item_identity_not_verified"})
            continue
        matched.append(
            {**candidate, "normalized_path": path, "normalized_row": ordinal, "proof": "hash_and_item_identity_verified"}
        )
    return {
        "source": recipe.normalization.source,
        "status": "passed" if matched else "incomplete",
        "trial_kind": "bounded_book_reconstruction_hash_and_item_identity",
        "transform_module_sha256": recipe.transform_module_sha256,
        "transform_provenance": recipe.transform_provenance,
        "raw_rows_examined": budget.rows,
        "raw_projected_bytes": budget.projected_bytes,
        "raw_budget_exhausted": budget.exhausted,
        "normalized_projected_bytes": match_budget.projected_bytes,
        "normalized_budget_exhausted_paths": match_budget.exhausted_paths,
        "candidate_books": len(candidates),
        "verified_books": len(matched),
        "samples": matched,
        "rejected_candidates": rejected,
        "limitations": [
            "Proof covers normalized text reconstruction for individually verified items only.",
            "Physical locators cover contributing candidate pages, not certified complete original lineage.",
            "Discarded empty pages and whitespace compaction can make distinct page inputs normalize identically.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--biodiversity-candidate-proof", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-raw-rows", type=int, default=32)
    parser.add_argument("--max-documents", type=int, default=4)
    parser.add_argument("--max-projected-bytes", type=int, default=64 * 1024 * 1024)
    args = parser.parse_args()
    manifest_fs, manifest_path = filesystem_for(args.manifest)
    with manifest_fs.open(manifest_path, "r") as stream:
        entries = json.load(stream)["sources"]
    reports = []
    for entry in entries:
        source = entry["normalization"]["source"]
        try:
            entry["normalization"] = SourceReplay(**entry["normalization"])
            runner = (
                run_biodiversity_candidate_replay
                if source == "cp/biodiversity" and args.biodiversity_candidate_proof
                else run_transform_replay
            )
            report = runner(
                TransformReplay(**entry),
                max_raw_rows=args.max_raw_rows,
                max_documents=args.max_documents,
                max_projected_bytes=args.max_projected_bytes,
            )
        except Exception as error:
            report = {"source": source, "status": "failed", "error": str(error)}
        reports.append(report)
        print(json.dumps({"source": source, "status": report["status"]}), flush=True)
    passed = bool(reports) and all(report["status"] == "passed" for report in reports)
    output_fs, output_path = filesystem_for(args.output)
    with output_fs.open(output_path, "w") as stream:
        json.dump({"status": "passed" if passed else "incomplete", "sources": reports}, stream, indent=2)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
