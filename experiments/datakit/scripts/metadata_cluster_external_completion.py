# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind verified Common Corpus reconstruction to a complete-cluster source plan."""

import dataclasses
import json
import uuid
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.scripts.metadata_cluster_external_reconstruction import (
    aggregate_common_corpus_scan,
    common_corpus_recipe,
    reconstruct_common_corpus_selection,
)
from experiments.datakit.scripts.metadata_cluster_reconstruction import _verify_plan, build_selected_index
from experiments.datakit.scripts.metadata_cluster_selection import _digest, _read_json, _write_json
from experiments.datakit.store.reconstruct_selected_source import RECONSTRUCTED_TEXT_SCHEMA

COMMON_CORPUS_SOURCE = "common_corpus/english"
SELECTED_IDENTITY_COLUMNS = ["id", "source_id", "normalized_shard", "normalized_row"]


def complete_common_corpus_source(
    plan: dict, provider_manifest: dict, selected_rows: list[dict], reports: list[dict]
) -> dict:
    """Publish a source marker only after original-file coverage and selected-ledger identity agree.

    This accepts content reconstruction for the supplied Common Corpus source.
    It does not certify historical provider occurrence lineage or the other
    sources in the cluster.
    """
    _verify_plan(plan)
    source = next(source for source in plan["sources"] if source["marin_source"] == COMMON_CORPUS_SOURCE)
    if source["blocked_reason"] or source["provider_paths"] != provider_manifest["files"]:
        raise ValueError("Common Corpus completion requires a plan bound to the complete original provider inventory")
    recipe = json.loads(json.dumps(source["recipe"]))
    expected_recipe = json.loads(json.dumps(dataclasses.asdict(common_corpus_recipe())))
    if recipe != expected_recipe or len(selected_rows) != source["selected_occurrences"]:
        raise ValueError("Common Corpus recipe or selected occurrence count differs from the reconstruction plan")
    observed: Counter[tuple] = Counter()
    for ledger in source["ledgers"]:
        fs, path = filesystem_for(ledger["output"])
        with fs.open(path, "rb") as stream:
            for batch in pq.ParquetFile(stream).iter_batches(columns=SELECTED_IDENTITY_COLUMNS, batch_size=8192):
                observed.update(tuple(row[key] for key in SELECTED_IDENTITY_COLUMNS) for row in batch.to_pylist())
    expected = Counter(tuple(row[key] for key in SELECTED_IDENTITY_COLUMNS) for row in selected_rows)
    if observed != expected:
        raise ValueError("Common Corpus supplied targets do not cover the exact selected-ledger occurrences")
    coverage = aggregate_common_corpus_scan(provider_manifest, selected_rows, reports)
    if not coverage["content_complete_for_supplied_selection"]:
        raise ValueError("Common Corpus original files did not recover the complete selected source")
    index = build_selected_index(plan, COMMON_CORPUS_SOURCE)
    binding = {"source_sha256": source["manifest_sha256"], "index_sha256": index["index_sha256"]}
    root = plan["source_roots"][COMMON_CORPUS_SOURCE]
    marker_path = prefix_join(root, "reconstructed/_complete.json")
    marker_fs, marker_key = filesystem_for(marker_path)
    if marker_fs.exists(marker_key):
        previous = _read_json(marker_path)
        if (
            previous["binding"] != binding
            or previous["external_proof"]["provider_manifest_sha256"] != coverage["provider_manifest_sha256"]
            or previous["external_proof"]["selected_rows_sha256"] != coverage["selected_rows_sha256"]
        ):
            raise ValueError("Existing Common Corpus completion belongs to different pinned inputs")
        return previous
    attempt = uuid.uuid4().hex
    content = reconstruct_common_corpus_selection(
        provider_manifest, selected_rows, reports, prefix_join(root, f"external-content/attempt={attempt}")
    )
    fs, path = filesystem_for(content["output"])
    rows = []
    with fs.open(path, "rb") as stream:
        for batch in pq.ParquetFile(stream).iter_batches(columns=["id", "text"], batch_size=64):
            for row in batch.to_pylist():
                if generate_id(row["text"]) != row["id"]:
                    raise ValueError("Common Corpus reconstructed text changed before source completion")
                rows.append({"marin_source": COMMON_CORPUS_SOURCE, **row})
    if Counter(row["id"] for row in rows) != Counter(row["id"] for row in selected_rows):
        raise ValueError("Common Corpus reconstructed text does not preserve the selected document multiplicity")
    output = prefix_join(root, f"reconstructed/attempt={attempt}/text-00000.parquet")
    output_fs, output_key = filesystem_for(output)
    output_fs.makedirs(output_key.rsplit("/", 1)[0], exist_ok=True)
    with output_fs.open(output_key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=RECONSTRUCTED_TEXT_SCHEMA), stream, compression="zstd")
    candidate_counts = Counter(
        match["id"] for task in reports for item in task["completed"] for match in item["matches"]
    )
    completion = {
        "binding": binding,
        "marin_source": COMMON_CORPUS_SOURCE,
        "provider_files": len(provider_manifest["files"]),
        "selected_occurrences": source["selected_occurrences"],
        "reconstructed_occurrences": len(rows),
        "missing_ids": 0,
        "missing_occurrences": 0,
        "duplicate_candidate_ids": sum(count - 1 for count in candidate_counts.values()),
        "historical_occurrence_provenance": "not_established",
        "outputs": [{"path": output, "occurrences": len(rows)}],
        "complete": True,
        "external_proof": {
            "provider_manifest_sha256": coverage["provider_manifest_sha256"],
            "selected_rows_sha256": coverage["selected_rows_sha256"],
            "complete_original_file_coverage": True,
            "task_reports_sha256": _digest(reports),
            "content_report": prefix_join(root, f"external-content/attempt={attempt}/_complete.json"),
        },
    }
    _write_json(prefix_join(root, f"reconstructed/attempt={attempt}/report.json"), completion)
    _write_json(marker_path, completion)
    return completion
