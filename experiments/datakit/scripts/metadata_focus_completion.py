# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind complete original WARC replay to the frozen selected Focus Crawl ledger."""

import argparse
import hashlib
import json
import uuid
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import generate_id
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.scripts.metadata_cluster_reconstruction import _verify_plan, build_selected_index
from experiments.datakit.scripts.metadata_cluster_selection import _digest, _read_json, _write_json
from experiments.datakit.scripts.metadata_focus_provider import (
    FOCUS_SOURCE,
    JUSTEXT_REVISION,
    MODEL_SETTINGS,
    RUNTIME_DEPENDENCIES,
    focus_provider_evidence,
)
from experiments.datakit.store.reconstruct_selected_source import RECONSTRUCTED_TEXT_SCHEMA


def reconcile_focus_source(plan: dict, targets: dict[str, list[dict]], replay_output: str) -> dict:
    """Require every original target and selected occurrence before source completion."""
    _verify_plan(plan)
    source = next(source for source in plan["sources"] if source["marin_source"] == FOCUS_SOURCE)
    filenames = sorted(targets)
    original_paths = ["https://data.commoncrawl.org/" + filename for filename in filenames]
    evidence = focus_provider_evidence(targets)
    if (
        source["blocked_reason"]
        or source["provider_paths"] != original_paths
        or source["original_provider_evidence"] != evidence
    ):
        raise ValueError("Focus completion requires a plan bound to its exact original WARC targets and replay settings")
    expected_targets = {}
    for filename in filenames:
        for target in targets[filename]:
            key = (target["normalized_id"], target["record_id"])
            if key in expected_targets or any(key[1] == existing[1] for existing in expected_targets):
                raise ValueError("A Focus Crawl UUID is repeated or assigned conflicting content")
            expected_targets[key] = (filename, target["url"])
    if not expected_targets:
        raise ValueError("Focus Crawl completion requires selected original WARC targets")
    selected: Counter[tuple[str, str]] = Counter()
    for ledger in source["ledgers"]:
        fs, key = filesystem_for(ledger["output"])
        with fs.open(key, "rb", block_size=65536, cache_type="readahead") as stream:
            for batch in pq.ParquetFile(stream, pre_buffer=False).iter_batches(columns=["id", "source_id"]):
                for row in batch.to_pylist():
                    source_id = json.loads(row["source_id"])
                    if not isinstance(source_id, str):
                        raise ValueError("Selected Focus Crawl record lacks its original UUID")
                    selected[(row["id"], source_id)] += 1
    if set(selected) != set(expected_targets) or sum(selected.values()) != source["selected_occurrences"]:
        raise ValueError("Original WARC targets do not cover every selected-ledger identity and occurrence")
    # The shared index also checks the actual ledger ordering, counts, hashes,
    # normalized-shard identity, and canonical ID format against frozen reports.
    index = build_selected_index(plan, FOCUS_SOURCE)
    binding = {"source_sha256": source["manifest_sha256"], "index_sha256": index["index_sha256"]}
    candidate_text = {}
    observed: Counter[tuple[str, str]] = Counter()
    markers = []
    range_hashes = {}
    for file_index, filename in enumerate(filenames):
        root = prefix_join(replay_output, f"file={file_index:06d}")
        marker = _read_json(prefix_join(root, "_complete.json"))
        if (
            marker["binding"]
            != {"warc_filename": filename, "targets": targets[filename], "justext_revision": JUSTEXT_REVISION}
            or marker["runtime_dependencies"] != RUNTIME_DEPENDENCIES
            or marker["model_settings"] != MODEL_SETTINGS
        ):
            raise ValueError("Original WARC completion belongs to different targets or replay settings")
        expected_file = Counter((target["normalized_id"], target["record_id"]) for target in targets[filename])
        file_observed: Counter[tuple[str, str]] = Counter()
        text_hashes = {}
        fs, key = filesystem_for(prefix_join(root, "candidates.parquet"))
        with fs.open(key, "rb", block_size=65536, cache_type="readahead") as stream:
            for row in pq.read_table(stream).to_pylist():
                identity = (row["id"], row["source_id"])
                if identity not in expected_targets or expected_targets[identity] != (filename, row["url"]):
                    raise ValueError("Original WARC candidate UUID, URL, or content ID differs from its selected target")
                if row["warc_filename"] != filename or generate_id(row["text"]) != row["id"]:
                    raise ValueError("Original WARC candidate failed filename or canonical text verification")
                if len(row["warc_bytes_sha256"]) != 64 or len(bytes.fromhex(row["warc_bytes_sha256"])) != 32:
                    raise ValueError("Original WARC candidate lacks its response-byte digest")
                text_hashes[row["id"]] = hashlib.sha256(row["text"].encode()).hexdigest()
                if row["id"] in candidate_text and candidate_text[row["id"]] != row["text"]:
                    raise ValueError("Original WARC records disagree on text for the same canonical ID")
                candidate_text[row["id"]] = row["text"]
                range_hashes[row["source_id"]] = row["warc_bytes_sha256"]
                file_observed[identity] += 1
        if (
            file_observed != expected_file
            or sum(file_observed.values()) != marker["unique_candidates"]
            or text_hashes != marker["text_sha256"]
        ):
            raise ValueError("Original WARC candidate bytes or counts differ from the completed target set")
        observed.update(file_observed)
        markers.append(marker)
    if set(observed) != set(selected):
        raise ValueError("Completed original WARC files leave selected identities unresolved")
    rows = [
        {"marin_source": FOCUS_SOURCE, "id": doc_id, "text": candidate_text[doc_id]}
        for (doc_id, _), count in sorted(selected.items())
        for _ in range(count)
    ]
    root = plan["source_roots"][FOCUS_SOURCE]
    marker_path = prefix_join(root, "reconstructed/_complete.json")
    fs, key = filesystem_for(marker_path)
    if fs.exists(key):
        previous = _read_json(marker_path)
        if previous["binding"] != binding or previous["external_proof"]["replay_output"] != replay_output:
            raise ValueError("Existing Focus completion belongs to different selected or original replay inputs")
        return previous
    attempt = uuid.uuid4().hex
    output = prefix_join(root, f"reconstructed/attempt={attempt}/text-00000.parquet")
    output_fs, output_key = filesystem_for(output)
    output_fs.makedirs(output_key.rsplit("/", 1)[0], exist_ok=True)
    with output_fs.open(output_key, "wb") as stream:
        pq.write_table(pa.Table.from_pylist(rows, schema=RECONSTRUCTED_TEXT_SCHEMA), stream, compression="zstd")
    report = {
        "binding": binding,
        "marin_source": FOCUS_SOURCE,
        "provider_files": len(filenames),
        "selected_occurrences": source["selected_occurrences"],
        "reconstructed_occurrences": len(rows),
        "missing_ids": 0,
        "missing_occurrences": 0,
        "duplicate_candidate_ids": sum(observed.values()) - len(candidate_text),
        "historical_occurrence_provenance": "not_established",
        "outputs": [{"path": output, "occurrences": len(rows)}],
        "complete": True,
        "external_proof": {
            **evidence,
            "replay_output": replay_output,
            "complete_original_target_coverage": True,
            "selected_UUIDs": len(selected),
            "file_markers_sha256": _digest(markers),
            "response_byte_hashes_sha256": _digest(range_hashes),
        },
    }
    _write_json(prefix_join(root, f"reconstructed/attempt={attempt}/report.json"), report)
    _write_json(marker_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--replay-output", required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile_focus_source(_read_json(args.plan), _read_json(args.targets), args.replay_output)))


if __name__ == "__main__":
    main()
