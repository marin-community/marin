# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact counts and nesting in the Luxical ladder manifest."""

import hashlib
import json
import logging
import string
from collections import Counter
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ladder_config import (
    EVAL_ROWS_PER_SOURCE,
    MANIFEST_ROOT,
    MIN_SOURCES,
    SAMPLE_BLOCKS_PER_SOURCE,
    SAMPLING_METHOD,
    STACK_V3_OUTPUT_HASH,
    TRAIN_TARGET_3M,
    TRAIN_TARGET_750K,
)
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
AUDIT_URL = f"{MANIFEST_ROOT}/manifest-audit.json"
RESULT_FILE = Path("/tmp/luxical-arctic-manifest-audit")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical manifest digest without its digest field."""
    payload = {key: value for key, value in manifest.items() if key != "sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_sha256(value: str) -> bool:
    """Return true when a value is a lowercase SHA-256 digest."""
    return len(value) == 64 and all(character in string.hexdigits.lower() for character in value)


def source_counts(url: str, selected_input_files: list[dict[str, Any]]) -> dict[str, int]:
    """Return exact split, rung, and provenance counts for one source file."""
    filesystem, path = fsspec.core.url_to_fs(url)
    if not filesystem.exists(path):
        raise FileNotFoundError(f"Missing source manifest file: {url}")
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=[
            "split",
            "in_750k",
            "in_3m",
            "input_path",
            "input_row_group",
            "input_row_in_group",
            "raw_sha256",
            "normalized_sha256",
        ],
    )
    eval_mask = pc.equal(table["split"], "eval")
    train_mask = pc.equal(table["split"], "train")
    invalid_small = pc.any(pc.and_(table["in_750k"], pc.invert(table["in_3m"]))).as_py()
    invalid_eval = pc.any(
        pc.and_(
            eval_mask,
            pc.or_(table["in_750k"], table["in_3m"]),
        )
    ).as_py()
    if invalid_small:
        raise ValueError(f"The 0.75M rows are not nested in the 3M rows: {url}")
    if invalid_eval:
        raise ValueError(f"Evaluation rows appear in a training rung: {url}")
    expected_path_counts = {result["path"]: result["selected_rows"] for result in selected_input_files}
    actual_path_counts = Counter(table["input_path"].to_pylist())
    if actual_path_counts != expected_path_counts:
        raise ValueError(f"Selected input-file counts differ for {url}")
    positions = set(
        zip(
            table["input_path"].to_pylist(),
            table["input_row_group"].to_pylist(),
            table["input_row_in_group"].to_pylist(),
            strict=True,
        )
    )
    if len(positions) != len(table):
        raise ValueError(f"Selected input positions are not unique for {url}")
    for column in ("raw_sha256", "normalized_sha256"):
        if not all(valid_sha256(value) for value in table[column].to_pylist()):
            raise ValueError(f"Column {column} contains an invalid digest: {url}")
    return {
        "rows": len(table),
        "eval": pc.sum(pc.cast(eval_mask, "int64")).as_py(),
        "train": pc.sum(pc.cast(train_mask, "int64")).as_py(),
        "train_750k": pc.sum(pc.cast(table["in_750k"], "int64")).as_py(),
        "train_3m": pc.sum(pc.cast(table["in_3m"], "int64")).as_py(),
        "selected_input_file_count": len(actual_path_counts),
    }


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def main() -> None:
    """Audit every source file and all global counts."""
    manifest = read_json(MANIFEST_URL)
    if manifest_digest(manifest) != manifest["sha256"]:
        raise ValueError("The stored manifest digest does not match its content")
    if len(manifest["sources"]) < MIN_SOURCES:
        raise ValueError(f"Manifest has only {len(manifest['sources'])} sources")
    if manifest["sampling_method"] != SAMPLING_METHOD:
        raise ValueError(f"Manifest has sampling method {manifest['sampling_method']}")
    if manifest["sampling_blocks_per_source"] != SAMPLE_BLOCKS_PER_SOURCE:
        raise ValueError(f"Manifest has {manifest['sampling_blocks_per_source']} sampling blocks")
    stack_v3 = manifest["sources"].get("stack-v3")
    if stack_v3 is None:
        raise ValueError("The manifest does not contain Stack v3")
    if f"stack-v3_{STACK_V3_OUTPUT_HASH}" not in stack_v3["main_output_dir"]:
        raise ValueError(f"Stack v3 uses an unexpected output: {stack_v3['main_output_dir']}")
    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Auditing manifest source %d/%d: %s", index, len(manifest["sources"]), source)
        counts = source_counts(result["output_url"], result["selected_input_files"])
        if counts["eval"] != EVAL_ROWS_PER_SOURCE:
            raise ValueError(f"Source {source} has {counts['eval']} evaluation rows")
        if counts["train_750k"] != result["counts"]["train_750k"]:
            raise ValueError(f"Source {source} has an incorrect 0.75M quota")
        if counts["train_3m"] != result["counts"]["train_3m"]:
            raise ValueError(f"Source {source} has an incorrect 3M quota")
        sources[source] = counts
    totals = {
        key: sum(counts[key] for counts in sources.values())
        for key in ("rows", "eval", "train", "train_750k", "train_3m", "selected_input_file_count")
    }
    if totals["train_750k"] != TRAIN_TARGET_750K:
        raise ValueError(f"0.75M total is {totals['train_750k']}")
    if totals["train_3m"] != TRAIN_TARGET_3M:
        raise ValueError(f"3M total is {totals['train_3m']}")
    report = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_count": len(sources),
        "stack_v3_output_hash": STACK_V3_OUTPUT_HASH,
        "totals": totals,
        "sources": sources,
        "passed": True,
    }
    write_json(AUDIT_URL, report)
    summary = {
        "audit_url": AUDIT_URL,
        "manifest_sha256": manifest["sha256"],
        "source_count": len(sources),
        "stack_v3_output_hash": STACK_V3_OUTPUT_HASH,
        "totals": totals,
        "passed": True,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_MANIFEST_AUDIT=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
