# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact counts and nesting in the Luxical ladder manifest."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import fsspec
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ladder_config import (
    EVAL_ROWS_PER_SOURCE,
    MANIFEST_ROOT,
    MIN_SOURCES,
    SURVEY_ROWS_PER_SOURCE,
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


def source_counts(url: str) -> dict[str, int]:
    """Return exact split and rung counts for one source file."""
    filesystem, path = fsspec.core.url_to_fs(url)
    if not filesystem.exists(path):
        raise FileNotFoundError(f"Missing source manifest file: {url}")
    table = pq.read_table(
        path,
        filesystem=filesystem,
        columns=["split", "in_750k", "in_3m", "in_survey"],
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
    return {
        "rows": len(table),
        "eval": pc.sum(pc.cast(eval_mask, "int64")).as_py(),
        "train": pc.sum(pc.cast(train_mask, "int64")).as_py(),
        "train_750k": pc.sum(pc.cast(table["in_750k"], "int64")).as_py(),
        "train_3m": pc.sum(pc.cast(table["in_3m"], "int64")).as_py(),
        "survey": pc.sum(pc.cast(table["in_survey"], "int64")).as_py(),
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
    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Auditing manifest source %d/%d: %s", index, len(manifest["sources"]), source)
        counts = source_counts(result["output_url"])
        if counts["eval"] != EVAL_ROWS_PER_SOURCE:
            raise ValueError(f"Source {source} has {counts['eval']} evaluation rows")
        if counts["survey"] != SURVEY_ROWS_PER_SOURCE:
            raise ValueError(f"Source {source} has {counts['survey']} survey rows")
        if counts["train_750k"] != result["counts"]["train_750k"]:
            raise ValueError(f"Source {source} has an incorrect 0.75M quota")
        if counts["train_3m"] != result["counts"]["train_3m"]:
            raise ValueError(f"Source {source} has an incorrect 3M quota")
        sources[source] = counts
    totals = {
        key: sum(counts[key] for counts in sources.values())
        for key in ("rows", "eval", "train", "train_750k", "train_3m", "survey")
    }
    if totals["train_750k"] != TRAIN_TARGET_750K:
        raise ValueError(f"0.75M total is {totals['train_750k']}")
    if totals["train_3m"] != TRAIN_TARGET_3M:
        raise ValueError(f"3M total is {totals['train_3m']}")
    report = {
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "source_count": len(sources),
        "totals": totals,
        "sources": sources,
        "passed": True,
    }
    write_json(AUDIT_URL, report)
    summary = {
        "audit_url": AUDIT_URL,
        "manifest_sha256": manifest["sha256"],
        "source_count": len(sources),
        "totals": totals,
        "passed": True,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_MANIFEST_AUDIT=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
