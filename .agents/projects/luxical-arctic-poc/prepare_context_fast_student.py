# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pack the fixed 3M rows for one longer-context fast student."""

import json
import logging
from pathlib import Path

import fsspec
import numpy as np
from fast_student import BASELINE_FILE, BASELINE_REPO, BASELINE_REVISION, TOKENIZER_NAME, fast_student_config
from ladder_config import MANIFEST_ROOT, TEACHER_ID, TEACHER_REVISION
from prepare_fast_student import prepare_source, read_json, write_json

CONFIG_NAME = "context512"
MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_AUDIT_URL = f"{MANIFEST_ROOT}/teacher-arctic-v1/audit.json"
BASE_PREPARED_MANIFEST_URL = f"{MANIFEST_ROOT}/fast-student/prepared-3m/manifest.json"
OUTPUT_ROOT = f"{MANIFEST_ROOT}/fast-student/prepared-3m-{CONFIG_NAME}"
OUTPUT_MANIFEST_URL = f"{OUTPUT_ROOT}/manifest.json"
RESULT_FILE = Path("/tmp/luxical-context-fast-student-prepare")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def load_numpy(url: str) -> np.ndarray:
    """Load one NumPy array from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path, "rb") as file:
        return np.load(file)


def main() -> None:
    """Build the longer-context packed 3M training set."""
    manifest = read_json(MANIFEST_URL)
    audit = read_json(TEACHER_AUDIT_URL)
    base_prepared = read_json(BASE_PREPARED_MANIFEST_URL)
    if audit["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("The teacher audit has a different manifest digest")
    if audit["teacher_id"] != TEACHER_ID or audit["teacher_revision"] != TEACHER_REVISION:
        raise ValueError("The teacher audit does not identify the fixed Arctic Medium teacher")
    if base_prepared["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("The tokenizer map has a different manifest digest")

    config = fast_student_config(CONFIG_NAME, vocab_size=int(base_prepared["actual_compact_vocab_size"]))
    raw_to_compact = load_numpy(base_prepared["raw_to_compact_url"])
    sources = {}
    for index, (source, result) in enumerate(sorted(manifest["sources"].items()), start=1):
        logger.info("Packing context source %d/%d: %s", index, len(manifest["sources"]), source)
        sources[source] = prepare_source(
            source,
            result["output_url"],
            raw_to_compact,
            output_root=OUTPUT_ROOT,
            max_tokens=config.max_tokens,
            characters_per_source_window=config.max_tokens,
        )
    row_count = sum(result["rows"] for result in sources.values())
    if row_count != 3_000_000:
        raise ValueError(f"Prepared {row_count} rows; expected 3000000")
    output = {
        "config_name": CONFIG_NAME,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "teacher_audit_url": TEACHER_AUDIT_URL,
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "tokenizer": TOKENIZER_NAME,
        "tokenizer_artifact": {
            "repository": BASELINE_REPO,
            "file": BASELINE_FILE,
            "revision": BASELINE_REVISION,
        },
        "actual_compact_vocab_size": int(raw_to_compact.max()) + 1,
        "raw_to_compact_url": base_prepared["raw_to_compact_url"],
        "raw_to_compact_sha256": base_prepared["raw_to_compact_sha256"],
        "maximum_tokens": config.max_tokens,
        "characters_per_source_window": config.max_tokens,
        "view_regions": ["head", "middle", "tail"],
        "rows": row_count,
        "sources": sources,
    }
    write_json(OUTPUT_MANIFEST_URL, output)
    summary = {
        "config_name": CONFIG_NAME,
        "output_url": OUTPUT_MANIFEST_URL,
        "rows": row_count,
        "sources": len(sources),
        "maximum_tokens": config.max_tokens,
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("CONTEXT_FAST_STUDENT_PREPARE=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
