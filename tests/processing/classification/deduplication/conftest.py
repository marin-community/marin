# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
from zephyr import load_vortex

from pathlib import Path

import pytest
from zephyr.readers import load_jsonl


@pytest.fixture(scope="module")
def docs():
    test_resources = Path(__file__).parent.joinpath("resources", "docs")
    docs = {}
    for doc_file in test_resources.glob("*.txt"):
        docs[doc_file.stem] = doc_file.read_text()
    return docs


def load_dedup_outputs(output_dir: str) -> dict[str, dict]:
    """Load all dedupe output files and return as id->doc mapping.

    Args:
        output_dir: Directory containing .jsonl.gz output files

    Returns:
        Dictionary mapping document IDs to document records
    """
    output_files = list(Path(output_dir).glob("**/*.jsonl.gz"))
    results = []
    for output_file in output_files:
        results.extend(load_jsonl(str(output_file)))
    return {r["id"]: r for r in results}


def load_dedup_vortex_outputs(output_dir: str) -> dict[str, list[dict]]:
    """Load all dedup vortex output files keyed by output filename stem.

    Returns:
        Dictionary mapping output file stem (e.g. "test_shard_0") to list of records.
    """
    output_files = sorted(Path(output_dir).glob("**/*.vortex"))
    by_file: dict[str, list[dict]] = {}
    for output_file in output_files:
        by_file[output_file.stem] = list(load_vortex(str(output_file)))
    return by_file
