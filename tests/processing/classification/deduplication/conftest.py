# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
