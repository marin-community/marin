# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for repository file filtering and chunking."""

from itertools import pairwise
from pathlib import PurePosixPath

import repository_files


def test_indexed_file_rejects_generated_vendored_secret_binary_and_oversized_content():
    rejected = {
        "vendor/copied.py": b"print('vendored')",
        "config/external/model.py": b"MODEL = 'external'",
        "keys/deploy.pem": b"private material",
        "package-lock.json": b"{}",
        "src/generated.min.js": b"minified()",
        "src/binary.py": b"prefix\0suffix",
        "docs/huge.md": b"x" * (repository_files.MAX_FILE_BYTES + 1),
    }

    assert all(
        repository_files.indexed_file(PurePosixPath(path), contents) is None for path, contents in rejected.items()
    )
    kept = repository_files.indexed_file(
        PurePosixPath("docs/runbook.md"),
        b"# Collective diagnosis\n\nInspect the topology.",
    )
    assert kept is not None
    assert kept.title == "Collective diagnosis"


def test_text_chunks_cover_long_files_with_bounded_overlap():
    text = "\n".join(f"line {line}: {'x' * 90}" for line in range(100))
    chunks = list(repository_files.text_chunks(text))

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].text.endswith("line 99: " + "x" * 90)
    assert all(len(chunk.text) <= repository_files.CHUNK_CHARACTERS + 100 for chunk in chunks)
    for previous, current in pairwise(chunks):
        assert current.start_line < previous.start_line + len(previous.text.splitlines())
