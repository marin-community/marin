# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sqlite3

import pytest

from experiments.post_training.task_curation.datasets.openresearcher import image_context
from experiments.post_training.task_curation.datasets.openresearcher.document_verification import verify_corpus_documents


@pytest.mark.parametrize("defect", [None, "duplicate", "missing", "changed"])
def test_index_agreement_requires_every_original_document_exactly_once(tmp_path, defect):
    database = tmp_path / "corpus.sqlite"
    rows = [{"id": "b", "contents": "Second document."}, {"id": "a", "contents": "First document."}]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE docs (docid TEXT PRIMARY KEY, sha256 TEXT NOT NULL) WITHOUT ROWID")
        connection.executemany(
            "INSERT INTO docs VALUES (?, ?)",
            [(row["id"], hashlib.sha256(row["contents"].encode()).hexdigest()) for row in rows],
        )
    if defect == "duplicate":
        rows = [rows[0], rows[0]]
    elif defect == "missing":
        rows = rows[:1]
    elif defect == "changed":
        rows[1] = {**rows[1], "contents": "Replaced document."}
    if defect is None:
        # Stored Lucene order may differ from external corpus-ID order.
        assert verify_corpus_documents(rows, str(database), 2) is None
    else:
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            verify_corpus_documents(rows, str(database), 2)


def test_layer_partition_preserves_nested_index_files_under_the_registry_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(image_context, "MAX_LAYER_BYTES", 8)
    source, context = tmp_path / "index", tmp_path / "context"
    original = {"one.cfs": b"123456", "nested/two.tim": b"abcdef", "three.si": b"xx"}
    for name, content in original.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    sizes = image_context.index_layers(source, context)
    restored = {}
    for layer in context.iterdir():
        for path in layer.rglob("*"):
            if path.is_file():
                restored[path.relative_to(layer).as_posix()] = path.read_bytes()
    assert restored == original
    assert sum(sizes) == sum(map(len, original.values())) and max(sizes) <= 8
