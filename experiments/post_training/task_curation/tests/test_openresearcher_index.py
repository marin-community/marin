# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sqlite3

import pytest

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
