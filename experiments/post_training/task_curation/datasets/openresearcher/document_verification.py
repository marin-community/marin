# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a bijection between stored index documents and the corpus SHA inventory."""

import hashlib
import logging
import sqlite3
from collections.abc import Iterable


def verify_corpus_documents(documents: Iterable[dict[str, str]], database: str, expected_count: int) -> None:
    with sqlite3.connect(database) as connection:
        if connection.execute("SELECT COUNT(*) FROM docs").fetchone()[0] != expected_count:
            raise ValueError("Corpus inventory count differs from the index declaration")
        connection.execute("PRAGMA cache_size = -2097152")
        connection.execute("CREATE TEMP TABLE verified (docid TEXT PRIMARY KEY) WITHOUT ROWID")
        count = 0
        for raw in documents:
            docid, text = raw["id"], raw["contents"]
            expected = connection.execute("SELECT sha256 FROM docs WHERE docid = ?", (docid,)).fetchone()
            if expected is None or hashlib.sha256(text.encode()).hexdigest() != expected[0]:
                raise ValueError(f"Stored index document differs from corpus: {docid}")
            connection.execute("INSERT INTO verified VALUES (?)", (docid,))
            count += 1
            if count % 10_000 == 0:
                logging.info("Verified index documents=%d/%d", count, expected_count)
        if count != expected_count:
            raise ValueError(f"Index yielded {count} unique documents; expected {expected_count}")
