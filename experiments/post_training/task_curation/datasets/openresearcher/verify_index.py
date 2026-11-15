# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every stored Lucene document in the isolated Pyserini runtime."""

import argparse
import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    searcher = LuceneSearcher(args.index)
    if searcher.num_docs != args.count:
        raise ValueError(f"Index contains {searcher.num_docs} documents; corpus contains {args.count}")
    with sqlite3.connect(args.database) as connection:
        for position, (docid, expected) in enumerate(connection.execute("SELECT docid, sha256 FROM docs"), 1):
            document = searcher.doc(docid)
            if document is None:
                raise ValueError(f"Index is missing document {docid}")
            text = json.loads(document.raw())["contents"]
            if hashlib.sha256(text.encode()).hexdigest() != expected:
                raise ValueError(f"Stored index document differs from corpus: {docid}")
            if position % 100_000 == 0:
                logging.info("Verified index documents=%d/%d", position, args.count)
    searcher.close()
    Path(args.output).write_text(json.dumps({"documents": args.count, "content_mismatches": 0}) + "\n")


if __name__ == "__main__":
    main()
