# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify every stored Lucene document in the isolated Pyserini runtime."""

import argparse
import json
import logging
from collections.abc import Iterator
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

from experiments.post_training.task_curation.datasets.openresearcher.document_verification import verify_corpus_documents


def indexed_documents(searcher: LuceneSearcher, count: int) -> Iterator[dict[str, str]]:
    for position in range(count):
        # Sequential Lucene IDs avoid random stored-field reads for external-ID order.
        document = searcher.doc(position)
        if document is None:
            raise ValueError(f"Index is missing internal document {position}")
        yield json.loads(document.raw())


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
    verify_corpus_documents(indexed_documents(searcher, args.count), args.database, args.count)
    searcher.close()
    Path(args.output).write_text(json.dumps({"documents": args.count, "content_mismatches": 0}) + "\n")


if __name__ == "__main__":
    main()
