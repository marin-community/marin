# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private grounding retrieval in the index's isolated Java/Pyserini environment."""

import argparse
import json
import sys
from importlib import import_module

from experiments.post_training.task_curation.datasets.browsecomp.tool import source_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    args = parser.parse_args()
    with source_diagnostics():
        backend = import_module("pyserini.search.lucene")
        searcher = backend.LuceneSearcher(args.index)
    try:
        for line in sys.stdin:
            queries = json.loads(line)
            documents, searches = {}, []
            with source_diagnostics():
                for query in queries:
                    hits = searcher.search(query, 5)
                    searches.append({"query": query, "hits": [{"docid": hit.docid, "score": hit.score} for hit in hits]})
                    for hit in hits:
                        documents[hit.docid] = json.loads(searcher.doc(hit.docid).raw())["contents"]
            sys.stdout.write(json.dumps({"searches": searches, "documents": documents}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
