# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grounding retrieval over the hybrid BM25+dense index in the pinned runtime.

Same JSON-line protocol as the BM25 search server: each request line is a JSON
array of queries, each response is ``{"searches": [...], "documents": {...}}``.
This module must stay importable without the Marin workspace: the pinned
retrieval runtime only carries pyserini, faiss, torch and transformers.
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from experiments.post_training.task_curation.datasets.browsecomp.tool import source_diagnostics
from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import (
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_QUERY_PREFIX,
)
from experiments.post_training.task_curation.datasets.openresearcher.hybrid_fusion import (
    CANDIDATE_DEPTH,
    RRF_K,
    TOP_K,
    rrf_fuse,
)

logger = logging.getLogger(__name__)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


class HybridRetriever:
    """One BM25 searcher, one dense flat index and the pinned query encoder."""

    def __init__(
        self,
        bm25_index: str,
        dense_index: str,
        docids: str,
        model_dir: str,
        candidates: int = CANDIDATE_DEPTH,
        rrf_k: int = RRF_K,
        top_k: int = TOP_K,
    ):
        lucene = importlib.import_module("pyserini.search.lucene")
        self.searcher = lucene.LuceneSearcher(bm25_index)
        self.faiss = importlib.import_module("faiss")
        self.dense = self.faiss.read_index(dense_index)
        self.docids = pq.read_table(docids, columns=["docid"]).column("docid").to_pylist()
        if self.dense.ntotal != len(self.docids):
            raise ValueError("Dense index rows disagree with the ordered docid table")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32, local_files_only=True)
        self.model.eval()
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.top_k = top_k

    def encode(self, query: str) -> np.ndarray:
        encoded = self.tokenizer(
            EMBEDDING_QUERY_PREFIX + query,
            truncation=True,
            max_length=EMBEDDING_MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state[0, 0]
        return normalize(hidden.numpy())

    def dense_search(self, query: str) -> list[str]:
        scores, rows = self.dense.search(self.encode(query)[None, :], self.candidates)
        return [self.docids[row] for row in rows[0] if row >= 0]

    def bm25_search(self, query: str) -> list[str]:
        return [hit.docid for hit in self.searcher.search(query, self.candidates)]

    def search(self, query: str) -> dict:
        """Fuse one BM25 list and one dense list into the sealed five-hit shape."""
        fused = rrf_fuse([self.bm25_search(query), self.dense_search(query)], self.rrf_k, self.top_k)
        documents = {}
        for docid, score in fused:
            documents[docid] = json.loads(self.searcher.doc(docid).raw())["contents"]
        return {
            "query": query,
            "hits": [{"docid": docid, "score": score} for docid, score in fused],
            "documents": documents,
        }

    def close(self) -> None:
        self.searcher.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-index", required=True)
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--docids", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--candidates", type=int, default=CANDIDATE_DEPTH)
    parser.add_argument("--rrf-k", type=int, default=RRF_K)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    with source_diagnostics():
        retriever = HybridRetriever(
            args.bm25_index, args.dense_index, args.docids, args.model_dir, args.candidates, args.rrf_k, args.top_k
        )
    logger.info("Hybrid retriever ready: %d dense rows", retriever.dense.ntotal)
    try:
        for line in sys.stdin:
            queries = json.loads(line)
            documents, searches = {}, []
            with source_diagnostics():
                for query in queries:
                    result = retriever.search(query)
                    documents.update(result.pop("documents"))
                    searches.append(result)
            sys.stdout.write(json.dumps({"searches": searches, "documents": documents}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
