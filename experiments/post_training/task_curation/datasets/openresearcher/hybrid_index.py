# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the sealed BM25 and dense indexes into one versioned hybrid provider."""

import argparse
import json
from dataclasses import dataclass
from functools import partial

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS_REVISION,
    REGIONAL_PREFIX,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.build_dense_index import DenseIndex
from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import (
    EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
)
from experiments.post_training.task_curation.datasets.openresearcher.embedding import EmbeddingShards
from experiments.post_training.task_curation.datasets.openresearcher.hybrid_fusion import (
    CANDIDATE_DEPTH,
    RRF_K,
    TOP_K,
)
from experiments.post_training.task_curation.datasets.openresearcher.index import BM25Index
from experiments.post_training.task_curation.local import artifact_prefix

PROVIDER_VERSION = "openresearcher-hybrid:v1"
SNIPPET_MAX_TOKENS = 512


class FusionParameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str = "rrf"
    rrf_k: int
    candidate_depth: int
    bm25_lists_per_query: int = 1
    dense_lists_per_query: int = 1


class HybridIndex(Artifact):
    bm25_index: BM25Index
    dense_index: DenseIndex
    embedding: EmbeddingShards
    bm25_index_receipt_sha256: str
    dense_index_receipt_sha256: str
    embedding_receipt_sha256: str
    corpus_revision: str
    document_count: int
    provider_version: str
    fusion: FusionParameters
    embedding_model: str
    embedding_model_revision: str
    embedding_dim: int
    max_embed_tokens: int
    license_receipt_sha256: str


def provider_state(index: HybridIndex, implementation_revision: str) -> dict:
    """The frozen retrieval state the TaskSpec and hosted service both pin."""
    return {
        "implementation_revision": implementation_revision,
        "corpus_revision": index.corpus_revision,
        "bm25_index_manifest_sha256": index.bm25_index.index_manifest_sha256,
        "dense_index_manifest_sha256": index.dense_index.index_manifest_sha256,
        "embedding_model": index.embedding_model,
        "embedding_model_revision": index.embedding_model_revision,
        "embedding_dim": index.embedding_dim,
        "max_embed_tokens": index.max_embed_tokens,
        "fusion": index.fusion.method,
        "fusion_rrf_k": index.fusion.rrf_k,
        "candidate_depth": index.fusion.candidate_depth,
        "top_k": TOP_K,
        "snippet_max_tokens": SNIPPET_MAX_TOKENS,
        "include_get_document": True,
    }


@dataclass(frozen=True)
class HybridIndexConfig:
    bm25_receipt_uri: str
    bm25_receipt_sha256: str
    dense_receipt_uri: str
    dense_receipt_sha256: str
    embedding_receipt_uri: str
    embedding_receipt_sha256: str
    output_path: str


def build_hybrid_index(config: HybridIndexConfig) -> HybridIndex:
    bm25 = BM25Index.model_validate_json(pinned_document(config.bm25_receipt_uri, config.bm25_receipt_sha256))
    dense = DenseIndex.model_validate_json(pinned_document(config.dense_receipt_uri, config.dense_receipt_sha256))
    embedding = EmbeddingShards.model_validate_json(
        pinned_document(config.embedding_receipt_uri, config.embedding_receipt_sha256)
    )
    if bm25.corpus_revision != CORPUS_REVISION or dense.corpus_revision != CORPUS_REVISION:
        raise ValueError("Hybrid provider requires the approved immutable corpus")
    if bm25.document_count != dense.document_count or dense.document_count != embedding.document_count:
        raise ValueError("Hybrid provider requires both indexes to cover the same complete documents")
    if dense.corpus_revision != embedding.corpus_revision:
        raise ValueError("Dense index and embedding artifact cover different corpora")
    if (
        dense.embedding_dim != EMBEDDING_DIM
        or dense.embedding_model != EMBEDDING_MODEL
        or dense.embedding_model_revision != EMBEDDING_MODEL_REVISION
        or embedding.embedding_model_revision != EMBEDDING_MODEL_REVISION
    ):
        raise ValueError("Dense index differs from the pinned embedding contract")
    if dense.embedding_receipt_sha256 != config.embedding_receipt_sha256:
        raise ValueError("Dense index was not built from the bound embedding artifact")
    if bm25.index_manifest_sha256 == dense.index_manifest_sha256:
        raise ValueError("Independent index manifests must not collide")
    return HybridIndex(
        path=config.output_path,
        bm25_index=bm25,
        dense_index=dense,
        embedding=embedding,
        bm25_index_receipt_sha256=config.bm25_receipt_sha256,
        dense_index_receipt_sha256=config.dense_receipt_sha256,
        embedding_receipt_sha256=config.embedding_receipt_sha256,
        corpus_revision=CORPUS_REVISION,
        document_count=bm25.document_count,
        provider_version=PROVIDER_VERSION,
        fusion=FusionParameters(rrf_k=RRF_K, candidate_depth=CANDIDATE_DEPTH),
        embedding_model=EMBEDDING_MODEL,
        embedding_model_revision=EMBEDDING_MODEL_REVISION,
        embedding_dim=EMBEDDING_DIM,
        max_embed_tokens=EMBEDDING_MAX_TOKENS,
        license_receipt_sha256=embedding.license_receipt_sha256,
    )


def hybrid_config(
    bm25_uri: str,
    bm25_sha256: str,
    dense_uri: str,
    dense_sha256: str,
    embedding_uri: str,
    embedding_sha256: str,
    context: StepContext,
) -> HybridIndexConfig:
    return HybridIndexConfig(
        bm25_uri, bm25_sha256, dense_uri, dense_sha256, embedding_uri, embedding_sha256, context.output_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-receipt", required=True)
    parser.add_argument("--bm25-receipt-sha256", required=True)
    parser.add_argument("--dense-receipt", required=True)
    parser.add_argument("--dense-receipt-sha256", required=True)
    parser.add_argument("--embedding-receipt", required=True)
    parser.add_argument("--embedding-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/hybrid-index",
            version=args.artifact_version,
            artifact_type=HybridIndex,
            run=build_hybrid_index,
            build_config=partial(
                hybrid_config,
                args.bm25_receipt,
                args.bm25_receipt_sha256,
                args.dense_receipt,
                args.dense_receipt_sha256,
                args.embedding_receipt,
                args.embedding_receipt_sha256,
            ),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
