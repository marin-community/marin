# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the dense flat inner-product index from the sealed embedding shards.

Runs in the regional Iris CPU job. Every vector part and docid part is verified
against the embedding artifact manifest, and every docid row is compared in
order against the approved corpus before it enters the index.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS,
    CORPUS_REVISION,
    REGIONAL_PREFIX,
    RegionalInputs,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.embedding import (
    EMBEDDING_DIM,
    EXPECTED_DOCUMENTS,
    EmbeddingShards,
)
from experiments.post_training.task_curation.local import artifact_prefix, local_client

FAISS_INDEX_TYPE = "IndexFlatIP"
SPOT_CHECK_STRIDE = 100_000
logger = logging.getLogger(__name__)


class DenseIndex(Artifact):
    corpus_revision: str
    corpus_root: str
    stage_receipt_sha256: str
    embedding_receipt_sha256: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dim: int
    document_count: int
    faiss_index_type: str
    faiss_version: str
    index_manifest_sha256: str
    index_bytes: int
    iris_task_id: str


@dataclass(frozen=True)
class DenseIndexConfig:
    embedding_receipt_uri: str
    embedding_receipt_sha256: str
    stage_receipt_uri: str
    stage_receipt_sha256: str
    workdir: str
    output_path: str


def stage_embedding(config: DenseIndexConfig, work: Path) -> tuple[EmbeddingShards, list[dict]]:
    embedding = EmbeddingShards.model_validate_json(
        pinned_document(config.embedding_receipt_uri, config.embedding_receipt_sha256)
    )
    if embedding.corpus_revision != CORPUS_REVISION or embedding.document_count != EXPECTED_DOCUMENTS:
        raise ValueError("Embedding artifact differs from the approved complete corpus")
    if embedding.embedding_dim != EMBEDDING_DIM:
        raise ValueError("Embedding dimension differs from the dense index contract")
    manifest = json.loads(
        pinned_document(str(StoragePath(embedding.path) / "shard-manifest.json"), embedding.shard_manifest_sha256)
    )
    directory = work / "shards"
    directory.mkdir(parents=True, exist_ok=False)
    for entry in manifest:
        staged = (
            ("vectors", f"vectors/{entry['vectors']}", entry["vectors_sha256"], entry["vectors_bytes"]),
            ("docids", f"docids/{entry['docids']}", entry["docids_sha256"], None),
        )
        for kind, relative, expected_sha, expected_bytes in staged:
            source = StoragePath(embedding.path) / relative
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            if target.stat().st_size == 0:
                raise ValueError(f"Sealed embedding part is empty: {relative}")
            actual = file_sha256(StoragePath(str(target)))
            if actual != expected_sha or (expected_bytes is not None and target.stat().st_size != expected_bytes):
                raise ValueError(f"Sealed embedding part differs from the manifest: {relative}")
            entry[f"local_{kind}"] = str(target)
    return embedding, manifest


def corpus_docids(staged: RegionalInputs, shard_name: str) -> list[str]:
    root = StoragePath(staged.source_roots[CORPUS])
    with (root / "data" / shard_name).open("rb") as stream:
        parquet = pq.ParquetFile(stream)
        return [
            row["docid"]
            for batch in parquet.iter_batches(batch_size=4096, columns=["docid"])
            for row in batch.to_pylist()
        ]


def build_index(config: DenseIndexConfig) -> DenseIndex:
    import faiss

    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("The dense index must be built inside the regional Iris CPU job")
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt_uri, config.stage_receipt_sha256))
    embedding, manifest = stage_embedding(config, Path(config.workdir))
    work = Path(config.workdir)
    manifest.sort(key=lambda entry: entry["shard"])
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    ordered_docids: list[str] = []
    for entry in manifest:
        vectors = np.fromfile(entry["local_vectors"], dtype=np.float16).reshape(-1, EMBEDDING_DIM).astype(np.float32)
        docids = pq.read_table(entry["local_docids"], columns=["docid"]).column("docid").to_pylist()
        expected = corpus_docids(staged, entry["shard"])
        if docids != expected:
            raise ValueError(f"Dense docid order differs from the approved corpus shard: {entry['shard']}")
        if len(vectors) != len(docids):
            raise ValueError(f"Vector rows disagree with docid rows: {entry['shard']}")
        index.add(vectors)
        ordered_docids.extend(docids)
        logger.info("Added %s: %d rows (total %d)", entry["shard"], len(vectors), index.ntotal)
    if index.ntotal != EXPECTED_DOCUMENTS or len(ordered_docids) != EXPECTED_DOCUMENTS:
        raise ValueError("Dense index row count differs from the approved corpus")
    prefixes = {}
    consumed = 0
    for entry in manifest:
        prefixes[entry["shard"]] = consumed
        consumed += entry["documents"]
    for number in range(0, EXPECTED_DOCUMENTS, SPOT_CHECK_STRIDE):
        entry = next(item for item in manifest if prefixes[item["shard"]] <= number < prefixes[item["shard"]] + item["documents"])
        reconstructed = np.empty(EMBEDDING_DIM, dtype=np.float32)
        index.reconstruct(number, reconstructed)
        local_row = number - prefixes[entry["shard"]]
        stored = np.fromfile(
            entry["local_vectors"], dtype=np.float16, count=EMBEDDING_DIM, offset=local_row * EMBEDDING_DIM * 2
        )
        if not np.array_equal(reconstructed, stored.astype(np.float32)):
            raise ValueError(f"Dense index reconstruction differs at row {number}")
    docids_path = work / "dense-docids.parquet"
    with docids_path.open("wb") as writer:
        pq.write_table(pa.Table.from_arrays([pa.array(ordered_docids)], names=["docid"]), writer, compression="zstd")
    index_path = work / "dense.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("Dense index built: %d rows, %s bytes", index.ntotal, index_path.stat().st_size)
    output = StoragePath(config.output_path)
    files = []
    for source, relative in (
        (index_path, "dense.faiss"),
        (docids_path, "dense-docids.parquet"),
    ):
        destination = output / relative
        with source.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if file_sha256(destination) != file_sha256(StoragePath(str(source))):
            raise ValueError(f"Regional dense upload differs: {relative}")
        files.append({"path": relative, "sha256": file_sha256(destination), "bytes": source.stat().st_size})
    for relative in ("shard-manifest.json", "model-manifest.json", "license-receipt.json"):
        source = StoragePath(embedding.path) / relative
        target = output / relative
        with source.open("rb") as reader, target.open("wb", auto_mkdir=True) as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
    manifest_json = (json.dumps(files, indent=2) + "\n").encode()
    (output / "dense-manifest.json").write_bytes(manifest_json)
    return DenseIndex(
        path=config.output_path,
        corpus_revision=CORPUS_REVISION,
        corpus_root=staged.source_roots[CORPUS],
        stage_receipt_sha256=config.stage_receipt_sha256,
        embedding_receipt_sha256=config.embedding_receipt_sha256,
        embedding_model=embedding.embedding_model,
        embedding_model_revision=embedding.embedding_model_revision,
        embedding_dim=EMBEDDING_DIM,
        document_count=index.ntotal,
        faiss_index_type=FAISS_INDEX_TYPE,
        faiss_version=faiss.__version__,
        index_manifest_sha256=hashlib.sha256(manifest_json).hexdigest(),
        index_bytes=sum(file["bytes"] for file in files),
        iris_task_id=os.environ["IRIS_TASK_ID"],
    )


def dense_config(
    embedding_uri: str,
    embedding_sha256: str,
    stage_uri: str,
    stage_sha256: str,
    workdir: str,
    context: StepContext,
) -> DenseIndexConfig:
    return DenseIndexConfig(embedding_uri, embedding_sha256, stage_uri, stage_sha256, workdir, context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-receipt", required=True)
    parser.add_argument("--embedding-receipt-sha256", required=True)
    parser.add_argument("--stage-receipt", required=True)
    parser.add_argument("--stage-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/dense-index",
            version=args.artifact_version,
            artifact_type=DenseIndex,
            run=build_index,
            build_config=partial(
                dense_config,
                args.embedding_receipt,
                args.embedding_receipt_sha256,
                args.stage_receipt,
                args.stage_receipt_sha256,
                args.workdir,
            ),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))
