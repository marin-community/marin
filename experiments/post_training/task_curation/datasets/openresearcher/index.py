# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and verify a fresh BM25 index over the licensed regional corpus."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
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
from experiments.post_training.task_curation.local import artifact_prefix, local_client

EXPECTED_DOCUMENTS = 14_878_084
PYSERINI_VERSION = "1.2.0"
JAVA_ARCHIVE = (
    "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.9%2B10/"
    "OpenJDK21U-jre_x64_linux_hotspot_21.0.9_10.tar.gz"
)
JAVA_SHA256 = "aeab55d064a1a27a3744b0880b9b414077b4ed2b1790817eea3df60aec946431"
logger = logging.getLogger(__name__)


class BM25Index(Artifact):
    corpus_revision: str
    corpus_root: str
    stage_receipt_sha256: str
    document_count: int
    content_mismatches: int
    index_manifest_sha256: str
    index_bytes: int
    pyserini_version: str
    java_sha256: str


@dataclass(frozen=True)
class IndexConfig:
    stage_receipt_uri: str
    stage_receipt_sha256: str
    workdir: str
    threads: int
    runtime_python: str
    output_path: str


def corpus_collection(root: str, output: Path, database: Path) -> int:
    output.mkdir(parents=True)
    count = 0
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE docs (docid TEXT PRIMARY KEY, sha256 TEXT NOT NULL) WITHOUT ROWID")
        for source in sorted((StoragePath(root) / "data/*.parquet").glob(), key=str):
            target = output / (Path(str(source)).stem + ".jsonl")
            with source.open("rb") as stream, target.open("w") as writer:
                for batch in pq.ParquetFile(stream).iter_batches(batch_size=256, columns=["docid", "text"]):
                    for row in batch.to_pylist():
                        docid, text = row["docid"], row["text"]
                        if not isinstance(docid, str) or not docid or not isinstance(text, str) or not text.strip():
                            raise ValueError("Corpus document requires a nonempty string identifier and body")
                        connection.execute(
                            "INSERT INTO docs VALUES (?, ?)", (docid, hashlib.sha256(text.encode()).hexdigest())
                        )
                        writer.write(json.dumps({"id": docid, "contents": text}, ensure_ascii=False) + "\n")
                        count += 1
            connection.commit()
            logger.info("Prepared corpus shard %s, cumulative documents=%d", target.name, count)
    if count != EXPECTED_DOCUMENTS:
        raise ValueError(f"Corpus contains {count} documents; expected {EXPECTED_DOCUMENTS}")
    return count


def build_index(config: IndexConfig) -> BM25Index:
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("The full corpus and index must stay inside the regional Iris job")
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt_uri, config.stage_receipt_sha256))
    root = StoragePath(staged.path)
    manifest = json.loads(pinned_document(str(root / "source-manifest.json"), staged.source_manifest_sha256))
    corpus = next(asset for asset in manifest if asset["dataset_id"] == CORPUS)
    if corpus["revision"] != CORPUS_REVISION:
        raise ValueError("Staging receipt differs from the approved immutable corpus")
    work = Path(config.workdir)
    work.mkdir(parents=True, exist_ok=False)
    collection, index, database = work / "collection", work / "bm25", work / "document-hashes.sqlite"
    count = corpus_collection(staged.source_roots[CORPUS], collection, database)
    subprocess.run(
        [
            config.runtime_python,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(collection),
            "--index",
            str(index),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            str(config.threads),
            "--storePositions",
            "--storeDocvectors",
            "--storeRaw",
        ],
        check=True,
    )
    subprocess.run(
        [
            config.runtime_python,
            "-m",
            "experiments.post_training.task_curation.datasets.openresearcher.verify_index",
            "--index",
            str(index),
            "--database",
            str(database),
            "--count",
            str(count),
            "--output",
            str(work / "verification.json"),
        ],
        check=True,
    )
    output = StoragePath(config.output_path)
    files, total_bytes = [], 0
    for path in sorted(index.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(index).as_posix()
        expected = file_sha256(StoragePath(str(path)))
        destination = output / "bm25" / relative
        with path.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if file_sha256(destination) != expected:
            raise ValueError(f"Regional index upload differs: {relative}")
        files.append({"path": relative, "sha256": expected, "bytes": path.stat().st_size})
        total_bytes += path.stat().st_size
    encoded = (json.dumps(files, indent=2) + "\n").encode()
    (output / "index-manifest.json").write_bytes(encoded)
    (output / "verification.json").write_bytes((work / "verification.json").read_bytes())
    return BM25Index(
        path=config.output_path,
        corpus_revision=CORPUS_REVISION,
        corpus_root=staged.source_roots[CORPUS],
        stage_receipt_sha256=config.stage_receipt_sha256,
        document_count=count,
        content_mismatches=0,
        index_manifest_sha256=hashlib.sha256(encoded).hexdigest(),
        index_bytes=total_bytes,
        pyserini_version=PYSERINI_VERSION,
        java_sha256=JAVA_SHA256,
    )


def index_config(
    uri: str, sha256: str, workdir: str, threads: int, runtime_python: str, context: StepContext
) -> IndexConfig:
    return IndexConfig(uri, sha256, workdir, threads, runtime_python, context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", required=True)
    parser.add_argument("--stage-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/bm25-index",
            version=args.artifact_version,
            artifact_type=BM25Index,
            run=build_index,
            build_config=partial(
                index_config,
                args.stage_receipt,
                args.stage_receipt_sha256,
                args.workdir,
                args.threads,
                args.runtime_python,
            ),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        with local_client():
            result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))
