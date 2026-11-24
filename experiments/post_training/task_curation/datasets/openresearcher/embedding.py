# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed the complete approved corpus with the pinned, permissively licensed encoder.

The orchestrator runs in the Iris job's workspace environment; GPU work runs in
the pinned dense-embedding venv through ``embedding_worker``. Large corpus and
vector artifacts stay in-region.
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq
from finestore.rl.licenses import LicenseReceipt, require_license
from huggingface_hub import snapshot_download
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, run
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import file_sha256
from transformers import AutoTokenizer

from experiments.post_training.task_curation.datasets.openresearcher.assets import (
    CORPUS,
    CORPUS_REVISION,
    REGIONAL_PREFIX,
    RegionalInputs,
    pinned_document,
)
from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import (
    EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    MODEL_FILES,
)
from experiments.post_training.task_curation.local import artifact_prefix

EMBEDDING_MODEL_LICENSE = Path(__file__).with_name("embedding_model_license.json")
EXPECTED_DOCUMENTS = 14_878_084
logger = logging.getLogger(__name__)


class EmbeddingShards(Artifact):
    corpus_revision: str
    corpus_root: str
    stage_receipt_sha256: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dim: int
    max_embed_tokens: int
    document_count: int
    shard_manifest_sha256: str
    model_manifest_sha256: str
    license_receipt_sha256: str
    iris_task_id: str


@dataclass(frozen=True)
class EmbeddingConfig:
    stage_receipt_uri: str
    stage_receipt_sha256: str
    workdir: str
    embed_python: str
    batch_size: int
    max_embed_tokens: int
    output_path: str


def validate_model_license() -> LicenseReceipt:
    """Fail closed unless the pinned encoder's recorded license is permissive."""
    receipt = LicenseReceipt.model_validate_json(EMBEDDING_MODEL_LICENSE.read_bytes())
    require_license(receipt, EMBEDDING_MODEL, EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION, "train", ban=None)
    return receipt


def corpus_shards(staged: RegionalInputs) -> list[tuple[str, str, int]]:
    """Return the approved corpus shard sources with their pinned LFS digests."""
    root = StoragePath(staged.path)
    manifest = json.loads(pinned_document(str(root / "source-manifest.json"), staged.source_manifest_sha256))
    asset = next(asset for asset in manifest if asset["dataset_id"] == CORPUS)
    if asset["revision"] != CORPUS_REVISION:
        raise ValueError("Embedding input differs from the approved immutable corpus")
    files = [file for file in asset["files"] if file["path"].startswith("data/")]
    return [
        (str(StoragePath(staged.source_roots[CORPUS]) / file["path"]), file["sha256"], file["bytes"])
        for file in sorted(files, key=lambda file: file["path"])
    ]


def stage_corpus(staged_shards: list[tuple[str, str, int]], work: Path) -> list[Path]:
    """Copy the digest-pinned corpus shards into the job's local work directory."""
    corpus = work / "corpus"
    corpus.mkdir(parents=True, exist_ok=False)
    local = []
    for source, expected_sha, expected_bytes in staged_shards:
        target = corpus / Path(source).name
        with StoragePath(source).open("rb") as reader, target.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        if target.stat().st_size != expected_bytes or file_sha256(StoragePath(str(target))) != expected_sha:
            raise ValueError(f"Staged corpus shard differs from the pinned LFS digest: {target.name}")
        local.append(target)
        logger.info("Staged corpus shard %s", target.name)
    return local


def download_model(work: Path) -> list[dict]:
    directory = work / "model"
    snapshot_download(
        EMBEDDING_MODEL,
        revision=EMBEDDING_MODEL_REVISION,
        local_dir=directory,
        allow_patterns=list(MODEL_FILES),
    )
    manifest = []
    for name in MODEL_FILES:
        target = directory / name
        if not target.is_file():
            raise ValueError(f"Pinned embedding model file is missing: {name}")
        manifest.append(
            {
                "path": name,
                "sha256": file_sha256(StoragePath(str(target))),
                "bytes": target.stat().st_size,
            }
        )
    return manifest


def visible_gpus(embed_python: str) -> int:
    output = subprocess.check_output([embed_python, "-c", "import torch; print(torch.cuda.device_count())"], text=True)
    count = int(output.strip())
    if count < 1:
        raise RuntimeError("The embedding job requires at least one visible GPU")
    return count


def verify_entries(entries: list[dict], shards: list[Path]) -> int:
    """Require one manifest entry per corpus shard, with internally consistent digests."""
    expected = sorted(shard.name for shard in shards)
    if sorted(entry["shard"] for entry in entries) != expected:
        raise ValueError("Embedding produced a different shard set than the approved corpus")
    total = 0
    for entry in entries:
        rows = entry["vectors_bytes"] // (EMBEDDING_DIM * 2)
        if rows != entry["documents"] or rows <= 0:
            raise ValueError(f"Vector shard row count disagrees with its manifest: {entry['vectors']}")
        total += entry["documents"]
    if total != EXPECTED_DOCUMENTS:
        raise ValueError(f"Corpus contains {total} embedded documents; expected {EXPECTED_DOCUMENTS}")
    return total


def run_workers(
    config: EmbeddingConfig, shards: list[Path], model_manifest: list[dict], work: Path, output: StoragePath
) -> list[dict]:
    """Embed every corpus shard once across the visible GPUs, then verify each part."""
    staged_output = work / "embedding-output"
    gpus = visible_gpus(config.embed_python)
    logger.info("Embedding %d corpus shards across %d GPUs", len(shards), gpus)
    workers = []
    for worker in range(min(gpus, len(shards))):
        plan = work / f"plan-{worker}.json"
        # Round-robin shard assignment keeps part naming derived from the corpus
        # shard names, so processing order never affects the artifact layout.
        plan.write_text(
            json.dumps(
                {
                    "worker": worker,
                    "shards": [str(shard) for shard in shards[worker::gpus]],
                    "model_manifest": model_manifest,
                    "max_tokens": config.max_embed_tokens,
                }
            )
        )
        workers.append(
            subprocess.Popen(
                [
                    config.embed_python,
                    "-m",
                    "experiments.post_training.task_curation.datasets.openresearcher.embedding_worker",
                    "--plan",
                    str(plan),
                    "--model-dir",
                    str(work / "model"),
                    "--output",
                    str(staged_output),
                    "--batch-size",
                    str(config.batch_size),
                ],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(worker),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            )
        )
    failures = [number for number, worker in enumerate(workers) if worker.wait() != 0]
    if failures:
        raise RuntimeError(f"Embedding workers failed: {failures}")
    entries = []
    for worker in range(len(workers)):
        entries.extend(json.loads((staged_output / f"worker-manifest-{worker}.json").read_text()))
    entries.sort(key=lambda entry: entry["shard"])
    total = verify_entries(entries, shards)
    logger.info("Embedded %d documents; uploading vector shards", total)
    for relative in ("vectors", "docids"):
        destination = output / relative
        for source in sorted((staged_output / relative).iterdir()):
            with source.open("rb") as reader, (destination / source.name).open("wb", auto_mkdir=True) as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            if file_sha256(StoragePath(str(destination / source.name))) != file_sha256(StoragePath(str(source))):
                raise ValueError(f"Regional embedding upload differs: {source.name}")
    manifest = json.dumps(entries, indent=2) + "\n"
    (output / "shard-manifest.json").write_text(manifest)
    return entries


def run_embedding(config: EmbeddingConfig) -> EmbeddingShards:
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Corpus embedding must run inside the regional Iris GPU job")
    if config.max_embed_tokens < 128 or config.max_embed_tokens > 32768:
        raise ValueError("The document token cap must stay within the encoder's context budget")
    validate_model_license()
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt_uri, config.stage_receipt_sha256))
    work = Path(config.workdir)
    work.mkdir(parents=True, exist_ok=False)
    shards = stage_corpus(corpus_shards(staged), work)
    model_manifest = download_model(work)
    output = StoragePath(config.output_path)
    entries = run_workers(config, shards, model_manifest, work, output)
    model_manifest_json = json.dumps(model_manifest, indent=2) + "\n"
    (output / "model-manifest.json").write_text(model_manifest_json)
    (output / "license-receipt.json").write_text(EMBEDDING_MODEL_LICENSE.read_text())
    return EmbeddingShards(
        path=config.output_path,
        corpus_revision=CORPUS_REVISION,
        corpus_root=staged.source_roots[CORPUS],
        stage_receipt_sha256=config.stage_receipt_sha256,
        embedding_model=EMBEDDING_MODEL,
        embedding_model_revision=EMBEDDING_MODEL_REVISION,
        embedding_dim=EMBEDDING_DIM,
        max_embed_tokens=config.max_embed_tokens,
        document_count=sum(entry["documents"] for entry in entries),
        shard_manifest_sha256=hashlib.sha256((output / "shard-manifest.json").read_bytes()).hexdigest(),
        model_manifest_sha256=hashlib.sha256(model_manifest_json.encode()).hexdigest(),
        license_receipt_sha256=file_sha256(StoragePath(str(EMBEDDING_MODEL_LICENSE))),
        iris_task_id=os.environ["IRIS_TASK_ID"],
    )


def measure_token_stats(config: EmbeddingConfig) -> None:
    """Measure the corpus token distribution under the pinned tokenizer (CPU only).

    Fixes the document token cap before the GPU job: the stats record total
    tokens and full-coverage counts at each candidate cap, and the chosen cap
    is then recorded in the embedding artifact and retrieval state.
    """
    if not os.environ.get("IRIS_TASK_ID"):
        raise RuntimeError("Corpus statistics must be measured inside the regional Iris CPU job")
    validate_model_license()
    staged = RegionalInputs.model_validate_json(pinned_document(config.stage_receipt_uri, config.stage_receipt_sha256))
    work = Path(config.workdir)
    work.mkdir(parents=True, exist_ok=False)
    shards = stage_corpus(corpus_shards(staged), work)
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL, revision=EMBEDDING_MODEL_REVISION, trust_remote_code=False
    )
    caps = (1024, 2048, 4096, 8192, 16384, 32768)
    documents, total_tokens, total_chars = 0, 0, 0
    per_cap = {cap: {"tokens": 0, "documents_fully_embedded": 0, "documents_truncated": 0} for cap in caps}
    percentiles_source: list[int] = []
    for shard in shards:
        with pq.ParquetFile(shard) as parquet:
            for batch in parquet.iter_batches(batch_size=1024, columns=["docid", "text"]):
                rows = batch.to_pylist()
                # Batch encoding parallelizes inside the fast tokenizer.
                lengths = [
                    len(ids) for ids in tokenizer([row["text"] for row in rows], add_special_tokens=False)["input_ids"]
                ]
                documents += len(rows)
                total_tokens += sum(lengths)
                total_chars += sum(len(row["text"]) for row in rows)
                percentiles_source.extend(lengths)
                for cap in caps:
                    bucket = per_cap[cap]
                    for length in lengths:
                        bucket["tokens"] += min(length, cap)
                        if length <= cap:
                            bucket["documents_fully_embedded"] += 1
                        else:
                            bucket["documents_truncated"] += 1
        logger.info("Measured shard %s: %d documents so far", shard.name, documents)
    if documents != EXPECTED_DOCUMENTS:
        raise ValueError(f"Corpus contains {documents} documents; expected {EXPECTED_DOCUMENTS}")
    percentiles_source.sort()

    def percentile(fraction: float) -> int:
        return percentiles_source[min(int(fraction * documents), documents - 1)]

    stats = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "documents": documents,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "token_percentiles": {str(p): percentile(p) for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "caps": {str(cap): per_cap[cap] for cap in caps},
    }
    (StoragePath(config.output_path) / "token-stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    logger.info("Token stats: %s", json.dumps(stats))


def embedding_config(
    uri: str, sha256: str, workdir: str, embed_python: str, batch_size: int, max_embed_tokens: int, context: StepContext
) -> EmbeddingConfig:
    return EmbeddingConfig(uri, sha256, workdir, embed_python, batch_size, max_embed_tokens, context.output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", required=True)
    parser.add_argument("--stage-receipt-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--embed-python", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-embed-tokens", type=int, default=EMBEDDING_MAX_TOKENS)
    parser.add_argument("--mode", choices=("embed", "stats"), default="embed")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    StoragePath(args.artifact_root).relative_to(StoragePath(REGIONAL_PREFIX))
    configure_coreweave_s3()
    if args.mode == "stats":
        if not args.run:
            raise SystemExit("Token statistics run in-place; pass --run")
        measure_token_stats(
            EmbeddingConfig(
                args.stage_receipt,
                args.stage_receipt_sha256,
                args.workdir,
                args.embed_python,
                args.batch_size,
                args.max_embed_tokens,
                args.output,
            )
        )
        return
    with artifact_prefix(args.artifact_root):
        step = ArtifactStep(
            name="rl-data/openresearcher/embedding-shards",
            version=args.artifact_version,
            artifact_type=EmbeddingShards,
            run=run_embedding,
            build_config=partial(
                embedding_config,
                args.stage_receipt,
                args.stage_receipt_sha256,
                args.workdir,
                args.embed_python,
                args.batch_size,
                args.max_embed_tokens,
            ),
        )
        if not args.run:
            print(json.dumps({"path": step.path(), "fingerprint": step.fingerprint()}))
            return
        result = run(step)[0]
        StoragePath(args.output).write_text(result.model_dump_json(indent=2))
