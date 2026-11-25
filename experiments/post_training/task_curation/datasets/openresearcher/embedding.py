# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed the complete approved corpus with the pinned, permissively licensed encoder.

The orchestrator runs in the Iris job's workspace environment; GPU work runs in
the pinned dense-embedding venv through ``embedding_worker``. Large corpus and
vector artifacts stay in-region.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
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
    """Copy the digest-pinned corpus shards into the job's local work directory.

    An existing staged set with matching sizes (verified once at write time)
    is adopted so preempted attempts resume instead of recopying.
    """
    corpus = work / "corpus"
    if corpus.is_dir():
        existing = {path.name: path.stat().st_size for path in corpus.iterdir() if path.is_file()}
        if {
            Path(source).name: expected_bytes for source, _, expected_bytes in staged_shards
        }.items() <= existing.items():
            logger.info("Adopting %d previously staged corpus shards", len(existing))
            return [corpus / Path(source).name for source, _, _ in staged_shards]
        shutil.rmtree(corpus)
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


def verify_entries(entries: list[dict], shards: list[Path], expected_documents: int = EXPECTED_DOCUMENTS) -> int:
    """Require one manifest entry per corpus-shard chunk, with consistent digests.

    Chunk coverage must be contiguous from zero within every shard, so the
    dense index can concatenate chunk vectors in corpus order.
    """
    by_shard: dict[str, list[dict]] = {}
    for entry in entries:
        by_shard.setdefault(entry["shard"], []).append(entry)
    if sorted(by_shard) != sorted(shard.name for shard in shards):
        raise ValueError("Embedding produced a different shard set than the approved corpus")
    total = 0
    for shard_name, shard_entries in by_shard.items():
        chunk_indices = sorted(entry["chunk"] for entry in shard_entries)
        if chunk_indices != list(range(len(chunk_indices))):
            raise ValueError(f"Chunk coverage is not contiguous for {shard_name}: {chunk_indices}")
        for entry in shard_entries:
            rows = entry["vectors_bytes"] // (EMBEDDING_DIM * 2)
            if rows != entry["documents"] or rows <= 0:
                raise ValueError(f"Vector chunk row count disagrees with its manifest: {entry['vectors']}")
            total += entry["documents"]
    if total != expected_documents:
        raise ValueError(f"Corpus contains {total} embedded documents; expected {expected_documents}")
    return total


def adopt_regional_parts(output: StoragePath, staged_output: Path) -> set[str]:
    """Adopt parts a previous attempt already uploaded, so restarts resume anywhere."""
    receipts = staged_output / "part-receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    adopted: set[str] = set()
    regional_receipts = output / "part-receipts"
    if not regional_receipts.exists():
        return adopted
    for receipt_path in sorted((regional_receipts / "*.json").glob(), key=str):
        entry = json.loads(receipt_path.read_text())
        vectors_target = staged_output / "vectors" / entry["vectors"]
        docids_target = staged_output / "docids" / entry["docids"]
        if vectors_target.exists() and docids_target.exists():
            with pq.ParquetFile(docids_target) as existing:
                docid_rows = existing.metadata.num_rows
            if docid_rows == entry["documents"] and vectors_target.stat().st_size == entry["vectors_bytes"]:
                adopted.add(entry["shard"])
                continue
        for relative in (f"vectors/{entry['vectors']}", f"docids/{entry['docids']}"):
            target = staged_output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with (output / relative).open("rb") as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        vectors_target = staged_output / "vectors" / entry["vectors"]
        docids_target = staged_output / "docids" / entry["docids"]
        if (
            vectors_target.stat().st_size != entry["vectors_bytes"]
            or file_sha256(StoragePath(str(vectors_target))) != entry["vectors_sha256"]
            or file_sha256(StoragePath(str(docids_target))) != entry["docids_sha256"]
        ):
            raise ValueError(f"Regional part differs from its receipt: {entry['shard']}")
        (receipts / receipt_path.name).write_text(receipt_path.read_text())
        adopted.add(entry["shard"])
    if adopted:
        logger.info("Adopted %d previously uploaded parts", len(adopted))
    return adopted


def upload_completed_parts(output: StoragePath, staged_output: Path, uploaded: set[str], stop: threading.Event) -> None:
    """Upload finished parts to the region as workers complete them."""
    receipts = staged_output / "part-receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    while True:
        for receipt_path in sorted(receipts.glob("*.json")):
            name = receipt_path.stem
            if name in uploaded:
                continue
            entry = json.loads(receipt_path.read_text())
            for relative, expected_sha, expected_bytes in (
                (f"vectors/{entry['vectors']}", entry["vectors_sha256"], entry["vectors_bytes"]),
                (f"docids/{entry['docids']}", entry["docids_sha256"], None),
            ):
                source = staged_output / relative
                if not source.exists():
                    break
                destination = output / relative
                with source.open("rb") as reader, destination.open("wb", auto_mkdir=True) as writer:
                    shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
                actual = file_sha256(StoragePath(str(destination)))
                if actual != expected_sha or (expected_bytes is not None and destination.size() != expected_bytes):
                    raise ValueError(f"Regional part upload differs: {relative}")
            (output / "part-receipts" / receipt_path.name).write_text(receipt_path.read_text())
            uploaded.add(name)
            logger.info("Uploaded part %s", name)
        pending = {path.stem for path in receipts.glob("*.json")} - uploaded
        if stop.is_set() and not pending:
            break
        stop.wait(timeout=20)


def preflight(embed_python: str) -> None:
    """Fail fast with a clear error when the pinned runtime cannot reach the GPUs."""
    subprocess.run(
        [
            embed_python,
            "-c",
            "import torch; assert torch.cuda.is_available(), 'cuda unavailable'; "
            "x = torch.zeros(8, device='cuda'); assert int(x.sum()) == 0; import transformers; "
            "print('preflight ok', torch.cuda.device_count())",
        ],
        check=True,
    )


def run_workers(
    config: EmbeddingConfig,
    shards: list[Path],
    model_manifest: list[dict],
    work: Path,
    output: StoragePath,
    *,
    gpu_count: int | None = None,
    device: str = "cuda",
    expected_documents: int = EXPECTED_DOCUMENTS,
) -> list[dict]:
    """Embed every corpus shard once across the visible GPUs, then verify each part."""
    if device == "cuda":
        preflight(config.embed_python)
    staged_output = work / "embedding-output"
    staged_output.mkdir(parents=True, exist_ok=False)
    uploaded = adopt_regional_parts(output, staged_output)
    gpus = gpu_count if gpu_count is not None else visible_gpus(config.embed_python)
    logger.info("Embedding %d corpus shards across %d %s workers", len(shards), gpus, device)
    workers = []
    worker_logs = []
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
                    "device": device,
                }
            )
        )
        # The worker log handle lives as long as the subprocess; closed after wait().
        worker_logs.append(open(work / f"worker-{worker}.log", "wb"))  # noqa: SIM115
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
                ],
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": str(worker),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
                stdout=worker_logs[-1],
                stderr=subprocess.STDOUT,
            )
        )
    stop = threading.Event()
    uploader_error: list[BaseException] = []

    def upload_until_done() -> None:
        try:
            upload_completed_parts(output, staged_output, uploaded, stop)
        except BaseException as error:  # surfaced after the workers finish
            uploader_error.append(error)

    uploader = threading.Thread(target=upload_until_done, daemon=True)
    uploader.start()
    try:
        failures = [number for number, worker in enumerate(workers) if worker.wait() != 0]
        stop.set()
        uploader.join()
    finally:
        for handle in worker_logs:
            handle.close()
    if uploader_error:
        raise RuntimeError(f"Regional part upload failed: {uploader_error[0]!r}") from uploader_error[0]
    if failures:
        tails = []
        for number in failures:
            log = (work / f"worker-{number}.log").read_text(errors="replace")
            tails.append(f"worker {number}: ...{log[-2000:]}")
        raise RuntimeError(f"Embedding workers failed: {failures}; tails: {tails}")
    entries = []
    for worker in range(len(workers)):
        entries.extend(json.loads((staged_output / f"worker-manifest-{worker}.json").read_text()))
    entries.sort(key=lambda entry: (entry["shard"], entry["chunk"]))
    total = verify_entries(entries, shards, expected_documents)
    logger.info("Embedded %d documents; verifying regional uploads", total)
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
    is then recorded in the embedding artifact and retrieval state. Per-shard
    length checkpoints in the region let preempted attempts resume.
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
    checkpoint_dir = StoragePath(config.output_path) / "checkpoints"
    measured: dict[str, np.ndarray] = {}
    for shard in shards:
        checkpoint = checkpoint_dir / (shard.stem + ".npz")
        if checkpoint.exists():
            with checkpoint.open("rb") as reader:
                lengths = np.load(io.BytesIO(reader.read()))["lengths"]
            logger.info("Resuming measured shard %s: %d documents", shard.name, len(lengths))
        else:
            computed: list[int] = []
            with pq.ParquetFile(shard) as parquet:
                for batch in parquet.iter_batches(batch_size=1024, columns=["docid", "text"]):
                    rows = batch.to_pylist()
                    # Batch encoding parallelizes inside the fast tokenizer.
                    computed.extend(
                        len(ids)
                        for ids in tokenizer([row["text"] for row in rows], add_special_tokens=False)["input_ids"]
                    )
            lengths = np.asarray(computed, dtype=np.int32)
            payload = io.BytesIO()
            np.savez(payload, lengths=lengths)
            checkpoint.write_bytes(payload.getvalue())
            logger.info("Measured shard %s: %d documents", shard.name, len(lengths))
        measured[shard.stem] = lengths
    all_lengths = np.concatenate([measured[key] for key in sorted(measured)])
    documents = int(all_lengths.size)
    if documents != EXPECTED_DOCUMENTS:
        raise ValueError(f"Corpus contains {documents} documents; expected {EXPECTED_DOCUMENTS}")
    stats = {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_model_revision": EMBEDDING_MODEL_REVISION,
        "documents": documents,
        "total_tokens": int(all_lengths.sum()),
        "token_percentiles": {str(p): int(np.percentile(all_lengths, p * 100)) for p in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "caps": {
            str(cap): {
                "tokens": int(np.minimum(all_lengths, cap).sum()),
                "documents_fully_embedded": int((all_lengths <= cap).sum()),
                "documents_truncated": int((all_lengths > cap).sum()),
            }
            for cap in caps
        },
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


if __name__ == "__main__":
    main()
