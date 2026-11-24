# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed assigned corpus shards on one GPU inside the pinned dense-embedding runtime.

Documents embed whole (empty passage prefix) with last-token pooling and left
padding, following the pinned encoder's documented usage. Each shard is
processed in fixed document-order chunks so preempted attempts resume from the
last completed chunk on any node. Shards process in length-sorted batches to
bound padding waste, and vectors return to original row order before writing.
This module must stay importable without the Marin workspace: the embedding
runtime venv only carries torch, transformers, pyarrow and numpy.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoModel, AutoTokenizer

from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import EMBEDDING_DIM

logger = logging.getLogger(__name__)

CHUNK_DOCS = 6_400


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_dir: Path, model_manifest: list[dict]) -> None:
    """Require every pinned model file to match the artifact manifest before loading."""
    for entry in model_manifest:
        target = model_dir / entry["path"]
        if not target.is_file() or target.stat().st_size != entry["bytes"]:
            raise ValueError(f"Pinned embedding model file differs: {entry['path']}")
        if sha256_file(target) != entry["sha256"]:
            raise ValueError(f"Pinned embedding model digest differs: {entry['path']}")


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def shard_rows(shard: Path) -> int:
    with pq.ParquetFile(shard) as parquet:
        return parquet.metadata.num_rows


def read_rows(shard: Path, start: int, end: int) -> tuple[list[str], list[str]]:
    """Read one half-open row range of a corpus shard in corpus order."""
    docids: list[str] = []
    texts: list[str] = []
    position = 0
    with pq.ParquetFile(shard) as parquet:
        for batch in parquet.iter_batches(batch_size=1024, columns=["docid", "text"]):
            rows = batch.to_pylist()
            overlap_start = max(start - position, 0)
            overlap_end = min(end - position, len(rows))
            if overlap_end > overlap_start:
                docids.extend(row["docid"] for row in rows[overlap_start:overlap_end])
                texts.extend(row["text"] for row in rows[overlap_start:overlap_end])
            position += len(rows)
            if position >= end:
                break
    return docids, texts


def write_part(
    output: Path,
    shard: Path,
    chunk: int,
    docids: list[str],
    vectors: np.ndarray,
) -> dict:
    stem = shard.stem
    vectors_path = output / "vectors" / f"{stem}-chunk-{chunk:04d}.f16"
    docids_path = output / "docids" / f"{stem}-chunk-{chunk:04d}.parquet"
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    docids_path.parent.mkdir(parents=True, exist_ok=True)
    vectors.tofile(vectors_path)
    table = pa.Table.from_arrays([pa.array(docids)], names=["docid"])
    with docids_path.open("wb") as writer:
        pq.write_table(table, writer, compression="zstd")
    entry = {
        "shard": shard.name,
        "chunk": chunk,
        "vectors": vectors_path.name,
        "docids": docids_path.name,
        "documents": len(docids),
        "vectors_sha256": sha256_file(vectors_path),
        "vectors_bytes": vectors_path.stat().st_size,
        "docids_sha256": sha256_file(docids_path),
    }
    receipts = output / "part-receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / f"{stem}-chunk-{chunk:04d}.json").write_text(json.dumps(entry))
    return entry


def adopt_part(output: Path, vectors_path: Path, docids_path: Path, documents: int) -> dict | None:
    """Adopt a completed part pair from a previous attempt on this node."""
    if not (vectors_path.exists() and docids_path.exists()):
        return None
    with pq.ParquetFile(docids_path) as existing:
        docid_rows = existing.metadata.num_rows
    expected_bytes = docid_rows * EMBEDDING_DIM * 2
    if docid_rows != documents or vectors_path.stat().st_size != expected_bytes:
        return None
    return {
        "vectors_sha256": sha256_file(vectors_path),
        "vectors_bytes": vectors_path.stat().st_size,
        "docids_sha256": sha256_file(docids_path),
    }


def embed_chunk(
    model: torch.nn.Module,
    tokenizer,
    shard: Path,
    chunk: int,
    output: Path,
    batch_size: int,
    max_tokens: int,
) -> dict:
    """Embed rows [chunk*CHUNK_DOCS, ...) of one shard, resuming completed chunks."""
    total_rows = shard_rows(shard)
    start = chunk * CHUNK_DOCS
    end = min(start + CHUNK_DOCS, total_rows)
    if start >= total_rows:
        raise ValueError(f"Chunk {chunk} of {shard.name} is outside the corpus shard")
    stem = shard.stem
    vectors_path = output / "vectors" / f"{stem}-chunk-{chunk:04d}.f16"
    docids_path = output / "docids" / f"{stem}-chunk-{chunk:04d}.parquet"
    adopted = adopt_part(output, vectors_path, docids_path, end - start)
    if adopted is not None:
        logger.info("Resuming %s chunk %d: adopting %d embedded documents", shard.name, chunk, end - start)
        entry = {
            "shard": shard.name,
            "chunk": chunk,
            "vectors": vectors_path.name,
            "docids": docids_path.name,
            "documents": end - start,
            **adopted,
        }
        receipts = output / "part-receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        (receipts / f"{stem}-chunk-{chunk:04d}.json").write_text(json.dumps(entry))
        return entry
    docids, texts = read_rows(shard, start, end)
    # Length-sorted batches minimize padding waste; vectors scatter back to the
    # corpus row order before writing so the docid mapping never changes.
    order = sorted(range(len(docids)), key=lambda index: len(texts[index]))
    vectors = np.empty((len(docids), model.config.hidden_size), dtype=np.float16)
    embedded = 0
    for batch_start in range(0, len(docids), batch_size):
        rows = order[batch_start : batch_start + batch_size]
        encoded = tokenizer(
            [texts[row] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state[:, -1]
        vectors[rows] = normalize_rows(hidden.to(torch.float32).cpu().numpy()).astype(np.float16)
        embedded += len(rows)
        if embedded % (batch_size * 100) < len(rows):
            logger.info("Embedded %s chunk %d: %d/%d documents", shard.name, chunk, embedded, len(docids))
    entry = write_part(output, shard, chunk, docids, vectors)
    logger.info("Embedded %s chunk %d: %d documents", shard.name, chunk, entry["documents"])
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON file with this worker's corpus shard assignments")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    plan = json.loads(Path(args.plan).read_text())
    model_dir, output = Path(args.model_dir), Path(args.output)
    verify_model(model_dir, plan["model_manifest"])
    device = plan.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The embedding worker requires a CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side="left")
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.bfloat16, local_files_only=True)
    model.to(device).eval()
    if model.config.hidden_size != EMBEDDING_DIM:
        raise ValueError(
            f"Pinned encoder hidden size {model.config.hidden_size} differs from the dense contract {EMBEDDING_DIM}"
        )
    entries = []
    for shard in plan["shards"]:
        shard_path = Path(shard)
        total_rows = shard_rows(shard_path)
        for chunk in range((total_rows + CHUNK_DOCS - 1) // CHUNK_DOCS):
            entries.append(embed_chunk(model, tokenizer, shard_path, chunk, output, args.batch_size, plan["max_tokens"]))
    (output / f"worker-manifest-{plan['worker']}.json").write_text(json.dumps(entries, indent=2) + "\n")
    if device == "cuda":
        torch.cuda.empty_cache()
    sys.stdout.write(json.dumps({"worker": plan["worker"], "entries": len(entries)}) + "\n")


if __name__ == "__main__":
    main()
