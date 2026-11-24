# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed assigned corpus shards on one GPU inside the pinned dense-embedding runtime.

Documents embed whole (empty passage prefix) with last-token pooling and left
padding, following the pinned encoder's documented usage. Shards process in
length-sorted batches to bound padding waste, and vectors return to original
row order before writing. This module must stay importable without the Marin
workspace: the embedding runtime venv only carries torch, transformers,
pyarrow and numpy.
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


def embed_shard(
    model: torch.nn.Module,
    tokenizer,
    shard: Path,
    vectors_path: Path,
    docids_path: Path,
    batch_size: int,
    max_tokens: int,
) -> dict:
    """Stream one corpus shard, writing raw fp16 vectors and the ordered docids.

    Completed parts are resumed after preemption: an existing pair whose sizes
    and row counts agree is adopted instead of recomputed, and the manifest
    digests are recomputed from the bytes on disk.
    """
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    docids_path.parent.mkdir(parents=True, exist_ok=True)
    if vectors_path.exists() and docids_path.exists():
        with pq.ParquetFile(docids_path) as existing:
            docid_rows = existing.metadata.num_rows
        expected_bytes = docid_rows * EMBEDDING_DIM * 2
        if docid_rows > 0 and vectors_path.stat().st_size == expected_bytes:
            logger.info("Resuming %s: adopting %d embedded documents", shard.name, docid_rows)
            entry = {
                "shard": shard.name,
                "vectors": vectors_path.name,
                "docids": docids_path.name,
                "documents": docid_rows,
                "vectors_sha256": sha256_file(vectors_path),
                "vectors_bytes": vectors_path.stat().st_size,
                "docids_sha256": sha256_file(docids_path),
            }
            receipts = vectors_path.parent.parent / "part-receipts"
            receipts.mkdir(parents=True, exist_ok=True)
            (receipts / (Path(shard).stem + ".json")).write_text(json.dumps(entry))
            return entry
        if vectors_path.exists():
            vectors_path.unlink()
        if docids_path.exists():
            docids_path.unlink()
    docids: list[str] = []
    texts: list[str] = []
    with pq.ParquetFile(shard) as parquet:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["docid", "text"]):
            for row in batch.to_pylist():
                docids.append(row["docid"])
                texts.append(row["text"])
    documents = len(docids)
    # Length-sorted batches minimize padding waste; vectors scatter back to the
    # corpus row order before writing so the docid mapping never changes.
    order = sorted(range(documents), key=lambda index: len(texts[index]))
    vectors = np.empty((documents, model.config.hidden_size), dtype=np.float16)
    embedded = 0
    for start in range(0, documents, batch_size):
        rows = order[start : start + batch_size]
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
            logger.info("Embedded %s: %d/%d documents", shard.name, embedded, documents)
    vectors.tofile(vectors_path)
    table = pa.Table.from_arrays([pa.array(docids)], names=["docid"])
    with docids_path.open("wb") as writer:
        pq.write_table(table, writer, compression="zstd")
    entry = {
        "shard": shard.name,
        "vectors": vectors_path.name,
        "docids": docids_path.name,
        "documents": documents,
        "vectors_sha256": sha256_file(vectors_path),
        "vectors_bytes": vectors_path.stat().st_size,
        "docids_sha256": sha256_file(docids_path),
    }
    # The per-part receipt lets the orchestrator upload and resume this part
    # independently of the rest of the worker's assignment.
    (vectors_path.parent.parent / "part-receipts").mkdir(parents=True, exist_ok=True)
    (vectors_path.parent.parent / "part-receipts" / (Path(shard).stem + ".json")).write_text(json.dumps(entry))
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
        entry = embed_shard(
            model,
            tokenizer,
            Path(shard),
            output / "vectors" / (Path(shard).stem + ".f16"),
            output / "docids" / (Path(shard).stem + ".parquet"),
            args.batch_size,
            plan["max_tokens"],
        )
        entries.append(entry)
        logger.info("Embedded shard %s: %d documents", entry["shard"], entry["documents"])
    (output / f"worker-manifest-{plan['worker']}.json").write_text(json.dumps(entries, indent=2) + "\n")
    torch.cuda.empty_cache()
    sys.stdout.write(json.dumps({"worker": plan["worker"], "entries": len(entries)}) + "\n")


if __name__ == "__main__":
    main()
