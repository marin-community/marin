# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed assigned corpus shards on one GPU inside the pinned dense-embedding runtime.

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

from experiments.post_training.task_curation.datasets.openresearcher.dense_contract import EMBEDDING_MAX_TOKENS

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
) -> dict:
    """Stream one corpus shard, writing raw fp16 vectors and the ordered docids."""
    if vectors_path.exists() or docids_path.exists():
        raise ValueError(f"Embedding outputs already exist for {shard.name}")
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    docids_path.parent.mkdir(parents=True, exist_ok=True)
    documents = 0
    with vectors_path.open("wb") as vectors_writer:
        docids: list[str] = []
        with pq.ParquetFile(shard) as parquet:
            for batch in parquet.iter_batches(batch_size=batch_size, columns=["docid", "text"]):
                rows = batch.to_pylist()
                encoded = tokenizer(
                    [row["text"] for row in rows],
                    padding=True,
                    truncation=True,
                    max_length=EMBEDDING_MAX_TOKENS,
                    return_tensors="pt",
                ).to(model.device)
                with torch.no_grad():
                    hidden = model(**encoded).last_hidden_state[:, 0]
                vectors = normalize_rows(hidden.to(torch.float32).cpu().numpy()).astype(np.float16)
                vectors.tofile(vectors_writer)
                docids.extend(row["docid"] for row in rows)
                documents += len(rows)
                if len(docids) % (batch_size * 200) < len(rows):
                    logger.info("Embedded %s: %d documents", shard.name, documents)
    table = pa.Table.from_arrays([pa.array(docids)], names=["docid"])
    with docids_path.open("wb") as writer:
        pq.write_table(table, writer, compression="zstd")
    return {
        "shard": shard.name,
        "vectors": vectors_path.name,
        "docids": docids_path.name,
        "documents": documents,
        "vectors_sha256": sha256_file(vectors_path),
        "vectors_bytes": vectors_path.stat().st_size,
        "docids_sha256": sha256_file(docids_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON file with this worker's corpus shard paths")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("The embedding worker requires a CUDA device")
    plan = json.loads(Path(args.plan).read_text())
    model_dir, output = Path(args.model_dir), Path(args.output)
    verify_model(model_dir, plan["model_manifest"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, local_files_only=True)
    model.to("cuda").eval()
    entries = []
    for shard in plan["shards"]:
        entry = embed_shard(
            model,
            tokenizer,
            Path(shard),
            output / "vectors" / (Path(shard).stem + ".f16"),
            output / "docids" / (Path(shard).stem + ".parquet"),
            args.batch_size,
        )
        entries.append(entry)
        logger.info("Embedded shard %s: %d documents", entry["shard"], entry["documents"])
    (output / f"worker-manifest-{plan['worker']}.json").write_text(json.dumps(entries, indent=2) + "\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.stdout.write(json.dumps({"worker": plan["worker"], "entries": len(entries)}) + "\n")


if __name__ == "__main__":
    main()
