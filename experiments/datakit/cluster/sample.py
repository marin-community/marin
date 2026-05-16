# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stratified sample across all per-source EmbeddingAttrData → centroid training input.

For each source we pull up to ``n_per_source`` rows uniformly across its
embedding shards. The per-source cap (not strict proportional) keeps
long-tail sources audible against the giants. At ~100 active sources x
n_per_source=100_000 the result is ~10M rows.

Output is a single ``sample.npz`` (numpy fast-load is what FAISS K-means
wants; we don't need parquet's columnar machinery for the training set).
"""

import logging
import os
import tempfile

import numpy as np
import pyarrow.parquet as pq
from marin.execution.artifact import Artifact
from rigging.filesystem import open_url

from experiments.datakit.embeddings.luxical.embed_source import EmbeddingAttrData, dequantize_to_fp32

logger = logging.getLogger(__name__)


def _sample_from_shard(
    shard_uri: str,
    take: int,
    dim: int,
    scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Load int8 embeddings from one parquet shard, dequantize, return ``take`` random rows as fp32."""
    table = pq.read_table(shard_uri, columns=["embedding"])
    # ChunkedArray -> FixedSizeListArray -> flat int8 values array.
    fsl = table["embedding"].combine_chunks()
    flat_int8 = fsl.values.to_numpy(zero_copy_only=False)
    embeddings_int8 = flat_int8.reshape(-1, dim)
    if take < len(embeddings_int8):
        idx = rng.choice(len(embeddings_int8), size=take, replace=False)
        embeddings_int8 = embeddings_int8[idx]
    return dequantize_to_fp32(embeddings_int8, scale=scale)


def sample_centroid_inputs(
    output_path: str,
    embed_step_outputs: dict[str, str],
    n_per_source: int,
    seed: int = 42,
) -> None:
    """Concatenate up to ``n_per_source`` rows from each source's embedding parquet shards."""
    rng = np.random.default_rng(seed)
    all_emb: list[np.ndarray] = []
    all_src: list[np.ndarray] = []
    total = 0

    for source_name, step_output in sorted(embed_step_outputs.items()):
        attr = Artifact.from_path(step_output, EmbeddingAttrData)
        shards = attr.shard_paths()
        if not shards:
            logger.warning("No embedding shards for %s under %s", source_name, attr.output_dir)
            continue

        per_shard = max(n_per_source // len(shards), 1)
        chunks: list[np.ndarray] = []
        collected = 0
        for shard_uri in shards:
            if collected >= n_per_source:
                break
            take = min(per_shard, n_per_source - collected)
            chunk = _sample_from_shard(shard_uri, take, attr.embedding_dim, attr.quantization_scale, rng)
            chunks.append(chunk)
            collected += len(chunk)

        if not chunks:
            continue
        block = np.vstack(chunks).astype(np.float32, copy=False)
        all_emb.append(block)
        all_src.append(np.full(len(block), source_name, dtype=object))
        total += len(block)
        logger.info("Sampled %d from %s (over %d shards)", len(block), source_name, len(shards))

    if not all_emb:
        raise RuntimeError("No embeddings found across any source")

    sample_embeddings = np.vstack(all_emb).astype(np.float32, copy=False)
    sample_sources = np.concatenate(all_src)
    logger.info("Total sample: %d x %d across %d sources", *sample_embeddings.shape, len(all_emb))

    local = os.path.join(tempfile.gettempdir(), "sample.npz")
    np.savez_compressed(local, embeddings=sample_embeddings, sources=sample_sources)
    with open(local, "rb") as src, open_url(os.path.join(output_path, "sample.npz"), "wb") as dst:
        dst.write(src.read())
    os.remove(local)
