# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream a datakit-normalized source through Luxical-One, write co-partitioned embedding parquet shards.

Output is a datakit attribute dataset (mirrors :class:`TokenizedAttrData`):
each output parquet shard has the same basename as its source shard, with
columns ``id: string`` and ``embedding: FixedSizeList<int8, 192>``.
Row order matches the source, so the sort-by-id invariant carries through.

We quantize fp32 embeddings symmetrically to int8 with
``scale = 0.6 / 127`` (255 levels covering [-0.6, 0.6]). That gives
guaranteed 1-byte-per-value storage AND 4x in-memory savings when
loaded; consumers dequantize on read via :func:`dequantize_to_fp32`
(one line: ``int8.astype(np.float32) * scale``). Mean cos sim of an
int8 round trip is ~0.9998 on real Luxical-One output (see the
QUANT_RANGE comment below for the envelope sweep). ``scale`` is
recorded on the :class:`EmbeddingAttrData` artifact so consumers
don't have to hard-code it.

NOTE: at full scale, ``embed_source_shard`` should be invoked once per
shard via Zephyr (one worker per parquet shard); ``embed_source`` here
processes a whole source sequentially and is the right call only for
small sources.
"""

import logging
import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import Artifact
from marin.utils import fsspec_glob
from pydantic import BaseModel
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

LUXICAL_MODEL = "DatologyAI/luxical-one"
LUXICAL_DIM = 192
DEFAULT_BATCH_SIZE = 256
DEFAULT_CHUNK_DOCS = 50_000

# Quantization envelope. Empirically sweeping the per-vector cos-sim of an
# int8 round trip over 10k real Luxical-One embeddings of nemotron_cc_v2/
# high_quality showed:
#
#    range    mean_cos   min_cos   clip_pct
#    +/-0.3   0.9809     0.9546     1.015%   <- too tight; tails clipped
#    +/-0.5   0.9997     0.9948     0.249%
#    +/-0.6   0.99982    0.9997     0.001%   <- best
#    +/-0.7   0.99976    0.9997     0.000%
#    +/-1.0   0.99951    0.9994     0.000%
#
# Luxical-One's sparse-to-dense projection produces per-dim values whose
# p99.9 abs is ~0.53 and max abs is ~0.62 on real CC text — wider tails
# than the original "or so" quote suggested. +/-0.6 covers the whole
# observed range with effectively zero clipping while keeping the
# quantization step as fine as possible.
QUANT_RANGE = 0.6
QUANT_SCALE: float = QUANT_RANGE / 127  # int8 [-127, 127] -> fp32 [-0.6, 0.6] (255 levels, symmetric)

_EMBEDDING_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("embedding", pa.list_(pa.int8(), LUXICAL_DIM)),
    ]
)


class EmbeddingAttrData(BaseModel):
    """Co-partitioned per-source embedding parquet shards.

    Persisted as the step's ``.artifact``. Load via
    ``Artifact.from_path(step, EmbeddingAttrData)``.

    Attributes:
        output_dir: Directory containing ``<basename>.parquet`` shards
            (same basenames as ``source_main_dir``).
        source_main_dir: ``NormalizedData.main_output_dir`` this dataset
            mirrors. Co-partitioning means consumers can join
            ``(basename, row_idx)`` directly without an id index.
        model_name: HuggingFace model id used for encoding.
        embedding_dim: Vector dimension (192 for Luxical-One).
        quantization_scale: Multiply the stored int8 by this to recover fp32
            (i.e. ``fp32 = int8.astype(np.float32) * quantization_scale``).
        quantization_range: Original clipping envelope before quantization
            (informational; ``quantization_range == quantization_scale * 127``).
        counters: aggregate `{shards_out, docs_out}`.
    """

    version: str = "v1"
    output_dir: str
    source_main_dir: str
    model_name: str
    embedding_dim: int
    quantization_scale: float
    quantization_range: float
    counters: dict[str, int] = {}

    def shard_paths(self) -> list[str]:
        return sorted(fsspec_glob(f"{self.output_dir.rstrip('/')}/*.parquet"))


def quantize_to_int8(arr: np.ndarray) -> np.ndarray:
    """Quantize fp32 to int8 using ``QUANT_SCALE`` (symmetric, 255 levels in [-0.3, 0.3])."""
    return np.clip(np.round(arr / QUANT_SCALE), -127, 127).astype(np.int8)


def dequantize_to_fp32(arr: np.ndarray, scale: float = QUANT_SCALE) -> np.ndarray:
    """Inverse of :func:`quantize_to_int8`. Consumers call this on the loaded int8 column."""
    return arr.astype(np.float32) * scale


def _encode_chunk(model, texts: list[str], batch_size: int) -> np.ndarray:
    raw = np.asarray(
        model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True),
        dtype=np.float32,
    )
    return quantize_to_int8(raw)


def _embedding_record_batch(ids: list[str], embeddings_int8: np.ndarray) -> pa.RecordBatch:
    flat = pa.array(embeddings_int8.reshape(-1), type=pa.int8())
    emb_col = pa.FixedSizeListArray.from_arrays(flat, LUXICAL_DIM)
    return pa.RecordBatch.from_arrays([pa.array(ids, type=pa.string()), emb_col], schema=_EMBEDDING_SCHEMA)


def embed_source_shard(
    output_path: str,
    shard_uri: str,
    model_name: str = LUXICAL_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
) -> int:
    """Embed one normalized parquet shard → parquet shard with the same basename.

    Returns the number of docs written.
    """
    from sentence_transformers import SentenceTransformer

    basename = os.path.basename(shard_uri)
    out_uri = f"{output_path.rstrip('/')}/{basename}"
    logger.info("Embedding %s -> %s", shard_uri, out_uri)

    model = SentenceTransformer(model_name, trust_remote_code=True)
    pf = pq.ParquetFile(shard_uri)

    local_out = os.path.join(tempfile.gettempdir(), basename)
    # use_dictionary is redundant for int8 (already 1 byte/value) but zstd still helps
    # on repeated bytes from low-entropy dimensions.
    writer = pq.ParquetWriter(local_out, _EMBEDDING_SCHEMA, compression="zstd", use_dictionary=False)

    total = 0
    buf_texts: list[str] = []
    buf_ids: list[str] = []

    def _flush() -> None:
        nonlocal total
        if not buf_texts:
            return
        emb = _encode_chunk(model, buf_texts, batch_size)
        writer.write_batch(_embedding_record_batch(buf_ids, emb))
        total += len(buf_ids)
        buf_texts.clear()
        buf_ids.clear()

    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i, columns=["id", "text"]).to_pylist()
        for row in rg:
            buf_texts.append(row["text"])
            buf_ids.append(row["id"])
            if len(buf_texts) >= chunk_docs:
                _flush()
    _flush()
    writer.close()

    with open(local_out, "rb") as src, open_url(out_uri, "wb") as dst:
        dst.write(src.read())
    os.remove(local_out)

    logger.info("Encoded %d docs from %s", total, basename)
    return total


def embed_source(
    output_path: str,
    normalized_path: str,
    model_name: str = LUXICAL_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    chunk_docs: int = DEFAULT_CHUNK_DOCS,
) -> EmbeddingAttrData:
    """Embed every shard under normalized.main_output_dir; persist EmbeddingAttrData.

    For huge sources, replace this with a Zephyr-driven per-shard fan-out
    invoking :func:`embed_source_shard` directly.
    """
    normalized = Artifact.from_path(normalized_path, NormalizedData)
    shards = sorted(fsspec_glob(f"{normalized.main_output_dir.rstrip('/')}/**/*.parquet"))
    logger.info("Embedding %d shards from %s with %s", len(shards), normalized.main_output_dir, model_name)

    total = 0
    for shard_uri in shards:
        total += embed_source_shard(
            output_path=output_path,
            shard_uri=shard_uri,
            model_name=model_name,
            batch_size=batch_size,
            chunk_docs=chunk_docs,
        )

    artifact = EmbeddingAttrData(
        output_dir=output_path,
        source_main_dir=normalized.main_output_dir,
        model_name=model_name,
        embedding_dim=LUXICAL_DIM,
        quantization_scale=QUANT_SCALE,
        quantization_range=QUANT_RANGE,
        counters={"shards_out": len(shards), "docs_out": total},
    )
    Artifact.save(artifact, output_path)
    return artifact
