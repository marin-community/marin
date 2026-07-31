# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create one shard of Arctic teacher embeddings for the Luxical ladder."""

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from arctic import PinnedArcticEmbedder
from iris.cluster.client.job_info import get_job_info
from ladder_config import MANIFEST_ROOT, SEED, TEACHER_ID, TEACHER_REVISION, teacher_windows_from_view
from luxical.teacher_embedder import fast_8bit_uniform_scalar_quantize
from rigging.filesystem import atomic_rename

MANIFEST_URL = f"{MANIFEST_ROOT}/manifest.json"
TEACHER_ROOT = f"{MANIFEST_ROOT}/teacher-arctic-v1"
MAX_TEACHER_TOKENS = 512
TABLE_BATCH_SIZE = 512
INFERENCE_BATCH_SIZE = 128
QUANTIZATION_LIMIT = 0.3
EMBEDDING_DIMENSION = 256
RESULT_FILE = Path("/tmp/luxical-arctic-teacher-shard")
MANIFEST_METADATA_KEY = b"luxical_manifest_sha256"
TEACHER_ID_METADATA_KEY = b"luxical_teacher_id"
TEACHER_REVISION_METADATA_KEY = b"luxical_teacher_revision"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def read_json(url: str) -> dict[str, Any]:
    """Read one JSON object from private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with filesystem.open(path) as file:
        return json.load(file)


def assigned_sources(manifest: dict[str, Any], shard_index: int, num_shards: int) -> list[str]:
    """Assign whole sources to balanced teacher shards."""
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"Shard index {shard_index} is outside [0, {num_shards})")
    totals = [0] * num_shards
    assignments: dict[int, list[str]] = defaultdict(list)
    source_rows = {
        source: result["counts"]["train_3m"] + result["counts"]["eval"] for source, result in manifest["sources"].items()
    }
    for source in sorted(source_rows, key=lambda name: (-source_rows[name], name)):
        target = min(range(num_shards), key=lambda index: (totals[index], index))
        assignments[target].append(source)
        totals[target] += source_rows[source]
    logger.info("Teacher shard row totals: %s", totals)
    return sorted(assignments[shard_index])


def load_selected_source(url: str) -> pa.Table:
    """Read evaluation and 3M training rows for one source."""
    filesystem, path = fsspec.core.url_to_fs(url)
    table = pq.read_table(path, filesystem=filesystem)
    selected = pc.or_(pc.equal(table["split"], "eval"), table["in_3m"])
    return table.filter(selected)


def new_teacher() -> PinnedArcticEmbedder:
    """Load the pinned teacher and check startup inference."""
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    teacher = PinnedArcticEmbedder(
        model_id=TEACHER_ID,
        revision=TEACHER_REVISION,
        max_seq_len=MAX_TEACHER_TOKENS,
    )
    teacher.to("cuda", dtype=torch.float32)
    teacher.model.eval()
    control_vectors = teacher.embed_texts(
        (
            "A short English document about data processing.",
            "def add(left: int, right: int) -> int:\n    return left + right",
            "これは日本語の短い文書です。",
            "word " * 128,
        ),
        is_query=False,
        batch_size=4,
        mrl=True,
        progress_bar=False,
    )
    if not np.isfinite(control_vectors).all():
        raise ValueError("Arctic startup inference returned non-finite vectors")
    if np.unique(control_vectors, axis=0).shape[0] != len(control_vectors):
        raise ValueError("Arctic startup inference returned duplicate control vectors")
    return teacher


def teacher_batch(teacher: PinnedArcticEmbedder, texts: list[str], start: int) -> np.ndarray:
    """Return one checked and quantized teacher batch."""
    windows = [window for text in texts for window in teacher_windows_from_view(text)]
    window_embeddings = teacher.embed_texts(
        windows,
        is_query=False,
        batch_size=INFERENCE_BATCH_SIZE,
        mrl=True,
        progress_bar=False,
    ).reshape(len(texts), 3, EMBEDDING_DIMENSION)
    if not np.isfinite(window_embeddings).all():
        raise ValueError(f"Arctic returned non-finite vectors for batch starting at {start}")
    pooled = window_embeddings.mean(axis=1)
    pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
    quantized = fast_8bit_uniform_scalar_quantize(pooled, QUANTIZATION_LIMIT)
    return quantized


def expected_metadata(manifest_sha256: str) -> dict[bytes, bytes]:
    """Return the metadata that binds a teacher file to its inputs."""
    return {
        MANIFEST_METADATA_KEY: manifest_sha256.encode(),
        TEACHER_ID_METADATA_KEY: TEACHER_ID.encode(),
        TEACHER_REVISION_METADATA_KEY: TEACHER_REVISION.encode(),
    }


def embed_source(
    teacher: PinnedArcticEmbedder,
    source: str,
    input_url: str,
    manifest_sha256: str,
) -> tuple[str, dict[str, Any]]:
    """Embed and write all selected rows for one source."""
    input_table = load_selected_source(input_url)
    expected_rows = len(input_table)
    output_url = f"{TEACHER_ROOT}/sources/{Path(input_url).name}"
    output_filesystem, output_path = fsspec.core.url_to_fs(output_url)
    if output_filesystem.exists(output_path):
        with pq.ParquetFile(output_path, filesystem=output_filesystem) as parquet_file:
            actual_rows = parquet_file.metadata.num_rows
            metadata = parquet_file.schema_arrow.metadata or {}
        if actual_rows != expected_rows:
            raise ValueError(f"Existing teacher output has {actual_rows} rows; expected {expected_rows}: {output_url}")
        if any(metadata.get(key) != value for key, value in expected_metadata(manifest_sha256).items()):
            raise ValueError(f"Existing teacher output has different input metadata: {output_url}")
        logger.info("Reusing complete teacher output for %s", source)
        return output_url, {"rows": actual_rows, "reused": True}

    quantized_batches = []
    texts = input_table["text"].to_pylist()
    for start in range(0, len(texts), TABLE_BATCH_SIZE):
        batch = texts[start : start + TABLE_BATCH_SIZE]
        quantized_batches.append(teacher_batch(teacher, batch, start))
        logger.info("Teacher embedded %s: %d/%d", source, start + len(batch), len(texts))
    quantized = np.concatenate(quantized_batches)
    if quantized.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise ValueError(f"Unexpected teacher shape for {source}: {quantized.shape}")

    embedding_array = pa.FixedSizeListArray.from_arrays(
        pa.array(quantized.ravel()),
        EMBEDDING_DIMENSION,
    )
    output_table = input_table.drop(["text"]).append_column("embedding", embedding_array)
    metadata = dict(output_table.schema.metadata or {})
    metadata.update(expected_metadata(manifest_sha256))
    output_table = output_table.replace_schema_metadata(metadata)
    with atomic_rename(output_path, fs=output_filesystem) as temporary_path:
        pq.write_table(
            output_table,
            temporary_path,
            filesystem=output_filesystem,
            compression="zstd",
        )
    unique_rows = int(np.unique(quantized, axis=0).shape[0])
    varying_dimensions = int(np.count_nonzero(quantized.max(axis=0) > quantized.min(axis=0)))
    return output_url, {
        "rows": expected_rows,
        "reused": False,
        "unique_quantized_rows": unique_rows,
        "varying_dimensions": varying_dimensions,
        "minimum_quantized_value": int(quantized.min()),
        "maximum_quantized_value": int(quantized.max()),
    }


def write_json(url: str, value: dict[str, Any]) -> None:
    """Write one JSON object atomically to private storage."""
    filesystem, path = fsspec.core.url_to_fs(url)
    with atomic_rename(path, fs=filesystem) as temporary_path:
        with filesystem.open(temporary_path, "w") as file:
            json.dump(value, file, indent=2, sort_keys=True)


def shard_coordinates(arguments: argparse.Namespace) -> tuple[int, int]:
    """Return explicit or Iris-provided shard coordinates."""
    if arguments.shard_index is not None or arguments.num_shards is not None:
        if arguments.shard_index is None or arguments.num_shards is None:
            raise ValueError("Both explicit shard arguments are required")
        return arguments.shard_index, arguments.num_shards
    job_info = get_job_info()
    if job_info is None:
        raise ValueError("Shard arguments are required outside an Iris job")
    return job_info.task_index, job_info.num_tasks


def parse_args() -> argparse.Namespace:
    """Parse optional standalone shard coordinates."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    return parser.parse_args()


def main() -> None:
    """Embed every source assigned to this teacher shard."""
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    shard_index, num_shards = shard_coordinates(parse_args())
    manifest = read_json(MANIFEST_URL)
    sources = assigned_sources(manifest, shard_index, num_shards)
    teacher = new_teacher()
    source_reports: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(sources, start=1):
        logger.info("Embedding source %d/%d on shard %d/%d: %s", index, len(sources), shard_index, num_shards, source)
        result = manifest["sources"][source]
        output_url, metrics = embed_source(teacher, source, result["output_url"], manifest["sha256"])
        source_reports[source] = {"output_url": output_url, "metrics": metrics}

    report = {
        "teacher_id": TEACHER_ID,
        "teacher_revision": TEACHER_REVISION,
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": manifest["sha256"],
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_count": len(sources),
        "row_count": sum(result["metrics"]["rows"] for result in source_reports.values()),
        "sources": source_reports,
    }
    report_url = f"{TEACHER_ROOT}/shards/shard-{shard_index:02d}-of-{num_shards:02d}.json"
    write_json(report_url, report)
    summary = {
        "report_url": report_url,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "source_count": len(sources),
        "row_count": report["row_count"],
    }
    RESULT_FILE.write_text(json.dumps(summary, sort_keys=True))
    logger.info("LUXICAL_ARCTIC_TEACHER_SHARD=%s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
