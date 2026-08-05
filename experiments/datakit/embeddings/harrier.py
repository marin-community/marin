# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a proportional 50M-document Harrier embedding sample."""

import hashlib
import json
import logging
import tarfile
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from iris.cluster.client.job_info import get_job_info
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from rigging.filesystem import StoragePath, atomic_rename, marin_temp_bucket, url_to_fs

MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
MODEL_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
MODEL_DIMENSION = 1_024
TARGET_ROWS = 50_000_000
MAX_TOKENS = 8_192
PLAN_SHA256 = "791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c"
DATASET_ARTIFACT: ArtifactStep[Artifact] = ArtifactStep.adopt(
    name="datakit/sample-10pct-91269634",
    version="2026.08.04",
    source="s3://marin-us-east-02a/marin/datakit/sample_10pct_91269634",
)
HARRIER_EMBEDDINGS_ARTIFACT: ArtifactStep[Artifact] = ArtifactStep.adopt(
    name="datakit/embeddings/harrier-oss-v1-0.6b-50m",
    version="2026.08.04",
    source="s3://marin-us-east-02a/marin/user/held/harrier-oss-v1-0.6b-50m",
    config={
        "dataset": DATASET_ARTIFACT.path(),
        "input_plan_sha256": PLAN_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rows": TARGET_ROWS,
        "max_tokens": MAX_TOKENS,
    },
)
DATASET_ROOT = DATASET_ARTIFACT.path()
OUTPUT_ROOT = HARRIER_EMBEDDINGS_ARTIFACT.path()
OUTPUT_PATH = StoragePath(OUTPUT_ROOT)
TOKENIZE_BATCH_SIZE = 128
INPUT_BATCH_SIZE = 8
MAX_RAW_TEXT_CHARS = 1_048_576
MAX_INFERENCE_BATCH_TOKENS = 32_768
MAX_INFERENCE_BATCH_SIZE = 64
PARQUET_ROW_GROUP_SIZE = 8_192
INVENTORY_WORKERS = 16
INFERENCE_DTYPE = "bfloat16"
STORAGE_DTYPE = np.float16

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceQuota:
    """One source's available and selected document counts."""

    source: str
    available_rows: int
    selected_rows: int


@dataclass(frozen=True)
class SourceFile:
    """One sampled Parquet file and its footer row count."""

    input_url: str
    row_count: int


@dataclass(frozen=True)
class InputPart:
    """A prefix of one input Parquet file and its paired output."""

    source: str
    input_url: str
    row_count: int
    output_url: str


@dataclass(frozen=True)
class EmbedPartResult:
    output_url: str
    rows: int
    reused: bool
    duration_seconds: float


@dataclass(frozen=True)
class EmbeddingPlan:
    """Deterministic inputs for the proportional Harrier embedding run."""

    dataset_root: str
    output_root: str
    target_rows: int
    model_id: str
    model_revision: str
    model_dimension: int
    max_tokens: int
    sources: tuple[SourceQuota, ...]
    parts: tuple[InputPart, ...]
    sha256: str

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("sha256")
        return value


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def allocate_source_quotas(source_counts: dict[str, int], target_rows: int) -> dict[str, int]:
    """Allocate an exact target proportionally with deterministic largest remainders."""
    total_rows = sum(source_counts.values())
    if target_rows <= 0 or target_rows > total_rows:
        raise ValueError(f"target_rows must be in [1, {total_rows}]; got {target_rows}")
    numerators = {source: target_rows * count for source, count in source_counts.items()}
    quotas = {source: numerator // total_rows for source, numerator in numerators.items()}
    remaining = target_rows - sum(quotas.values())
    order = sorted(source_counts, key=lambda source: (-(numerators[source] % total_rows), source))
    for source in order[:remaining]:
        quotas[source] += 1
    if any(quotas[source] > count for source, count in source_counts.items()):
        raise ValueError("A proportional quota exceeds its source size")
    return quotas


def inference_groups(
    lengths: list[int],
    max_batch_tokens: int = MAX_INFERENCE_BATCH_TOKENS,
    max_batch_size: int = MAX_INFERENCE_BATCH_SIZE,
) -> list[list[int]]:
    """Group sequence indices into length-sorted padded batches."""
    if any(length <= 0 or length > MAX_TOKENS for length in lengths):
        raise ValueError(f"Token lengths must be in [1, {MAX_TOKENS}]")
    groups: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        next_max = max(current_max, lengths[index])
        if current and (len(current) >= max_batch_size or next_max * (len(current) + 1) > max_batch_tokens):
            groups.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        groups.append(current)
    return groups


def assigned_parts(parts: tuple[InputPart, ...], num_shards: int) -> tuple[tuple[InputPart, ...], ...]:
    """Assign input parts to balanced deterministic worker shards."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    totals = [0] * num_shards
    assignments: list[list[InputPart]] = [[] for _ in range(num_shards)]
    for part in sorted(parts, key=lambda item: (-item.row_count, item.source, item.input_url)):
        target = min(range(num_shards), key=lambda index: (totals[index], index))
        assignments[target].append(part)
        totals[target] += part.row_count
    logger.info("Worker row totals: %s", totals)
    return tuple(tuple(sorted(items, key=lambda item: (item.source, item.input_url))) for items in assignments)


def _source_files(source: str) -> tuple[SourceFile, ...]:
    input_root = StoragePath(DATASET_ROOT) / source / "outputs" / "main"
    if not input_root.exists():
        return ()
    urls = sorted(str(item) for item in input_root.ls() if item.name.endswith(".parquet"))
    files = []
    for input_url in urls:
        input_filesystem, input_path = url_to_fs(input_url)
        with pq.ParquetFile(input_path, filesystem=input_filesystem) as parquet_file:
            files.append(SourceFile(input_url=input_url, row_count=parquet_file.metadata.num_rows))
    logger.info("Found %d rows across %d files for %s", sum(file.row_count for file in files), len(files), source)
    return tuple(files)


def _source_inventory() -> dict[str, tuple[SourceFile, ...]]:
    sources = sorted(all_sources())
    with ThreadPoolExecutor(max_workers=INVENTORY_WORKERS) as pool:
        file_lists = pool.map(_source_files, sources)
    inventory = {source: files for source, files in zip(sources, file_lists, strict=True) if files}
    missing = sorted(set(sources) - set(inventory))
    if missing:
        raise ValueError(f"Missing Parquet outputs for {len(missing)} canonical sources: {', '.join(missing)}")
    logger.info("Found Parquet outputs for all %d canonical sources", len(inventory))
    return inventory


def _source_parts(source: str, selected_rows: int, files: tuple[SourceFile, ...]) -> list[InputPart]:
    parts = []
    remaining = selected_rows
    for file in files:
        row_count = min(remaining, file.row_count)
        output_url = str(OUTPUT_PATH / "embeddings" / source / StoragePath(file.input_url).name)
        parts.append(InputPart(source=source, input_url=file.input_url, row_count=row_count, output_url=output_url))
        remaining -= row_count
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"Source {source} is missing {remaining} selected rows")
    return parts


def build_plan() -> EmbeddingPlan:
    """Build the exact proportional 50M-row input plan."""
    inventory = _source_inventory()
    source_counts = {source: sum(file.row_count for file in files) for source, files in inventory.items()}
    quotas = allocate_source_quotas(source_counts, TARGET_ROWS)
    part_lists = [_source_parts(source, quotas[source], inventory[source]) for source in sorted(source_counts)]
    sources = tuple(SourceQuota(source, source_counts[source], quotas[source]) for source in sorted(source_counts))
    parts = tuple(part for part_list in part_lists for part in part_list)
    if sum(part.row_count for part in parts) != TARGET_ROWS:
        raise ValueError("Input parts do not sum to the target row count")
    plan = EmbeddingPlan(
        dataset_root=DATASET_ROOT,
        output_root=OUTPUT_ROOT,
        target_rows=TARGET_ROWS,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_dimension=MODEL_DIMENSION,
        max_tokens=MAX_TOKENS,
        sources=sources,
        parts=parts,
        sha256="",
    )
    digest = hashlib.sha256(_canonical_json(plan.payload())).hexdigest()
    if digest != PLAN_SHA256:
        raise ValueError(f"Built input plan digest is {digest}; expected {PLAN_SHA256}")
    return replace(plan, sha256=digest)


def _model_archive_url() -> str:
    root = StoragePath(marin_temp_bucket(ttl_days=1, prefix="harrier-staging", source_prefix=OUTPUT_ROOT))
    return str(root / MODEL_REVISION / "model.tar")


def stage_model() -> dict[str, Any]:
    """Download the pinned public model once and stage it in-region."""
    archive_url = _model_archive_url()
    if StoragePath(archive_url).exists():
        return {"model_archive_url": archive_url, "reused": True}
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_root = Path(temporary_directory) / "model"
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=local_root)
        archive_path = Path(temporary_directory) / "model.tar"
        with tarfile.open(archive_path, "w") as archive:
            for model_file in sorted(path for path in local_root.rglob("*") if path.is_file()):
                if ".cache" not in model_file.parts:
                    archive.add(model_file, arcname=model_file.relative_to(local_root))
        with atomic_rename(archive_url) as temporary_path:
            StoragePath(temporary_path).upload_from(str(archive_path))
    return {"model_archive_url": archive_url, "reused": False}


def _download_staged_model(local_root: Path) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "model.tar"
        StoragePath(_model_archive_url()).download_to(str(archive_path))
        with tarfile.open(archive_path) as archive:
            archive.extractall(local_root, filter="data")


class HarrierEmbedder:
    """Run the pinned Harrier checkpoint on CUDA."""

    def __init__(self, model_path: Path) -> None:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.torch = torch
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"Expected a fast tokenizer, got {type(tokenizer).__name__}")
        self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=getattr(torch, INFERENCE_DTYPE),
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        if int(self.model.config.hidden_size) != MODEL_DIMENSION:
            raise ValueError(f"Harrier hidden size is {self.model.config.hidden_size}; expected {MODEL_DIMENSION}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return normalized float16 embeddings in input order."""
        embeddings = np.empty((len(texts), MODEL_DIMENSION), dtype=STORAGE_DTYPE)
        for chunk_start in range(0, len(texts), TOKENIZE_BATCH_SIZE):
            chunk = texts[chunk_start : chunk_start + TOKENIZE_BATCH_SIZE]
            tokenized = self.tokenizer(
                chunk,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=MAX_TOKENS,
                return_attention_mask=False,
            )["input_ids"]
            lengths = [len(input_ids) for input_ids in tokenized]
            for group in inference_groups(lengths):
                inputs = self.tokenizer.pad(
                    [{"input_ids": tokenized[index]} for index in group],
                    padding=True,
                    return_tensors="pt",
                )
                device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
                with self.torch.inference_mode():
                    model_output = self.model(**device_inputs, use_cache=False)
                vectors = self.torch.nn.functional.normalize(model_output.last_hidden_state[:, -1].float(), p=2, dim=1)
                if not self.torch.isfinite(vectors).all():
                    raise ValueError(f"Harrier returned non-finite vectors at input row {chunk_start}")
                embeddings[chunk_start + np.asarray(group)] = vectors.half().cpu().numpy()
        return embeddings


def _output_metadata(part: InputPart, plan: EmbeddingPlan) -> dict[bytes, bytes]:
    return {
        b"harrier_manifest_sha256": plan.sha256.encode(),
        b"harrier_model_id": MODEL_ID.encode(),
        b"harrier_model_revision": MODEL_REVISION.encode(),
        b"harrier_max_tokens": str(MAX_TOKENS).encode(),
        b"harrier_pooling": b"last_token_l2_normalized",
        b"harrier_storage_dtype": b"float16",
        b"harrier_input_url": part.input_url.encode(),
        b"harrier_input_rows": str(part.row_count).encode(),
        b"harrier_source": part.source.encode(),
    }


def _output_is_complete(part: InputPart, plan: EmbeddingPlan) -> bool:
    output_path = StoragePath(part.output_url)
    if not output_path.exists():
        return False
    filesystem, path = url_to_fs(str(output_path))
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        rows = parquet_file.metadata.num_rows
        metadata = parquet_file.schema_arrow.metadata or {}
    if rows != part.row_count:
        raise ValueError(f"Existing output has {rows} rows; expected {part.row_count}: {part.output_url}")
    expected = _output_metadata(part, plan)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing output has different metadata: {part.output_url}")
    return True


def _input_batches(part: InputPart) -> Iterator[pa.RecordBatch]:
    filesystem, path = url_to_fs(part.input_url)
    remaining = part.row_count
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=INPUT_BATCH_SIZE, columns=["id", "text"]):
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
            text_index = batch.schema.get_field_index("text")
            text = pc.utf8_slice_codeunits(batch.column(text_index), start=0, stop=MAX_RAW_TEXT_CHARS)
            batch = batch.set_column(text_index, batch.schema.field(text_index), text)
            remaining -= len(batch)
            yield batch
    if remaining:
        raise ValueError(f"Input ended with {remaining} selected rows missing: {part.input_url}")


def embed_part(embedder: HarrierEmbedder, part: InputPart, plan: EmbeddingPlan) -> EmbedPartResult:
    """Embed one selected input prefix and atomically publish its output."""
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float16(), MODEL_DIMENSION)),
        ],
        metadata=_output_metadata(part, plan),
    )
    started = time.perf_counter()
    rows = 0
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "embeddings.parquet"
        with pq.ParquetWriter(local_path, schema, compression="zstd") as writer:
            output_tables = []
            buffered_rows = 0
            for batch in _input_batches(part):
                ids = batch.column(batch.schema.get_field_index("id"))
                texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
                vectors = embedder.embed(texts)
                embedding = pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), MODEL_DIMENSION)
                output_tables.append(pa.table({"id": ids, "embedding": embedding}, schema=schema))
                buffered_rows += len(batch)
                if buffered_rows >= PARQUET_ROW_GROUP_SIZE:
                    writer.write_table(pa.concat_tables(output_tables), row_group_size=PARQUET_ROW_GROUP_SIZE)
                    output_tables = []
                    buffered_rows = 0
                rows += len(batch)
                logger.info("Embedded %d/%d rows for %s", rows, part.row_count, part.input_url)
            if output_tables:
                writer.write_table(pa.concat_tables(output_tables), row_group_size=PARQUET_ROW_GROUP_SIZE)
        if rows != part.row_count:
            raise ValueError(f"Embedded {rows} rows; expected {part.row_count}: {part.input_url}")
        with atomic_rename(part.output_url) as temporary_path:
            StoragePath(temporary_path).upload_from(str(local_path))
    return EmbedPartResult(
        output_url=part.output_url,
        rows=rows,
        reused=False,
        duration_seconds=time.perf_counter() - started,
    )


@contextmanager
def _harrier_embedder() -> Iterator[HarrierEmbedder]:
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model"
        _download_staged_model(model_path)
        yield HarrierEmbedder(model_path)


def run_embed(shard_index: int, num_shards: int) -> dict[str, Any]:
    """Run one independent Harrier embedding worker."""
    plan = build_plan()
    assignments = assigned_parts(plan.parts, num_shards)
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    parts = assignments[shard_index]
    results = []
    pending_parts = []
    for part in parts:
        if _output_is_complete(part, plan):
            results.append(EmbedPartResult(part.output_url, part.row_count, True, 0.0))
        else:
            pending_parts.append(part)
    if pending_parts:
        with _harrier_embedder() as embedder:
            for index, part in enumerate(pending_parts, start=1):
                logger.info(
                    "Embedding part %d/%d on worker %d/%d: %s",
                    index,
                    len(pending_parts),
                    shard_index,
                    num_shards,
                    part.input_url,
                )
                results.append(embed_part(embedder, part, plan))
    return {
        "input_plan_sha256": plan.sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "part_count": len(parts),
        "row_count": sum(result.rows for result in results),
    }


def build() -> ArtifactStep[Artifact]:
    """Return the adopted artifact for the completed Harrier embedding run."""
    return HARRIER_EMBEDDINGS_ARTIFACT


def main() -> None:
    """Run the fixed Harrier embedding worker for this Iris task."""
    job_info = get_job_info()
    if job_info is None:
        raise ValueError("Harrier embedding must run as an Iris job")
    result = run_embed(job_info.task_index, job_info.num_tasks)
    logger.info("HARRIER_50M=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
