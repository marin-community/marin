# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a proportional 50M-document Harrier embedding sample."""

import argparse
import hashlib
import json
import logging
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
from rigging.filesystem import StoragePath, atomic_rename, url_to_fs

MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
MODEL_REVISION = "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
MODEL_DIMENSION = 1_024
TARGET_ROWS = 50_000_000
MAX_TOKENS = 8_192
MANIFEST_SHA256 = "791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c"
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
        "manifest_sha256": MANIFEST_SHA256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "rows": TARGET_ROWS,
        "max_tokens": MAX_TOKENS,
    },
)
DATASET_ROOT = DATASET_ARTIFACT.path()
OUTPUT_ROOT = HARRIER_EMBEDDINGS_ARTIFACT.path()
OUTPUT_PATH = StoragePath(OUTPUT_ROOT)
MANIFEST_URL = str(OUTPUT_PATH / "manifest-canonical-sources.json")
SANITY_URL = str(OUTPUT_PATH / "sanity.json")
TOKENIZE_BATCH_SIZE = 128
INPUT_BATCH_SIZE = 8
MAX_RAW_TEXT_CHARS = 1_048_576
MAX_INFERENCE_BATCH_TOKENS = 32_768
MAX_INFERENCE_BATCH_SIZE = 64
PARQUET_ROW_GROUP_SIZE = 8_192
MANIFEST_WORKERS = 16
INFERENCE_DTYPE = "bfloat16"
STORAGE_DTYPE = np.float16
SANITY_REEMBED_ROWS_PER_PART = 4
SANITY_RETRIEVAL_ROWS_PER_PART = 192
SANITY_REEMBED_BATCH_SIZE = 64
SANITY_RANDOM_PAIR_COUNT = 100_000
SANITY_NEIGHBOR_QUERY_COUNT = 256
SANITY_PROBES = (
    (
        "A kitten is a young domestic cat.",
        "A young cat is called a kitten.",
        "A database transaction groups updates atomically.",
    ),
    (
        "Python code can sort a list of integers into ascending order.",
        "JavaScript code can sort an array of numbers from smallest to largest.",
        "Melt chocolate and butter to make a rich cake.",
    ),
)

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
class ModelFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class InputPart:
    """A prefix of one input Parquet file and its paired output."""

    source: str
    input_url: str
    row_count: int
    output_url: str


@dataclass(frozen=True)
class SampledInput:
    ids: list[str]
    texts: list[str]
    raw_lengths: list[int]


@dataclass(frozen=True)
class SemanticProbeReport:
    anchor: str
    related: str
    unrelated: str
    related_cosine: float
    unrelated_cosine: float
    margin: float


@dataclass(frozen=True)
class EmbedPartResult:
    output_url: str
    rows: int
    reused: bool
    duration_seconds: float


@dataclass(frozen=True)
class SampleManifest:
    """Pinned inputs for the proportional Harrier embedding run."""

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


def _sha256_file(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _read_json(url: str) -> dict[str, Any]:
    return json.loads(StoragePath(url).read_text())


def _write_json(url: str, value: dict[str, Any]) -> None:
    with atomic_rename(url) as temporary_path:
        StoragePath(temporary_path).write_text(json.dumps(value, indent=2, sort_keys=True))


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


def embedding_sanity_metrics(stored: np.ndarray, recomputed: np.ndarray) -> dict[str, int | float]:
    """Measure stored-vector norms and agreement with recomputed embeddings."""
    if stored.ndim != 2 or stored.shape != recomputed.shape or not len(stored):
        raise ValueError(f"Expected matching nonempty embedding matrices; got {stored.shape} and {recomputed.shape}")
    stored_float = stored.astype(np.float32)
    recomputed_float = recomputed.astype(np.float32)
    nonfinite_value_count = int(np.size(stored_float) - np.isfinite(stored_float).sum())
    nonfinite_value_count += int(np.size(recomputed_float) - np.isfinite(recomputed_float).sum())
    if nonfinite_value_count:
        raise ValueError(f"Found {nonfinite_value_count} non-finite embedding values")
    stored_norms = np.linalg.norm(stored_float, axis=1)
    recomputed_norms = np.linalg.norm(recomputed_float, axis=1)
    cosine = np.sum(stored_float * recomputed_float, axis=1) / (stored_norms * recomputed_norms)
    return {
        "nonfinite_value_count": nonfinite_value_count,
        "norm_min": float(stored_norms.min()),
        "norm_mean": float(stored_norms.mean()),
        "norm_max": float(stored_norms.max()),
        "norm_max_error": float(np.max(np.abs(stored_norms - 1.0))),
        "reembed_cosine_min": float(cosine.min()),
        "reembed_cosine_p01": float(np.quantile(cosine, 0.01)),
        "reembed_cosine_mean": float(cosine.mean()),
    }


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
    with ThreadPoolExecutor(max_workers=MANIFEST_WORKERS) as pool:
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


def build_manifest() -> dict[str, Any]:
    """Build and persist the exact proportional 50M-row input manifest."""
    inventory = _source_inventory()
    source_counts = {source: sum(file.row_count for file in files) for source, files in inventory.items()}
    quotas = allocate_source_quotas(source_counts, TARGET_ROWS)
    part_lists = [_source_parts(source, quotas[source], inventory[source]) for source in sorted(source_counts)]
    sources = tuple(SourceQuota(source, source_counts[source], quotas[source]) for source in sorted(source_counts))
    parts = tuple(part for part_list in part_lists for part in part_list)
    if sum(part.row_count for part in parts) != TARGET_ROWS:
        raise ValueError("Manifest parts do not sum to the target row count")
    manifest = SampleManifest(
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
    digest = hashlib.sha256(_canonical_json(manifest.payload())).hexdigest()
    if digest != MANIFEST_SHA256:
        raise ValueError(f"Built manifest digest is {digest}; expected {MANIFEST_SHA256}")
    manifest = replace(manifest, sha256=digest)
    if StoragePath(MANIFEST_URL).exists():
        existing = read_manifest()
        if existing.sha256 != manifest.sha256:
            raise ValueError(f"Existing manifest has a different digest: {existing.sha256}")
        return {"manifest_url": MANIFEST_URL, "reused": True, **manifest_summary(existing)}
    _write_json(MANIFEST_URL, asdict(manifest))
    return {"manifest_url": MANIFEST_URL, "reused": False, **manifest_summary(manifest)}


def read_manifest() -> SampleManifest:
    """Read and verify the persisted sample manifest."""
    value = _read_json(MANIFEST_URL)
    manifest = SampleManifest(
        dataset_root=value["dataset_root"],
        output_root=value["output_root"],
        target_rows=int(value["target_rows"]),
        model_id=value["model_id"],
        model_revision=value["model_revision"],
        model_dimension=int(value["model_dimension"]),
        max_tokens=int(value["max_tokens"]),
        sources=tuple(SourceQuota(**source) for source in value["sources"]),
        parts=tuple(InputPart(**part) for part in value["parts"]),
        sha256=value["sha256"],
    )
    digest = hashlib.sha256(_canonical_json(manifest.payload())).hexdigest()
    if digest != manifest.sha256:
        raise ValueError(f"Manifest digest is {digest}; expected {manifest.sha256}")
    if manifest.sha256 != MANIFEST_SHA256:
        raise ValueError(f"Manifest is {manifest.sha256}; expected pinned digest {MANIFEST_SHA256}")
    if manifest.dataset_root != DATASET_ROOT or manifest.output_root != OUTPUT_ROOT:
        raise ValueError("Manifest storage roots do not match this run")
    if manifest.target_rows != TARGET_ROWS or sum(part.row_count for part in manifest.parts) != TARGET_ROWS:
        raise ValueError("Manifest row count does not match the fixed target")
    return manifest


def manifest_summary(manifest: SampleManifest) -> dict[str, Any]:
    """Return the compact manifest fields used by job logs and handoff."""
    return {
        "manifest_sha256": manifest.sha256,
        "source_count": len(manifest.sources),
        "part_count": len(manifest.parts),
        "available_rows": sum(source.available_rows for source in manifest.sources),
        "selected_rows": sum(source.selected_rows for source in manifest.sources),
    }


def stage_model() -> dict[str, Any]:
    """Download the pinned public model once and stage it in-region."""
    manifest_url = str(OUTPUT_PATH / "model" / "manifest.json")
    if StoragePath(manifest_url).exists():
        manifest = _read_json(manifest_url)
        if manifest["model_id"] != MODEL_ID or manifest["model_revision"] != MODEL_REVISION:
            raise ValueError("Existing staged model identifies a different checkpoint")
        return {"model_manifest_url": manifest_url, "reused": True, **manifest}
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_root = Path(temporary_directory) / "model"
        snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=local_root)
        files: list[ModelFile] = []
        for local_path in sorted(
            path for path in local_root.rglob("*") if path.is_file() and ".cache" not in path.parts
        ):
            relative = local_path.relative_to(local_root).as_posix()
            remote_path = OUTPUT_PATH / "model" / "files" / relative
            with atomic_rename(str(remote_path)) as temporary_path:
                StoragePath(temporary_path).upload_from(str(local_path))
            files.append(ModelFile(path=relative, size=local_path.stat().st_size, sha256=_sha256_file(local_path)))
    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "files": [asdict(file) for file in files],
    }
    manifest["sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
    _write_json(manifest_url, manifest)
    return {"model_manifest_url": manifest_url, "reused": False, **manifest}


def _download_staged_model(local_root: Path) -> dict[str, Any]:
    manifest = _read_json(str(OUTPUT_PATH / "model" / "manifest.json"))
    if manifest["model_id"] != MODEL_ID or manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("Staged model identifies a different checkpoint")
    expected_sha256 = manifest.pop("sha256")
    if hashlib.sha256(_canonical_json(manifest)).hexdigest() != expected_sha256:
        raise ValueError("Staged model manifest failed digest verification")
    manifest["sha256"] = expected_sha256
    files = [ModelFile(**item) for item in manifest["files"]]
    for item in files:
        local_path = local_root / item.path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path = OUTPUT_PATH / "model" / "files" / item.path
        remote_path.download_to(str(local_path))
        if local_path.stat().st_size != item.size or _sha256_file(local_path) != item.sha256:
            raise ValueError(f"Staged model file failed verification: {item.path}")
    return manifest


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


def _output_metadata(part: InputPart, manifest: SampleManifest) -> dict[bytes, bytes]:
    return {
        b"harrier_manifest_sha256": manifest.sha256.encode(),
        b"harrier_model_id": MODEL_ID.encode(),
        b"harrier_model_revision": MODEL_REVISION.encode(),
        b"harrier_max_tokens": str(MAX_TOKENS).encode(),
        b"harrier_pooling": b"last_token_l2_normalized",
        b"harrier_storage_dtype": b"float16",
        b"harrier_input_url": part.input_url.encode(),
        b"harrier_input_rows": str(part.row_count).encode(),
        b"harrier_source": part.source.encode(),
    }


def _complete_output(part: InputPart, manifest: SampleManifest) -> bool:
    output_path = StoragePath(part.output_url)
    if not output_path.exists():
        return False
    filesystem, path = url_to_fs(str(output_path))
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        rows = parquet_file.metadata.num_rows
        metadata = parquet_file.schema_arrow.metadata or {}
    if rows != part.row_count:
        raise ValueError(f"Existing output has {rows} rows; expected {part.row_count}: {part.output_url}")
    expected = _output_metadata(part, manifest)
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


def _sample_input(part: InputPart, row_count: int) -> SampledInput:
    filesystem, path = url_to_fs(part.input_url)
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        batch = next(parquet_file.iter_batches(batch_size=min(row_count, part.row_count), columns=["id", "text"]))
    ids = batch.column("id").to_pylist()
    text = batch.column("text")
    raw_lengths = pc.utf8_length(text).to_pylist()
    truncated = pc.utf8_slice_codeunits(text, start=0, stop=MAX_RAW_TEXT_CHARS).to_pylist()
    return SampledInput(ids=ids, texts=truncated, raw_lengths=raw_lengths)


def _sample_output(part: InputPart, row_count: int) -> tuple[list[str], np.ndarray]:
    filesystem, path = url_to_fs(part.output_url)
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        batch = next(parquet_file.iter_batches(batch_size=min(row_count, part.row_count), columns=["id", "embedding"]))
    ids = batch.column("id").to_pylist()
    embedding = batch.column("embedding")
    vectors = embedding.values.to_numpy(zero_copy_only=False).reshape(len(embedding), MODEL_DIMENSION)
    return ids, vectors.astype(STORAGE_DTYPE, copy=False)


def _retrieval_distribution(vectors: np.ndarray) -> dict[str, int | float]:
    values = vectors.astype(np.float32)
    nonfinite_value_count = int(values.size - np.isfinite(values).sum())
    if nonfinite_value_count:
        raise ValueError(f"Found {nonfinite_value_count} non-finite sampled values")
    norms = np.linalg.norm(values, axis=1)
    normalized = values / norms[:, None]
    dimension_std = normalized.std(axis=0)
    random = np.random.default_rng(0)
    left = random.integers(0, len(normalized), size=SANITY_RANDOM_PAIR_COUNT)
    right = random.integers(0, len(normalized), size=SANITY_RANDOM_PAIR_COUNT)
    right[left == right] = (right[left == right] + 1) % len(normalized)
    pair_cosine = np.sum(normalized[left] * normalized[right], axis=1)
    return {
        "row_count": len(values),
        "nonfinite_value_count": nonfinite_value_count,
        "norm_min": float(norms.min()),
        "norm_p01": float(np.quantile(norms, 0.01)),
        "norm_mean": float(norms.mean()),
        "norm_p99": float(np.quantile(norms, 0.99)),
        "norm_max": float(norms.max()),
        "dimension_std_min": float(dimension_std.min()),
        "dimension_std_median": float(np.median(dimension_std)),
        "dimension_std_mean": float(dimension_std.mean()),
        "dimension_std_max": float(dimension_std.max()),
        "centroid_norm": float(np.linalg.norm(normalized.mean(axis=0))),
        "random_pair_cosine_p01": float(np.quantile(pair_cosine, 0.01)),
        "random_pair_cosine_median": float(np.median(pair_cosine)),
        "random_pair_cosine_p99": float(np.quantile(pair_cosine, 0.99)),
        "random_pair_cosine_mean": float(pair_cosine.mean()),
    }


def _nearest_neighbor_metrics(embedder: HarrierEmbedder, vectors: np.ndarray) -> dict[str, int | float]:
    torch = embedder.torch
    corpus = torch.from_numpy(vectors.astype(np.float32)).to("cuda")
    corpus = torch.nn.functional.normalize(corpus, p=2, dim=1)
    query_indices = np.linspace(
        0,
        len(vectors) - 1,
        num=min(SANITY_NEIGHBOR_QUERY_COUNT, len(vectors)),
        dtype=np.int64,
    )
    with torch.inference_mode():
        scores = corpus[query_indices] @ corpus.T
        scores[torch.arange(len(query_indices), device="cuda"), torch.from_numpy(query_indices).to("cuda")] = -1
        nearest = scores.max(dim=1).values.cpu().numpy()
    return {
        "query_count": len(query_indices),
        "cosine_min": float(nearest.min()),
        "cosine_p01": float(np.quantile(nearest, 0.01)),
        "cosine_median": float(np.median(nearest)),
        "cosine_p99": float(np.quantile(nearest, 0.99)),
        "cosine_max": float(nearest.max()),
    }


def _semantic_probe_metrics(embedder: HarrierEmbedder) -> list[SemanticProbeReport]:
    texts = [text for probe in SANITY_PROBES for text in probe]
    vectors = embedder.embed(texts).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    reports = []
    for index, (anchor, related, unrelated) in enumerate(SANITY_PROBES):
        start = index * 3
        related_cosine = float(vectors[start] @ vectors[start + 1])
        unrelated_cosine = float(vectors[start] @ vectors[start + 2])
        reports.append(
            SemanticProbeReport(
                anchor=anchor,
                related=related,
                unrelated=unrelated,
                related_cosine=related_cosine,
                unrelated_cosine=unrelated_cosine,
                margin=related_cosine - unrelated_cosine,
            )
        )
    return reports


def embed_part(embedder: HarrierEmbedder, part: InputPart, manifest: SampleManifest) -> EmbedPartResult:
    """Embed one selected input prefix and atomically publish its output."""
    if _complete_output(part, manifest):
        return EmbedPartResult(output_url=part.output_url, rows=part.row_count, reused=True, duration_seconds=0.0)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float16(), MODEL_DIMENSION)),
        ],
        metadata=_output_metadata(part, manifest),
    )
    started = time.perf_counter()
    rows = 0
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "embeddings.parquet"
        with pq.ParquetWriter(local_path, schema, compression="zstd") as writer:
            for batch in _input_batches(part):
                ids = batch.column(batch.schema.get_field_index("id"))
                texts = batch.column(batch.schema.get_field_index("text")).to_pylist()
                vectors = embedder.embed(texts)
                embedding = pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), MODEL_DIMENSION)
                writer.write_table(
                    pa.table({"id": ids, "embedding": embedding}, schema=schema),
                    row_group_size=PARQUET_ROW_GROUP_SIZE,
                )
                rows += len(batch)
                logger.info("Embedded %d/%d rows for %s", rows, part.row_count, part.input_url)
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
    manifest = read_manifest()
    assignments = assigned_parts(manifest.parts, num_shards)
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards})")
    parts = assignments[shard_index]
    with _harrier_embedder() as embedder:
        results = []
        for index, part in enumerate(parts, start=1):
            logger.info(
                "Embedding part %d/%d on worker %d/%d: %s", index, len(parts), shard_index, num_shards, part.input_url
            )
            result = embed_part(embedder, part, manifest)
            results.append({"source": part.source, "input_url": part.input_url, **asdict(result)})
    report = {
        "manifest_sha256": manifest.sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "part_count": len(parts),
        "row_count": sum(result["rows"] for result in results),
        "parts": results,
    }
    report_url = str(OUTPUT_PATH / "reports" / f"worker-{shard_index:03d}-of-{num_shards:03d}.json")
    _write_json(report_url, report)
    return {"report_url": report_url, **report}


def resolve_shard_index(shard_index: int | None) -> int:
    """Resolve an explicit shard or the current Iris replica index."""
    if shard_index is not None:
        return shard_index
    job_info = get_job_info()
    if job_info is None:
        raise ValueError("No --shard-index and no Iris task identity")
    return job_info.task_index


def run_smoke(max_rows: int) -> dict[str, Any]:
    """Embed a bounded real-data prefix before the full run."""
    manifest = read_manifest()
    source_part = max(manifest.parts, key=lambda part: (part.row_count, part.source, part.input_url))
    part = replace(
        source_part,
        row_count=min(max_rows, source_part.row_count),
        output_url=str(OUTPUT_PATH / "smoke" / f"{manifest.sha256[:12]}.parquet"),
    )
    with _harrier_embedder() as embedder:
        result = embed_part(embedder, part, manifest)
    report = {
        **asdict(result),
        "source": part.source,
        "input_url": part.input_url,
        "manifest_sha256": manifest.sha256,
    }
    _write_json(str(OUTPUT_PATH / "smoke" / "report.json"), report)
    return report


def run_audit(num_shards: int) -> dict[str, Any]:
    """Verify worker coverage, output metadata, and the exact 50M row count."""
    manifest = read_manifest()
    expected_assignments = assigned_parts(manifest.parts, num_shards)
    reported_inputs = set()
    reported_rows = 0
    for shard_index, expected_parts in enumerate(expected_assignments):
        report_url = str(OUTPUT_PATH / "reports" / f"worker-{shard_index:03d}-of-{num_shards:03d}.json")
        report = _read_json(report_url)
        if report["manifest_sha256"] != manifest.sha256:
            raise ValueError(f"Worker {shard_index} has a different manifest")
        expected_inputs = {part.input_url for part in expected_parts}
        actual_inputs = {part["input_url"] for part in report["parts"]}
        if actual_inputs != expected_inputs:
            raise ValueError(f"Worker {shard_index} reports different inputs")
        for part in expected_parts:
            if not _complete_output(part, manifest):
                raise FileNotFoundError(part.output_url)
        reported_inputs.update(actual_inputs)
        reported_rows += int(report["row_count"])
    if len(reported_inputs) != len(manifest.parts) or reported_rows != TARGET_ROWS:
        raise ValueError(f"Audit found {len(reported_inputs)} parts and {reported_rows} rows")
    report = {
        "manifest_sha256": manifest.sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_count": len(manifest.sources),
        "part_count": len(manifest.parts),
        "row_count": reported_rows,
        "num_shards": num_shards,
    }
    report_url = str(OUTPUT_PATH / "audit.json")
    _write_json(report_url, report)
    return {"audit_url": report_url, **report}


def run_sanity() -> dict[str, Any]:
    """Check sampled stored embeddings, reembedding agreement, and semantic separation."""
    manifest = read_manifest()
    audit = _read_json(str(OUTPUT_PATH / "audit.json"))
    if audit["manifest_sha256"] != manifest.sha256 or int(audit["row_count"]) != TARGET_ROWS:
        raise ValueError("The completed audit does not match the pinned 50M manifest")

    retrieval_vectors = []
    reembed_stored = []
    reembed_recomputed = []
    pending_texts: list[str] = []
    pending_stored: list[np.ndarray] = []
    sampled_sources = set()
    sampled_parts = 0
    raw_prefix_truncated_rows = 0
    truncated_underfilled_rows = 0

    with _harrier_embedder() as embedder:

        def flush_reembedding() -> None:
            if not pending_texts:
                return
            reembed_stored.append(np.stack(pending_stored))
            reembed_recomputed.append(embedder.embed(pending_texts))
            pending_texts.clear()
            pending_stored.clear()

        for index, part in enumerate(manifest.parts, start=1):
            if part.row_count == 0:
                continue
            input_count = min(SANITY_REEMBED_ROWS_PER_PART, part.row_count)
            output_count = min(SANITY_RETRIEVAL_ROWS_PER_PART, part.row_count)
            sampled_input = _sample_input(part, input_count)
            output_ids, output_vectors = _sample_output(part, output_count)
            if sampled_input.ids != output_ids[:input_count]:
                raise ValueError(f"Sampled input/output IDs differ: {part.input_url}")
            retrieval_vectors.append(output_vectors)
            pending_texts.extend(sampled_input.texts)
            pending_stored.extend(output_vectors[:input_count])
            sampled_sources.add(part.source)
            sampled_parts += 1
            for text, raw_length in zip(sampled_input.texts, sampled_input.raw_lengths, strict=True):
                if raw_length <= MAX_RAW_TEXT_CHARS:
                    continue
                raw_prefix_truncated_rows += 1
                token_count = len(
                    embedder.tokenizer(
                        text,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=MAX_TOKENS,
                    )["input_ids"]
                )
                if token_count < MAX_TOKENS:
                    truncated_underfilled_rows += 1
            if len(pending_texts) >= SANITY_REEMBED_BATCH_SIZE:
                flush_reembedding()
            if index % 25 == 0 or index == len(manifest.parts):
                logger.info("Sampled %d/%d embedding parts", index, len(manifest.parts))
        flush_reembedding()

        stored = np.concatenate(reembed_stored)
        recomputed = np.concatenate(reembed_recomputed)
        retrieval = np.concatenate(retrieval_vectors)
        reembedding = embedding_sanity_metrics(stored, recomputed)
        distribution = _retrieval_distribution(retrieval)
        nearest_neighbors = _nearest_neighbor_metrics(embedder, retrieval)
        semantic_probes = _semantic_probe_metrics(embedder)

    checks = {
        "all_sources_sampled": len(sampled_sources) == len(manifest.sources),
        "all_parts_sampled": sampled_parts == len(manifest.parts),
        "finite_values": reembedding["nonfinite_value_count"] == 0 and distribution["nonfinite_value_count"] == 0,
        "unit_norms": reembedding["norm_max_error"] <= 0.01,
        "reembedding_matches": reembedding["reembed_cosine_min"] >= 0.995,
        "character_cap_preserves_8k_prefix": truncated_underfilled_rows == 0,
        "sample_not_collapsed": distribution["random_pair_cosine_median"] < 0.99,
        "semantic_probes_separate": all(probe.margin > 0 for probe in semantic_probes),
    }
    report = {
        "manifest_sha256": manifest.sha256,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_count": len(sampled_sources),
        "part_count": sampled_parts,
        "reembedded_row_count": len(stored),
        "retrieval_row_count": len(retrieval),
        "raw_prefix_truncated_rows": raw_prefix_truncated_rows,
        "truncated_underfilled_rows": truncated_underfilled_rows,
        "reembedding": reembedding,
        "distribution": distribution,
        "nearest_neighbors": nearest_neighbors,
        "semantic_probes": [asdict(probe) for probe in semantic_probes],
        "checks": checks,
        "passed": all(checks.values()),
    }
    _write_json(SANITY_URL, report)
    if not report["passed"]:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Embedding sanity checks failed: {failed}; report: {SANITY_URL}")
    return {"sanity_url": SANITY_URL, **report}


def build() -> ArtifactStep[Artifact]:
    """Return the adopted artifact for the completed Harrier embedding run."""
    return HARRIER_EMBEDDINGS_ARTIFACT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("stage-model", "manifest", "smoke", "embed", "audit", "sanity"), required=True
    )
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    arguments = parser.parse_args()
    if arguments.output_root != OUTPUT_ROOT:
        raise ValueError(f"This run is pinned to {OUTPUT_ROOT}")
    if arguments.mode == "embed" and arguments.num_shards is None:
        parser.error("embed mode requires --num-shards")
    if arguments.mode == "audit" and arguments.num_shards is None:
        parser.error("audit mode requires --num-shards")
    if arguments.max_rows < 1:
        parser.error("--max-rows must be positive")
    return arguments


def main() -> None:
    """Run one stage of the fixed Harrier embedding workflow."""
    arguments = parse_args()
    if arguments.mode == "stage-model":
        result = stage_model()
    elif arguments.mode == "manifest":
        result = build_manifest()
    elif arguments.mode == "smoke":
        result = run_smoke(arguments.max_rows)
    elif arguments.mode == "embed":
        result = run_embed(resolve_shard_index(arguments.shard_index), arguments.num_shards)
    elif arguments.mode == "audit":
        result = run_audit(arguments.num_shards)
    else:
        result = run_sanity()
    Path(f"/tmp/harrier-50m-{arguments.mode}").write_text(json.dumps(result, sort_keys=True))
    logger.info("HARRIER_50M=%s", json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
