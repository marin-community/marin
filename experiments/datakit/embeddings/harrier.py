# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Create a proportional 50M-document Harrier embedding sample."""

import hashlib
import json
import logging
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from iris.cluster.client.job_info import get_job_info
from levanter.model_cache import cache_hf_model
from marin.datakit.sources import all_sources
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from rigging.filesystem import StoragePath, atomic_rename, marin_temp_bucket, url_to_fs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def build() -> ArtifactStep[Artifact]:
    """Return the completed Harrier embedding artifact."""
    return ArtifactStep.adopt(
        name="datakit/embeddings/harrier-oss-v1-0.6b-50m",
        version="2026.08.04",
        source="s3://marin-us-east-02a/marin/user/held/harrier-oss-v1-0.6b-50m",
        config={
            "dataset": "s3://marin-us-east-02a/marin/datakit/sample_10pct_91269634",
            "input_plan_sha256": "791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c",
            "model_id": "microsoft/harrier-oss-v1-0.6b",
            "model_revision": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
            "rows": 50_000_000,
            "max_tokens": 8_192,
        },
    )


@dataclass(frozen=True)
class InputPart:
    source: str
    input_url: str
    row_count: int
    output_url: str


def _allocate_source_quotas(source_counts: dict[str, int], target_rows: int) -> dict[str, int]:
    total_rows = sum(source_counts.values())
    numerators = {source: target_rows * count for source, count in source_counts.items()}
    quotas = {source: numerator // total_rows for source, numerator in numerators.items()}
    remaining = target_rows - sum(quotas.values())
    order = sorted(source_counts, key=lambda source: (-(numerators[source] % total_rows), source))
    for source in order[:remaining]:
        quotas[source] += 1
    return quotas


def _inference_groups(lengths: list[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in sorted(range(len(lengths)), key=lambda i: (lengths[i], i)):
        next_max = max(current_max, lengths[index])
        if current and (len(current) >= 64 or next_max * (len(current) + 1) > 32_768):
            groups.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, lengths[index])
    if current:
        groups.append(current)
    return groups


def _source_files(dataset_root: str, source: str) -> tuple[tuple[str, int], ...]:
    input_root = StoragePath(dataset_root) / source / "outputs" / "main"
    if not input_root.exists():
        return ()
    files = []
    for input_url in sorted(str(item) for item in input_root.ls() if item.name.endswith(".parquet")):
        filesystem, path = url_to_fs(input_url)
        with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
            files.append((input_url, parquet_file.metadata.num_rows))
    logger.info("Found %d rows across %d files for %s", sum(rows for _, rows in files), len(files), source)
    return tuple(files)


def _output_metadata(part: InputPart) -> dict[bytes, bytes]:
    return {
        b"harrier_manifest_sha256": b"791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c",
        b"harrier_model_id": b"microsoft/harrier-oss-v1-0.6b",
        b"harrier_model_revision": b"f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        b"harrier_max_tokens": b"8192",
        b"harrier_pooling": b"last_token_l2_normalized",
        b"harrier_storage_dtype": b"float16",
        b"harrier_input_url": part.input_url.encode(),
        b"harrier_input_rows": str(part.row_count).encode(),
        b"harrier_source": part.source.encode(),
    }


def _output_is_complete(part: InputPart) -> bool:
    output_path = StoragePath(part.output_url)
    if not output_path.exists():
        return False
    filesystem, path = url_to_fs(str(output_path))
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        rows = parquet_file.metadata.num_rows
        metadata = parquet_file.schema_arrow.metadata or {}
    if rows != part.row_count:
        raise ValueError(f"Existing output has {rows} rows; expected {part.row_count}: {part.output_url}")
    if any(metadata.get(key) != value for key, value in _output_metadata(part).items()):
        raise ValueError(f"Existing output has different metadata: {part.output_url}")
    return True


def _input_batches(part: InputPart) -> Iterator[pa.RecordBatch]:
    filesystem, path = url_to_fs(part.input_url)
    remaining = part.row_count
    with pq.ParquetFile(path, filesystem=filesystem) as parquet_file:
        for batch in parquet_file.iter_batches(batch_size=8, columns=["id", "text"]):
            if remaining <= 0:
                break
            if len(batch) > remaining:
                batch = batch.slice(0, remaining)
            text_index = batch.schema.get_field_index("text")
            text = pc.utf8_slice_codeunits(batch.column(text_index), start=0, stop=1_048_576)
            batch = batch.set_column(text_index, batch.schema.field(text_index), text)
            remaining -= len(batch)
            yield batch
    if remaining:
        raise ValueError(f"Input ended with {remaining} selected rows missing: {part.input_url}")


class HarrierEmbedder:
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
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        if int(self.model.config.hidden_size) != 1_024:
            raise ValueError(f"Harrier hidden size is {self.model.config.hidden_size}; expected 1024")

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = np.empty((len(texts), 1_024), dtype=np.float16)
        for chunk_start in range(0, len(texts), 128):
            chunk = texts[chunk_start : chunk_start + 128]
            tokenized = self.tokenizer(
                chunk,
                add_special_tokens=True,
                padding=False,
                truncation=True,
                max_length=8_192,
                return_attention_mask=False,
            )["input_ids"]
            lengths = [len(input_ids) for input_ids in tokenized]
            for group in _inference_groups(lengths):
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


def _embed_part(embedder: HarrierEmbedder, part: InputPart) -> int:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("embedding", pa.list_(pa.float16(), 1_024)),
        ],
        metadata=_output_metadata(part),
    )
    rows = 0
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_path = Path(temporary_directory) / "embeddings.parquet"
        with pq.ParquetWriter(local_path, schema, compression="zstd") as writer:
            output_tables = []
            buffered_rows = 0
            for batch in _input_batches(part):
                ids = batch.column(batch.schema.get_field_index("id"))
                vectors = embedder.embed(batch.column(batch.schema.get_field_index("text")).to_pylist())
                embedding = pa.FixedSizeListArray.from_arrays(pa.array(vectors.reshape(-1)), 1_024)
                output_tables.append(pa.table({"id": ids, "embedding": embedding}, schema=schema))
                buffered_rows += len(batch)
                if buffered_rows >= 8_192:
                    writer.write_table(pa.concat_tables(output_tables), row_group_size=8_192)
                    output_tables = []
                    buffered_rows = 0
                rows += len(batch)
            if output_tables:
                writer.write_table(pa.concat_tables(output_tables), row_group_size=8_192)
        if rows != part.row_count:
            raise ValueError(f"Embedded {rows} rows; expected {part.row_count}: {part.input_url}")
        with atomic_rename(part.output_url) as temporary_path:
            StoragePath(temporary_path).upload_from(str(local_path))
    logger.info("Embedded %d rows from %s", rows, part.input_url)
    return rows


@contextmanager
def _harrier_embedder(output_root: str) -> Iterator[HarrierEmbedder]:
    staged_model = cache_hf_model(
        marin_temp_bucket(
            ttl_days=30,
            prefix="harrier-staging/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
            source_prefix=output_root,
        ),
        "microsoft/harrier-oss-v1-0.6b",
        revision="f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model"
        StoragePath(staged_model).download_to(str(model_path), recursive=True)
        yield HarrierEmbedder(model_path)


def main() -> None:
    """Run the fixed Harrier embedding pipeline for this Iris task."""
    job_info = get_job_info()
    if job_info is None:
        raise ValueError("Harrier embedding must run as an Iris job")

    dataset_root = "s3://marin-us-east-02a/marin/datakit/sample_10pct_91269634"
    output_root = "s3://marin-us-east-02a/marin/user/held/harrier-oss-v1-0.6b-50m"
    sources = sorted(all_sources())
    with ThreadPoolExecutor(max_workers=16) as pool:
        source_files = dict(
            zip(sources, pool.map(lambda source: _source_files(dataset_root, source), sources), strict=True)
        )
    missing = [source for source in sources if not source_files[source]]
    if missing:
        raise ValueError(f"Missing Parquet outputs for {len(missing)} canonical sources: {', '.join(missing)}")

    source_counts = {source: sum(rows for _, rows in source_files[source]) for source in sources}
    source_quotas = _allocate_source_quotas(source_counts, 50_000_000)
    input_parts = []
    for source in sources:
        remaining = source_quotas[source]
        for input_url, available_rows in source_files[source]:
            row_count = min(remaining, available_rows)
            input_parts.append(
                InputPart(
                    source=source,
                    input_url=input_url,
                    row_count=row_count,
                    output_url=str(StoragePath(output_root) / "embeddings" / source / StoragePath(input_url).name),
                )
            )
            remaining -= row_count
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"Source {source} is missing {remaining} selected rows")
    if sum(part.row_count for part in input_parts) != 50_000_000:
        raise ValueError("Input parts do not sum to 50,000,000 rows")

    plan = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "target_rows": 50_000_000,
        "model_id": "microsoft/harrier-oss-v1-0.6b",
        "model_revision": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
        "model_dimension": 1_024,
        "max_tokens": 8_192,
        "sources": [
            {"source": source, "available_rows": source_counts[source], "selected_rows": source_quotas[source]}
            for source in sources
        ],
        "parts": [asdict(part) for part in input_parts],
    }
    plan_sha256 = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if plan_sha256 != "791ce33496e7e99d54c17c4dfb5d71ce20a1273f021fef4f67c54da72e71e97c":
        raise ValueError(f"Input plan digest changed: {plan_sha256}")

    if not 0 <= job_info.task_index < job_info.num_tasks:
        raise ValueError(f"Task index {job_info.task_index} is outside [0, {job_info.num_tasks})")
    assignments: list[list[InputPart]] = [[] for _ in range(job_info.num_tasks)]
    assigned_rows = [0] * job_info.num_tasks
    for part in sorted(input_parts, key=lambda item: (-item.row_count, item.source, item.input_url)):
        task_index = min(range(job_info.num_tasks), key=lambda index: (assigned_rows[index], index))
        assignments[task_index].append(part)
        assigned_rows[task_index] += part.row_count
    parts = sorted(assignments[job_info.task_index], key=lambda part: (part.source, part.input_url))

    completed_rows = 0
    pending_parts = []
    for part in parts:
        if _output_is_complete(part):
            completed_rows += part.row_count
        else:
            pending_parts.append(part)
    if pending_parts:
        with _harrier_embedder(output_root) as embedder:
            for index, part in enumerate(pending_parts, start=1):
                logger.info(
                    "Embedding part %d/%d on worker %d/%d: %s",
                    index,
                    len(pending_parts),
                    job_info.task_index,
                    job_info.num_tasks,
                    part.input_url,
                )
                completed_rows += _embed_part(embedder, part)

    logger.info(
        "HARRIER_50M=%s",
        json.dumps(
            {
                "input_plan_sha256": plan_sha256,
                "model_id": "microsoft/harrier-oss-v1-0.6b",
                "model_revision": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
                "shard_index": job_info.task_index,
                "num_shards": job_info.num_tasks,
                "part_count": len(parts),
                "row_count": completed_rows,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
