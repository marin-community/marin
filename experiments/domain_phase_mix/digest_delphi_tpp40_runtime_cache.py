# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec>=2025.7.0",
#   "gcsfs>=2025.7.0",
#   "google-cloud-storage>=3.8.0",
#   "numpy>=2.0.0",
# ]
# ///

"""Hash a tokenized runtime cache independently of its physical TreeCache layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
from dataclasses import asdict, dataclass
from typing import Any

import fsspec
import numpy as np
from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from levanter.store.cache import CACHE_LAYOUT_CONSOLIDATED, CACHE_LAYOUT_SHARDED, TreeCache
from levanter.store.tree_store import TreeStore

ALGORITHM = "treecache-input-ids-logical-row-block-sha256-v4"
DEFAULT_BLOCK_ROWS = 4_096
EXEMPLAR = {"input_ids": np.zeros((0,), dtype=np.int32)}
FIELD = "input_ids"
EXPECTED_FIELDS = frozenset({FIELD})
SUPPORTED_DTYPES = frozenset({"int32", "int64"})
LOG_EVERY_BLOCKS = 128

logger = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class RowRange:
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class DigestBlock:
    output_row_start: int
    output_row_stop: int
    token_count: int
    sha256: str


@dataclass(frozen=True)
class ShardRuntimeStats:
    rows: int
    tokens: int


@dataclass(frozen=True)
class RuntimeEvidence:
    source_rows: int
    source_tokens: int
    dtype: str
    field_names: tuple[str, ...]
    shard_stats: dict[str, ShardRuntimeStats]


@dataclass(frozen=True)
class ExcludedShard:
    name: str
    source_row_start: int
    source_row_stop: int
    rows: int
    tokens: int


@dataclass(frozen=True)
class RuntimeObjectManifest:
    sha256: str
    objects: int
    bytes: int
    field_names: tuple[str, ...]


def runtime_object_manifest_binding(manifest: RuntimeObjectManifest) -> dict[str, object]:
    return {
        **asdict(manifest),
        "field_names": list(manifest.field_names),
    }


def normalize_exclusions(total_rows: int, exclusions: tuple[RowRange, ...]) -> tuple[RowRange, ...]:
    ordered = tuple(sorted(exclusions))
    previous_stop = 0
    for exclusion in ordered:
        if exclusion.start < 0 or exclusion.stop <= exclusion.start or exclusion.stop > total_rows:
            raise ValueError(f"Invalid excluded row range {exclusion} for cache with {total_rows} rows")
        if exclusion.start < previous_stop:
            raise ValueError(f"Excluded row ranges overlap: {ordered}")
        previous_stop = exclusion.stop
    return ordered


def included_row_ranges(total_rows: int, exclusions: tuple[RowRange, ...]) -> tuple[RowRange, ...]:
    exclusions = normalize_exclusions(total_rows, exclusions)
    included: list[RowRange] = []
    start = 0
    for exclusion in exclusions:
        if start < exclusion.start:
            included.append(RowRange(start, exclusion.start))
        start = exclusion.stop
    if start < total_rows:
        included.append(RowRange(start, total_rows))
    return tuple(included)


def _canonical_array(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(array, dtype=np.dtype(dtype)))


def _field_store_stats(field_store: Any) -> tuple[ShardRuntimeStats, str]:
    dtype = str(field_store.data.dtype.numpy_dtype)
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"Runtime cache field has unsupported dtype {dtype!r}; expected {sorted(SUPPORTED_DTYPES)}")
    return ShardRuntimeStats(rows=int(field_store.num_rows), tokens=int(field_store.data_size)), dtype


def validate_runtime_evidence(cache: TreeCache) -> RuntimeEvidence:
    ledger = cache.ledger
    if not ledger.is_finished:
        raise ValueError("Runtime cache ledger is not finalized")
    if cache.is_sharded:
        if set(ledger.field_counts) != EXPECTED_FIELDS:
            raise ValueError(f"Sharded cache ledger fields differ from {sorted(EXPECTED_FIELDS)}: {ledger.field_counts}")
        shard_stats: dict[str, ShardRuntimeStats] = {}
        runtime_dtype: str | None = None
        for shard_name in ledger.finished_shards:
            shard_path = f"{cache.cache_dir.rstrip('/')}/{shard_name}"
            shard_store = TreeStore.open(EXEMPLAR, shard_path, mode="r", cache_metadata=True)
            stats, shard_dtype = _field_store_stats(shard_store.tree[FIELD])
            if runtime_dtype is not None and shard_dtype != runtime_dtype:
                raise ValueError(
                    f"Shard {shard_name!r} dtype differs from earlier shards: {shard_dtype!r} != {runtime_dtype!r}"
                )
            runtime_dtype = shard_dtype
            expected_rows = ledger.shard_rows[shard_name]
            expected_tokens = ledger.field_counts_by_shard.get(shard_name, {}).get(FIELD)
            if expected_tokens is None and expected_rows == 0:
                expected_tokens = 0
            if stats.rows != expected_rows:
                raise ValueError(
                    f"Shard {shard_name!r} row count differs from its ledger: {stats.rows} != {expected_rows}"
                )
            if stats.tokens != expected_tokens:
                raise ValueError(
                    f"Shard {shard_name!r} token count differs from its ledger: {stats.tokens} != {expected_tokens}"
                )
            shard_stats[shard_name] = stats
        source_rows = sum(stats.rows for stats in shard_stats.values())
        source_tokens = sum(stats.tokens for stats in shard_stats.values())
        if source_rows != ledger.total_num_rows:
            raise ValueError(f"Runtime shard rows differ from root ledger: {source_rows} != {ledger.total_num_rows}")
        if source_tokens != ledger.field_counts[FIELD]:
            raise ValueError(
                f"Runtime shard tokens differ from root ledger: {source_tokens} != {ledger.field_counts[FIELD]}"
            )
        if runtime_dtype is None:
            raise ValueError("Sharded cache has no finished shards")
    else:
        if ledger.field_counts and set(ledger.field_counts) != EXPECTED_FIELDS:
            raise ValueError(
                f"Consolidated cache ledger fields differ from {sorted(EXPECTED_FIELDS)}: {ledger.field_counts}"
            )
        stats, runtime_dtype = _field_store_stats(cache.store.tree[FIELD])
        source_rows = stats.rows
        source_tokens = stats.tokens
        shard_stats = {}
        if source_rows != ledger.total_num_rows:
            raise ValueError(f"Runtime rows differ from consolidated ledger: {source_rows} != {ledger.total_num_rows}")
        if ledger.field_counts and source_tokens != ledger.field_counts[FIELD]:
            raise ValueError(
                f"Runtime tokens differ from consolidated ledger: {source_tokens} != {ledger.field_counts[FIELD]}"
            )

    return RuntimeEvidence(
        source_rows=source_rows,
        source_tokens=source_tokens,
        dtype=runtime_dtype,
        field_names=tuple(sorted(EXPECTED_FIELDS)),
        shard_stats=shard_stats,
    )


def excluded_shard_ranges(
    cache: TreeCache,
    runtime: RuntimeEvidence,
    excluded_shard_names: tuple[str, ...],
) -> tuple[tuple[RowRange, ...], tuple[ExcludedShard, ...]]:
    if not excluded_shard_names:
        return (), ()
    if not cache.is_sharded:
        raise ValueError("Shard exclusions require a sharded cache")
    if len(set(excluded_shard_names)) != len(excluded_shard_names):
        raise ValueError(f"Excluded shard names contain duplicates: {excluded_shard_names}")

    requested = set(excluded_shard_names)
    unknown = requested - set(cache.ledger.finished_shards)
    if unknown:
        raise ValueError(f"Excluded shard names are not in the runtime ledger: {sorted(unknown)}")

    row_cursor = 0
    exclusions: list[RowRange] = []
    evidence: list[ExcludedShard] = []
    for shard_name in cache.ledger.finished_shards:
        stats = runtime.shard_stats[shard_name]
        row_stop = row_cursor + stats.rows
        if shard_name in requested:
            exclusions.append(RowRange(row_cursor, row_stop))
            evidence.append(
                ExcludedShard(
                    name=shard_name,
                    source_row_start=row_cursor,
                    source_row_stop=row_stop,
                    rows=stats.rows,
                    tokens=stats.tokens,
                )
            )
        row_cursor = row_stop
    return tuple(exclusions), tuple(evidence)


def _read_row_segment(store: Any, row_range: RowRange, token_dtype: str) -> tuple[np.ndarray, np.ndarray]:
    if row_range.length <= 0:
        raise ValueError(f"Cannot read empty row range: {row_range}")

    row_ends = _canonical_array(
        store.offsets[row_range.start + 1 : row_range.stop + 1].read().result(),
        "<i8",
    )
    if len(row_ends) != row_range.length:
        raise ValueError(f"Offset slice returned {len(row_ends)} rows for requested range {row_range}")
    if row_range.start == 0:
        token_start = 0
    else:
        token_start_value = store.offsets[row_range.start : row_range.start + 1].read().result()
        token_start = int(np.asarray(token_start_value)[0])
    token_stop = int(row_ends[-1])
    row_lengths = _canonical_array(
        np.diff(np.concatenate((np.asarray([token_start], dtype=np.dtype("<i8")), row_ends))),
        "<i8",
    )
    if np.any(row_lengths < 0):
        raise ValueError(f"Offsets are not monotonic in requested range {row_range}")

    canonical_token_dtype = np.dtype(token_dtype).newbyteorder("<")
    tokens = _canonical_array(store.data[token_start:token_stop].read().result(), str(canonical_token_dtype))
    if int(row_lengths.sum()) != len(tokens):
        raise ValueError(
            f"Token count does not match row lengths for {row_range}: {len(tokens)} != {int(row_lengths.sum())}"
        )
    return row_lengths, tokens


def _source_segments_for_output_block(
    included: tuple[RowRange, ...],
    output_start: int,
    output_stop: int,
) -> tuple[RowRange, ...]:
    if output_start < 0 or output_stop <= output_start:
        raise ValueError(f"Invalid output row block [{output_start}, {output_stop})")

    segments: list[RowRange] = []
    output_cursor = 0
    for source_range in included:
        next_output_cursor = output_cursor + source_range.length
        overlap_start = max(output_start, output_cursor)
        overlap_stop = min(output_stop, next_output_cursor)
        if overlap_start < overlap_stop:
            source_start = source_range.start + overlap_start - output_cursor
            source_stop = source_range.start + overlap_stop - output_cursor
            segments.append(RowRange(source_start, source_stop))
        output_cursor = next_output_cursor
        if output_cursor >= output_stop:
            break
    if sum(segment.length for segment in segments) != output_stop - output_start:
        raise ValueError(f"Output block [{output_start}, {output_stop}) exceeds selected cache rows")
    return tuple(segments)


def digest_tree_cache(
    cache: TreeCache,
    *,
    expected_rows: int,
    expected_tokens: int,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    exclusions: tuple[RowRange, ...] = (),
    runtime: RuntimeEvidence | None = None,
) -> dict[str, object]:
    if block_rows <= 0:
        raise ValueError(f"block_rows must be positive, got {block_rows}")

    runtime = runtime or validate_runtime_evidence(cache)
    included = included_row_ranges(runtime.source_rows, exclusions)
    selected_rows = sum(row_range.length for row_range in included)
    if selected_rows <= 0:
        raise ValueError("Logical digest cannot select zero rows")
    if selected_rows != expected_rows:
        raise ValueError(f"Selected row count differs from expectation: {selected_rows} != {expected_rows}")

    store = cache.jagged_array_tree()[FIELD]
    blocks: list[DigestBlock] = []
    selected_tokens = 0
    total_blocks = (selected_rows + block_rows - 1) // block_rows
    for block_index, output_start in enumerate(range(0, selected_rows, block_rows)):
        output_stop = min(output_start + block_rows, selected_rows)
        source_segments = _source_segments_for_output_block(included, output_start, output_stop)
        lengths_parts: list[np.ndarray] = []
        token_parts: list[np.ndarray] = []
        for source_segment in source_segments:
            row_lengths, tokens = _read_row_segment(store, source_segment, runtime.dtype)
            lengths_parts.append(row_lengths)
            token_parts.append(tokens)

        token_count = sum(len(tokens) for tokens in token_parts)
        hasher = hashlib.sha256()
        hasher.update(struct.pack("<QQ", output_stop - output_start, token_count))
        for row_lengths in lengths_parts:
            hasher.update(row_lengths.tobytes(order="C"))
        for tokens in token_parts:
            hasher.update(tokens.tobytes(order="C"))
        blocks.append(
            DigestBlock(
                output_row_start=output_start,
                output_row_stop=output_stop,
                token_count=token_count,
                sha256=hasher.hexdigest(),
            )
        )
        selected_tokens += token_count
        if block_index % LOG_EVERY_BLOCKS == 0 or block_index + 1 == total_blocks:
            logger.info(
                "Hashed block %d/%d: rows %d:%d (%d tokens)",
                block_index + 1,
                total_blocks,
                output_start,
                output_stop,
                token_count,
            )

    if selected_tokens != expected_tokens:
        raise ValueError(f"Selected token count differs from expectation: {selected_tokens} != {expected_tokens}")

    root_payload = {
        "algorithm": ALGORITHM,
        "block_rows": block_rows,
        "selected_rows": selected_rows,
        "selected_tokens": selected_tokens,
        "dtype": runtime.dtype,
        "field_names": list(runtime.field_names),
        "blocks": [asdict(block) for block in blocks],
    }
    root_sha256 = digest_payload_sha256(root_payload)
    return {
        **root_payload,
        "source_rows": runtime.source_rows,
        "source_tokens": runtime.source_tokens,
        "excluded_row_ranges": [
            asdict(exclusion) for exclusion in normalize_exclusions(runtime.source_rows, exclusions)
        ],
        "logical_payload_sha256": root_sha256,
    }


def digest_payload_sha256(report: dict[str, object]) -> str:
    payload = {
        field: report.get(field)
        for field in ("algorithm", "block_rows", "selected_rows", "selected_tokens", "dtype", "field_names", "blocks")
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_contract_sha256(report: dict[str, object]) -> str:
    payload = {
        field: report.get(field)
        for field in (
            "status",
            "binding",
            "algorithm",
            "block_rows",
            "selected_rows",
            "source_rows",
            "selected_tokens",
            "source_tokens",
            "dtype",
            "field_names",
            "blocks",
            "excluded_row_ranges",
            "logical_payload_sha256",
        )
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_digest_artifact(
    report: dict[str, object],
    *,
    expected_binding: dict[str, object] | None = None,
) -> None:
    if report.get("status") != "complete":
        raise ValueError(f"Digest artifact is not complete: {report.get('status')!r}")
    binding = report.get("binding")
    if not isinstance(binding, dict):
        raise ValueError("Digest artifact lacks a binding object")
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("Digest artifact binding differs from the requested digest")
    if report.get("algorithm") != ALGORITHM or binding.get("algorithm") != report.get("algorithm"):
        raise ValueError("Digest artifact algorithm does not match its binding")
    if report.get("selected_rows") != binding.get("expected_rows"):
        raise ValueError("Digest selected_rows does not match its expected_rows binding")
    if report.get("selected_tokens") != binding.get("expected_tokens"):
        raise ValueError("Digest selected_tokens does not match its expected_tokens binding")
    if report.get("block_rows") != binding.get("block_rows"):
        raise ValueError("Digest block_rows does not match its binding")
    if report.get("dtype") not in SUPPORTED_DTYPES or report.get("field_names") != [FIELD]:
        raise ValueError(
            f"Digest artifact does not cover exactly supported {FIELD}: "
            f"dtype={report.get('dtype')!r}, fields={report.get('field_names')!r}"
        )

    source_rows = report.get("source_rows")
    selected_rows = report.get("selected_rows")
    source_tokens = report.get("source_tokens")
    selected_tokens = report.get("selected_tokens")
    block_rows = report.get("block_rows")
    if not isinstance(source_rows, int) or not isinstance(selected_rows, int) or not 0 < selected_rows <= source_rows:
        raise ValueError("Digest artifact has invalid source/selected row counts")
    if (
        not isinstance(source_tokens, int)
        or not isinstance(selected_tokens, int)
        or not 0 <= selected_tokens <= source_tokens
    ):
        raise ValueError("Digest artifact has invalid source/selected token counts")
    if not isinstance(block_rows, int) or block_rows <= 0:
        raise ValueError("Digest artifact has an invalid row-block size")

    excluded_row_ranges = report.get("excluded_row_ranges")
    if not isinstance(excluded_row_ranges, list) or not all(isinstance(item, dict) for item in excluded_row_ranges):
        raise ValueError("Digest artifact lacks an excluded_row_ranges list")
    try:
        exclusion_values = tuple((item["start"], item["stop"]) for item in excluded_row_ranges)
    except KeyError as error:
        raise ValueError("Digest artifact has malformed excluded row ranges") from error
    if not all(isinstance(start, int) and isinstance(stop, int) for start, stop in exclusion_values):
        raise ValueError("Digest artifact has malformed excluded row ranges")
    exclusions = tuple(RowRange(start=start, stop=stop) for start, stop in exclusion_values)
    normalized_exclusions = normalize_exclusions(source_rows, exclusions)
    if selected_rows != source_rows - sum(exclusion.length for exclusion in normalized_exclusions):
        raise ValueError("Digest selected_rows does not match its excluded row ranges")
    if not normalized_exclusions and selected_tokens != source_tokens:
        raise ValueError("Zero-exclusion digest selected_tokens does not match source_tokens")

    if not isinstance(binding.get("cache_path"), str):
        raise ValueError("Digest binding lacks a cache path")
    object_manifest = binding.get("runtime_object_manifest")
    if not isinstance(object_manifest, dict) or object_manifest.get("field_names") != [FIELD]:
        raise ValueError(f"Digest binding lacks an exact {FIELD} runtime-object manifest")
    if not isinstance(binding.get("ledger_sha256"), str):
        raise ValueError("Digest binding lacks a ledger SHA-256")
    if not isinstance(binding.get("preprocessor_metadata_sha256"), str):
        raise ValueError("Digest binding lacks a preprocessing-metadata SHA-256")
    excluded_shards = binding.get("excluded_shards")
    if not isinstance(excluded_shards, list) or not all(isinstance(item, dict) for item in excluded_shards):
        raise ValueError("Digest binding lacks an excluded_shards list")

    blocks = report.get("blocks")
    if not isinstance(blocks, list) or not blocks or not all(isinstance(block, dict) for block in blocks):
        raise ValueError("Digest artifact lacks a block list")
    row_cursor = 0
    token_total = 0
    for block in blocks:
        row_start = block.get("output_row_start")
        row_stop = block.get("output_row_stop")
        token_count = block.get("token_count")
        sha256 = block.get("sha256")
        if (
            not isinstance(row_start, int)
            or not isinstance(row_stop, int)
            or row_start != row_cursor
            or row_stop <= row_start
            or row_stop != min(row_start + block_rows, selected_rows)
            or row_stop > selected_rows
        ):
            raise ValueError("Digest block rows are not a contiguous partition of the selected rows")
        if not isinstance(token_count, int) or token_count < 0:
            raise ValueError("Digest block has an invalid token count")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("Digest block has an invalid SHA-256")
        row_cursor = row_stop
        token_total += token_count
    if row_cursor != selected_rows or token_total != selected_tokens:
        raise ValueError("Digest blocks do not cover the declared selected rows and tokens")
    if report.get("logical_payload_sha256") != digest_payload_sha256(report):
        raise ValueError("Digest artifact logical payload SHA-256 is invalid")
    if report.get("artifact_contract_sha256") != artifact_contract_sha256(report):
        raise ValueError("Digest artifact contract SHA-256 is invalid")


def _split_gcs_path(path: str) -> tuple[str, str]:
    if not path.startswith("gs://"):
        raise ValueError(f"Production digest inputs must use GCS: {path!r}")
    bucket, separator, prefix = path.removeprefix("gs://").partition("/")
    if not separator or not prefix:
        raise ValueError(f"Expected a GCS object prefix, got {path!r}")
    return bucket, prefix.rstrip("/")


def runtime_object_manifest(
    client: storage.Client,
    cache_path: str,
    *,
    layout: str,
    finished_shards: tuple[str, ...],
) -> RuntimeObjectManifest:
    bucket_name, cache_prefix = _split_gcs_path(cache_path)
    train_prefix = f"{cache_prefix}/train/"
    shard_names = set(finished_shards)
    payload: list[dict[str, object]] = []
    field_names: set[str] = set()
    blobs = client.list_blobs(
        bucket_name,
        prefix=train_prefix,
        fields="items(name,size,crc32c,generation),nextPageToken",
    )
    for blob in blobs:
        relative_path = blob.name.removeprefix(train_prefix)
        include = relative_path == "shard_ledger.json"
        field_name: str | None = None
        if layout == CACHE_LAYOUT_SHARDED:
            shard_name, separator, shard_relative = relative_path.partition("/")
            if separator and shard_name in shard_names:
                include = shard_relative == "shard_ledger.json" or shard_relative.startswith(f"{FIELD}/")
                if shard_relative.endswith("/data/zarr.json"):
                    field_name = shard_relative.removesuffix("/data/zarr.json")
        else:
            include = include or relative_path.startswith(f"{FIELD}/")
            if relative_path.endswith("/data/zarr.json"):
                field_name = relative_path.removesuffix("/data/zarr.json")
        if not include:
            continue
        if blob.crc32c is None or blob.generation is None:
            raise ValueError(f"Runtime object lacks immutable metadata: gs://{bucket_name}/{blob.name}")
        payload.append(
            {
                "name": relative_path,
                "size": int(blob.size or 0),
                "crc32c": blob.crc32c,
                "generation": int(blob.generation),
            }
        )
        if field_name is not None:
            field_names.add(field_name)
    if not payload:
        raise ValueError(f"Runtime cache has no payload objects: {cache_path}")
    if field_names != EXPECTED_FIELDS:
        raise ValueError(f"On-disk runtime fields differ from {sorted(EXPECTED_FIELDS)}: {sorted(field_names)}")
    payload.sort(key=lambda item: str(item["name"]))
    return RuntimeObjectManifest(
        sha256=hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        objects=len(payload),
        bytes=sum(int(item["size"]) for item in payload),
        field_names=tuple(sorted(field_names)),
    )


def _read_ledger(cache_path: str) -> tuple[bytes, dict[str, object]]:
    ledger_path = cache_path.rstrip("/") + "/train/shard_ledger.json"
    with fsspec.open(ledger_path, "rb") as handle:
        payload = handle.read()
    ledger = json.loads(payload)
    if not isinstance(ledger, dict):
        raise ValueError(f"Expected a JSON object at {ledger_path}")
    return payload, ledger


def _metadata_sha256(ledger: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(ledger.get("metadata"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _finished_shards(ledger: dict[str, object]) -> tuple[str, ...]:
    value = ledger.get("finished_shards", [])
    if not isinstance(value, list):
        raise ValueError("Runtime cache ledger has malformed finished_shards")
    return tuple(str(name) for name in value)


def _read_json(path: str) -> dict[str, object]:
    with fsspec.open(path, "rt") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _write_json_create_if_absent(
    client: storage.Client,
    path: str,
    payload: dict[str, object],
) -> bool:
    bucket_name, object_name = _split_gcs_path(path)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    blob = client.bucket(bucket_name).blob(object_name)
    try:
        blob.upload_from_string(serialized, content_type="application/json", if_generation_match=0)
    except PreconditionFailed as error:
        existing = _read_json(path)
        validate_digest_artifact(existing, expected_binding=payload["binding"])
        if existing != payload:
            raise FileExistsError(f"Concurrent digest artifact differs from the computed payload: {path}") from error
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", required=True, help="Runtime cache root containing train/")
    parser.add_argument("--output", required=True, help="GCS JSON output path")
    parser.add_argument("--expect-rows", type=int, required=True)
    parser.add_argument("--expect-tokens", type=int, required=True)
    parser.add_argument("--block-rows", type=int, default=DEFAULT_BLOCK_ROWS)
    parser.add_argument("--exclude-shard", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    cache_path = args.cache_path.rstrip("/")
    _split_gcs_path(args.output)
    client = storage.Client()

    ledger_bytes_before, ledger_before = _read_ledger(cache_path)
    layout = str(ledger_before.get("layout") or CACHE_LAYOUT_CONSOLIDATED)
    if layout not in {CACHE_LAYOUT_CONSOLIDATED, CACHE_LAYOUT_SHARDED}:
        raise ValueError(f"Unknown runtime cache layout: {layout!r}")
    finished_shards = _finished_shards(ledger_before)
    manifest_before = runtime_object_manifest(
        client,
        cache_path,
        layout=layout,
        finished_shards=finished_shards,
    )
    cache = TreeCache.load(cache_path + "/train", EXEMPLAR)
    runtime = validate_runtime_evidence(cache)
    exclusions, excluded_shards = excluded_shard_ranges(cache, runtime, tuple(args.exclude_shard))
    binding = {
        "algorithm": ALGORITHM,
        "cache_path": cache_path,
        "ledger_sha256": hashlib.sha256(ledger_bytes_before).hexdigest(),
        "preprocessor_metadata_sha256": _metadata_sha256(ledger_before),
        "runtime_object_manifest": runtime_object_manifest_binding(manifest_before),
        "block_rows": args.block_rows,
        "expected_rows": args.expect_rows,
        "expected_tokens": args.expect_tokens,
        "excluded_shards": [asdict(excluded) for excluded in excluded_shards],
    }

    fs, fs_path = fsspec.core.url_to_fs(args.output)
    if fs.exists(fs_path):
        existing = _read_json(args.output)
        validate_digest_artifact(existing, expected_binding=binding)
        print(
            json.dumps(
                {
                    "status": "skipped",
                    "output": args.output,
                    "logical_payload_sha256": existing.get("logical_payload_sha256"),
                    "binding": binding,
                },
                sort_keys=True,
            )
        )
        return

    digest = digest_tree_cache(
        cache,
        expected_rows=args.expect_rows,
        expected_tokens=args.expect_tokens,
        block_rows=args.block_rows,
        exclusions=exclusions,
        runtime=runtime,
    )
    ledger_bytes_after, ledger_after = _read_ledger(cache_path)
    manifest_after = runtime_object_manifest(
        client,
        cache_path,
        layout=layout,
        finished_shards=finished_shards,
    )
    if ledger_bytes_after != ledger_bytes_before:
        raise ValueError("Runtime cache ledger changed while the digest was running")
    if manifest_after != manifest_before:
        raise ValueError("Runtime cache payload objects changed while the digest was running")
    if _metadata_sha256(ledger_after) != binding["preprocessor_metadata_sha256"]:
        raise ValueError("Runtime cache preprocessing metadata changed while the digest was running")

    report = {"status": "complete", "binding": binding, **digest}
    report["artifact_contract_sha256"] = artifact_contract_sha256(report)
    created = _write_json_create_if_absent(client, args.output, report)
    print(
        json.dumps(
            {
                "status": "complete" if created else "skipped-after-race",
                "output": args.output,
                "selected_rows": report["selected_rows"],
                "selected_tokens": report["selected_tokens"],
                "logical_payload_sha256": report["logical_payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
