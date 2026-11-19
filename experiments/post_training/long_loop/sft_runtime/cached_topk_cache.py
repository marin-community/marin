# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sidecar cache writer and multi-host MixtureDataset reader for cached teacher top-k logprobs.

Format ``cached_teacher_topk_v1`` (fsspec-compatible, local or object-store URI):
  <cache_dir>/
    manifest.json        # schema metadata, dtype, shard layout, optional per-row token SHA-256
    shard-000000.npz     # ids: int32[R, S, K], logprobs_bits (uint16 for bf16/fp16) or logprobs (fp32)

Alignment contract:
  Stored position ``i`` MUST hold the teacher distribution predicting ``tokens[i + 1]`` (aligned
  with ``loss_weight[i]`` in ``Transformer.next_token_loss_from_hidden``). Use
  :func:`left_shift_prompt_logprobs` when converting raw vLLM ``prompt_logprobs`` arrays.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import io
import json
import logging
from typing import Literal

import fsspec
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.utils.jax_utils import local_cpu_mesh
from levanter.utils.thread_utils import blocking_wait

from experiments.post_training.long_loop.sft_runtime.temperature_matching import CachedTeacherTopK

CACHED_TOPK_FORMAT = "cached_teacher_topk_v1"
LogprobStorageDtype = Literal["bfloat16", "float16", "float32"]
_SUPPORTED_LP_DTYPES: dict[str, np.dtype] = {
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}


def row_tokens_sha256(tokens_row: np.ndarray) -> str:
    """Return the canonical SHA-256 hex digest for one int32 token sequence row."""
    arr = np.ascontiguousarray(tokens_row, dtype=np.int32)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def left_shift_prompt_logprobs(
    prompt_ids: np.ndarray,
    prompt_logprobs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-shift vLLM-style ``prompt_logprobs`` by 1 so position ``i`` predicts ``token[i + 1]``.

    The trailing position ``S - 1`` has no next token and is filled with ``ids = -1``, ``logprobs = 0.0``.
    """
    if prompt_ids.shape != prompt_logprobs.shape or prompt_ids.ndim < 2:
        raise ValueError(
            f"prompt_ids {prompt_ids.shape} and prompt_logprobs {prompt_logprobs.shape} must match with ndim >= 2"
        )
    shifted_ids = np.full_like(prompt_ids, -1, dtype=np.int32)
    shifted_lp = np.zeros_like(prompt_logprobs)
    if prompt_ids.shape[-2] > 1:
        shifted_ids[..., :-1, :] = prompt_ids[..., 1:, :].astype(np.int32, copy=False)
        shifted_lp[..., :-1, :] = prompt_logprobs[..., 1:, :]
    return shifted_ids, shifted_lp


def _join_uri(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}"


def _encode_logprobs_for_npz(logprobs: np.ndarray, dtype_name: str) -> tuple[str, np.ndarray]:
    if dtype_name not in _SUPPORTED_LP_DTYPES:
        raise ValueError(f"Unsupported logprobs_dtype {dtype_name!r}; expected one of {sorted(_SUPPORTED_LP_DTYPES)}")
    if dtype_name == "bfloat16":
        casted = np.ascontiguousarray(logprobs, dtype=ml_dtypes.bfloat16)
        return "logprobs_bf16_u16", casted.view(np.uint16)
    if dtype_name == "float16":
        return "logprobs_fp16", np.ascontiguousarray(logprobs, dtype=np.float16)
    return "logprobs_fp32", np.ascontiguousarray(logprobs, dtype=np.float32)


def _decode_logprobs_from_npz(npz_data, dtype_name: str) -> np.ndarray:
    if dtype_name == "bfloat16":
        return np.ascontiguousarray(npz_data["logprobs_bf16_u16"], dtype=np.uint16).view(ml_dtypes.bfloat16)
    if dtype_name == "float16":
        return np.asarray(npz_data["logprobs_fp16"], dtype=np.float16)
    if dtype_name == "float32":
        return np.asarray(npz_data["logprobs_fp32"], dtype=np.float32)
    raise ValueError(f"Unsupported logprobs_dtype {dtype_name!r} in manifest")


def write_cached_topk_source(
    out_dir: str,
    *,
    ids: np.ndarray,
    logprobs: np.ndarray,
    vocab_size: int,
    tokens: np.ndarray,
    logprobs_dtype: LogprobStorageDtype = "bfloat16",
    rows_per_shard: int = 16,
) -> dict[str, object]:
    """Write a ``cached_teacher_topk_v1`` directory readable by :class:`CachedTopKSourceReader`.

    ``tokens`` is required so every manifest records per-row ``token_sha256_per_row`` digests
    for fail-closed token-ID parity verification at load time.
    """
    ids_arr = np.asarray(ids, dtype=np.int32)
    if ids_arr.ndim != 3:
        raise ValueError(f"ids must have shape [N, S, K], got {ids_arr.shape}")
    if np.asarray(logprobs).shape != ids_arr.shape:
        raise ValueError(f"logprobs shape {np.asarray(logprobs).shape} does not match ids shape {ids_arr.shape}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if rows_per_shard <= 0:
        raise ValueError(f"rows_per_shard must be positive, got {rows_per_shard}")

    num_rows, seq_len, k_slots = ids_arr.shape
    tok_arr = np.asarray(tokens, dtype=np.int32)
    if tok_arr.shape != (num_rows, seq_len):
        raise ValueError(f"tokens shape {tok_arr.shape} must match (num_rows={num_rows}, seq_len={seq_len})")
    token_hashes = [row_tokens_sha256(tok_arr[i]) for i in range(num_rows)]

    fs, root_path = fsspec.core.url_to_fs(out_dir)
    fs.makedirs(root_path, exist_ok=True)

    num_shards = (num_rows + rows_per_shard - 1) // rows_per_shard
    for shard_idx in range(num_shards):
        start = shard_idx * rows_per_shard
        stop = min(start + rows_per_shard, num_rows)
        lp_key, lp_arr = _encode_logprobs_for_npz(np.asarray(logprobs)[start:stop], logprobs_dtype)
        buf = io.BytesIO()
        np.savez(buf, ids=np.ascontiguousarray(ids_arr[start:stop], dtype=np.int32), **{lp_key: lp_arr})
        shard_uri = _join_uri(out_dir, f"shard-{shard_idx:06d}.npz")
        with fsspec.open(shard_uri, "wb") as handle:
            handle.write(buf.getvalue())

    manifest: dict[str, object] = {
        "format": CACHED_TOPK_FORMAT,
        "num_rows": int(num_rows),
        "seq_len": int(seq_len),
        "k_slots": int(k_slots),
        "vocab_size": int(vocab_size),
        "ids_dtype": "int32",
        "logprobs_dtype": logprobs_dtype,
        "rows_per_shard": int(rows_per_shard),
        "num_shards": int(num_shards),
        "token_sha256_per_row": token_hashes,
    }
    manifest_uri = _join_uri(out_dir, "manifest.json")
    with fsspec.open(manifest_uri, "wb") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))
    return manifest


@dataclass(frozen=True)
class CachedTopKManifest:
    num_rows: int
    seq_len: int
    k_slots: int
    vocab_size: int
    logprobs_dtype: str
    rows_per_shard: int
    token_sha256_per_row: tuple[str, ...] | None


class CachedTopKSourceReader:
    """Random-access shard reader for one ``cached_teacher_topk_v1`` source directory."""

    def __init__(self, cache_dir: str, *, max_cached_shards: int = 4) -> None:
        if max_cached_shards <= 0:
            raise ValueError(f"max_cached_shards must be positive, got {max_cached_shards}")
        self.cache_dir = cache_dir
        self._max_cached_shards = max_cached_shards
        self._shard_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

        manifest_uri = _join_uri(cache_dir, "manifest.json")
        with fsspec.open(manifest_uri, "rb") as handle:
            raw = json.loads(handle.read().decode("utf-8"))
        if raw.get("format") != CACHED_TOPK_FORMAT:
            raise ValueError(
                f"Unsupported cached top-k manifest format {raw.get('format')!r} at {manifest_uri}; "
                f"expected {CACHED_TOPK_FORMAT!r}"
            )
        lp_dtype = str(raw["logprobs_dtype"])
        if lp_dtype not in _SUPPORTED_LP_DTYPES:
            raise ValueError(f"Unsupported logprobs_dtype {lp_dtype!r} in {manifest_uri}")
        raw_hashes = raw.get("token_sha256_per_row")
        token_hashes = tuple(str(h) for h in raw_hashes) if raw_hashes is not None else None
        num_rows = int(raw["num_rows"])
        if token_hashes is not None and len(token_hashes) != num_rows:
            raise ValueError(
                f"token_sha256_per_row length ({len(token_hashes)}) != num_rows ({num_rows}) in {manifest_uri}"
            )
        self.manifest = CachedTopKManifest(
            num_rows=num_rows,
            seq_len=int(raw["seq_len"]),
            k_slots=int(raw["k_slots"]),
            vocab_size=int(raw["vocab_size"]),
            logprobs_dtype=lp_dtype,
            rows_per_shard=int(raw["rows_per_shard"]),
            token_sha256_per_row=token_hashes,
        )

    def _load_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached
        shard_uri = _join_uri(self.cache_dir, f"shard-{shard_idx:06d}.npz")
        with fsspec.open(shard_uri, "rb") as handle:
            raw_bytes = handle.read()
        with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as npz_data:
            ids = np.asarray(npz_data["ids"], dtype=np.int32)
            logprobs = _decode_logprobs_from_npz(npz_data, self.manifest.logprobs_dtype)
        self._shard_cache[shard_idx] = (ids, logprobs)
        while len(self._shard_cache) > self._max_cached_shards:
            self._shard_cache.popitem(last=False)
        return ids, logprobs

    def read_row(self, row_idx: int) -> tuple[np.ndarray, np.ndarray, str | None]:
        """Return ``(ids[S, K], logprobs[S, K], expected_token_sha256 | None)`` for ``row_idx``."""
        if row_idx < 0 or row_idx >= self.manifest.num_rows:
            raise IndexError(
                f"Cached top-k row {row_idx} out of range [0, {self.manifest.num_rows}) for {self.cache_dir}"
            )
        shard_idx = row_idx // self.manifest.rows_per_shard
        local_row = row_idx % self.manifest.rows_per_shard
        shard_ids, shard_lp = self._load_shard(shard_idx)
        expected_hash = (
            self.manifest.token_sha256_per_row[row_idx]
            if self.manifest.token_sha256_per_row is not None
            else None
        )
        return shard_ids[local_row], shard_lp[local_row], expected_hash


logger = logging.getLogger(__name__)


def _extract_local_token_rows(batch_tokens: jax.Array) -> dict[int, np.ndarray]:
    """Read locally addressable full-sequence batch rows from ``batch_tokens`` (zero cross-host traffic).

    Fails closed if any addressable shard carries a partial or strided sequence slice so token-ID
    parity verification can never silently skip context-sharded batches.
    """
    local_rows: dict[int, np.ndarray] = {}
    seq_len = batch_tokens.shape[1]
    for shard in batch_tokens.addressable_shards:
        row_slice, pos_slice = shard.index
        pos_start, pos_stop, pos_step = pos_slice.indices(seq_len)
        if pos_start != 0 or pos_stop != seq_len or pos_step != 1:
            raise ValueError(
                "Token-hash verification requires full-sequence batch_tokens shards "
                f"(got position slice {pos_slice} for seq_len={seq_len})"
            )
        row_start, row_stop, row_step = row_slice.indices(batch_tokens.shape[0])
        if row_step != 1:
            raise ValueError(f"Token-hash verification requires contiguous batch_tokens row slices (got {row_slice})")
        data_np = np.asarray(shard.data, dtype=np.int32)
        for offset, global_row in enumerate(range(row_start, row_stop)):
            if global_row not in local_rows:
                local_rows[global_row] = data_np[offset]
    return local_rows


class CachedTopKMixtureReader:
    """Attaches sharded :class:`CachedTeacherTopK` tensors to ``MixtureDataset`` batches.

    Resolves each global mixture index to its exact ``(source_name, source_row_idx)`` matching
    ``MixtureDataset.get_batch`` (including finite-dataset wraparound under ``RESTART_STRATEGY``),
    reads only the rows owned by this host's batch shards, and enforces fail-closed per-row token
    SHA-256 verification against ``batch.tokens`` when ``verify_token_hashes=True``.
    """

    def __init__(
        self,
        dataset: MixtureDataset,
        source_paths: Mapping[str, str],
        *,
        seq_len: int,
        vocab_size: int,
        verify_token_hashes: bool = True,
        require_full_source_coverage: bool = False,
        max_cached_shards: int = 4,
    ) -> None:
        if not source_paths:
            raise ValueError("CachedTopKMixtureReader requires at least one source path")
        available = set(dataset.dataset_index)
        missing = sorted(set(source_paths) - available)
        if missing:
            raise ValueError(
                f"cached_topk_paths references unknown mixture sources {missing}; available are {sorted(available)}"
            )
        self.dataset = dataset
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.verify_token_hashes = verify_token_hashes
        self.require_full_source_coverage = require_full_source_coverage
        self.last_uncovered_rows = 0

        self._readers: dict[str, CachedTopKSourceReader] = {}
        self._source_lengths: dict[str, int | None] = {}
        k_slots = 0
        lp_dtypes: set[str] = set()
        for source_name, path in source_paths.items():
            reader = CachedTopKSourceReader(path, max_cached_shards=max_cached_shards)
            if reader.manifest.seq_len != seq_len:
                raise ValueError(
                    f"Cached top-k source {source_name!r} has seq_len={reader.manifest.seq_len}, "
                    f"expected {seq_len}"
                )
            if reader.manifest.vocab_size != vocab_size:
                raise ValueError(
                    f"Cached top-k source {source_name!r} has vocab_size={reader.manifest.vocab_size}, "
                    f"expected {vocab_size}"
                )
            if verify_token_hashes and reader.manifest.token_sha256_per_row is None:
                raise ValueError(
                    f"Cached top-k source {source_name!r} at {path} is missing token_sha256_per_row in manifest.json; "
                    "token-hash verification cannot run without per-row digests"
                )
            self._readers[source_name] = reader
            k_slots = max(k_slots, reader.manifest.k_slots)
            lp_dtypes.add(reader.manifest.logprobs_dtype)

        for ds_id, source_name in enumerate(dataset.dataset_index):
            sub_ds = dataset._dataset_of_id(ds_id)
            if (
                dataset.stop_strategy in (StopStrategy.RESTART_STRATEGY, StopStrategy.ALL_STOP_STRATEGY)
                and sub_ds.is_finite()
            ):
                source_len = int(blocking_wait(sub_ds.async_len()))
                if source_len <= 0:
                    raise ValueError(f"Mixture source {source_name!r} reported non-positive length {source_len}")
                self._source_lengths[source_name] = source_len
            else:
                self._source_lengths[source_name] = None

        self.k_slots = k_slots
        if len(lp_dtypes) == 1:
            self.logprobs_dtype_name = next(iter(lp_dtypes))
        else:
            self.logprobs_dtype_name = "float32"
        self._np_lp_dtype = _SUPPORTED_LP_DTYPES[self.logprobs_dtype_name]

    def resolve_mixture_indices(self, indices: Sequence[int]) -> list[tuple[str, int]]:
        """Map global mixture indices to ``(source_name, source_row_idx)`` matching ``MixtureDataset``."""
        resolved: list[tuple[str, int]] = []
        block_size = self.dataset.block_size
        dataset_index = self.dataset.dataset_index
        # MixtureDataset.__init__ places self.key under local_cpu_mesh() (Auto); match that mesh
        # so cold-cache _get_block() calls work inside an active Explicit training mesh.
        with local_cpu_mesh():
            for raw_index in indices:
                rel_index = self.dataset._relative_index(int(raw_index))
                block_id = rel_index // block_size
                block = self.dataset._get_block(int(block_id))
                packed_id = int(block[rel_index % block_size])
                dataset_id, source_row_idx = self.dataset._index_into_dataset_for_id(packed_id, int(block_id))
                source_name = dataset_index[dataset_id]
                source_len = self._source_lengths.get(source_name)
                if source_len is not None:
                    source_row_idx = source_row_idx % source_len
                resolved.append((source_name, int(source_row_idx)))
        return resolved

    def _read_batch_row(
        self,
        batch_row_idx: int,
        source_name: str,
        source_row_idx: int,
        local_tokens: Mapping[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        reader = self._readers.get(source_name)
        if reader is None:
            return (
                np.full((self.seq_len, self.k_slots), -1, dtype=np.int32),
                np.zeros((self.seq_len, self.k_slots), dtype=self._np_lp_dtype),
                False,
            )
        if source_row_idx >= reader.manifest.num_rows:
            if self.require_full_source_coverage:
                raise IndexError(
                    f"Cached top-k source {source_name!r} covers {reader.manifest.num_rows} rows, "
                    f"but mixture requested row {source_row_idx}"
                )
            return (
                np.full((self.seq_len, self.k_slots), -1, dtype=np.int32),
                np.zeros((self.seq_len, self.k_slots), dtype=self._np_lp_dtype),
                True,
            )

        row_ids, row_lp, expected_hash = reader.read_row(source_row_idx)
        if self.verify_token_hashes:
            if expected_hash is None:
                raise ValueError(
                    f"Cached top-k source {source_name!r} row {source_row_idx} is missing a token SHA-256 digest"
                )
            if batch_row_idx not in local_tokens:
                raise ValueError(
                    f"Cannot verify token-ID parity for cached top-k source {source_name!r} row {source_row_idx}: "
                    f"batch row {batch_row_idx} is not present in locally addressable batch_tokens shards"
                )
            actual_hash = row_tokens_sha256(local_tokens[batch_row_idx])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Token-ID parity mismatch for cached top-k source {source_name!r} row {source_row_idx} "
                    f"(batch row {batch_row_idx}): expected sha256={expected_hash[:12]}, got {actual_hash[:12]}"
                )

        if row_ids.shape[1] < self.k_slots:
            padded_ids = np.full((self.seq_len, self.k_slots), -1, dtype=np.int32)
            padded_lp = np.zeros((self.seq_len, self.k_slots), dtype=self._np_lp_dtype)
            padded_ids[:, : row_ids.shape[1]] = row_ids
            padded_lp[:, : row_lp.shape[1]] = row_lp.astype(self._np_lp_dtype, copy=False)
            return padded_ids, padded_lp, False
        return row_ids, row_lp.astype(self._np_lp_dtype, copy=False), False

    def load_batch_topk(
        self,
        indices: Sequence[int],
        *,
        mesh: Mesh,
        batch_axis_spec,
        batch_tokens: jax.Array | None = None,
    ) -> CachedTeacherTopK:
        """Load a sharded :class:`CachedTeacherTopK` matching ``indices`` onto ``mesh``."""
        resolved = self.resolve_mixture_indices(indices)
        batch_size = len(resolved)
        if self.verify_token_hashes:
            if batch_tokens is None:
                raise ValueError("batch_tokens must be provided when verify_token_hashes=True")
            local_tokens = _extract_local_token_rows(batch_tokens)
        else:
            local_tokens = {}

        row_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        uncovered_rows: set[int] = set()

        def _get_row(r: int) -> tuple[np.ndarray, np.ndarray]:
            cached = row_cache.get(r)
            if cached is None:
                source_name, source_row_idx = resolved[r]
                ids_row, lp_row, was_uncovered = self._read_batch_row(r, source_name, source_row_idx, local_tokens)
                if was_uncovered:
                    uncovered_rows.add(r)
                cached = (ids_row, lp_row)
                row_cache[r] = cached
            return cached

        def _slice_ids(index):
            row_slice, pos_slice, k_slice = index
            row_indices = range(*row_slice.indices(batch_size))
            return np.stack([_get_row(r)[0][pos_slice, k_slice] for r in row_indices], axis=0)

        def _slice_lp(index):
            row_slice, pos_slice, k_slice = index
            row_indices = range(*row_slice.indices(batch_size))
            return np.stack([_get_row(r)[1][pos_slice, k_slice] for r in row_indices], axis=0)

        shape = (batch_size, self.seq_len, self.k_slots)
        sharding = NamedSharding(mesh, P(batch_axis_spec, None, None))
        ids = jax.make_array_from_callback(shape, sharding, _slice_ids)
        logprobs = jax.make_array_from_callback(shape, sharding, _slice_lp)
        self.last_uncovered_rows = len(uncovered_rows)
        if self.last_uncovered_rows > 0:
            logger.info(
                "Cached top-k batch had %d local row(s) beyond configured source coverage (masked out of KL)",
                self.last_uncovered_rows,
            )
        return CachedTeacherTopK(ids=ids, logprobs=logprobs)
