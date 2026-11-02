# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact per-step training batches, exported once from a run's data loader and replayed elsewhere.

Each optimizer step is one zstd-compressed ``.npz`` holding the step's ``[batch, seq]`` tokens, loss
weights and segment ids. Replaying the files in order reproduces the source run's batches without
access to its source caches (for example on CoreWeave, which cannot read the GCS caches).
"""

import io
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import zstandard
from levanter.data.dataset import AsyncDataset
from levanter.data.text.examples import GrugAttentionMask, GrugLmExample
from rigging.filesystem.buckets import filesystem_for

logger = logging.getLogger(__name__)

ZSTD_LEVEL = 3
# A stalled object-store read on one host stalls every host at the next collective (the fused-CE
# plan allgather times out after 200 s), so each read is abandoned and retried well before that.
READ_TIMEOUT = 45
READ_ATTEMPTS = 4


def step_file(root: str, step: int) -> str:
    return f"{root.rstrip('/')}/step-{step:06d}.npz.zst"


@dataclass(frozen=True)
class StepBatch:
    tokens: np.ndarray  # [B, S] uint32
    loss_weight: np.ndarray  # [B, S] uint8 when binary, else float16
    segment_ids: np.ndarray  # [B, S] int32
    sliding_window: int | None


def batch_from_examples(examples: Sequence[GrugLmExample]) -> StepBatch:
    """Pack one step's examples, checking that the encoding loses nothing."""
    masks = {(ex.attn_mask.is_causal, ex.attn_mask.sliding_window, ex.attn_mask.fa4_bounds is None) for ex in examples}
    if len(masks) != 1:
        raise ValueError(f"Mixed attention masks in one batch: {masks}")
    is_causal, sliding_window, no_bounds = masks.pop()
    if not is_causal or not no_bounds:
        raise ValueError("Replay supports causal masks without precomputed FA4 bounds")
    segment_ids = []
    for ex in examples:
        if ex.attn_mask.segment_ids is None:
            raise ValueError("Replay expects block-cross-document segment ids")
        q, kv = (np.asarray(x) for x in ex.attn_mask.segment_ids)
        if not np.array_equal(q, kv):
            raise ValueError("Replay expects identical query and key segment ids")
        segment_ids.append(q.astype(np.int32))
    tokens = np.stack([np.asarray(ex.tokens) for ex in examples]).astype(np.uint32)
    weights = np.stack([np.asarray(ex.loss_weight, dtype=np.float32) for ex in examples])
    if np.isin(weights, (0.0, 1.0)).all():
        loss_weight = weights.astype(np.uint8)
    else:
        loss_weight = weights.astype(np.float16)
        if not np.array_equal(loss_weight.astype(np.float32), weights):
            raise ValueError("Loss weights are not representable in float16")
    return StepBatch(tokens, loss_weight, np.stack(segment_ids), sliding_window)


def encode(batch: StepBatch) -> bytes:
    buffer = io.BytesIO()
    np.savez(
        buffer,
        tokens=batch.tokens,
        loss_weight=batch.loss_weight,
        segment_ids=batch.segment_ids,
        sliding_window=np.array(-1 if batch.sliding_window is None else batch.sliding_window),
    )
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL, threads=-1).compress(buffer.getvalue())


def decode(blob: bytes) -> StepBatch:
    arrays = np.load(io.BytesIO(zstandard.ZstdDecompressor().decompress(blob)))
    window = int(arrays["sliding_window"])
    return StepBatch(
        tokens=arrays["tokens"],
        loss_weight=arrays["loss_weight"],
        segment_ids=arrays["segment_ids"],
        sliding_window=None if window < 0 else window,
    )


def write_step(root: str, step: int, batch: StepBatch) -> None:
    fs, path = filesystem_for(step_file(root, step))
    fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    fs.pipe(path, encode(batch))


def read_step(root: str, step: int) -> StepBatch:
    fs, path = filesystem_for(step_file(root, step))
    for attempt in range(1, READ_ATTEMPTS + 1):
        # A stalled read can't be cancelled; leave its thread behind instead of waiting on it.
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(fs.cat, path)
        pool.shutdown(wait=False)
        try:
            return decode(future.result(timeout=READ_TIMEOUT))
        except FutureTimeout:
            logger.warning("Read of %s stalled for %d s (attempt %d/%d)", path, READ_TIMEOUT, attempt, READ_ATTEMPTS)
    raise TimeoutError(f"{path} did not arrive in {READ_ATTEMPTS} attempts of {READ_TIMEOUT} s")


def example_at(batch: StepBatch, row: int) -> GrugLmExample:
    mask = GrugAttentionMask.causal(sliding_window=batch.sliding_window).with_segment_ids(
        jnp.asarray(batch.segment_ids[row])
    )
    return GrugLmExample(
        tokens=jnp.asarray(batch.tokens[row].astype(np.int32)),
        loss_weight=jnp.asarray(batch.loss_weight[row].astype(np.float32)),
        attn_mask=mask,
    )


class ReplayBatchDataset(AsyncDataset[GrugLmExample]):
    """Sequences of exported steps ``0..num_steps-1``, in loader order (index = step * batch_size + row)."""

    def __init__(self, root: str, *, batch_size: int, num_steps: int):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_steps = num_steps
        self._cached: tuple[int, StepBatch] | None = None

    async def async_len(self) -> int:
        return self.batch_size * self.num_steps

    def is_finite(self) -> bool:
        return True

    def _step(self, step: int) -> StepBatch:
        if self._cached is None or self._cached[0] != step:
            batch = read_step(self.root, step)
            if batch.tokens.shape[0] != self.batch_size:
                raise ValueError(f"step {step} has {batch.tokens.shape[0]} rows, expected {self.batch_size}")
            self._cached = (step, batch)
        return self._cached[1]

    async def get_batch(self, indices: Sequence[int]) -> Sequence[GrugLmExample]:
        return [example_at(self._step(i // self.batch_size), i % self.batch_size) for i in indices]
