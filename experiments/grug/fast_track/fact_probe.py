# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fact probe: how a run learns, and keeps, short facts from the batches it trains on.

Two passes over the same seed (the data order is a pure function of the seed):

* ``train_batch_dump_steps`` writes the exact train batches of chosen steps (``write_train_batch``) and stops
  before training, so facts can be picked from them offline.
* ``fact_probe_input`` is an npz built offline (``write_fact_probe_input``): rows of tokens (e.g. a fact's
  original training window, the fact alone after BOS, never-seen control text) and spans naming the fact tokens
  inside them. Every ``fact_probe_every`` steps the run scores those rows with the raw weights (and, every
  ``fact_probe_ema_every`` steps once it is live, the EMA weights) and ``FactProbeWriter`` records, for every span
  token, its loss and the model's top-``TOP_K`` next-token predictions, plus every row's mean loss.

A span ``(row, start, end)`` scores the prediction of ``tokens[row, start:end]``, i.e. the model's outputs at
positions ``start - 1 .. end - 2``. A batch dumped at ``data_step`` is consumed by the update that takes the step
count from ``data_step`` to ``data_step + 1``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

import fsspec
import jax
import numpy as np
from levanter.data.text.examples import GrugLmExample, causal_example_on_host

logger = logging.getLogger(__name__)

TRAIN_BATCH_DUMP_FILE = "train_batch_step{step}.npz"
TRAIN_TEXT_COUNTS_FILE = "train_text_counts.npz"
FACT_PROBE_CHUNK_FILE = "fact_probe_{kind}_{index:04d}.npz"
TOP_K = 5
RAW = "raw"
EMA = "ema"


def write_train_batch(path: str, step: int, tokens: np.ndarray, loss_weight: np.ndarray) -> None:
    with fsspec.open(path, "wb") as f:
        np.savez(f, step=step, tokens=tokens.astype(np.int32), loss_weight=loss_weight.astype(np.float32))
    logger.info("wrote train batch of step %d (%s) to %s", step, tokens.shape, path)


@dataclass(frozen=True)
class FactProbeInput:
    tokens: np.ndarray  # [N, S] int32
    span_row: np.ndarray  # [K]
    span_start: np.ndarray  # [K]
    span_end: np.ndarray  # [K]

    def __post_init__(self):
        if np.any(self.span_start < 1) or np.any(self.span_end <= self.span_start):
            raise ValueError("every fact span needs 1 <= start < end")
        if np.any(self.span_end > self.tokens.shape[1]) or np.any(self.span_row >= self.tokens.shape[0]):
            raise ValueError("fact span outside the probe rows")

    @classmethod
    def load(cls, path: str) -> FactProbeInput:
        with fsspec.open(path, "rb") as f:
            data = np.load(f)
            return cls(
                tokens=data["tokens"].astype(np.int32),
                span_row=data["span_row"].astype(np.int64),
                span_start=data["span_start"].astype(np.int64),
                span_end=data["span_end"].astype(np.int64),
            )

    def positions(self) -> np.ndarray:
        """``[T, 2]`` (row, position) of every span token's prediction, spans in order."""
        return np.concatenate(
            [
                np.stack([np.full(end - start, row), np.arange(start - 1, end - 1)], axis=1)
                for row, start, end in zip(self.span_row, self.span_start, self.span_end, strict=True)
            ]
        ).astype(np.int32)

    def example(self, eos_id: int) -> GrugLmExample:
        """The rows as one host batch, masked by document like the train loader (a segment starts after each EOS)."""
        rows = [causal_example_on_host(row, eos_id=eos_id) for row in self.tokens]
        return jax.tree.map(lambda *leaves: np.stack(leaves), *rows)


def write_fact_probe_input(path: str, probe: FactProbeInput) -> None:
    with fsspec.open(path, "wb") as f:
        np.savez(
            f,
            tokens=probe.tokens,
            span_row=probe.span_row,
            span_start=probe.span_start,
            span_end=probe.span_end,
        )


class FactProbeWriter:
    """Buffers per-step probe results and writes them in chunks of ``chunk_size`` steps per weight kind:
    ``<directory>/fact_probe_<kind>_<index>.npz`` with ``steps`` [C], ``loss`` [C, T], ``top_ids`` [C, T, TOP_K],
    ``top_probs`` [C, T, TOP_K] (float16) and ``row_loss`` [C, N]."""

    def __init__(self, directory: str, chunk_size: int):
        self.directory = directory.rstrip("/")
        self.chunk_size = chunk_size
        self.buffers: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {RAW: [], EMA: []}
        self.chunks_written = {RAW: 0, EMA: 0}

    def add(
        self,
        kind: str,
        step: int,
        loss: np.ndarray,
        top_ids: np.ndarray,
        top_probs: np.ndarray,
        row_loss: np.ndarray,
    ) -> None:
        self.buffers[kind].append((step, loss, top_ids, top_probs, row_loss))
        if len(self.buffers[kind]) >= self.chunk_size:
            self.flush(kind)

    def flush(self, kind: str) -> None:
        records = self.buffers[kind]
        if not records:
            return
        steps, loss, top_ids, top_probs, row_loss = zip(*records, strict=True)
        path = f"{self.directory}/{FACT_PROBE_CHUNK_FILE.format(kind=kind, index=self.chunks_written[kind])}"
        with fsspec.open(path, "wb") as f:
            np.savez(
                f,
                steps=np.asarray(steps, np.int64),
                loss=np.stack(loss).astype(np.float32),
                top_ids=np.stack(top_ids).astype(np.int32),
                top_probs=np.stack(top_probs).astype(np.float16),
                row_loss=np.stack(row_loss).astype(np.float32),
            )
        self.chunks_written[kind] += 1
        self.buffers[kind] = []

    def flush_all(self) -> None:
        for kind in (RAW, EMA):
            self.flush(kind)


def row_mean_loss(loss: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (loss * weight).sum(axis=1) / weight.sum(axis=1)


def count_text_patterns(tokens: np.ndarray, decode: Callable[[list[int]], str], patterns: tuple[str, ...]) -> np.ndarray:
    """Matches of each regex in the decoded rows of ``tokens`` [B, S] (special tokens kept, so a match never spans a
    document boundary unnoticed)."""
    compiled = [re.compile(pattern) for pattern in patterns]
    counts = np.zeros(len(patterns), np.int64)
    for row in tokens:
        text = decode(row.tolist())
        for index, regex in enumerate(compiled):
            counts[index] += sum(1 for _ in regex.finditer(text))
    return counts


def write_text_counts(path: str, patterns: tuple[str, ...], steps: list[int], counts: list[np.ndarray]) -> None:
    with fsspec.open(path, "wb") as f:
        np.savez(f, patterns=np.asarray(patterns), steps=np.asarray(steps, np.int64), counts=np.stack(counts))
