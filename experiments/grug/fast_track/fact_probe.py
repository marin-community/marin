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
    attn_queries: np.ndarray  # [Q, 2] (row, position) whose MLA attention is recorded; Q may be 0
    # The first ``attn_full_queries`` queries keep their full attention; every query also gets, per layer and head,
    # its attention mass on the absolute key positions of its ``attn_key_sets`` row (-1 pads) and its entropy.
    attn_full_queries: int
    attn_key_sets: np.ndarray  # [Q, M]
    spot: np.ndarray  # [2] (row, position) whose whole forward is recorded (``model.forward_probe``), or [0]
    spot_token_ids: np.ndarray  # tokens whose lm_head columns are recorded at the spot

    def __post_init__(self):
        if np.any(self.span_start < 1) or np.any(self.span_end <= self.span_start):
            raise ValueError("every fact span needs 1 <= start < end")
        if np.any(self.span_end > self.tokens.shape[1]) or np.any(self.span_row >= self.tokens.shape[0]):
            raise ValueError("fact span outside the probe rows")
        if not 0 <= self.attn_full_queries <= len(self.attn_queries):
            raise ValueError("attn_full_queries must be between 0 and the number of attention queries")
        if len(self.attn_key_sets) != len(self.attn_queries):
            raise ValueError("attn_key_sets needs one row per attention query")

    @classmethod
    def load(cls, path: str) -> FactProbeInput:
        with fsspec.open(path, "rb") as f:
            data = np.load(f)
            return cls(
                tokens=data["tokens"].astype(np.int32),
                span_row=data["span_row"].astype(np.int64),
                span_start=data["span_start"].astype(np.int64),
                span_end=data["span_end"].astype(np.int64),
                attn_queries=(data["attn_queries"] if "attn_queries" in data else np.zeros((0, 2))).astype(np.int64),
                attn_full_queries=(
                    int(data["attn_full_queries"]) if "attn_full_queries" in data else len(data.get("attn_queries", []))
                ),
                attn_key_sets=(
                    data["attn_key_sets"]
                    if "attn_key_sets" in data
                    else np.full((len(data["attn_queries"]) if "attn_queries" in data else 0, 0), -1)
                ).astype(np.int64),
                spot=(data["spot"] if "spot" in data else np.zeros(0)).astype(np.int64),
                spot_token_ids=(data["spot_token_ids"] if "spot_token_ids" in data else np.zeros(0)).astype(np.int64),
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
            attn_queries=probe.attn_queries,
            attn_full_queries=probe.attn_full_queries,
            attn_key_sets=probe.attn_key_sets,
            spot=probe.spot,
            spot_token_ids=probe.spot_token_ids,
        )


class FactProbeWriter:
    """Buffers per-step probe results and writes them in chunks of ``chunk_size`` steps per weight kind:
    ``<directory>/fact_probe_<kind>_<index>.npz`` with ``steps`` [C], ``loss`` [C, T], ``top_ids`` [C, T, TOP_K],
    ``top_probs`` [C, T, TOP_K] (float16), ``row_loss`` [C, N] and every ``model.forward_probe`` recording as
    ``probe/<stat>`` [C, ...] (float16)."""

    def __init__(self, directory: str, chunk_size: int):
        self.directory = directory.rstrip("/")
        self.chunk_size = chunk_size
        self.buffers: dict[str, list[tuple]] = {RAW: [], EMA: []}
        self.chunks_written = {RAW: 0, EMA: 0}

    def add(
        self,
        kind: str,
        step: int,
        loss: np.ndarray,
        top_ids: np.ndarray,
        top_probs: np.ndarray,
        row_loss: np.ndarray,
        attention: dict[str, np.ndarray],
    ) -> None:
        self.buffers[kind].append((step, loss, top_ids, top_probs, row_loss, attention))
        if len(self.buffers[kind]) >= self.chunk_size:
            self.flush(kind)

    def flush(self, kind: str) -> None:
        records = self.buffers[kind]
        if not records:
            return
        steps, loss, top_ids, top_probs, row_loss, attention = zip(*records, strict=True)
        path = f"{self.directory}/{FACT_PROBE_CHUNK_FILE.format(kind=kind, index=self.chunks_written[kind])}"
        with fsspec.open(path, "wb") as f:
            np.savez(
                f,
                steps=np.asarray(steps, np.int64),
                loss=np.stack(loss).astype(np.float32),
                top_ids=np.stack(top_ids).astype(np.int32),
                top_probs=np.stack(top_probs).astype(np.float16),
                row_loss=np.stack(row_loss).astype(np.float32),
                **{f"probe/{name}": np.stack([a[name] for a in attention]).astype(np.float16) for name in attention[0]},
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


def summarize_attention(
    probs: np.ndarray, queries: np.ndarray, key_sets: np.ndarray, num_keys: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per query and head: the attention mass on that query's key set and the attention entropy. ``probs`` is
    [Q, H, num_keys] over keys ``position - num_keys + 1 .. position`` (``model.ATTN_PROBE_KEYS``)."""
    mass = np.zeros(probs.shape[:2], np.float32)
    for q, ((_, position), keys) in enumerate(zip(queries, key_sets, strict=True)):
        index = keys[keys >= 0] - (position - num_keys + 1)
        if np.any(index < 0) or np.any(index >= num_keys):
            raise ValueError(f"key set of query {q} falls outside its attention window")
        mass[q] = probs[q][:, index].sum(-1)
    entropy = -(probs * np.log(np.maximum(probs, 1e-30))).sum(-1)
    return mass, entropy.astype(np.float32)
