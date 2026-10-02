# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fact probe: how a run learns, and keeps, the content of the batches it trained on.

Two passes over the same seed (the data order is a pure function of the seed):

* ``train_batch_dump_steps`` writes the exact train batches of chosen steps (``write_train_batch``) and stops
  before training, so a fact can be picked from them offline.
* ``fact_probe_spans`` names token ranges inside those batches. After each of ``fact_probe_steps`` the run re-scores
  the batches with the raw weights (and the EMA weights once they are live), and ``FactProbeRecord`` writes the
  per-row mean loss and the per-token loss on every span to one npz.

A span ``(data_step, row, start, end)`` scores the prediction of ``tokens[row, start:end]`` in the batch of
``data_step``, i.e. the losses at positions ``start - 1 .. end - 2``. That batch is consumed by the update that
takes the step count from ``data_step`` to ``data_step + 1``, so the probe at count ``data_step`` is the last one
before the model sees it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import fsspec
import numpy as np

logger = logging.getLogger(__name__)

TRAIN_BATCH_DUMP_FILE = "train_batch_step{step}.npz"
RAW = "raw"
EMA = "ema"


@dataclass(frozen=True)
class FactSpan:
    data_step: int
    row: int
    start: int
    end: int

    def __post_init__(self):
        if self.start < 1 or self.end <= self.start:
            raise ValueError(f"fact span needs 1 <= start < end, got {self}")

    @classmethod
    def parse(cls, text: str) -> FactSpan:
        """``"data_step:row:start:end"``."""
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(f"fact span must be data_step:row:start:end, got {text!r}")
        return cls(*(int(part) for part in parts))


def write_train_batch(path: str, step: int, tokens: np.ndarray, loss_weight: np.ndarray) -> None:
    with fsspec.open(path, "wb") as f:
        np.savez(f, step=step, tokens=tokens.astype(np.int32), loss_weight=loss_weight.astype(np.float32))
    logger.info("wrote train batch of step %d (%s) to %s", step, tokens.shape, path)


class FactProbeRecord:
    """Accumulates probe results and rewrites one npz after every probe step (so a crash keeps what it measured).

    Arrays: ``probe_steps`` [P]; ``data_steps`` [D]; ``{raw,ema}_row_loss`` [P, D, B] (mean loss per batch row);
    ``{raw,ema}_span_loss`` [P, T] (every span's per-token losses, concatenated); ``span_id`` / ``span_token`` [T];
    ``spans`` [K, 4]. EMA entries are NaN before the EMA is live.
    """

    def __init__(self, spans: tuple[FactSpan, ...], path: str):
        if not spans:
            raise ValueError("fact probe needs at least one span")
        self.spans = spans
        self.path = path
        self.data_steps = tuple(sorted({span.data_step for span in spans}))
        self.probe_steps: list[int] = []
        self.row_loss: dict[str, list[np.ndarray]] = {RAW: [], EMA: []}
        self.span_loss: dict[str, list[np.ndarray]] = {RAW: [], EMA: []}
        self.span_token: np.ndarray | None = None

    def add(
        self,
        step: int,
        losses: dict[str, dict[int, np.ndarray]],
        tokens: dict[int, np.ndarray],
        weights: dict[int, np.ndarray],
    ) -> dict[str, float]:
        """Record ``losses[kind][data_step]`` (per-token [B, S]) at ``step``; returns scalars to log."""
        if self.span_token is None:
            self.span_token = np.concatenate([tokens[s.data_step][s.row, s.start : s.end] for s in self.spans])
        self.probe_steps.append(step)
        scalars = {}
        for kind in (RAW, EMA):
            by_step = losses.get(kind)
            if by_step is None:
                num_rows = next(iter(weights.values())).shape[0]
                self.row_loss[kind].append(np.full((len(self.data_steps), num_rows), np.nan, np.float32))
                self.span_loss[kind].append(np.full(len(self.span_token), np.nan, np.float32))
                continue
            rows = np.stack([_row_mean(by_step[d], weights[d]) for d in self.data_steps])
            spans = [by_step[s.data_step][s.row, s.start - 1 : s.end - 1] for s in self.spans]
            self.row_loss[kind].append(rows.astype(np.float32))
            self.span_loss[kind].append(np.concatenate(spans).astype(np.float32))
            for index, span_losses in enumerate(spans):
                scalars[f"fact_probe/{kind}/span{index}"] = float(np.mean(span_losses))
            for index, data_step in enumerate(self.data_steps):
                scalars[f"fact_probe/{kind}/batch{data_step}"] = float(np.mean(rows[index]))
        self._write()
        return scalars

    def _write(self) -> None:
        span_id = np.concatenate([np.full(s.end - s.start, i, np.int32) for i, s in enumerate(self.spans)])
        with fsspec.open(self.path, "wb") as f:
            np.savez(
                f,
                probe_steps=np.asarray(self.probe_steps, np.int64),
                data_steps=np.asarray(self.data_steps, np.int64),
                spans=np.asarray([[s.data_step, s.row, s.start, s.end] for s in self.spans], np.int64),
                span_id=span_id,
                span_token=self.span_token,
                **{f"{kind}_row_loss": np.stack(self.row_loss[kind]) for kind in (RAW, EMA)},
                **{f"{kind}_span_loss": np.stack(self.span_loss[kind]) for kind in (RAW, EMA)},
            )


def _row_mean(loss: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (loss * weight).sum(axis=1) / weight.sum(axis=1)
