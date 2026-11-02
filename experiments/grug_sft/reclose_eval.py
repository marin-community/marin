# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced re-close probe run inside the Grug SFT loop (marin-community/marin#10400).

Scores a fixed set of trajectories (prompt, reasoning, ``<|end_think|>``, lead-in, continuation; written by
``fewshot_think_probe_channels/reclose_trajectories.py``) with the live training parameters. One forward
pass gives p(``<|end_think|>``) at every position. Per trajectory it reports:

* ``hazard``: ``1 - prod(1 - p)`` over the continuation, from the close through the token before the
  sampling model's first re-close (or the end): the chance of closing again along this fixed text;
* ``p_reclose``: p at the sampling model's own re-close, when it had one;
* ``p_first_close``: p at the first close, i.e. whether the close token is learned at all;
* ``p_premature``: mean p over the reasoning before the first close.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jmp
import levanter.tracker
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from rigging.filesystem.buckets import filesystem_for

logger = logging.getLogger(__name__)

PAD_SEGMENT = -1


@dataclass(frozen=True)
class RecloseEvalConfig:
    trajectories: str
    """JSONL of trajectories with ``ids``, ``group``, ``bridge``, ``reasoning_start``, ``close`` and ``reclose``."""
    close_token_id: int
    every: int = 1
    output: str | None = None
    """Directory for per-trajectory ``step-N.jsonl`` dumps; None logs only aggregates."""


@dataclass(frozen=True)
class Placement:
    row: int
    offset: int


def load_trajectories(path: str) -> list[dict]:
    fs, fs_path = filesystem_for(path)
    with fs.open(fs_path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def pack(
    trajectories: list[dict], *, seq_len: int, rows_multiple: int
) -> tuple[np.ndarray, np.ndarray, list[Placement]]:
    """First-fit pack trajectories into rows, one segment each; pad rows to a multiple of ``rows_multiple``."""
    fill: list[int] = []
    placements = []
    for t in trajectories:
        length = len(t["ids"])
        if length > seq_len:
            raise ValueError(f"trajectory of {length} tokens exceeds sequence length {seq_len}")
        row = next((r for r, used in enumerate(fill) if used + length <= seq_len), None)
        if row is None:
            row = len(fill)
            fill.append(0)
        placements.append(Placement(row, fill[row]))
        fill[row] += length
    num_rows = -(-len(fill) // rows_multiple) * rows_multiple
    tokens = np.zeros((num_rows, seq_len), dtype=np.int32)
    segment_ids = np.full((num_rows, seq_len), PAD_SEGMENT, dtype=np.int32)
    for i, (t, place) in enumerate(zip(trajectories, placements, strict=True)):
        end = place.offset + len(t["ids"])
        tokens[place.row, place.offset : end] = t["ids"]
        segment_ids[place.row, place.offset : end] = i
    return tokens, segment_ids, placements


def trajectory_stats(t: dict, p: np.ndarray) -> dict:
    """Stats for one trajectory from ``p[j]`` = p(next token is the close | ids[: j + 1])."""
    close, reclose = t["close"], t["reclose"]
    last = reclose if reclose is not None else len(t["ids"]) - 1
    continuation = p[close:last].astype(np.float64)
    return {
        "hazard": float(1.0 - np.prod(1.0 - continuation)),
        "mean_p": float(continuation.mean()),
        "p_reclose": float(p[reclose - 1]) if reclose is not None else None,
        "p_first_close": float(p[close - 1]),
        "p_premature": float(p[t["reasoning_start"] - 1 : close - 1].mean()),
    }


def aggregate(trajectories: list[dict], stats: list[dict]) -> dict[str, float]:
    cells: dict[str, list[dict]] = defaultdict(list)
    for t, s in zip(trajectories, stats, strict=True):
        cells[t["group"]].append(s)
        cells[f"{t['group']}/lead_in" if t["bridge"] else f"{t['group']}/no_lead_in"].append(s)
    out = {}
    for cell, rows in cells.items():
        for key in ("hazard", "mean_p", "p_first_close", "p_premature"):
            out[f"reclose/{cell}/{key}"] = float(np.mean([r[key] for r in rows]))
        recloses = [r["p_reclose"] for r in rows if r["p_reclose"] is not None]
        if recloses:
            out[f"reclose/{cell}/p_reclose"] = float(np.mean(recloses))
    return out


class RecloseEvaluator:
    def __init__(
        self,
        config: RecloseEvalConfig,
        *,
        mesh: Mesh,
        batch_axes: tuple[str, ...],
        seq_len: int,
        rows_per_pass: int,
        mp: jmp.Policy,
        apply_qb_betas,
    ):
        self.config = config
        self.trajectories = load_trajectories(config.trajectories)
        tokens, segment_ids, self.placements = pack(self.trajectories, seq_len=seq_len, rows_multiple=rows_per_pass)
        sharding = NamedSharding(mesh, P(batch_axes, None))
        self.passes = [
            tuple(
                jax.make_array_from_callback(chunk.shape, sharding, lambda index, chunk=chunk: chunk[index])
                for chunk in (tokens[start : start + rows_per_pass], segment_ids[start : start + rows_per_pass])
            )
            for start in range(0, tokens.shape[0], rows_per_pass)
        ]
        self.rows_per_pass = rows_per_pass
        logger.info(
            "Re-close probe: %d trajectories, %d tokens, %d pass(es) of %d rows",
            len(self.trajectories),
            sum(len(t["ids"]) for t in self.trajectories),
            len(self.passes),
            rows_per_pass,
        )
        close_id = config.close_token_id

        @jax.jit
        def close_probability(params, pending_qb_betas, tokens, segment_ids):
            model = mp.cast_to_compute(apply_qb_betas(params, pending_qb_betas))
            mask = GrugAttentionMask.causal().with_segment_ids(segment_ids)
            hidden, _ = model(tokens, mask=mask)
            nll = fused_linear_softmax_cross_entropy_loss(
                hidden,
                model.output_proj,
                jnp.full(tokens.shape, close_id, dtype=jnp.int32),
                reduction="none",
                implementation=model.config.ce_implementation,
            )
            return jnp.exp(-nll.astype(jnp.float32))

        self._close_probability = close_probability

    def run(self, state, *, log_step: int) -> dict[str, float]:
        probabilities = np.concatenate(
            [
                multihost_utils.process_allgather(
                    self._close_probability(state.params, state.pending_qb_betas, tokens, segment_ids), tiled=True
                )
                for tokens, segment_ids in self.passes
            ]
        )
        stats = []
        for t, place in zip(self.trajectories, self.placements, strict=True):
            p = probabilities[place.row, place.offset : place.offset + len(t["ids"])]
            stats.append(trajectory_stats(t, p))
        metrics = aggregate(self.trajectories, stats)
        levanter.tracker.log(metrics, step=log_step)
        if jax.process_index() != 0:
            return metrics
        logger.info("re-close probe at step %d: %s", log_step, {k: round(v, 4) for k, v in metrics.items()})
        if self.config.output is not None:
            path = f"{self.config.output.rstrip('/')}/step-{int(state.step):06d}.jsonl"
            rows = [
                {"group": t["group"], "id": t["id"], "sample": t["sample"], "bridge": t["bridge"], **s}
                for t, s in zip(self.trajectories, stats, strict=True)
            ]
            fs, fs_path = filesystem_for(path)
            fs.makedirs(fs_path.rsplit("/", 1)[0], exist_ok=True)
            fs.pipe(fs_path, "\n".join(json.dumps(r) for r in rows).encode())
        return metrics
