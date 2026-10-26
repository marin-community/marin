# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-sequence scoring passes: batch building, KL reduction, and row writing.

One scoring job loads one checkpoint into the Megatron policy worker
(``probe_score_forward``) and runs a pass over fixed token sequences:

- **reference**: the model's top-256 token ids and fp32 logprobs per position,
  written as parquet, one file per (model, probe set);
- **candidate**: the model's logprobs gathered at a stored reference's ids —
  plus the realized token's own logprob — reduced against that reference through
  :func:`~experiments.post_training.mismatch_probe.sequence_kl.binned_kl` into
  :class:`SequenceKLRow` rows;
- **exact**: full-vocabulary log-softmax rows on the seeded 1% position
  subsample, reduced to ``exact_kl`` scalars and never stored whole;
- **noise floor**: a model scored against its own reference pass under a
  different batch layout — micro-batch size and sequence order.

Position ``t`` of a pass's output carries the distribution after prefix
``[.. t]``, i.e. it predicts the token at ``t + 1``. Every KL row is keyed by
``(probe_hash, sample_id, position)`` where ``position`` is the scored token's
index; its support comes from output position ``position - 1``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from experiments.post_training.mismatch_probe.sequence_kl import (
    KLForm,
    ProbeBucket,
    SequenceFlipRow,
    SequenceKLRow,
    binned_kl,
    exact_kl,
    exact_subsample_seed,
    position_subsample_mask,
)

TOP_K = 256

# The bucket an unscored token carries (WP8's fixed context); it produces no row.
CONTEXT_TAG = "context"


def sequence_probe_hash(token_ids: list[int]) -> str:
    """sha256 of one sequence's token ids — the scorer's probe identity."""
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int64).tobytes()).hexdigest()


@dataclass(frozen=True)
class ProbeSequence:
    """One fixed token sequence with its bucket tags, as the passes consume it."""

    sample_id: str
    token_ids: list[int]
    token_type_tags: list[str]
    position_tags: list[int]
    bucket_group: str

    @property
    def probe_hash(self) -> str:
        return sequence_probe_hash(self.token_ids)

    def scored_positions(self) -> list[int]:
        """Positions whose tokens are bucketed; position 0 has no predecessor distribution."""
        return [position for position in range(1, len(self.token_ids)) if self.token_type_tags[position] != CONTEXT_TAG]


def load_probe_sequences(path: str | Path) -> list[ProbeSequence]:
    """Probe sequences from a JSONL of ``{sample_id, token_ids, token_type_tags,
    position_tags, bucket_group}`` rows."""
    sequences = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence = ProbeSequence(
            sample_id=str(row["sample_id"]),
            token_ids=[int(token) for token in row["token_ids"]],
            token_type_tags=[str(tag) for tag in row["token_type_tags"]],
            position_tags=[int(tag) for tag in row["position_tags"]],
            bucket_group=str(row["bucket_group"]),
        )
        if len(sequence.token_type_tags) != len(sequence.token_ids):
            raise ValueError(
                f"{sequence.sample_id}: {len(sequence.token_type_tags)} tags for {len(sequence.token_ids)} tokens"
            )
        if len(sequence.position_tags) != len(sequence.token_ids):
            raise ValueError(f"{sequence.sample_id}: position tags do not cover the sequence")
        sequences.append(sequence)
    if not sequences:
        raise ValueError(f"{path} holds no probe sequences")
    return sequences


def build_score_batch(
    sequences: list[ProbeSequence],
    *,
    pad_token_id: int,
    gather_ids: np.ndarray | None = None,
    exact_positions: bool,
    exact_rate: float,
    master_seed: int,
) -> dict[str, np.ndarray | None]:
    """The padded batch arrays one ``probe_score_forward`` call consumes.

    Sequences are left-padded to one common width; the worker's forward strips
    the padding. ``gather_ids`` is ``[n, padded_len, K]`` of candidate-pass ids
    (padding positions carry 0 and are never scored); ``exact_positions`` selects
    the seeded subsample of each sequence's scored positions for the exact pass,
    tagged with the sequence's row plus one so the worker reports batch-wide rows.
    """
    widths = [len(sequence.token_ids) for sequence in sequences]
    padded_len = max(widths)
    count = len(sequences)
    token_ids = np.full((count, padded_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((count, padded_len), dtype=np.int64)
    positions: np.ndarray | None = np.zeros((count, padded_len), dtype=np.int64) if exact_positions else None
    for row, (sequence, width) in enumerate(zip(sequences, widths, strict=True)):
        start = padded_len - width
        token_ids[row, start:] = sequence.token_ids
        attention_mask[row, start:] = 1
        if positions is not None:
            scored = sequence.scored_positions()
            if scored:
                seed = exact_subsample_seed(master_seed, sequence.probe_hash)
                mask = position_subsample_mask(len(scored), rate=exact_rate, seed=seed)
                for selected, position in zip(mask.tolist(), scored, strict=True):
                    # The distribution that scores token ``position`` sits at the logits
                    # position before it.
                    positions[row, start + position - 1] = row + 1 if selected else 0
    padded_gather: np.ndarray | None = None
    if gather_ids is not None:
        if gather_ids.shape[:2] != (count, padded_len):
            raise ValueError(f"gather ids {gather_ids.shape[:2]} do not match the batch {count}x{padded_len}")
        padded_gather = gather_ids
    return {
        "sequences": token_ids,
        "attention_mask": attention_mask,
        "probe_score_positions": positions,
        "probe_score_gather_ids": padded_gather,
    }


def candidate_gather_ids(
    sequence: ProbeSequence,
    reference_ids: np.ndarray,
    padded_len: int,
) -> np.ndarray:
    """One sequence's candidate-pass gather ids, ``[padded_len, K + 1]``, left-padded.

    Logits position ``j`` (which predicts the token at ``j + 1``) gathers the
    reference's K stored ids, then the realized token — so the candidate pass
    also returns the realized token's own logprob. ``reference_ids`` are the
    reference pass's ids for this sequence, ``[len(token_ids) - 1, K]``.
    """
    logits_positions = len(sequence.token_ids) - 1
    if reference_ids.shape[0] != logits_positions:
        raise ValueError(
            f"reference ids cover {reference_ids.shape[0]} logits positions; the sequence has {logits_positions}"
        )
    top_k = reference_ids.shape[1]
    ids = np.zeros((padded_len, top_k + 1), dtype=np.int64)
    start = padded_len - len(sequence.token_ids)
    ids[start : start + logits_positions, :top_k] = reference_ids
    ids[start : start + logits_positions, top_k] = np.asarray(sequence.token_ids[1:], dtype=np.int64)
    return ids


def kl_rows_from_passes(
    sequences: list[ProbeSequence],
    reference_topk_ids: np.ndarray,
    reference_topk_logprobs: np.ndarray,
    reference_tail_logmass: np.ndarray,
    candidate_gathered: np.ndarray,
    candidate_tail_logmass: np.ndarray,
    *,
    reference_id: str,
    candidate_id: str,
    candidate_step: int,
) -> list[SequenceKLRow]:
    """One :class:`SequenceKLRow` per scored position, forward KL(reference ‖ candidate).

    The pass arrays are ``[n, L, ...]`` in the padded batch layout;
    ``candidate_gathered`` holds K + 1 columns — the reference's K stored ids
    gathered on the candidate, then the realized token's own logprob. The KL's
    support is the reference's stored top-K logprobs; each pass's tail log mass
    (``[n, L]``) is its own logsumexp outside the support ids.
    """
    top_k = candidate_gathered.shape[2] - 1
    if reference_topk_ids.shape[2] != top_k or reference_topk_logprobs.shape[2] != top_k:
        raise ValueError("reference and candidate passes hold different supports")
    rows: list[SequenceKLRow] = []
    for row, sequence in enumerate(sequences):
        positions = np.asarray(sequence.scored_positions(), dtype=np.int64)
        if not len(positions):
            continue
        logits_positions = candidate_gathered.shape[1] - len(sequence.token_ids) + positions - 1
        token_ids = np.asarray(sequence.token_ids, dtype=np.int64)[positions]
        support_ids = reference_topk_ids[row, logits_positions]
        support_logprobs = reference_topk_logprobs[row, logits_positions].astype(np.float64)
        reference_tail = reference_tail_logmass[row, logits_positions].astype(np.float64)
        candidate_tail = candidate_tail_logmass[row, logits_positions].astype(np.float64)
        kls = binned_kl(
            torch.from_numpy(support_logprobs),
            torch.from_numpy(reference_tail),
            torch.from_numpy(candidate_gathered[row, logits_positions, :top_k].astype(np.float64)),
            torch.from_numpy(candidate_tail),
        ).numpy()
        bad = np.flatnonzero(~np.isfinite(kls))
        if len(bad):
            raise ValueError(f"{sequence.sample_id}@{int(positions[bad[0]])}: KL is not finite")
        in_support = support_ids == token_ids[:, None]
        reference_logprobs = np.where(
            in_support.any(-1), support_logprobs[np.arange(len(positions)), in_support.argmax(-1)], -np.inf
        )
        candidate_logprobs = candidate_gathered[row, logits_positions, top_k]
        probe_hash = sequence.probe_hash  # a sha256 over the whole sequence: once per sequence
        for index, position in enumerate(positions.tolist()):
            rows.append(
                SequenceKLRow(
                    sequence_id=sequence.sample_id,
                    position=position,
                    token_id=int(token_ids[index]),
                    kl=float(kls[index]),
                    reference_logprob=float(reference_logprobs[index]),
                    candidate_logprob=float(candidate_logprobs[index]),
                    candidate_step=candidate_step,
                    reference_id=reference_id,
                    probe_hash=probe_hash,
                    top_k=top_k,
                    reference_tail_mass=float(reference_tail[index]),
                    candidate_tail_mass=float(candidate_tail[index]),
                    form=KLForm.FORWARD,
                    token_type_tag=ProbeBucket(sequence.token_type_tags[position]),
                    position_tag=sequence.position_tags[position],
                )
            )
    return rows


def exact_rows_from_passes(
    sequences: list[ProbeSequence],
    reference_exact: np.ndarray,
    reference_positions: list[tuple[int, int]],
    candidate_exact: np.ndarray,
    candidate_positions: list[tuple[int, int]],
    *,
    padded_len: int,
) -> list[tuple[str, int, str, float]]:
    """(sample_id, token position, probe_hash, exact KL) per subsampled position.

    ``reference_exact``/``candidate_exact`` are the full-vocabulary log-softmax
    rows of the two passes, each in its own compacted order beside the
    ``(batch row, padded logits position)`` coordinates of every returned row.
    The reported position is the scored token's, as in the binned rows.
    """
    if len(reference_exact) != len(reference_positions) or len(candidate_exact) != len(candidate_positions):
        raise ValueError("exact rows and their coordinates disagree")
    reference_by_key = {tuple(key): value for key, value in zip(reference_positions, reference_exact, strict=True)}
    rows = []
    for (row, logits_position), logprobs in zip(candidate_positions, candidate_exact, strict=True):
        if (row, logits_position) not in reference_by_key:
            raise ValueError(
                f"the candidate pass scored logits position {logits_position} of row {row}; the reference pass did not"
            )
        kl = float(
            exact_kl(
                torch.from_numpy(reference_by_key[(row, logits_position)].astype(np.float64)),
                torch.from_numpy(logprobs.astype(np.float64)),
            )
        )
        sequence = sequences[row]
        # Logits position j of a row left-padded to padded_len scores the token at
        # sequence position j - (padded_len - len) + 1, the binned rows' position.
        position = logits_position - (padded_len - len(sequence.token_ids)) + 1
        rows.append((sequence.sample_id, position, sequence.probe_hash, kl))
    return rows


def write_rows_parquet(rows: list[SequenceKLRow], output_path: str | Path) -> int:
    """One parquet file of SequenceKLRow records; returns the row count."""
    if not rows:
        raise ValueError("refusing to write an empty parquet file")
    schema = pa.schema(
        [
            ("sequence_id", pa.string()),
            ("position", pa.int64()),
            ("token_id", pa.int64()),
            ("kl", pa.float64()),
            ("reference_logprob", pa.float64()),
            ("candidate_logprob", pa.float64()),
            ("candidate_step", pa.int64()),
            ("reference_id", pa.string()),
            ("probe_hash", pa.string()),
            ("top_k", pa.int64()),
            ("reference_tail_mass", pa.float64()),
            ("candidate_tail_mass", pa.float64()),
            ("form", pa.string()),
            ("token_type_tag", pa.string()),
            ("position_tag", pa.int64()),
            ("exact_kl", pa.float64()),
        ]
    )
    records = []
    for row in rows:
        record = dict(row.__dict__)
        record["form"] = row.form.value
        record["token_type_tag"] = row.token_type_tag.value
        records.append(record)
    table = pa.Table.from_pylist(records, schema=schema)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output))
    return len(rows)


def read_rows_parquet(input_path: str | Path) -> list[SequenceKLRow]:
    """The rows of one pass's parquet file, with the form and bucket back as their enums."""
    table = pq.read_table(str(input_path))
    return [
        SequenceKLRow(
            **{
                **record,
                "form": KLForm(record["form"]),
                "token_type_tag": ProbeBucket(record["token_type_tag"]),
            }
        )
        for record in table.to_pylist()
    ]


def write_flip_rows_parquet(rows: list[SequenceFlipRow], output_path: str | Path) -> int:
    """One parquet file of SequenceFlipRow records; returns the row count."""
    if not rows:
        raise ValueError("refusing to write an empty parquet file")
    schema = pa.schema(
        [
            ("sequence_id", pa.string()),
            ("position", pa.int64()),
            ("reference_top1", pa.int64()),
            ("candidate_top1", pa.int64()),
            ("flipped", pa.bool_()),
            ("reference_top1_logprob", pa.float64()),
            ("candidate_top1_logprob", pa.float64()),
            ("candidate_step", pa.int64()),
            ("reference_id", pa.string()),
            ("candidate_id", pa.string()),
            ("probe_hash", pa.string()),
            ("token_type_tag", pa.string()),
            ("position_tag", pa.int64()),
        ]
    )
    records = [{**row.__dict__, "token_type_tag": row.token_type_tag.value} for row in rows]
    table = pa.Table.from_pylist(records, schema=schema)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_path))
    return len(rows)


def read_flip_rows_parquet(input_path: str | Path) -> list[SequenceFlipRow]:
    """The flip rows of one pass's parquet file, with the bucket back as its enum."""
    table = pq.read_table(str(input_path))
    return [
        SequenceFlipRow(**{**record, "token_type_tag": ProbeBucket(record["token_type_tag"])})
        for record in table.to_pylist()
    ]
