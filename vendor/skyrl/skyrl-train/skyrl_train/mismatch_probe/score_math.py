# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-position scoring of frozen-sequence logits for the KL passes.

The Megatron policy runs on left-unpadded sequences, so its outputs sit in
compact space. This module holds the pure computation both score passes share:
the reference pass's top-K support, the candidate pass's gather at stored ids,
and the exact pass's full-vocabulary rows — each computed from the float32
log-softmax in sequence chunks and scattered back to the batch's padded layout.
Position t's distribution predicts the token at t+1.

The passes start from the final hidden states, not the logits: the output
projection runs one sequence chunk at a time, so a [tokens x vocab] logits
tensor is never materialized. At 131k tokens and Snowball's vocabulary that
tensor alone would be about 34 GB in bf16.

The reference and candidate passes also return each position's tail: the log
mass outside the support ids, a logsumexp over the complement of the full
log-softmax. Deriving it later as ``1 - sum(exp(support))`` cannot resolve a tail
below the fp32 rounding of that sum (about 1e-6), which rounds a candidate's true
tail to zero and its KL to infinity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompactHiddenStates:
    """A micro's final hidden states in compact space and the projection to its logits.

    ``hidden`` is ``[B, compact_len, H]``; ``project`` maps a sequence slice of it,
    ``[B, n, H]``, to that slice's logits, ``[B, n, V_padded]``.
    """

    hidden: torch.Tensor
    project: Callable[[torch.Tensor], torch.Tensor]

    @property
    def compact_len(self) -> int:
        return self.hidden.shape[1]


def score_compacted_hidden(
    outputs: CompactHiddenStates,
    attention_mask: torch.Tensor,
    *,
    top_k: int | None,
    gather_ids: torch.Tensor | None,
    probe_score_positions: torch.Tensor | None,
    vocabulary_size: int,
    chunk_size: int | None,
    support_width: int | None = None,
) -> dict[str, torch.Tensor]:
    """Top-K, gathered, or exact rows from one micro's compact-space hidden states.

    ``outputs`` holds the model's final hidden states for its left-unpadded
    sequences. Each sequence chunk of ``chunk_size`` positions is projected to
    logits and reduced before the next, so peak memory holds one chunk's logits.
    Every output is computed from the float32 log-softmax over the true
    vocabulary (padded columns dropped), and scattered
    back to the batch's padded layout, aligned with the model's logits positions —
    position t predicts the token at t+1.

    Exactly one mode must be set: ``top_k`` returns the top-K token ids and their
    logprobs per position; ``gather_ids`` ([B, T, K], padded layout) returns the
    log-softmax gathered at those ids, of which the first ``support_width`` are the
    support whose tail is returned; ``probe_score_positions`` ([B, T] int64)
    returns the full-vocabulary log-softmax rows of the selected positions,
    compacted in (row, position) order beside their coordinates. A selected
    position holds its sequence's batch-wide row id plus one (0 elsewhere), and the
    coordinates report that row id: the local row would be lost once micro-batches
    and data-parallel shards are concatenated.
    """
    modes = (top_k is not None, gather_ids is not None, probe_score_positions is not None)
    if sum(modes) != 1:
        raise ValueError("pass exactly one of top_k, gather_ids, or probe_score_positions")
    attention_mask = attention_mask.to(bool)
    batch_size, padded_len = attention_mask.shape
    if (gather_ids is not None) != (support_width is not None):
        raise ValueError("the gather pass needs support_width, and only the gather pass takes it")
    if gather_ids is not None and gather_ids.shape[:2] != (batch_size, padded_len):
        raise ValueError(f"gather ids {tuple(gather_ids.shape)} do not match the batch {batch_size}x{padded_len}")
    seq_lens = attention_mask.sum(dim=1, dtype=torch.int32)
    compact_len = outputs.compact_len
    device = outputs.hidden.device
    longest = max(int(seq_lens.max().item()), 1)
    if compact_len < longest:
        raise ValueError(f"hidden-state compact length {compact_len} is shorter than the longest sequence {longest}")

    compact_gather: torch.Tensor | None = None
    if gather_ids is not None:
        compact_gather = torch.zeros((batch_size, compact_len, gather_ids.shape[2]), dtype=torch.long, device=device)
        for row, length in enumerate(seq_lens.tolist()):
            compact_gather[row, :length] = gather_ids[row, attention_mask[row]].long()
    selected_rows_meta: list[tuple[int, int, int, int]] | None = None
    if probe_score_positions is not None:
        selected_rows_meta = []
        for row, length in enumerate(seq_lens.tolist()):
            padded_positions = attention_mask[row].nonzero(as_tuple=True)[0]
            tags = probe_score_positions[row, attention_mask[row]]
            for compact_index in range(length):
                tag = int(tags[compact_index].item())
                if tag > 0:
                    selected_rows_meta.append(
                        (row, compact_index, int(padded_positions[compact_index].item()), tag - 1)
                    )

    collected: dict[str, list[torch.Tensor]] = {}
    selected_rows: list[torch.Tensor] = []
    selected_coordinates: list[tuple[int, int]] = []
    width = max(1, chunk_size or compact_len)
    for start in range(0, compact_len, width):
        stop = min(start + width, compact_len)
        logits = outputs.project(outputs.hidden[:, start:stop])
        logprobs = torch.log_softmax(logits[..., :vocabulary_size].float(), dim=-1)
        del logits
        if top_k is not None:
            values, ids = torch.topk(logprobs, top_k, dim=-1)
            collected.setdefault("topk_ids", []).append(ids)
            collected.setdefault("topk_logprobs", []).append(values)
            collected.setdefault("topk_tail_logmass", []).append(complement_log_mass(logprobs, ids))
        elif gather_ids is not None:
            chunk_ids = compact_gather[:, start:stop]
            collected.setdefault("gathered_logprobs", []).append(torch.gather(logprobs, -1, chunk_ids))
            collected.setdefault("gathered_tail_logmass", []).append(
                complement_log_mass(logprobs, chunk_ids[..., :support_width])
            )
        else:
            for row, compact_index, padded_position, sequence_row in selected_rows_meta:
                if start <= compact_index < stop:
                    # A clone, so the stored row does not keep the chunk's logprobs alive.
                    selected_rows.append(logprobs[row, compact_index - start].clone())
                    selected_coordinates.append((sequence_row, padded_position))
    scores = {key: torch.cat(chunks, dim=1) for key, chunks in collected.items()}

    if probe_score_positions is not None:
        scores["exact_logprobs"] = (
            torch.stack(selected_rows, dim=0)
            if selected_rows
            else torch.zeros((0, vocabulary_size), dtype=torch.float32, device=device)
        )
        scores["exact_positions"] = torch.tensor(selected_coordinates, dtype=torch.long, device=device).reshape(-1, 2)
        return scores

    for key, values in scores.items():
        restored = torch.zeros((batch_size, padded_len, *values.shape[2:]), dtype=values.dtype, device=values.device)
        for row, length in enumerate(seq_lens.tolist()):
            restored[row, attention_mask[row]] = values[row, :length]
        scores[key] = restored
    return scores


def complement_log_mass(logprobs: torch.Tensor, support_ids: torch.Tensor) -> torch.Tensor:
    """Log mass of each position's distribution outside ``support_ids`` (``[..., V]`` -> ``[...]``)."""
    return torch.logsumexp(logprobs.scatter(-1, support_ids, float("-inf")), dim=-1)
