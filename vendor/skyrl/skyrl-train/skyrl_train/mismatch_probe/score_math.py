# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-position scoring of frozen-sequence logits for the KL passes.

The Megatron policy runs on left-unpadded sequences, so its logits sit in
compact space. This module holds the pure computation both score passes share:
the reference pass's top-K support, the candidate pass's gather at stored ids,
and the exact pass's full-vocabulary rows — each computed from the float32
log-softmax in sequence chunks and scattered back to the batch's padded layout.
Position t's distribution predicts the token at t+1.
"""

from __future__ import annotations

import torch


def score_compacted_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    top_k: int | None,
    gather_ids: torch.Tensor | None,
    probe_score_positions: torch.Tensor | None,
    vocabulary_size: int,
    chunk_size: int | None,
) -> dict[str, torch.Tensor]:
    """Top-K, gathered, or exact rows from one micro's compact-space logits.

    ``logits`` are the model's output for its left-unpadded sequences, ``[B,
    compact_len, V]``. Every output is computed from the float32 log-softmax over
    the true vocabulary (padded columns dropped) in sequence chunks, and scattered
    back to the batch's padded layout, aligned with the model's logits positions —
    position t predicts the token at t+1.

    Exactly one mode must be set: ``top_k`` returns the top-K token ids and their
    logprobs per position; ``gather_ids`` ([B, T, K], padded layout) returns the
    log-softmax gathered at those ids; ``probe_score_positions`` ([B, T] int64)
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
    if gather_ids is not None and gather_ids.shape[:2] != (batch_size, padded_len):
        raise ValueError(f"gather ids {tuple(gather_ids.shape)} do not match the batch {batch_size}x{padded_len}")
    seq_lens = attention_mask.sum(dim=1, dtype=torch.int32)
    compact_len = logits.shape[1]
    longest = max(int(seq_lens.max().item()), 1)
    if compact_len < longest:
        raise ValueError(f"logits compact length {compact_len} is shorter than the longest sequence {longest}")

    compact_gather: torch.Tensor | None = None
    if gather_ids is not None:
        compact_gather = torch.zeros(
            (batch_size, compact_len, gather_ids.shape[2]), dtype=torch.long, device=logits.device
        )
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
                    selected_rows_meta.append((row, compact_index, int(padded_positions[compact_index].item()), tag - 1))

    collected: dict[str, list[torch.Tensor]] = {}
    selected_rows: list[torch.Tensor] = []
    selected_coordinates: list[tuple[int, int]] = []
    width = max(1, chunk_size or compact_len)
    for start in range(0, compact_len, width):
        stop = min(start + width, compact_len)
        logprobs = torch.log_softmax(logits[:, start:stop, :vocabulary_size].float(), dim=-1)
        if top_k is not None:
            values, ids = torch.topk(logprobs, top_k, dim=-1)
            collected.setdefault("topk_ids", []).append(ids)
            collected.setdefault("topk_logprobs", []).append(values)
        elif gather_ids is not None:
            collected.setdefault("gathered_logprobs", []).append(
                torch.gather(logprobs, -1, compact_gather[:, start:stop])
            )
        else:
            for row, compact_index, padded_position, sequence_row in selected_rows_meta:
                if start <= compact_index < stop:
                    selected_rows.append(logprobs[row, compact_index - start])
                    selected_coordinates.append((sequence_row, padded_position))
    scores = {key: torch.cat(chunks, dim=1) for key, chunks in collected.items()}

    if probe_score_positions is not None:
        scores["exact_logprobs"] = (
            torch.stack(selected_rows, dim=0)
            if selected_rows
            else torch.zeros((0, vocabulary_size), dtype=logits.dtype, device=logits.device)
        )
        scores["exact_positions"] = torch.tensor(selected_coordinates, dtype=torch.long, device=logits.device).reshape(
            -1, 2
        )
        return scores

    for key, values in scores.items():
        restored = torch.zeros((batch_size, padded_len, *values.shape[2:]), dtype=values.dtype, device=values.device)
        for row, length in enumerate(seq_lens.tolist()):
            restored[row, attention_mask[row]] = values[row, :length]
        scores[key] = restored
    return scores
