# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert complete GRPO trajectories into pipeline execution batches."""

from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np

from levanter.data.packing import SequencePacker, pack_documents
from levanter.grpo_model import GrpoExample
from levanter.pipeline import reshape_batch_into_microbatches


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PipelineBatch:
    tokens: jax.Array
    attention_mask: jax.Array
    segment_ids: jax.Array
    position_ids: jax.Array
    advantages: jax.Array
    policy_weights: jax.Array
    kl_weights: jax.Array
    old_logprobs: jax.Array
    reference_logprobs: jax.Array


def pipeline_batch(example: GrpoExample, *, microbatches: int) -> PipelineBatch:
    """Convert named examples once, then add an explicit temporal batch axis."""

    def sequence(value):
        return value.rearrange(("batch", "position")).array

    def response(value):
        return value.rearrange(("batch", "response")).array

    batch = PipelineBatch(
        sequence(example.tokens),
        sequence(example.attention_mask),
        sequence(example.attention_mask),
        sequence(example.position_ids),
        response(example.advantages),
        response(example.policy_weights),
        response(example.kl_weights),
        response(example.old_logprobs),
        (
            jnp.zeros_like(response(example.old_logprobs))
            if example.reference_logprobs is None
            else response(example.reference_logprobs)
        ),
    )
    return reshape_batch_into_microbatches(batch, microbatches)


def packed_pipeline_batch(
    example: GrpoExample,
    *,
    sequence_length: int,
    rows_per_microbatch: int,
    pad_token_id: int,
    minimum_microbatches: int = 1,
) -> PipelineBatch:
    """Pack complete trajectories while preserving their precomputed objectives.

    Only masked input padding is removed. Response fields move to the predictor
    of their original target token; prompt, segment-boundary, and padding targets
    receive zero objective weight. Source position IDs are preserved verbatim.
    """
    if sequence_length < 2 or rows_per_microbatch < 1 or minimum_microbatches < 1:
        raise ValueError("Packing requires sequence_length >= 2 and positive batch dimensions")
    source = jax.tree.map(lambda value: np.asarray(value[0]), pipeline_batch(example, microbatches=1))
    response_mask = np.asarray(example.response_mask.rearrange(("batch", "response")).array)
    if np.any((source.attention_mask != 0) & (source.attention_mask != 1)):
        raise ValueError("Packing requires a binary input attention mask")
    valid = source.attention_mask.astype(bool)
    lengths = valid.sum(axis=1)
    packs = pack_documents(lengths, sequence_length, slice_strategy="raise")
    microbatches = max(minimum_microbatches, (len(packs) + rows_per_microbatch - 1) // rows_per_microbatch)
    rows = microbatches * rows_per_microbatch
    tokens = np.full((rows, sequence_length), pad_token_id, dtype=np.int32)
    segments = np.full((rows, sequence_length), -1, dtype=np.int32)
    positions = np.zeros((rows, sequence_length), dtype=np.int32)
    response_names = ("advantages", "policy_weights", "kl_weights", "old_logprobs", "reference_logprobs")
    objective = {
        name: np.zeros((rows, sequence_length - 1), dtype=getattr(source, name).dtype) for name in response_names
    }
    response_start = source.tokens.shape[1] - source.old_logprobs.shape[1]
    if response_start < 1:
        raise ValueError("Each response needs a preceding prompt token")
    if np.any(response_mask > valid[:, response_start:]) or np.any(response_mask > valid[:, response_start - 1 : -1]):
        raise ValueError("Each response target and its predecessor must be attended")
    if np.any((response_mask == 0) & ((source.policy_weights != 0) | (source.kl_weights != 0))):
        raise ValueError("Objective weights must be zero outside the response mask")
    for row, documents in enumerate(packs):
        packer = SequencePacker(hax.Axis("position", sequence_length), max(len(documents), 1), pad_token_id)
        offset = 0
        for document in documents:
            indices = np.flatnonzero(valid[document])
            count = len(indices)
            packer.add_example(source.tokens[document, indices].tolist(), np.zeros(count), segment_id=document)
            positions[row, offset : offset + count] = source.position_ids[document, indices]
            for compact_target, target in enumerate(indices):
                if target < response_start or compact_target == 0 or not valid[document, target - 1]:
                    continue
                response_index = target - response_start
                predictor = offset + compact_target - 1
                for name in response_names:
                    objective[name][row, predictor] = getattr(source, name)[document, response_index]
            offset += count
        packed = packer.pack()
        tokens[row] = np.asarray(packed.tokens.array)
        segments[row] = np.asarray(packed.attn_mask.segment_ids[0].array)
    batch = PipelineBatch(
        tokens, (segments >= 0).astype(np.int32), segments, positions, *(objective[name] for name in response_names)
    )
    return reshape_batch_into_microbatches(jax.tree.map(jnp.asarray, batch), microbatches)
