# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared scoring runtime for RL/post-training objectives."""

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import haliax as hax
import jax
import jax.numpy as jnp
from haliax import NamedArray
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty

from .types import BatchInfo, SequenceBatch, TrainingBatch

ScoredBatch: TypeAlias = SequenceBatch | TrainingBatch


@dataclass(frozen=True)
class TokenScoreInputs:
    """Canonical token-scoring inputs shared by sequence and training batches."""

    input_ids: NamedArray
    position_ids: NamedArray
    sampling_temperature: NamedArray


@dataclass(frozen=True)
class ScoreRequirements:
    """Scores required by an objective runtime."""

    student_logprobs: bool = True
    behavior_logprobs: bool = False
    reference_logprobs: bool = False
    teacher_token_logprobs: bool = False
    teacher_entropy: bool = False
    teacher_topk_support: int | None = None


@dataclass
class ScoreBundle:
    """Resolved scores plus compact execution metadata."""

    student_logprobs: jax.Array | None = None
    behavior_logprobs: jax.Array | None = None
    reference_logprobs: jax.Array | None = None
    teacher_logprobs: jax.Array | None = None
    teacher_entropy: jax.Array | None = None
    teacher_topk_ids: jax.Array | None = None
    teacher_topk_logprobs: jax.Array | None = None
    student_pass_count: int = 0
    reference_pass_count: int = 0
    teacher_pass_count: int = 0


@dataclass(frozen=True)
class ModelRoles:
    """Concrete model handles used by a score source."""

    student: LmHeadModel
    reference: LmHeadModel | None = None
    teacher: LmHeadModel | None = None


class ScoreSource(Protocol):
    """Backend protocol for scoring model-dependent quantities."""

    @property
    def backend_name(self) -> str:
        """Return a stable scorer backend identifier."""
        ...

    def requirements(self) -> ScoreRequirements:
        """Return the score requests handled by this source."""
        ...

    def score(
        self,
        batch: ScoredBatch,
        info: BatchInfo | None,
        roles: ModelRoles,
        *,
        key: jax.Array | None,
    ) -> ScoreBundle:
        """Compute requested scores for the provided batch."""
        ...


def token_score_inputs_from_batch(batch: ScoredBatch) -> TokenScoreInputs:
    """Project a batch into the canonical token-scoring view."""
    if isinstance(batch, SequenceBatch):
        return TokenScoreInputs(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            sampling_temperature=batch.sampling_temperature,
        )
    if isinstance(batch, TrainingBatch):
        return TokenScoreInputs(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            sampling_temperature=batch.temperature,
        )
    raise TypeError(f"Unsupported batch type for scoring: {type(batch)!r}")


def behavior_logprobs_from_batch(batch: ScoredBatch) -> jax.Array:
    """Extract the behavior-policy logprobs from the current batch view."""
    if isinstance(batch, SequenceBatch):
        return batch.behavior_logprobs.array
    if isinstance(batch, TrainingBatch):
        return batch.policy_logprobs.array
    raise TypeError(f"Unsupported batch type for scoring: {type(batch)!r}")


def compute_logprobs(
    model: LmHeadModel,
    batch: TokenScoreInputs,
    key: jax.Array | None,
) -> jax.Array:
    """Compute token log probabilities for the observed next-token targets."""
    batch_size, _seq_len = batch.input_ids.array.shape

    model_output = model(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )

    logits_array = model_output.array.astype(jnp.float32)[:, :-1, :]
    target_ids_array = batch.input_ids.array[:, 1:]
    safe_temperature = jnp.where(batch.sampling_temperature.array == 0, 1.0, batch.sampling_temperature.array).reshape(
        batch_size, 1, 1
    )
    logits_array = logits_array / safe_temperature.astype(jnp.float32)

    log_probs = jax.nn.log_softmax(logits_array, axis=-1)
    logprobs_shifted = jnp.take_along_axis(log_probs, target_ids_array[..., None], axis=-1).squeeze(-1)
    return jnp.concatenate(
        [jnp.zeros((logprobs_shifted.shape[0], 1), dtype=logprobs_shifted.dtype), logprobs_shifted],
        axis=1,
    )


def chunked_compute_logprobs(
    model: LmHeadModel,
    batch: TokenScoreInputs,
    key: jax.Array | None,
    block_size: int,
) -> jax.Array:
    """Compute token log probabilities with blockwise vocab processing."""
    activations = model.activations(
        input_ids=batch.input_ids,
        attn_mask=AttentionMask.causal(),
        pos_ids=batch.position_ids,
        key=key,
    )
    if isinstance(activations, tuple):
        activations, _aux = activations

    pred_embeddings = activations.astype(jnp.float32)
    pred_lm_head = model.get_lm_head().astype(jnp.float32)

    safe_temperature = hax.where(batch.sampling_temperature == 0, 1.0, batch.sampling_temperature).astype(jnp.float32)
    pred_embeddings = pred_embeddings / safe_temperature

    pos_axis = batch.input_ids.resolve_axis("position")
    target_y = hax.roll(batch.input_ids, -1, pos_axis)

    loss = fused_cross_entropy_loss_and_logsumexp_penalty(
        pred_embeddings,
        pred_lm_head,
        Contract=model.Embed,
        Label=model.Vocab,
        target_y=target_y,
        reduction=None,
        logsumexp_weight=0.0,
        block_size=block_size,
        dtype=jnp.float32,
    )

    logprobs_shifted = -loss.array[:, :-1]
    return jnp.concatenate(
        [jnp.zeros((logprobs_shifted.shape[0], 1), dtype=logprobs_shifted.dtype), logprobs_shifted],
        axis=1,
    )


def compute_model_logprobs(
    model: LmHeadModel,
    batch: ScoredBatch,
    key: jax.Array | None,
    *,
    vocab_tile_size: int | None = None,
) -> jax.Array:
    """Compute logprobs using either the eager or chunked backend."""
    score_inputs = token_score_inputs_from_batch(batch)
    if vocab_tile_size is None:
        return compute_logprobs(model, score_inputs, key)
    return chunked_compute_logprobs(model, score_inputs, key, vocab_tile_size)


@dataclass(frozen=True)
class LocalScoreSource:
    """Local Levanter-backed scorer for student, reference, and teacher models."""

    score_requirements: ScoreRequirements
    vocab_tile_size: int | None = None
    backend_name: str = "levanter-local"

    def requirements(self) -> ScoreRequirements:
        return self.score_requirements

    def score(
        self,
        batch: ScoredBatch,
        info: BatchInfo | None,
        roles: ModelRoles,
        *,
        key: jax.Array | None,
    ) -> ScoreBundle:
        del info

        requirements = self.score_requirements
        if requirements.teacher_entropy:
            raise NotImplementedError("Teacher entropy scoring is not implemented yet")
        if requirements.teacher_topk_support is not None:
            raise NotImplementedError("Teacher top-k scoring is not implemented yet")

        score_bundle = ScoreBundle()
        if requirements.behavior_logprobs:
            score_bundle.behavior_logprobs = behavior_logprobs_from_batch(batch)

        if requirements.student_logprobs:
            score_bundle.student_logprobs = compute_model_logprobs(
                roles.student,
                batch,
                key,
                vocab_tile_size=self.vocab_tile_size,
            )
            score_bundle.student_pass_count = 1

        if requirements.reference_logprobs:
            if roles.reference is None:
                raise ValueError("reference model is required to score reference logprobs")
            score_bundle.reference_logprobs = compute_model_logprobs(
                roles.reference,
                batch,
                key,
                vocab_tile_size=self.vocab_tile_size,
            )
            score_bundle.reference_pass_count = 1

        if requirements.teacher_token_logprobs:
            if roles.teacher is None:
                raise ValueError("teacher model is required to score teacher logprobs")
            score_bundle.teacher_logprobs = compute_model_logprobs(
                roles.teacher,
                batch,
                key,
                vocab_tile_size=self.vocab_tile_size,
            )
            score_bundle.teacher_pass_count = 1

        return score_bundle
