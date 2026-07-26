# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import haliax as hax
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import maybe_rng_split
from haliax.quantization import OverwriteWithGradient

from levanter.layers.attention import AttentionMask
from levanter.models.llama import LlamaLMHeadModel
from levanter.models.lm_model import LmExample, LmHeadModel


class DistillationObjective(StrEnum):
    FORWARD_KL = "forward_kl"
    PROJECTED_HIDDEN = "projected_hidden"
    TAID = "taid"


class DistillationModel(eqx.Module):
    student: LmHeadModel
    teacher: LmHeadModel
    hidden_projector: hnn.Linear | None = None
    taid_state: "TaidState | None" = None


@dataclass(frozen=True)
class TaidConfig:
    start: float = 0.4
    end: float = 1.0
    learning_rate: float = 5e-4
    momentum: float = 0.99


class TaidState(OverwriteWithGradient):
    interpolation: jax.Array
    loss_momentum: jax.Array
    previous_loss: jax.Array
    update_count: jax.Array
    has_previous_loss: jax.Array

    @staticmethod
    def init(config: TaidConfig, initial_loss: float = 0.0) -> "TaidState":
        return TaidState(
            interpolation=jnp.asarray(config.start, dtype=jnp.float32),
            loss_momentum=jnp.asarray(0.0, dtype=jnp.float32),
            previous_loss=jnp.asarray(initial_loss, dtype=jnp.float32),
            update_count=jnp.asarray(0.0, dtype=jnp.float32),
            has_previous_loss=jnp.asarray(float(initial_loss != 0.0), dtype=jnp.float32),
        )


def distillation_trainable_filter(model: DistillationModel) -> DistillationModel:
    """Return a filter that trains and checkpoints only the student."""
    return DistillationModel(
        student=True,  # type: ignore[arg-type]
        teacher=False,  # type: ignore[arg-type]
        hidden_projector=model.hidden_projector is not None,  # type: ignore[arg-type]
        taid_state=model.taid_state is not None,  # type: ignore[arg-type]
    )


def _weighted_mean(loss: NamedArray, loss_weight: NamedArray) -> jax.Array:
    total_weight = hax.sum(loss_weight).scalar()
    return hax.sum(loss * loss_weight).scalar() / total_weight


def forward_kl_loss(
    student_logits: NamedArray,
    teacher_logits: NamedArray,
    loss_weight: NamedArray,
    *,
    Vocab: Axis,
) -> jax.Array:
    """Compute full-vocabulary forward KL cross-entropy on scored positions.

    The omitted teacher entropy is constant with respect to the student, so this
    cross-entropy has the same student gradient as ``KL(teacher || student)``.
    """
    student_log_probs = hax.nn.log_softmax(student_logits.astype(jnp.float32), Vocab)
    teacher_probs = hax.nn.softmax(jax.lax.stop_gradient(teacher_logits).astype(jnp.float32), Vocab)
    per_position_loss = -hax.sum(teacher_probs * student_log_probs, axis=Vocab)
    return _weighted_mean(per_position_loss, loss_weight.astype(jnp.float32))


def hard_label_next_token_loss(
    logits: NamedArray,
    tokens: NamedArray,
    *,
    Vocab: Axis,
    Pos: Axis,
) -> NamedArray:
    """Compute unreduced next-token NLL from materialized logits."""
    logits = logits.astype(jnp.float32)
    target_ids = hax.roll(tokens, -1, Pos)
    log_normalizers = hax.nn.logsumexp(logits, Vocab)
    target_logits = logits.take(Vocab, target_ids)
    return log_normalizers - target_logits


def taid_target_logits(
    student_logits: NamedArray,
    teacher_logits: NamedArray,
    interpolation: jax.Array,
) -> NamedArray:
    interpolation = interpolation.astype(jnp.float32)
    student_target = jax.lax.stop_gradient(student_logits).astype(jnp.float32)
    teacher_target = jax.lax.stop_gradient(teacher_logits).astype(jnp.float32)
    return (1.0 - interpolation) * student_target + interpolation * teacher_target


def taid_loss(
    student_logits: NamedArray,
    teacher_logits: NamedArray,
    loss_weight: NamedArray,
    state: TaidState,
    *,
    Vocab: Axis,
) -> jax.Array:
    target_logits = taid_target_logits(student_logits, teacher_logits, state.interpolation)
    return forward_kl_loss(student_logits, target_logits, loss_weight, Vocab=Vocab)


def model_with_layer_anchors(
    model: LlamaLMHeadModel,
    input_ids: NamedArray,
    attn_mask: AttentionMask | NamedArray,
    anchor_indices: tuple[int, ...],
    *,
    key=None,
) -> tuple[NamedArray, NamedArray]:
    """Run a Llama-family model while retaining only configured residual-stream anchors."""
    if not anchor_indices:
        raise ValueError("anchor_indices must not be empty")
    if len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError("anchor_indices must be unique")
    if min(anchor_indices) < 0 or max(anchor_indices) >= model.config.num_layers:
        raise ValueError(f"anchor_indices must be in [0, {model.config.num_layers}), got {anchor_indices}")

    Anchor = Axis("anchor", len(anchor_indices))
    layer_indices = hax.arange(model.config.Layers)
    requested_indices = hax.named(jnp.asarray(anchor_indices, dtype=jnp.int32), Anchor)
    hidden = model.embeddings.embed(input_ids)
    anchors = hax.zeros((Anchor, *hidden.axes), dtype=hidden.dtype)
    layer_keys = maybe_rng_split(key, model.config.num_layers) if key is not None else None

    def capture_anchor(layer, carry, layer_index, layer_key):
        hidden, anchors = carry
        hidden = layer(hidden, mask=attn_mask, key=layer_key)
        matches = layer_index == requested_indices
        anchors = hax.where(matches, hidden.broadcast_axis(Anchor), anchors)
        return hidden, anchors

    hidden, anchors = model.transformer.layers.fold_via(capture_anchor)(
        (hidden, anchors),
        layer_indices,
        layer_keys,
    )
    hidden = model.transformer.norm(hidden)
    logits = hax.dot(hidden, model.get_lm_head(), axis=model.Embed)
    return logits, anchors


def projected_hidden_loss(
    student_anchors: NamedArray,
    teacher_anchors: NamedArray,
    loss_weight: NamedArray,
    projector: hnn.Linear,
    *,
    TeacherEmbed: Axis,
    Anchor: Axis,
) -> jax.Array:
    """Compute cosine distance between projected student and teacher residual streams."""
    teacher_anchors = jax.lax.stop_gradient(teacher_anchors).rename({TeacherEmbed.name: projector.Out})
    projected_student = projector(student_anchors)
    projected_student = projected_student / hax.sqrt(
        hax.mean(hax.square(projected_student), axis=projector.Out) + 1e-6
    )
    normalized_teacher = teacher_anchors / hax.sqrt(hax.mean(hax.square(teacher_anchors), axis=projector.Out) + 1e-6)
    per_anchor_loss = 1.0 - hax.mean(projected_student * normalized_teacher, axis=projector.Out)
    per_position_loss = hax.mean(per_anchor_loss, axis=Anchor)
    return _weighted_mean(per_position_loss, loss_weight.astype(jnp.float32))


def projected_hidden_distillation_loss(
    model: DistillationModel,
    example: LmExample,
    *,
    student_anchor_indices: tuple[int, ...],
    teacher_anchor_indices: tuple[int, ...],
    key=None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    if model.hidden_projector is None:
        raise ValueError("hidden_projector is required for projected hidden-state distillation")
    if not isinstance(model.student, LlamaLMHeadModel) or not isinstance(model.teacher, LlamaLMHeadModel):
        raise TypeError("Projected hidden-state distillation requires Llama-family teacher and student models")
    if len(student_anchor_indices) != len(teacher_anchor_indices):
        raise ValueError("Student and teacher anchor lists must have the same length")

    student_logits, student_anchors = model_with_layer_anchors(
        model.student,
        example.tokens,
        example.attn_mask,
        student_anchor_indices,
        key=key,
    )
    teacher_logits, teacher_anchors = model_with_layer_anchors(
        model.teacher,
        example.tokens,
        example.attn_mask,
        teacher_anchor_indices,
        key=None,
    )
    teacher_logits = jax.lax.stop_gradient(teacher_logits)
    teacher_anchors = jax.lax.stop_gradient(teacher_anchors)

    kd_loss = forward_kl_loss(
        student_logits,
        teacher_logits,
        example.loss_weight,
        Vocab=model.student.Vocab,
    )
    Anchor = student_anchors.resolve_axis("anchor")
    hidden_loss = projected_hidden_loss(
        student_anchors,
        teacher_anchors,
        example.loss_weight,
        model.hidden_projector,
        TeacherEmbed=model.teacher.Embed,
        Anchor=Anchor,
    )
    loss = 0.9 * kd_loss + 0.1 * hidden_loss
    return loss, {"kd_loss": kd_loss, "hidden_loss": hidden_loss}


def update_taid_state(
    state: TaidState,
    loss: jax.Array,
    num_train_steps: int,
    config: TaidConfig,
) -> TaidState:
    """Advance TAID from detached training loss, following the reference implementation."""
    loss = jax.lax.stop_gradient(loss).astype(jnp.float32)
    denominator = jnp.maximum(jnp.abs(state.previous_loss), jnp.finfo(jnp.float32).eps)
    relative_change = jnp.where(
        state.has_previous_loss > 0,
        (state.previous_loss - loss) / denominator,
        0.0,
    )
    loss_momentum = config.momentum * state.loss_momentum + (1.0 - config.momentum) * relative_change
    adaptive_delta = jax.nn.sigmoid(loss_momentum)
    progress = jnp.minimum(state.update_count.astype(jnp.float32) / float(num_train_steps), 1.0)
    linear_floor = config.start + (config.end - config.start) * progress
    increment = config.learning_rate * adaptive_delta * (1.0 - state.interpolation)
    interpolation = jnp.minimum(config.end, jnp.maximum(linear_floor, state.interpolation + increment))
    return TaidState(
        interpolation=interpolation,
        loss_momentum=loss_momentum,
        previous_loss=loss,
        update_count=state.update_count + 1,
        has_previous_loss=jnp.asarray(1.0, dtype=jnp.float32),
    )


@partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def taid_loss_with_state_update(
    loss: jax.Array,
    state: TaidState,
    num_train_steps: int,
    config: TaidConfig,
) -> jax.Array:
    return loss


def _taid_loss_with_state_update_fwd(
    loss: jax.Array,
    state: TaidState,
    num_train_steps: int,
    config: TaidConfig,
):
    del num_train_steps, config
    return loss, (loss, state)


def _taid_loss_with_state_update_bwd(
    num_train_steps: int,
    config: TaidConfig,
    residual,
    gradient,
):
    loss, state = residual
    updated_state = update_taid_state(state, loss, num_train_steps, config)
    return gradient, updated_state


taid_loss_with_state_update.defvjp(
    _taid_loss_with_state_update_fwd,
    _taid_loss_with_state_update_bwd,
)


def distillation_loss(
    model: DistillationModel,
    example: LmExample,
    *,
    objective: DistillationObjective = DistillationObjective.FORWARD_KL,
    taid_state: TaidState | None = None,
    key=None,
) -> jax.Array:
    """Compute an online-teacher next-token distillation loss."""
    if objective == DistillationObjective.PROJECTED_HIDDEN:
        raise ValueError("Use projected_hidden_distillation_loss with explicit anchor mappings")
    teacher_logits = jax.lax.stop_gradient(model.teacher(example.tokens, example.attn_mask, key=None))
    student_logits = model.student(example.tokens, example.attn_mask, key=key)
    if objective == DistillationObjective.FORWARD_KL:
        return forward_kl_loss(
            student_logits,
            teacher_logits,
            example.loss_weight,
            Vocab=model.student.Vocab,
        )
    if objective == DistillationObjective.TAID:
        if taid_state is None:
            raise ValueError("taid_state is required for the TAID objective")
        return taid_loss(
            student_logits,
            teacher_logits,
            example.loss_weight,
            taid_state,
            Vocab=model.student.Vocab,
        )
    raise ValueError(f"Unsupported distillation objective: {objective}")
