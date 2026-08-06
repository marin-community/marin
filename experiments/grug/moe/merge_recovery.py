# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Loss and update primitives for recovering a post-hoc merged expert bank."""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import register_dataclass
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MOE_REMAT_SAVE_NAMES
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss

from experiments.grug.moe.model import Transformer


class RecoveryStage(StrEnum):
    """The trainable surface and objective used by a merge-recovery phase."""

    LOCAL = "local"
    PRESERVATION = "preservation"


@dataclass(frozen=True)
class MergeRecoveryConfig:
    """Configuration shared by the local and preservation recovery objectives."""

    affected_layers: tuple[int, int]
    stage: RecoveryStage
    moe_loss_weight: float = 1.0
    logit_kl_weight: float = 0.0
    normalization_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.stage, RecoveryStage):
            raise ValueError(f"stage must be a RecoveryStage, got {self.stage!r}")
        if len(set(self.affected_layers)) != 2:
            raise ValueError(f"affected_layers must contain two distinct layers, got {self.affected_layers}")
        if tuple(sorted(self.affected_layers)) != self.affected_layers:
            raise ValueError(f"affected_layers must be in model order, got {self.affected_layers}")
        if self.moe_loss_weight < 0:
            raise ValueError("moe_loss_weight must be non-negative")
        if self.logit_kl_weight < 0:
            raise ValueError("logit_kl_weight must be non-negative")
        if self.stage is RecoveryStage.LOCAL and self.logit_kl_weight != 0:
            raise ValueError("local recovery does not include logit KL")
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")


class RecoveryForward(NamedTuple):
    hidden: jax.Array
    moe_loss: jax.Array
    moe_output_nrmse: jax.Array
    qb_beta_per_layer: jax.Array


class RecoveryLosses(NamedTuple):
    total: jax.Array
    cross_entropy: jax.Array
    moe: jax.Array
    logit_kl: jax.Array
    moe_output_nrmse: jax.Array
    qb_beta_per_layer: jax.Array


LogitKlLoss = Callable[
    [Transformer, Transformer, jax.Array, AttentionMask | jax.Array | None, jax.Array],
    jax.Array,
]


@register_dataclass
@dataclass(frozen=True)
class MergeRecoveryState:
    step: jax.Array
    params: Transformer
    opt_state: optax.OptState
    pending_qb_betas: jax.Array


def _validate_affected_layers(model: Transformer, affected_layers: tuple[int, int]) -> int:
    num_layers = len(model.blocks)
    if any(layer < 0 or layer >= num_layers for layer in affected_layers):
        raise IndexError(f"affected_layers must lie in [0, {num_layers}), got {affected_layers}")
    bank_indices = tuple(model.blocks[layer].expert_bank_index for layer in affected_layers)
    if len(set(bank_indices)) != 1:
        raise ValueError(f"affected student layers must share one expert bank, got bank IDs {bank_indices}")
    shared_bank = bank_indices[0]
    bank_use_count = sum(block.expert_bank_index == shared_bank for block in model.blocks)
    if bank_use_count != len(affected_layers):
        raise ValueError(
            "the initial recovery requires the merged bank to be used only by the two affected layers; "
            f"bank {shared_bank} is used by {bank_use_count} layers"
        )
    return shared_bank


def _validate_teacher(student: Transformer, teacher: Transformer, affected_layers: tuple[int, int]) -> None:
    if len(student.blocks) != len(teacher.blocks):
        raise ValueError("student and teacher must have the same number of layers")
    student_with_teacher_topology = dataclasses.replace(
        student.config,
        expert_bank_for_layer=teacher.config.expert_bank_for_layer,
    )
    if student_with_teacher_topology != teacher.config:
        raise ValueError("student and teacher configs may differ only in expert-bank topology")
    teacher_banks = tuple(teacher.blocks[layer].expert_bank_index for layer in affected_layers)
    if len(set(teacher_banks)) != len(affected_layers):
        raise ValueError("the recovery teacher must retain distinct expert banks at the affected layers")


def recovery_trainable_filter(model: Transformer, config: MergeRecoveryConfig) -> Transformer:
    """Return an Equinox filter selecting exactly the parameters trained in a phase."""
    shared_bank = _validate_affected_layers(model, config.affected_layers)
    filter_spec = jax.tree.map(lambda _: False, model)
    bank_filter = jax.tree.map(eqx.is_inexact_array, model.expert_banks[shared_bank])
    filter_spec = eqx.tree_at(
        lambda current: current.expert_banks[shared_bank],
        filter_spec,
        bank_filter,
    )
    if config.stage is RecoveryStage.PRESERVATION:
        for layer in config.affected_layers:
            filter_spec = eqx.tree_at(
                lambda current, layer=layer: current.blocks[layer].mlp.router,
                filter_spec,
                True,
            )
    return filter_spec


def apply_affected_qb_betas(
    model: Transformer,
    qb_betas: jax.Array,
    affected_layers: tuple[int, int],
) -> Transformer:
    """Apply pending QB statistics only to the routers unfrozen for recovery."""
    expected_shape = (len(model.blocks), model.config.num_experts)
    if qb_betas.shape != expected_shape:
        raise ValueError(f"qb_betas must have shape {expected_shape}, got {qb_betas.shape}")

    blocks = list(model.blocks)
    for layer in affected_layers:
        block = blocks[layer]
        bias = -qb_betas[layer]
        bias = bias - jnp.mean(bias)
        mlp = eqx.tree_at(lambda current: current.router_bias, block.mlp, bias)
        blocks[layer] = eqx.tree_at(lambda current: current.mlp, block, mlp)
    return eqx.tree_at(lambda current: current.blocks, model, tuple(blocks))


def update_affected_qb_betas(
    pending_qb_betas: jax.Array,
    measured_qb_betas: jax.Array,
    affected_layers: tuple[int, int],
) -> jax.Array:
    """Replace pending QB statistics only for affected layers."""
    if pending_qb_betas.shape != measured_qb_betas.shape:
        raise ValueError(
            "pending and measured QB statistics must have identical shapes, got "
            f"{pending_qb_betas.shape} and {measured_qb_betas.shape}"
        )
    updated = pending_qb_betas
    for layer in affected_layers:
        updated = updated.at[layer].set(measured_qb_betas[layer])
    return updated


def recovery_forward(
    student: Transformer,
    teacher: Transformer,
    token_ids: jax.Array,
    *,
    affected_layers: tuple[int, int],
    mask: AttentionMask | jax.Array | None = None,
    normalization_epsilon: float = 1e-8,
) -> RecoveryForward:
    """Roll out the student and match teacher MoEs on each current student state."""
    _validate_affected_layers(student, affected_layers)
    _validate_teacher(student, teacher, affected_layers)
    if normalization_epsilon <= 0:
        raise ValueError("normalization_epsilon must be positive")
    if mask is None:
        mask = AttentionMask.causal()

    if student.config.remat_mode == "save_moe":
        remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
    else:
        remat_policy = None

    hidden = student.embed_inputs(token_ids)
    affected_layer_set = set(affected_layers)
    normalized_errors: list[jax.Array] = []
    qb_betas: list[jax.Array] = []
    for layer_index, student_block in enumerate(student.blocks):
        options = student.block_call_options(mask, layer_index)
        student_bank = student.expert_banks[student_block.expert_bank_index]
        trace = eqx.filter_checkpoint(student_block.forward_with_moe_trace, policy=remat_policy)(
            hidden,
            options.mask,
            student_bank,
            options.use_pko,
            options.disable_rope,
        )
        hidden = trace.hidden
        qb_betas.append(trace.router_stats["qb_beta"])
        if layer_index not in affected_layer_set:
            continue

        teacher_block = teacher.blocks[layer_index]
        teacher_bank = teacher.expert_banks[teacher_block.expert_bank_index]
        teacher_output, _ = teacher_block.mlp(trace.mlp_input, teacher_bank)
        teacher_output = jax.lax.stop_gradient(teacher_output.astype(jnp.float32))
        student_output = trace.routed_output.astype(jnp.float32)
        numerator = jnp.mean(jnp.square(student_output - teacher_output))
        denominator = jax.lax.stop_gradient(jnp.mean(jnp.square(teacher_output))) + normalization_epsilon
        normalized_errors.append(numerator / denominator)

    normalized_error = jnp.stack(normalized_errors)
    return RecoveryForward(
        hidden=student.final_gated_norm(student.final_norm(hidden)),
        moe_loss=jnp.mean(normalized_error),
        moe_output_nrmse=jnp.sqrt(normalized_error),
        qb_beta_per_layer=jnp.stack(qb_betas),
    )


def recovery_objective(
    student: Transformer,
    teacher: Transformer,
    token_ids: jax.Array,
    loss_weight: jax.Array | None,
    *,
    config: MergeRecoveryConfig,
    mask: AttentionMask | jax.Array | None = None,
    logit_kl_loss: LogitKlLoss | None = None,
) -> RecoveryLosses:
    """Compute the stage objective without materializing full-vocabulary logits."""
    forward = recovery_forward(
        student,
        teacher,
        token_ids,
        affected_layers=config.affected_layers,
        mask=mask,
        normalization_epsilon=config.normalization_epsilon,
    )

    zero = jnp.zeros((), dtype=jnp.float32)
    if config.stage is RecoveryStage.LOCAL:
        cross_entropy = zero
    else:
        if loss_weight is None:
            raise ValueError("preservation recovery requires per-token loss weights")
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        cross_entropy = fused_linear_softmax_cross_entropy_loss(
            forward.hidden,
            student.output_proj,
            labels,
            weight=loss_weight.astype(jnp.float32),
            reduction="mean",
            logsumexp_weight=None,
            dtype=jnp.float32,
        )

    if config.logit_kl_weight == 0:
        logit_kl = zero
    else:
        if logit_kl_loss is None:
            raise NotImplementedError(
                "logit KL requires an explicit streaming or fused implementation; "
                "merge recovery does not materialize full-vocabulary logits"
            )
        logit_kl = logit_kl_loss(student, teacher, token_ids, mask, forward.hidden)

    total = cross_entropy + config.moe_loss_weight * forward.moe_loss + config.logit_kl_weight * logit_kl
    return RecoveryLosses(
        total=total,
        cross_entropy=cross_entropy,
        moe=forward.moe_loss,
        logit_kl=logit_kl,
        moe_output_nrmse=forward.moe_output_nrmse,
        qb_beta_per_layer=forward.qb_beta_per_layer,
    )


def initial_recovery_state(
    params: Transformer,
    *,
    optimizer: optax.GradientTransformation,
    pending_qb_betas: jax.Array,
    config: MergeRecoveryConfig,
) -> MergeRecoveryState:
    """Initialize optimizer state for only the parameters trained in this phase."""
    expected_shape = (len(params.blocks), params.config.num_experts)
    if pending_qb_betas.shape != expected_shape:
        raise ValueError(f"pending_qb_betas must have shape {expected_shape}, got {pending_qb_betas.shape}")
    trainable, _ = eqx.partition(params, recovery_trainable_filter(params, config))
    return MergeRecoveryState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=optimizer.init(trainable),
        pending_qb_betas=pending_qb_betas,
    )


def make_recovery_train_step(
    optimizer: optax.GradientTransformation,
    config: MergeRecoveryConfig,
    *,
    logit_kl_loss: LogitKlLoss | None = None,
):
    """Build one pure recovery update; callers may JIT it with their donation policy."""

    def train_step(
        state: MergeRecoveryState,
        teacher: Transformer,
        token_ids: jax.Array,
        loss_weight: jax.Array | None,
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[MergeRecoveryState, RecoveryLosses]:
        params = state.params
        if config.stage is RecoveryStage.PRESERVATION:
            params = apply_affected_qb_betas(params, state.pending_qb_betas, config.affected_layers)

        filter_spec = recovery_trainable_filter(params, config)
        trainable, frozen = eqx.partition(params, filter_spec)

        def loss_fn(current_trainable):
            current = eqx.combine(current_trainable, frozen)
            losses = recovery_objective(
                current,
                teacher,
                token_ids,
                loss_weight,
                config=config,
                mask=mask,
                logit_kl_loss=logit_kl_loss,
            )
            return losses.total, losses

        (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(trainable)
        updates, opt_state = optimizer.update(grads, state.opt_state, params=trainable)
        trainable = optax.apply_updates(trainable, updates)
        params = eqx.combine(trainable, frozen)

        if config.stage is RecoveryStage.LOCAL:
            pending_qb_betas = state.pending_qb_betas
        else:
            pending_qb_betas = update_affected_qb_betas(
                state.pending_qb_betas,
                losses.qb_beta_per_layer,
                config.affected_layers,
            )
        return (
            dataclasses.replace(
                state,
                step=state.step + jnp.array(1, dtype=jnp.int32),
                params=params,
                opt_state=opt_state,
                pending_qb_betas=pending_qb_betas,
            ),
            losses,
        )

    return train_step
