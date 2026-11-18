# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Deletion Tuning Levanter trainer main and its trainer subclass.

Mirrors ``levanter.main.train_lm`` with four Deletion-Tuning additions and no
other behavior change: the mask settings are injected into the model config
after any HF re-derivation, a Wanda calibration pass over the run's training
data initializes the gate logits before state construction, the state init
scopes optimizer state and checkpointing to the gate logits via
``is_trainable``, and the trainer ramps alpha per optimizer step while
evaluating with the hard mask.
"""

import dataclasses
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import levanter.callbacks
from levanter.models.snowball import SnowballLMHeadModel
from levanter.models.snowball_deletion import (
    down_norms,
    init_gate_logits_jax,
)
from levanter.trainer import Trainer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DeletionTrainSettings:
    """The runtime deletion settings the launcher passes beside the trainer config."""

    total_steps: int
    alpha_start: float
    alpha_end: float
    eta: float
    lam: float
    lr: float
    calibration_sequences: int
    seed: int


def _capture_model(model: SnowballLMHeadModel) -> SnowballLMHeadModel:
    """Turn on the Wanda capture flag on every block's MoE MLP (a static field)."""
    capture_mlps = tuple(dataclasses.replace(block.mlp, capture_input_sums=True) for block in model.transformer.blocks)
    return eqx.tree_at(lambda m: tuple(block.mlp for block in m.transformer.blocks), model, capture_mlps)


def _layer_logits(weight: jax.Array, norms: jax.Array, keys: jax.Array, settings, threshold: float) -> jax.Array:
    """Per-layer gate logits ``[L, ...]`` from a stacked weight and its input norms."""

    def per_layer(w_layer, norms_layer, key_layer):
        broadcast = norms_layer.reshape(norms_layer.shape + (1,) * (w_layer.ndim - norms_layer.ndim))
        scores = jnp.abs(w_layer) * broadcast
        return init_gate_logits_jax(scores, settings.eta, key_layer, threshold)

    return jax.vmap(per_layer)(weight, norms, keys)


def _reshard_like(logits: jax.Array, reference: jax.Array) -> jax.Array:
    return jax.sharding.reshard(logits, jax.typeof(reference).sharding.spec)


def capture_wanda_norms(model, batches, *, num_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the base model over calibration batches and return (routed, shared) input norms."""
    capture = _capture_model(model)

    @jax.jit
    def capture_step(tokens):
        result = capture.transformer._forward(tokens, None, return_routes=False)
        wanda = result.wanda
        assert wanda is not None
        return (
            wanda["wanda_input_sums_local"],
            wanda["wanda_input_counts_local"],
            wanda["wanda_channel_sums_local"],
            wanda["wanda_channel_counts_local"],
        )

    sums_total = counts_total = channel_sums_total = channel_counts_total = None
    seen = 0
    for batch in batches:
        tokens = np.asarray(batch.tokens.array).reshape(-1, batch.tokens.array.shape[-1])
        sums, counts, channel_sums, channel_counts = capture_step(jnp.asarray(tokens))
        sums_host = np.asarray(sums, dtype=np.float64)
        counts_host = np.asarray(counts, dtype=np.float64)
        sums_total = sums_host if sums_total is None else sums_total + sums_host
        counts_total = counts_host if counts_total is None else counts_total + counts_host
        cs = np.asarray(channel_sums, dtype=np.float64)
        cc = np.asarray(channel_counts, dtype=np.float64)
        channel_sums_total = cs if channel_sums_total is None else channel_sums_total + cs
        channel_counts_total = cc if channel_counts_total is None else channel_counts_total + cc
        seen += tokens.shape[0]
        if seen >= num_layers * 0 + 1 and seen >= 1:
            continue
    if sums_total is None:
        raise ValueError("calibration produced no batches")
    routed = counts_total > 0
    safe_counts = np.maximum(counts_total, 1.0)
    per_expert = np.divide(sums_total, safe_counts[..., None], out=np.zeros_like(sums_total), where=routed[..., None])
    routed_mean = per_expert[routed].mean(axis=0) if routed.any() else np.zeros(sums_total.shape[-1])
    filled = np.where(routed[..., None], per_expert, routed_mean[None, :])
    routed_norms = np.sqrt(np.maximum(filled, 0.0))
    shared_norms = np.sqrt(np.maximum(channel_sums_total / np.maximum(channel_counts_total, 1.0)[:, None], 0.0))
    return routed_norms, shared_norms


def apply_deletion_init(
    model: SnowballLMHeadModel,
    settings: DeletionTrainSettings,
    routed_norms: np.ndarray,
    shared_norms: np.ndarray,
    *,
    threshold: float,
    num_layers: int,
) -> SnowballLMHeadModel:
    """Insert stacked gate logits initialized from Wanda scores and set the ramp start."""
    layer_keys = jax.random.split(jax.random.PRNGKey(settings.seed), num_layers)
    norms_jax = jnp.asarray(routed_norms, dtype=jnp.float32)
    shared_norms_jax = jnp.asarray(shared_norms, dtype=jnp.float32)

    updated = model
    for index, block in enumerate(model.transformer.blocks):
        mlp = block.mlp
        gate = _reshard_like(
            _layer_logits(mlp.expert_mlp.w_gate, norms_jax, layer_keys, settings, threshold),
            mlp.expert_mlp.w_gate,
        )

        def gate_where(m, index=index):
            return m.transformer.blocks[index].mlp.gate_logits

        updated = eqx.tree_at(gate_where, updated, gate, is_leaf=lambda x: x is None)
        if settings.masks_up_and_down:
            up = _reshard_like(
                _layer_logits(mlp.expert_mlp.w_up, norms_jax, layer_keys, settings, threshold),
                mlp.expert_mlp.w_up,
            )
            updated = eqx.tree_at(
                lambda m, index=index: m.transformer.blocks[index].mlp.up_logits,
                updated,
                up,
                is_leaf=lambda x: x is None,
            )
            down = _reshard_like(
                _layer_logits(mlp.expert_mlp.w_down, down_norms(norms_jax), layer_keys, settings, threshold),
                mlp.expert_mlp.w_down,
            )
            updated = eqx.tree_at(
                lambda m, index=index: m.transformer.blocks[index].mlp.down_logits,
                updated,
                down,
                is_leaf=lambda x: x is None,
            )
        if block.shared.gate_logits is not None or settings.masks_shared_experts:
            shared_gate = _reshard_like(
                _layer_logits(block.shared.w_gate, shared_norms_jax, layer_keys, settings, threshold),
                block.shared.w_gate,
            )
            updated = eqx.tree_at(
                lambda m, index=index: m.transformer.blocks[index].shared.gate_logits,
                updated,
                shared_gate,
                is_leaf=lambda x: x is None,
            )
    updated = eqx.tree_at(
        lambda m: m.transformer.deletion_alpha,
        updated,
        jnp.asarray(settings.alpha_start, dtype=jnp.float32),
        is_leaf=lambda x: x is None,
    )
    updated = eqx.tree_at(lambda m: m.transformer.deletion_hard, updated, jnp.asarray(0.0), is_leaf=lambda x: x is None)
    return updated


class DeletionTrainer(Trainer):
    """Ramp alpha per optimizer step and evaluate with the hard mask."""

    deletion_settings: DeletionTrainSettings | None = None

    def _train_step(self, state, batch, batch_kwargs, _no_hooks=False):
        result = super()._train_step(state, batch, batch_kwargs, _no_hooks)
        settings = self.deletion_settings
        if settings is None:
            return result
        alpha = _alpha_ramp_jax(result.state.step, settings.total_steps, settings.alpha_start, settings.alpha_end)
        model = eqx.tree_at(lambda m: m.transformer.deletion_alpha, result.state.model, alpha)
        return dataclasses.replace(result, state=dataclasses.replace(result.state, model=model))

    def add_eval_hook(self, eval_dataset, name: str | None = None):
        """Evaluate with the hard mask (deploy semantics), unlike the training forward."""
        eval_loader = self.data_loader(eval_dataset, self.EvalBatch)
        if eval_loader and (self.config.max_eval_batches is None or self.config.max_eval_batches > 0):

            @eqx.filter_jit
            def eval_loss(model, *batch, **batch_kwargs):
                hard = eqx.tree_at(lambda m: m.transformer.deletion_hard, model, jnp.asarray(1.0))
                return self.loss_fn(hard, *batch, **batch_kwargs, key=None)

            self.add_hook(
                levanter.callbacks.compute_validation_loss(
                    eval_loss, eval_loader, max_batches=self.config.max_eval_batches, name=name
                ),
                every=self.config.steps_per_eval,
            )


def _alpha_ramp_jax(step, total_steps: int, alpha_start: float, alpha_end: float) -> jax.Array:
    fraction = jnp.clip(step.astype(jnp.float32) / jnp.asarray(total_steps, jnp.float32), 0.0, 1.0)
    return alpha_start + fraction * (alpha_end - alpha_start)
