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

``run_deletion_train_lm`` is the pod dispatch: marin's standard SFT preparation
runs unchanged and the Deletion seam installs over ``train_lm`` for the
duration. ``DeletionHFModel`` is the SFT-launcher model source that sends a
spec's pod config to that entrypoint.
"""

import dataclasses
import itertools
import logging
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import levanter.callbacks
import levanter.main.train_lm as train_lm_module
import numpy as np
from levanter.models.snowball import SnowballLMHeadModel
from levanter.models.snowball_deletion import (
    WANDA_SAMPLE_STRIDE,
    DeletionTuningScope,
    deletion_trainable_filter,
    down_norms,
    init_gate_logits_jax,
    wanda_delete_threshold,
)
from levanter.trainer import Trainer
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.sft.launcher import HFModel, _levanter_train_job

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
    scope: DeletionTuningScope = DeletionTuningScope.ROUTED_GATE

    @property
    def masks_up_and_down(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL

    @property
    def masks_shared_experts(self) -> bool:
        return self.scope == DeletionTuningScope.ROUTED_AND_SHARED_ALL


def _capture_model(model: SnowballLMHeadModel) -> SnowballLMHeadModel:
    """Turn on the Wanda capture flag on every block's MoE MLP (a static field)."""
    capture_mlps = tuple(dataclasses.replace(block.mlp, capture_input_sums=True) for block in model.transformer.blocks)
    return eqx.tree_at(lambda m: tuple(block.mlp for block in m.transformer.blocks), model, capture_mlps)


def _layer_logits(weight: jax.Array, norms: jax.Array, key: jax.Array, settings, threshold: float) -> jax.Array:
    """One layer's gate logits from its weights, per-channel input norms, and its layer key.

    ``init_gate_logits_jax`` draws independent noise per element from the single
    layer key, so the batched computation needs no vmap and no per-expert keys.
    """
    broadcast = norms.reshape((*norms.shape, *((1,) * (weight.ndim - norms.ndim))))
    scores = jnp.abs(weight) * broadcast
    return init_gate_logits_jax(scores, settings.eta, key, threshold)


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
            _layer_logits(mlp.expert_mlp.w_gate, norms_jax[index], layer_keys[index], settings, threshold),
            mlp.expert_mlp.w_gate,
        )

        def gate_where(m, index=index):
            return m.transformer.blocks[index].mlp.gate_logits

        updated = eqx.tree_at(gate_where, updated, gate, is_leaf=lambda x: x is None)
        if settings.masks_up_and_down:
            up = _reshard_like(
                _layer_logits(mlp.expert_mlp.w_up, norms_jax[index], layer_keys[index], settings, threshold),
                mlp.expert_mlp.w_up,
            )
            updated = eqx.tree_at(
                lambda m, index=index: m.transformer.blocks[index].mlp.up_logits,
                updated,
                up,
                is_leaf=lambda x: x is None,
            )
            down = _reshard_like(
                _layer_logits(
                    mlp.expert_mlp.w_down, down_norms(norms_jax[index]), layer_keys[index], settings, threshold
                ),
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
                _layer_logits(block.shared.w_gate, shared_norms_jax[index], layer_keys[index], settings, threshold),
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

    def initial_state(self, training_key, model=None, model_init=None, *, is_trainable=True):
        """Scope optimizer state, gradients, and checkpointing to the gate logits.

        ``train_lm`` passes the adapter's trainable filter, which trains
        everything by default; a Deletion run trains only the gate logits, so
        the filter comes from the initialized model whenever one is provided
        (the weight-source init path).
        """
        if model is not None:
            is_trainable = deletion_trainable_filter(model)
        return super().initial_state(training_key, model=model, model_init=model_init, is_trainable=is_trainable)

    def _train_step(self, state, batch, batch_kwargs, _no_hooks=False):
        result = super()._train_step(state, batch, batch_kwargs, _no_hooks)
        settings = self.deletion_settings
        if settings is None:
            return result
        alpha = _alpha_ramp_jax(result.new_state.step, settings.total_steps, settings.alpha_start, settings.alpha_end)
        model = eqx.tree_at(lambda m: m.transformer.deletion_alpha, result.new_state.model, alpha)
        return dataclasses.replace(result, new_state=dataclasses.replace(result.new_state, model=model))

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


def _initialize_deletion(model, *, config, trainer, settings: DeletionTrainSettings, num_layers: int):
    """Calibrate Wanda norms on the run's training data, then insert the gate logits."""
    seq_len = config.train_seq_len or config.model.max_seq_len
    pos = config.model.max_Pos.resize(seq_len)
    # The calibration data key follows the launcher's data seed (a TrainLmConfig
    # field), falling back to the trainer seed.
    data_seed = config.data_seed if config.data_seed is not None else config.trainer.seed
    data_key = jrandom.PRNGKey(data_seed)
    train_dataset = config.data.train_set(pos, config.trainer.batch_schedule, key=data_key)
    loader = trainer.data_loader(train_dataset)
    batches = itertools.islice(loader.iter_from_step(0), max(1, settings.calibration_sequences))
    routed_norms, shared_norms = capture_wanda_norms(model, batches, num_layers=num_layers)
    scores = None
    logger.info("calibrated Wanda norms over %d batches", settings.calibration_sequences)

    # The delete threshold comes from a strided host-pooled subsample of the scores.
    sample_parts = []
    norms_host = jnp.asarray(routed_norms, dtype=jnp.float32)
    for index, block in enumerate(model.transformer.blocks):
        norms_layer = norms_host[index]
        scores = jnp.abs(block.mlp.expert_mlp.w_gate) * norms_layer.reshape((*norms_layer.shape, 1))
        strided = scores[..., ::WANDA_SAMPLE_STRIDE]
        sample_parts.append(strided.reshape(strided.shape[0], -1))
    pooled = np.concatenate([np.asarray(part).reshape(-1) for part in sample_parts])
    threshold = wanda_delete_threshold(pooled, settings.eta)
    levanter.tracker.log_summary({"deletion/wanda_threshold": threshold, "deletion/wanda_samples": int(pooled.size)})
    return apply_deletion_init(model, settings, routed_norms, shared_norms, threshold=threshold, num_layers=num_layers)


def deletion_settings_from_config(config) -> DeletionTrainSettings:
    """Derive the runtime deletion settings from the model config and trainer step count.

    Read before ``train_lm.main`` runs: the HF config reload replaces the model
    config, and the reload keeps ``deletion_tuning`` but this decodes the
    launcher-requested settings verbatim.
    """
    return DeletionTrainSettings(
        total_steps=config.trainer.num_train_steps,
        alpha_start=config.model.deletion_tuning.alpha_start,
        alpha_end=config.model.deletion_tuning.alpha_end,
        eta=config.model.deletion_tuning.eta,
        lam=config.model.deletion_tuning.lam,
        lr=config.model.deletion_tuning.lr,
        calibration_sequences=config.model.deletion_tuning.calibration_sequences,
        seed=config.model.deletion_tuning.seed,
        scope=config.model.deletion_tuning.scope,
    )


@contextmanager
def _deletion_train_lm_seam(settings: DeletionTrainSettings, num_layers: int):
    """Install the DT trainer and DT model loader over ``levanter.main.train_lm``.

    The standard main constructs ``Trainer`` and loads the model through two
    module globals; this swaps them for the Deletion variants and restores them
    afterwards, so the whole standard init/eval/checkpoint flow is reused
    verbatim.
    """
    original_loader = train_lm_module._load_lm_model_from_configured_source
    original_trainer_cls = train_lm_module.Trainer

    def deletion_loader(**kwargs):
        model = original_loader(**kwargs)
        return _initialize_deletion(
            model, config=kwargs["config"], trainer=kwargs["trainer"], settings=settings, num_layers=num_layers
        )

    trainer_cls = type("DeletionTrainerPatched", (DeletionTrainer,), {"deletion_settings": settings})

    train_lm_module._load_lm_model_from_configured_source = deletion_loader
    train_lm_module.Trainer = trainer_cls
    try:
        yield
    finally:
        train_lm_module._load_lm_model_from_configured_source = original_loader
        train_lm_module.Trainer = original_trainer_cls


def main(config):
    """Run Levanter's ``train_lm.main`` with Deletion Tuning applied.

    The settings ride ``config.model.deletion_tuning``; ``total_steps`` is the
    trainer's step count.
    """
    settings = deletion_settings_from_config(config)
    num_layers = config.model.num_layers
    with _deletion_train_lm_seam(settings, num_layers):
        train_lm_module.main(config)


def run_deletion_train_lm(config: TrainLmOnPodConfig) -> None:
    """The pod entrypoint for a Deletion Tuning SFT run.

    marin's standard ``run_levanter_train_lm`` preparation (run ID, tracker,
    path checks, execution deadline) runs unchanged; the Deletion trainer and
    model loader install over ``levanter.main.train_lm`` for the duration, so
    the pod executes the DT main through the standard flow. The settings come
    from ``config.train_config.model.deletion_tuning`` before any mutation.
    """
    train_config = config.train_config
    settings = deletion_settings_from_config(train_config)
    num_layers = train_config.model.num_layers
    with _deletion_train_lm_seam(settings, num_layers):
        run_levanter_train_lm(config)


@dataclasses.dataclass(frozen=True)
class DeletionHFModel(HFModel):
    """An HF-init model source whose pod dispatch targets the Deletion Tuning main.

    Everything else matches :class:`HFModel` verbatim — same ``initialize_from_hf``
    + ``use_hf_model_config`` semantics and the same pod config — so a Deletion
    run differs from the equivalent standard SFT run only in which main executes.
    """

    @property
    def run(self) -> Callable[..., None]:
        return partial(
            _levanter_train_job,
            task_timeout_seconds=self.task_timeout_seconds,
            processes_per_task=self.processes_per_task,
            priority=self.priority,
            run_fn=run_deletion_train_lm,
        )
