# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote Iris job: score Snowball's Levanter training forward at every position against stored vLLM log-probs.

The serving parity check scores Levanter's inference load path at each prompt's last position. This job scores
the model ``train_lm`` builds, at every position of the same 64 representative prompts:

- ``training_lm_config`` is a ``TrainLmConfig`` with the identity recipe's model settings: ``initialize_from_hf``
  the Snowball export, ``use_hf_model_config``, ``FrozenParameterAdaptorConfig(("router_bias",))``,
  ``p=f32,c=bf16`` and ``fp32_fields=("router_bias",)``, on the recipe's expert-parallel mesh shrunk to one
  H100x8 host (data 2 x expert 4);
- ``build_training_state`` repeats ``train_lm.main``'s inline HF-config block (``preserve_hf_runtime_settings``),
  then calls ``train_lm``'s own ``_load_lm_model_from_configured_source`` and ``Trainer.initial_state``;
- the forward runs on the view the trainer's loss sees, ``trainer.mp.cast_to_compute(state.model)`` under the
  compute mapping (``WrappedLossFunction``'s cast), through ``activations_with_stats`` with every token valid,
  which dispatches exactly as the loss's ``activations`` and also counts MoE drops;
- log-probs come from the training loss itself: ``maybe_fused_next_token_loss`` unreduced, as
  ``compute_next_token_loss`` calls it (``loss_dtype=float32``, no z-loss), once for the actual next token and once
  per vLLM top-k rank with that rank's ids as the targets.

Forward only: no update and no export. ``TrainingForwardSettings`` holds the recipe's precision fixes; only the
all-on defaults are gated, and any other setting reports its results for an ablation.
"""

import dataclasses
import logging
import time
from dataclasses import dataclass

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jmp
import numpy as np
import optax
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.adaptor import FrozenParameterAdaptorConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.main.train_lm import TrainLmConfig, _load_lm_model_from_configured_source, preserve_hf_runtime_settings
from levanter.models.loss import maybe_fused_next_token_loss
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.tracker import NoopConfig
from levanter.trainer import Trainer, TrainerConfig
from levanter.trainer_state import TrainerState
from levanter.utils.mesh import MeshConfig

from marin.testing.inference.snowball import (
    BATCH_SIZE,
    SNOWBALL,
    PromptBatch,
    pad_prompt_batch,
    read_prompt_fixture,
    read_representative_goldens,
    read_vllm_position_reference,
)
from marin.testing.inference.snowball_backend_parity_jobs import GPU_COUNT, with_attention_implementation
from marin.testing.inference.snowball_position_parity import (
    ScoredPositions,
    check_reference_matches_prompts,
    compare_positions,
    dispatch_failures,
    expected_groups,
    logprob_failures,
    reference_id_grid,
)

logger = logging.getLogger(__name__)

# The identity recipe's settings (experiments/post_training/identity_context_distill/sft.py), repeated because
# lib/ cannot import experiments/.
DATA_AXIS = 2
EXPERT_AXIS = 4
FP32_FIELDS = ("router_bias",)
FROZEN_FIELDS = ("router_bias",)
TRAINING_PRECISION = "p=f32,c=bfloat16,o=f32"
RECIPE_ATTENTION = "torch_order"
RECIPE_TRAINING_SEQUENCE_LENGTH = 8192
"""The identity SFT's training length. ``torch_order`` materializes S x S FP32 scores (20 heads); the default
``TORCH_ORDER_MAX_TOKENS`` of 4096 does not reach it."""
TORCH_ORDER_MAX_TOKENS = 4096
"""Prompt buckets above this run ``LONG_CONTEXT_ATTENTION``: a 16K row's torch-order scores are about 21 GB per
layer. The prompt buckets are 256, 1K, 4K, 16K and 32K tokens, so no bucket sits at the recipe's 8192."""
LONG_CONTEXT_ATTENTION = "gpu_fa4_cute"
PAD_ID = 128001
"""Snowball's EOS/PAD. Padding is causally invisible to real positions and its scores are discarded."""
SEED = 0


@dataclass(frozen=True)
class TrainingForwardSettings:
    """The recipe's precision fixes, each switchable for the threshold ablation. Defaults are the recipe.

    ``train_lm`` refuses ``router_dot_fp32`` without the FP32 router bias, so the ``fp32_router_bias=False`` arm
    also needs ``router_dot_fp32=False``.
    """

    moe_capacity_factor: float = float(EXPERT_AXIS)
    router_dot_fp32: bool = True
    fp32_router_bias: bool = True
    router_combine_fp32: bool = True
    fp32_activations: bool = True
    attention_implementation: str = RECIPE_ATTENTION
    torch_order_max_tokens: int = TORCH_ORDER_MAX_TOKENS

    @property
    def gated(self) -> bool:
        return self == TrainingForwardSettings()


def training_mesh() -> MeshConfig:
    """The identity recipe's single-host mesh without its context axis: batch over data x expert."""
    return MeshConfig(
        axes={"data": DATA_AXIS, "expert": EXPERT_AXIS, "replica": 1, "model": 1, "context": 1},
        dcn_axes={"replica_dcn": 1},
        # No "position": "context": with context 1, Snowball leaves position unsharded, and the loss's shard_map
        # rejects a mapping that names an axis the activations do not carry.
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "vocab": "model"},
        param_mapping={"embed": ["data", "context"]},
    )


def training_lm_config(settings: TrainingForwardSettings) -> TrainLmConfig:
    """The smallest ``TrainLmConfig`` whose model path is the identity recipe's; data and optimizer are unused."""
    return TrainLmConfig(
        data=LmDataConfig(tokenizer=SNOWBALL.export_uri),
        model=SnowballConfig(
            moe_capacity_factor=settings.moe_capacity_factor,
            router_dot_fp32=settings.router_dot_fp32,
            router_combine_fp32=settings.router_combine_fp32,
            fp32_activations=settings.fp32_activations,
            attention_implementation=settings.attention_implementation,  # pyrefly: ignore[bad-argument-type]
            reference_checkpoint=SNOWBALL.export_uri,
            tokenizer=SNOWBALL.export_uri,
        ),
        initialize_from_hf=SNOWBALL.export_uri,
        use_hf_model_config=True,
        adapter=FrozenParameterAdaptorConfig(frozen_field_names=FROZEN_FIELDS),
        trainer=TrainerConfig(
            id="snowball-training-parity",
            seed=SEED,
            tracker=NoopConfig(),
            mp=jmp.get_policy(TRAINING_PRECISION),
            fp32_fields=FP32_FIELDS if settings.fp32_router_bias else (),
            mesh=training_mesh(),
            use_explicit_mesh_axes=True,
            train_batch_size=BATCH_SIZE,
            per_device_parallelism=1,
            load_checkpoint=False,
        ),
    )


def _router_biases(model: SnowballLMHeadModel) -> list[jax.Array]:
    return [block.mlp.router_bias for block in model.transformer.blocks]


def build_training_state(config: TrainLmConfig, trainer: Trainer) -> TrainerState:
    """``train_lm.main``'s model setup, inside ``trainer``'s context.

    The HF-config block is inline in ``main`` and repeated here; the load and the trainer state are ``train_lm``'s
    own functions. The optimizer is ``optax.identity()``: the recipe's Adam state (two FP32 copies of 67B
    parameters) does not fit one node, and a forward does not read it.
    """
    tokenizer = config.data.the_tokenizer
    assert isinstance(config.model, SnowballConfig) and isinstance(config.initialize_from_hf, str)
    converter = config.model.hf_checkpoint_converter()
    converter.warn_if_tokenizer_mismatch(tokenizer)
    converter = converter.replaced(reference_checkpoint=config.initialize_from_hf, tokenizer=tokenizer)
    config = dataclasses.replace(
        config,
        model=preserve_hf_runtime_settings(
            converter.config_from_hf_config(converter.default_hf_config), config.model, config.trainer.fp32_fields
        ),
    )
    _, _, model_key, training_key = jrandom.split(jrandom.PRNGKey(config.trainer.seed), 4)
    Vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), trainer.parameter_axis_mapping)
    model = _load_lm_model_from_configured_source(
        config=config,
        converter=converter,
        Vocab=Vocab,
        model_key=model_key,
        adapter_key=jrandom.fold_in(model_key, ord("a")),
        parameter_axis_mapping=trainer.parameter_axis_mapping,
        trainer=trainer,
    )
    return trainer.initial_state(training_key, model=model, is_trainable=config.adapter.trainable_filter(model))


def training_compute_view(trainer: Trainer, state: TrainerState) -> SnowballLMHeadModel:
    """The model the trainer's loss sees: ``WrappedLossFunction``'s cast of the stored state."""
    model = state.model
    assert isinstance(model, SnowballLMHeadModel)
    with hax.axis_mapping(trainer.compute_axis_mapping):
        view = trainer.mp.cast_to_compute(model)
    assert isinstance(view, SnowballLMHeadModel)
    if trainer.config.fp32_fields:
        for stored, viewed in zip(_router_biases(model), _router_biases(view), strict=True):
            assert stored.dtype == jnp.float32, f"stored router_bias is {stored.dtype}"
            assert viewed.dtype == jnp.float32, f"compute-view router_bias is {viewed.dtype}"
    assert view.transformer.blocks[0].mlp.router.dtype == jnp.bfloat16
    return view


def _scores(model: SnowballLMHeadModel, tokens: jax.Array, rank_ids: jax.Array):
    """Per grid position ``p``: the log-prob of token ``p + 1`` and of each vLLM rank's id, from the training loss.

    ``rank_ids[r, b, p + 1]`` is the id vLLM ranked ``r`` at position ``p``, placed where ``next_token_ids`` reads a
    target. Every token is valid: ``activations_with_stats`` then dispatches exactly as the loss's ``activations``
    (which passes no validity and treats every token as valid) and also counts drops, padding included.
    """
    Batch, Pos = Axis("batch", tokens.shape[0]), model.Pos.resize(tokens.shape[1])
    input_ids = hax.named(tokens, (Batch, Pos))
    hidden, stats = model.activations_with_stats(
        input_ids, hax.named(jnp.ones(tokens.shape, dtype=jnp.bool_), (Batch, Pos))
    )
    # The loss's shard_map expects batch-sharded targets and weights, as the trainer's data loader lays them out.
    loss_weight = hax.shard(hax.ones((Batch, Pos), dtype=jnp.float32))

    def logprobs(true_ids: hax.NamedArray) -> jax.Array:
        # compute_next_token_loss's call: loss_dtype float32, a float32 loss weight, no z-loss (the recipe's
        # z_loss_weight is 0). The unreduced loss is -log p(target) per position.
        loss = maybe_fused_next_token_loss(
            model.Pos,
            model.Embed,
            model.Vocab,
            hidden,
            model.get_lm_head(),
            hax.shard(true_ids),
            loss_weight=loss_weight,
            reduction=None,
            dtype=jnp.float32,
        )
        return -loss.array

    target = logprobs(input_ids)
    ranks = jnp.stack([logprobs(hax.named(rank_ids[r], (Batch, Pos))) for r in range(rank_ids.shape[0])], axis=-1)
    return tuple(reshard(x, P()) for x in (target, ranks, stats.sender_dropped, stats.receiver_dropped))


def _score_batch(
    scores,
    model: SnowballLMHeadModel,
    batch: PromptBatch,
    reference_grids: dict[str, np.ndarray],
) -> tuple[dict[str, ScoredPositions], np.ndarray, np.ndarray]:
    token_ids, _ = pad_prompt_batch(batch, PAD_ID)
    width = next(iter(reference_grids.values())).shape[1]
    reference = np.full((BATCH_SIZE, batch.max_tokens, width), -1, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        # Grid position p predicts token p + 1: reference entry p.
        reference[row, : len(case.prompt_token_ids) - 1] = reference_grids[case.id]
    # next_token_ids shifts targets left by one, so the id scored at position p sits at p + 1. Missing ids score
    # token 0 and are masked back to NaN below.
    rank_ids = np.zeros((width, BATCH_SIZE, batch.max_tokens), dtype=np.int32)
    rank_ids[:, :, 1:] = np.maximum(reference[:, :-1], 0).transpose(2, 0, 1)
    target, ranks, sender, receiver = (
        np.asarray(jax.device_get(x)) for x in scores(model, jnp.asarray(token_ids), jnp.asarray(rank_ids))
    )
    ranks = np.where(reference >= 0, ranks, np.nan)
    scored = {
        case.id: ScoredPositions(
            case_id=case.id,
            target_logprobs=target[row, : len(case.prompt_token_ids) - 1],
            reference_logprobs=ranks[row, : len(case.prompt_token_ids) - 1],
        )
        for row, case in enumerate(batch.cases)
    }
    return scored, sender, receiver


def score_training_forward_against_vllm_positions(settings: TrainingForwardSettings = TrainingForwardSettings()) -> dict:
    """Entry point: one H100x8 task. Returns the report; with the default settings, raises on any gate failure."""
    num_devices = jax.device_count()
    assert num_devices == GPU_COUNT, f"expected {GPU_COUNT} H100s, found {num_devices} devices"
    reference = read_vllm_position_reference()
    if reference.metadata.model_sha256 != SNOWBALL.export_sha256:
        raise ValueError(
            f"Reference model sha256 {reference.metadata.model_sha256} is not the scored export {SNOWBALL.export_sha256}"
        )
    logger.info("vLLM position reference metadata: %s", dataclasses.asdict(reference.metadata))
    prompt_fixture = read_prompt_fixture(read_representative_goldens())
    check_reference_matches_prompts(reference, {case.id: case.prompt_token_ids for case in prompt_fixture.cases})
    reference_grids = {case.id: reference_id_grid(case, reference.metadata.top_k) for case in reference.cases}

    config = training_lm_config(settings)
    assert config.trainer.data_axis_size == BATCH_SIZE, config.trainer.data_axis_size
    started = time.time()

    def loss_function(model, example, *, key=None):
        return model.compute_next_token_loss(example, key=key, logsumexp_weight=config.z_loss_weight)

    attention_by_case: dict[str, str] = {}
    scored: dict[str, ScoredPositions] = {}
    sender = receiver = None
    with Trainer(config.trainer, optax.identity(), loss_function, add_default_hooks=False) as trainer:
        state = build_training_state(config, trainer)
        view = training_compute_view(trainer, state)
        # Forward only: the FP32 master copy is not read again.
        del state
        long_view = with_attention_implementation(view, LONG_CONTEXT_ATTENTION)
        logger.info("Built the training compute view in %.0fs: %s", time.time() - started, view.config)
        with hax.axis_mapping(trainer.compute_axis_mapping):
            scores = jax.jit(_scores)
            for index, batch in enumerate(prompt_fixture.batches):
                model = view if batch.max_tokens <= settings.torch_order_max_tokens else long_view
                batch_scores, batch_sender, batch_receiver = _score_batch(scores, model, batch, reference_grids)
                scored |= batch_scores
                attention = str(model.config.attention_implementation)
                attention_by_case |= {case.id: attention for case in batch.cases}
                sender = batch_sender if sender is None else sender + batch_sender
                receiver = batch_receiver if receiver is None else receiver + batch_receiver
                logger.info(
                    "Training forward batch %d/%d: max_tokens=%d attention=%s (%.0fs)",
                    index + 1,
                    len(prompt_fixture.batches),
                    batch.max_tokens,
                    attention,
                    time.time() - started,
                )
    assert sender is not None and receiver is not None

    stats = compare_positions(reference, scored, attention_by_case)
    # TODO: layer-0 expert-route agreement with vLLM needs vLLM's per-position routes, which the reference does not
    # hold; add them to the reference format before gating routes here.
    failures = [
        f"logprobs: {failure}" for failure in logprob_failures(stats, expected_groups=expected_groups(reference))
    ]
    failures += [f"dispatch: {failure}" for failure in dispatch_failures(sender.tolist(), receiver.tolist())]
    report = {
        "settings": dataclasses.asdict(settings),
        "gated": settings.gated,
        "recipe_training_sequence_length": RECIPE_TRAINING_SEQUENCE_LENGTH,
        "reference": dataclasses.asdict(reference.metadata),
        "groups": {group: dataclasses.asdict(group_stats) for group, group_stats in stats.items()},
        "dispatch": {"sender_dropped": sender.tolist(), "receiver_dropped": receiver.tolist()},
        "failures": failures,
    }
    logger.info("Snowball training-forward parity report: %s", report)
    if settings.gated:
        assert not failures, "Snowball training forward differs from vLLM:\n" + "\n".join(failures)
    return report
