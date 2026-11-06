# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remote Iris job: score Snowball's Levanter training forward at every position against stored vLLM log-probs.

The serving parity check scores Levanter's inference load path at each prompt's last position. This job scores
the model the training path builds, at every position of the same 64 representative prompts:

- the HF export loads as ``train_lm`` loads it: HF config reload (``use_hf_model_config``), the Snowball runtime
  settings restored with ``preserve_hf_runtime_settings``, weights loaded at their stored dtype because
  ``fp32_fields`` is set, then ``cast_to_param`` under the trainer's parameter mapping;
- the forward runs on the trainer's compute view, ``precision_policy.cast_to_compute(model)`` (``p=f32,c=bf16``
  with ``router_bias`` kept FP32), on the identity recipe's expert-parallel mesh shrunk to one H100x8 host
  (data 2 x expert 4);
- the runtime settings are the identity recipe's: dropless capacity (factor = expert-axis size), FP32 router dot,
  FP32 routed combine, FP32 SiLU/sigmoid and torch-order attention. ``torch_order`` materializes S x S scores, so
  prompts in buckets above ``TORCH_ORDER_MAX_TOKENS`` use the recipe's other attention, FA4;
- the forward is ``activations_with_stats`` with a token-validity grid, so the same pass reports MoE drops.

Forward only: no optimizer state, no update, no export. The gate is ``snowball_position_parity``: per prompt
category and past the sliding window, p99 and max absolute log-prob difference over vLLM's top-k, and zero drops.
"""

import dataclasses
import logging
import time

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from haliax.partitioning import named_jit
from jax.sharding import PartitionSpec as P
from jax.sharding import reshard
from levanter.fp32_fields import check_fp32_fields
from levanter.main.train_lm import preserve_hf_runtime_settings
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.trainer import TrainerConfig
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

DATA_AXIS = 2
EXPERT_AXIS = 4
FP32_FIELDS = ("router_bias",)
TRAINING_PRECISION = "p=f32,c=bfloat16,o=f32"
TORCH_ORDER_MAX_TOKENS = 4096
LONG_CONTEXT_ATTENTION = "gpu_fa4_cute"
HEAD_CHUNK_TOKENS = 1024
"""Positions projected through the LM head at once; full-vocab FP32 logits of a 32K row do not fit."""
PAD_ID = 128001
"""Snowball's EOS/PAD. Padding is invalid for dispatch, causally invisible to real positions and discarded."""


def training_mesh() -> MeshConfig:
    """The identity recipe's single-host mesh without its context axis: batch over data x expert."""
    return MeshConfig(
        axes={"data": DATA_AXIS, "expert": EXPERT_AXIS, "replica": 1, "model": 1, "context": 1},
        dcn_axes={"replica_dcn": 1},
        compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        param_mapping={"embed": ["data", "context"]},
    )


def training_trainer_config() -> TrainerConfig:
    return TrainerConfig(
        mp=jmp.get_policy(TRAINING_PRECISION),
        fp32_fields=FP32_FIELDS,
        mesh=training_mesh(),
        use_explicit_mesh_axes=True,
        train_batch_size=BATCH_SIZE,
        per_device_parallelism=1,
    )


def requested_runtime_config() -> SnowballConfig:
    """The runtime-only settings the identity recipe trains with; the HF config supplies the architecture."""
    return SnowballConfig(
        moe_capacity_factor=float(EXPERT_AXIS),
        router_dot_fp32=True,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation="torch_order",
        reference_checkpoint=SNOWBALL.export_uri,
        tokenizer=SNOWBALL.export_uri,
    )


def load_training_compute_view(trainer: TrainerConfig) -> SnowballLMHeadModel:
    """Load the export as ``train_lm`` does and return the trainer's compute view. Call inside the mesh."""
    requested = requested_runtime_config()
    converter = requested.hf_checkpoint_converter()
    model_config = preserve_hf_runtime_settings(
        converter.config_from_hf_config(converter.default_hf_config), requested, trainer.fp32_fields
    )
    assert isinstance(model_config, SnowballConfig)
    model = converter.load_pretrained(
        model_config.model_type,
        config=model_config,
        axis_mapping=trainer.parameter_axis_mapping,
        dtype=None,
    )
    check_fp32_fields(model, trainer.fp32_fields)
    policy = trainer.precision_policy
    model = named_jit(policy.cast_to_param, trainer.parameter_axis_mapping)(model)
    with hax.axis_mapping(trainer.compute_axis_mapping):
        view = policy.cast_to_compute(model)
    del model
    assert isinstance(view, SnowballLMHeadModel)
    mlp = view.transformer.blocks[0].mlp
    assert mlp.router_bias.dtype == jnp.float32, mlp.router_bias.dtype
    assert mlp.router.dtype == jnp.bfloat16, mlp.router.dtype
    return view


def _forward(model: SnowballLMHeadModel, tokens: jax.Array, valid: jax.Array):
    Batch, Pos = Axis("batch", tokens.shape[0]), model.Pos.resize(tokens.shape[1])
    hidden, stats = model.activations_with_stats(hax.named(tokens, (Batch, Pos)), hax.named(valid, (Batch, Pos)))
    return hidden.array, reshard(stats.sender_dropped, P()), reshard(stats.receiver_dropped, P())


def _head_chunk(
    model: SnowballLMHeadModel,
    hidden: jax.Array,
    targets: jax.Array,
    reference_ids: jax.Array,
    start: jax.Array,
):
    """Target log-probs, top-1 ids and log-probs at the reference ids for ``HEAD_CHUNK_TOKENS`` positions from
    ``start``."""
    size = min(HEAD_CHUNK_TOKENS, hidden.shape[1])
    hidden = jax.lax.dynamic_slice_in_dim(hidden, start, size, axis=1)
    targets = jax.lax.dynamic_slice_in_dim(targets, start, size, axis=1)
    reference_ids = jax.lax.dynamic_slice_in_dim(reference_ids, start, size, axis=1)
    logits = jnp.einsum("bsd,dv->bsv", hidden, model.get_lm_head().array, preferred_element_type=jnp.float32)
    batch_spec, position_spec = jax.typeof(logits).sharding.spec[:2]
    logprobs = jax.nn.log_softmax(reshard(logits, P(batch_spec, position_spec, None)), axis=-1)
    targets = reshard(targets, P(batch_spec, position_spec))
    reference_ids = reshard(reference_ids, P(batch_spec, position_spec, None))
    target_logprobs = jnp.take_along_axis(logprobs, targets[..., None], axis=-1)[..., 0]
    top1 = jnp.argmax(logprobs, axis=-1).astype(jnp.int32)
    measured = jnp.take_along_axis(logprobs, jnp.maximum(reference_ids, 0), axis=-1)
    measured = jnp.where(reference_ids >= 0, measured, jnp.nan)
    return tuple(reshard(x, P()) for x in (target_logprobs, top1, measured))


def _score_batch(
    forward,
    head,
    model: SnowballLMHeadModel,
    batch: PromptBatch,
    reference_grids: dict[str, np.ndarray],
) -> tuple[dict[str, ScoredPositions], np.ndarray, np.ndarray]:
    token_ids, _ = pad_prompt_batch(batch, PAD_ID)
    length = batch.max_tokens
    valid = np.zeros((BATCH_SIZE, length), dtype=bool)
    targets = np.full_like(token_ids, PAD_ID)
    targets[:, :-1] = token_ids[:, 1:]
    width = next(iter(reference_grids.values())).shape[1]
    reference = np.full((BATCH_SIZE, length, width), -1, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        n = len(case.prompt_token_ids)
        valid[row, :n] = True
        # Grid position p predicts token p + 1: reference entry p.
        reference[row, : n - 1] = reference_grids[case.id]
    hidden, sender, receiver = forward(model, jnp.asarray(token_ids), jnp.asarray(valid))
    targets_device, reference_device = jnp.asarray(targets), jnp.asarray(reference)
    chunks = []
    for start in range(0, length, min(HEAD_CHUNK_TOKENS, length)):
        outputs = head(model, hidden, targets_device, reference_device, jnp.int32(start))
        chunks.append(tuple(np.asarray(jax.device_get(x)) for x in outputs))
    target_lp, top1, measured = (np.concatenate(parts, axis=1) for parts in zip(*chunks, strict=True))
    scored = {
        case.id: ScoredPositions(
            case_id=case.id,
            target_logprobs=target_lp[row, : len(case.prompt_token_ids) - 1],
            top1_token_ids=top1[row, : len(case.prompt_token_ids) - 1],
            reference_logprobs=measured[row, : len(case.prompt_token_ids) - 1],
        )
        for row, case in enumerate(batch.cases)
    }
    return scored, np.asarray(jax.device_get(sender)), np.asarray(jax.device_get(receiver))


def score_training_forward_against_vllm_positions() -> None:
    """Entry point: one H100x8 task. Raises ``AssertionError`` listing every gate failure."""
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

    trainer = training_trainer_config()
    assert trainer.data_axis_size == BATCH_SIZE, trainer.data_axis_size
    started = time.time()
    with trainer.use_device_mesh():
        view = load_training_compute_view(trainer)
        long_view = with_attention_implementation(view, LONG_CONTEXT_ATTENTION)
        logger.info("Loaded the training compute view in %.0fs: %s", time.time() - started, view.config)
        with hax.axis_mapping(trainer.compute_axis_mapping):
            forward = jax.jit(_forward)
            head = jax.jit(_head_chunk)
            scored: dict[str, ScoredPositions] = {}
            sender = receiver = None
            for index, batch in enumerate(prompt_fixture.batches):
                model = view if batch.max_tokens <= TORCH_ORDER_MAX_TOKENS else long_view
                batch_scores, batch_sender, batch_receiver = _score_batch(forward, head, model, batch, reference_grids)
                scored |= batch_scores
                sender = batch_sender if sender is None else sender + batch_sender
                receiver = batch_receiver if receiver is None else receiver + batch_receiver
                logger.info(
                    "Training forward batch %d/%d: max_tokens=%d attention=%s (%.0fs)",
                    index + 1,
                    len(prompt_fixture.batches),
                    batch.max_tokens,
                    model.config.attention_implementation,
                    time.time() - started,
                )
    assert sender is not None and receiver is not None

    stats = compare_positions(reference, scored)
    for group, group_stats in stats.items():
        logger.info("training-forward vs vLLM %s: %s", group, dataclasses.asdict(group_stats))
    # TODO: layer-0 expert-route agreement with vLLM needs vLLM's per-position routes, which the reference does not
    # hold; add them to the reference format before gating routes here.
    failures = [
        f"logprobs: {failure}" for failure in logprob_failures(stats, expected_groups=expected_groups(reference))
    ]
    failures += [f"dispatch: {failure}" for failure in dispatch_failures(sender.tolist(), receiver.tolist())]
    assert not failures, "Snowball training forward differs from vLLM:\n" + "\n".join(failures)
