# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.base.model import GrugModelConfig
from experiments.probabilistic_dataflow.compiler import (
    AttentionLayout,
    PackedBatch,
    ParallelQuery,
    TokenCodec,
    compile_query,
    lower_to_transformer,
    pack_transformer_calls,
)
from experiments.probabilistic_dataflow.scientific_model import CrossDomainTransformer
from experiments.probabilistic_dataflow.synthetic import (
    advection_example,
    advection_problem,
    symmetric_pairs_example,
    symmetric_pairs_problem,
)


@dataclass(frozen=True)
class TrainingResult:
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    supervised_tokens: int
    packed_rows: int
    task_families: tuple[str, ...]


@dataclass(frozen=True)
class TaskBatch:
    name: str
    token_ids: np.ndarray
    scientific_position_ids: np.ndarray
    rotary_position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    segment_ids: np.ndarray
    attention_layout: AttentionLayout


@dataclass(frozen=True)
class TaskTrainingMetrics:
    initial_loss: float
    final_loss: float
    initial_accuracy: float
    final_accuracy: float
    supervised_tokens: int


@dataclass(frozen=True)
class CrossDomainTrainingResult:
    initial_loss: float
    final_loss: float
    text: TaskTrainingMetrics
    science: TaskTrainingMetrics
    task_families: tuple[str, ...]
    shared_vocab_size: int


TEXT_SENTENCES = (
    ("bos", "the", "ocean", "field", "changes", "slowly", "eos"),
    ("bos", "the", "protein", "contact", "changes", "slowly", "eos"),
    ("bos", "the", "ocean", "contact", "changes", "today", "eos"),
    ("bos", "the", "protein", "field", "changes", "today", "eos"),
)


def record_order_equivariance_error(*, seed: int = 0) -> float:
    """Measure the maximum logit change after permuting and restoring scientific records."""
    problem = advection_problem()
    plan = compile_query(problem.query, ParallelQuery(problem.targets))
    codec = TokenCodec()
    sequence = (
        lower_to_transformer(
            problem.program,
            plan,
            advection_example(problem, seed=seed),
            codec,
        )
        .calls[0]
        .sequences[0]
    )
    order = tuple(int(index) for index in np.random.default_rng(seed).permutation(len(sequence.token_ids)))
    permuted = sequence.reordered(order)
    config = GrugModelConfig(
        vocab_size=codec.vocab_size,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=len(sequence.token_ids),
    )
    with jax.set_mesh(compact_grug_mesh()):
        model = CrossDomainTransformer.init(
            config,
            scientific_position_count=codec.scientific_position_count,
            key=jax.random.PRNGKey(seed),
        )
        segment_ids = jnp.zeros((1, len(sequence.token_ids)), dtype=jnp.int32)
        mask = AttentionMask().with_segment_ids(segment_ids)
        logits = model.logits(
            jnp.asarray((sequence.token_ids,)),
            jnp.asarray((sequence.scientific_position_ids,)),
            mask=mask,
            rotary_position_ids=jnp.asarray((sequence.rotary_position_ids,)),
        )
        permuted_logits = model.logits(
            jnp.asarray((permuted.token_ids,)),
            jnp.asarray((permuted.scientific_position_ids,)),
            mask=mask,
            rotary_position_ids=jnp.asarray((permuted.rotary_position_ids,)),
        )
    inverse = np.argsort(np.asarray(order))
    restored_logits = np.asarray(permuted_logits)[:, inverse]
    return float(np.max(np.abs(np.asarray(logits) - restored_logits)))


def build_mixed_synthetic_batch(
    *, examples_per_problem: int = 8, max_seq_len: int = 64
) -> tuple[PackedBatch, TokenCodec]:
    if examples_per_problem <= 0:
        raise ValueError(f"examples_per_problem must be positive, got {examples_per_problem}")
    codec = TokenCodec()
    advection = advection_problem()
    contacts = symmetric_pairs_problem()
    advection_plan = compile_query(advection.query, ParallelQuery(advection.targets))
    contacts_plan = compile_query(contacts.query, ParallelQuery(contacts.targets))

    executions = []
    for seed in range(examples_per_problem):
        executions.append(
            lower_to_transformer(
                advection.program,
                advection_plan,
                advection_example(advection, seed=seed),
                codec,
            )
        )
        executions.append(
            lower_to_transformer(
                contacts.program,
                contacts_plan,
                symmetric_pairs_example(contacts, seed=10_000 + seed),
                codec,
            )
        )
    return pack_transformer_calls(tuple(executions), max_seq_len=max_seq_len), codec


def build_synthetic_text_batch(codec: TokenCodec, *, repetitions: int = 4) -> TaskBatch:
    """Build a small causal next-token workload in the shared token vocabulary."""
    if repetitions <= 0:
        raise ValueError(f"repetitions must be positive, got {repetitions}")
    sentences = [sentence for _ in range(repetitions) for sentence in TEXT_SENTENCES]
    token_ids = np.asarray([[codec.token(word) for word in sentence] for sentence in sentences], dtype=np.int32)
    target_ids = np.full_like(token_ids, -1)
    target_ids[:, :-1] = token_ids[:, 1:]
    loss_weights = np.zeros_like(token_ids, dtype=np.float32)
    loss_weights[:, :-1] = 1.0
    rotary_position_ids = np.broadcast_to(
        np.arange(token_ids.shape[1], dtype=np.int32),
        token_ids.shape,
    ).copy()
    return TaskBatch(
        name="synthetic_text",
        token_ids=token_ids,
        scientific_position_ids=np.full_like(token_ids, -1),
        rotary_position_ids=rotary_position_ids,
        target_ids=target_ids,
        loss_weights=loss_weights,
        segment_ids=np.zeros_like(token_ids),
        attention_layout=AttentionLayout.CAUSAL,
    )


def build_synthetic_advection_batch(
    codec: TokenCodec,
    *,
    examples: int = 8,
    max_seq_len: int = 64,
) -> TaskBatch:
    """Build a full-attention scientific workload in the shared token vocabulary."""
    if examples <= 0:
        raise ValueError(f"examples must be positive, got {examples}")
    problem = advection_problem()
    plan = compile_query(problem.query, ParallelQuery(problem.targets))
    executions = tuple(
        lower_to_transformer(
            problem.program,
            plan,
            advection_example(problem, seed=seed),
            codec,
        )
        for seed in range(examples)
    )
    packed = pack_transformer_calls(executions, max_seq_len=max_seq_len)
    return TaskBatch(
        name="synthetic_advection",
        token_ids=packed.token_ids,
        scientific_position_ids=packed.scientific_position_ids,
        rotary_position_ids=packed.rotary_position_ids,
        target_ids=packed.target_ids,
        loss_weights=packed.loss_weights,
        segment_ids=packed.segment_ids,
        attention_layout=AttentionLayout.FULL,
    )


def train_smoke(
    *,
    steps: int = 80,
    examples_per_problem: int = 8,
    max_seq_len: int = 64,
    seed: int = 0,
) -> TrainingResult:
    """Train a tiny Marin Grug transformer on permutation-equivariant scientific records."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    batch, codec = build_mixed_synthetic_batch(
        examples_per_problem=examples_per_problem,
        max_seq_len=max_seq_len,
    )
    mesh = compact_grug_mesh()
    model_config = GrugModelConfig(
        vocab_size=codec.vocab_size,
        hidden_dim=48,
        intermediate_dim=96,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
    )
    optimizer = optax.adam(learning_rate=3e-3)

    with jax.set_mesh(mesh):
        model = CrossDomainTransformer.init(
            model_config,
            scientific_position_count=codec.scientific_position_count,
            key=jax.random.PRNGKey(seed),
        )
        opt_state = optimizer.init(model)
        token_ids = jnp.asarray(batch.token_ids)
        scientific_position_ids = jnp.asarray(batch.scientific_position_ids)
        rotary_position_ids = jnp.asarray(batch.rotary_position_ids)
        target_ids = jnp.asarray(batch.target_ids)
        loss_weights = jnp.asarray(batch.loss_weights)
        segment_ids = jnp.asarray(batch.segment_ids)

        initial_loss, initial_accuracy = _metrics(
            model,
            token_ids,
            scientific_position_ids,
            rotary_position_ids,
            target_ids,
            loss_weights,
            segment_ids,
        )

        @eqx.filter_jit
        def train_step(current_model: CrossDomainTransformer, current_opt_state: optax.OptState):
            def loss_fn(candidate: CrossDomainTransformer):
                mask = AttentionMask().with_segment_ids(segment_ids)
                return candidate.aligned_token_loss(
                    token_ids,
                    scientific_position_ids,
                    target_ids,
                    loss_weights,
                    mask=mask,
                    rotary_position_ids=rotary_position_ids,
                    reduction="mean",
                )

            loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
            next_model = eqx.apply_updates(current_model, updates)
            return next_model, next_opt_state, loss

        for _ in range(steps):
            model, opt_state, _ = train_step(model, opt_state)
        final_loss, final_accuracy = _metrics(
            model,
            token_ids,
            scientific_position_ids,
            rotary_position_ids,
            target_ids,
            loss_weights,
            segment_ids,
        )

    return TrainingResult(
        initial_loss=float(initial_loss),
        final_loss=float(final_loss),
        initial_accuracy=float(initial_accuracy),
        final_accuracy=float(final_accuracy),
        supervised_tokens=int(np.sum(batch.loss_weights)),
        packed_rows=batch.token_ids.shape[0],
        task_families=("synthetic_advection", "synthetic_contacts"),
    )


def train_cross_domain_smoke(
    *,
    steps: int = 100,
    examples_per_task: int = 8,
    max_seq_len: int = 64,
    seed: int = 0,
) -> CrossDomainTrainingResult:
    """Train one Grug parameter set on causal text and full-attention scientific calls."""
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    codec = TokenCodec()
    text_batch = build_synthetic_text_batch(codec, repetitions=max(1, examples_per_task // len(TEXT_SENTENCES)))
    science_batch = build_synthetic_advection_batch(
        codec,
        examples=examples_per_task,
        max_seq_len=max_seq_len,
    )
    model_config = GrugModelConfig(
        vocab_size=codec.vocab_size,
        hidden_dim=48,
        intermediate_dim=96,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=max_seq_len,
    )
    optimizer = optax.adam(learning_rate=3e-3)
    text_arrays = _task_arrays(text_batch)
    science_arrays = _task_arrays(science_batch)
    text_mask = _task_attention_mask(text_batch, text_arrays[-1])
    science_mask = _task_attention_mask(science_batch, science_arrays[-1])

    with jax.set_mesh(compact_grug_mesh()):
        model = CrossDomainTransformer.init(
            model_config,
            scientific_position_count=codec.scientific_position_count,
            key=jax.random.PRNGKey(seed),
        )
        opt_state = optimizer.init(model)
        initial_text = _aligned_metrics(model, *text_arrays[:-1], mask=text_mask)
        initial_science = _aligned_metrics(model, *science_arrays[:-1], mask=science_mask)

        @eqx.filter_jit
        def train_step(current_model: CrossDomainTransformer, current_opt_state: optax.OptState):
            def loss_fn(candidate: CrossDomainTransformer):
                text_loss = _task_loss(candidate, text_arrays, text_mask)
                science_loss = _task_loss(candidate, science_arrays, science_mask)
                return 0.5 * (text_loss + science_loss)

            loss, grads = eqx.filter_value_and_grad(loss_fn)(current_model)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_model)
            next_model = eqx.apply_updates(current_model, updates)
            return next_model, next_opt_state, loss

        for _ in range(steps):
            model, opt_state, _ = train_step(model, opt_state)

        final_text = _aligned_metrics(model, *text_arrays[:-1], mask=text_mask)
        final_science = _aligned_metrics(model, *science_arrays[:-1], mask=science_mask)

    text_metrics = TaskTrainingMetrics(
        initial_loss=float(initial_text[0]),
        final_loss=float(final_text[0]),
        initial_accuracy=float(initial_text[1]),
        final_accuracy=float(final_text[1]),
        supervised_tokens=int(np.sum(text_batch.loss_weights)),
    )
    science_metrics = TaskTrainingMetrics(
        initial_loss=float(initial_science[0]),
        final_loss=float(final_science[0]),
        initial_accuracy=float(initial_science[1]),
        final_accuracy=float(final_science[1]),
        supervised_tokens=int(np.sum(science_batch.loss_weights)),
    )
    return CrossDomainTrainingResult(
        initial_loss=0.5 * (text_metrics.initial_loss + science_metrics.initial_loss),
        final_loss=0.5 * (text_metrics.final_loss + science_metrics.final_loss),
        text=text_metrics,
        science=science_metrics,
        task_families=(text_batch.name, science_batch.name),
        shared_vocab_size=codec.vocab_size,
    )


def _metrics(
    model: CrossDomainTransformer,
    token_ids: jax.Array,
    scientific_position_ids: jax.Array,
    rotary_position_ids: jax.Array,
    target_ids: jax.Array,
    loss_weights: jax.Array,
    segment_ids: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    mask = AttentionMask().with_segment_ids(segment_ids)
    return _aligned_metrics(
        model,
        token_ids,
        scientific_position_ids,
        rotary_position_ids,
        target_ids,
        loss_weights,
        mask=mask,
    )


def _aligned_metrics(
    model: CrossDomainTransformer,
    token_ids: jax.Array,
    scientific_position_ids: jax.Array,
    rotary_position_ids: jax.Array,
    target_ids: jax.Array,
    loss_weights: jax.Array,
    *,
    mask: AttentionMask,
) -> tuple[jax.Array, jax.Array]:
    loss = model.aligned_token_loss(
        token_ids,
        scientific_position_ids,
        target_ids,
        loss_weights,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
        reduction="mean",
    )
    logits = model.logits(
        token_ids,
        scientific_position_ids,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
    )
    predictions = jnp.argmax(logits, axis=-1)
    supervised = loss_weights > 0
    correct = jnp.sum((predictions == target_ids) * supervised)
    accuracy = correct / jnp.maximum(jnp.sum(supervised), 1)
    return loss, accuracy


def _task_arrays(batch: TaskBatch) -> tuple[jax.Array, ...]:
    return (
        jnp.asarray(batch.token_ids),
        jnp.asarray(batch.scientific_position_ids),
        jnp.asarray(batch.rotary_position_ids),
        jnp.asarray(batch.target_ids),
        jnp.asarray(batch.loss_weights),
        jnp.asarray(batch.segment_ids),
    )


def _task_attention_mask(batch: TaskBatch, segment_ids: jax.Array) -> AttentionMask:
    if batch.attention_layout == AttentionLayout.CAUSAL:
        return AttentionMask.causal().with_segment_ids(segment_ids)
    return AttentionMask().with_segment_ids(segment_ids)


def _task_loss(
    model: CrossDomainTransformer,
    arrays: tuple[jax.Array, ...],
    mask: AttentionMask,
) -> jax.Array:
    token_ids, scientific_position_ids, rotary_position_ids, target_ids, loss_weights, _segment_ids = arrays
    return model.aligned_token_loss(
        token_ids,
        scientific_position_ids,
        target_ids,
        loss_weights,
        mask=mask,
        rotary_position_ids=rotary_position_ids,
        reduction="mean",
    )
