# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.testing.devices import use_cpu_devices

from experiments.post_training.long_loop.sft_runtime.model import GrugModelConfig, student_topk_logprobs
from experiments.post_training.long_loop.sft_runtime.teacher_targets import TeacherLmExample, TeacherTopK
from experiments.post_training.long_loop.sft_runtime.train import _accumulated_loss_and_grads, initial_state

VOCAB = 32
K = 3
CONFIG = GrugModelConfig(
    vocab_size=VOCAB,
    hidden_dim=16,
    intermediate_dim=8,
    shared_expert_intermediate_dim=8,
    num_shared_experts=1,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=2,
    num_kv_heads=1,
    local_kv_heads=1,
    global_kv_heads=1,
    head_dim=8,
    max_seq_len=8,
    sliding_window=4,
    global_every=2,
    capacity_factor=4.0,
    attention_implementation="reference",
    moe_implementation="fixed_all_to_all",
)
MP = jmp.get_policy("float32")


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()).reshape(1, 2, 2, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _batches(mesh: Mesh, teacher_mask: np.ndarray) -> tuple[GrugLmExample, TeacherLmExample]:
    rng = np.random.default_rng(0)
    tokens = rng.integers(1, VOCAB, size=(4, 8)).astype(np.int32)
    weights = np.array(
        [[0] * 8, [1, 1, 0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 0]], np.float32
    )
    weights[1, :2] = 0.5
    segments = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], np.int32), (4, 1))
    ids = np.stack([rng.permutation(VOCAB)[:K] for _ in range(32)]).reshape(4, 8, K).astype(np.int32)
    logprobs = np.log(rng.dirichlet(np.ones(K), size=(4, 8)) * 0.9).astype(np.float32)
    rows = NamedSharding(mesh, P("data", None))
    rows3 = NamedSharding(mesh, P("data", None, None))
    mask = AttentionMask.causal().with_segment_ids(jax.device_put(segments, rows))
    hard = GrugLmExample(jax.device_put(tokens, rows), jax.device_put(weights, rows), mask)
    teacher = TeacherTopK(
        ids=jax.device_put(ids, rows3),
        logprobs=jax.device_put(logprobs, rows3),
        mask=jax.device_put(teacher_mask.astype(np.float32), rows),
    )
    return hard, TeacherLmExample(
        tokens=hard.tokens, loss_weight=hard.loss_weight, attn_mask=hard.attn_mask, teacher=teacher
    )


def _dense_soft_cross_entropy(params, batch: TeacherLmExample) -> jax.Array:
    """Reference: full logits -> log_softmax -> gather, weighted like the hard loss."""
    logits = params.logits(batch.tokens, mask=batch.attn_mask)
    student = _dense_gather(jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1), batch.teacher.ids)
    q = _replicated(jax.nn.softmax(batch.teacher.logprobs, axis=-1))
    weight = _replicated(batch.loss_weight * batch.teacher.mask)
    return jnp.sum(-jnp.sum(q * student, axis=-1) * weight) / jnp.sum(batch.loss_weight)


def _replicated(x: jax.Array) -> jax.Array:
    return jax.sharding.reshard(x, P())


def _dense_gather(logp: jax.Array, ids: jax.Array) -> jax.Array:
    return jnp.take_along_axis(_replicated(logp), _replicated(ids), axis=-1)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("steps", [1, 2])
def test_zero_alpha_matches_hard_loss_bitwise(steps):
    with use_cpu_devices(4):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            state = initial_state(CONFIG, optimizer=optax.adam(1e-3), mp=MP, key=jax.random.key(1), ema_beta=None)
            hard, teacher = _batches(mesh, np.ones((4, 8)))

            def run(batch, alpha):
                return jax.jit(
                    lambda params, b: _accumulated_loss_and_grads(params, b, MP, 1e-4, steps, distill_alpha=alpha)
                )(state.params, batch)

            (hard_loss, hard_metrics), hard_grads = run(hard, 0.0)
            (loss, metrics), grads = run(teacher, 0.0)
            np.testing.assert_array_equal(loss, hard_loss)
            np.testing.assert_array_equal(metrics["train/cross_entropy_loss"], hard_metrics["train/cross_entropy_loss"])
            for actual, expected in zip(jax.tree.leaves(grads), jax.tree.leaves(hard_grads), strict=True):
                np.testing.assert_array_equal(actual, expected)

            (distill_loss, _), distill_grads = run(teacher, 0.5)
            assert float(distill_loss) > float(hard_loss)
            assert any(
                not np.array_equal(a, b)
                for a, b in zip(jax.tree.leaves(distill_grads), jax.tree.leaves(hard_grads), strict=True)
            )


@pytest.mark.timeout(300)
@pytest.mark.parametrize("steps", [1, 2])
def test_soft_term_matches_dense_reference_and_masked_rows_add_nothing(steps):
    with use_cpu_devices(4):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            state = initial_state(CONFIG, optimizer=optax.adam(1e-3), mp=MP, key=jax.random.key(1), ema_beta=None)
            # Row 1 has scored positions but no teacher data.
            teacher_mask = np.ones((4, 8))
            teacher_mask[1] = 0
            hard, teacher = _batches(mesh, teacher_mask)
            alpha = 0.5

            def run(batch, alpha):
                return jax.jit(
                    lambda params, b: _accumulated_loss_and_grads(params, b, MP, None, steps, distill_alpha=alpha)
                )(state.params, batch)

            (hard_loss, _), hard_grads = run(hard, 0.0)
            (loss, metrics), grads = run(teacher, alpha)
            reference_soft, reference_soft_grads = jax.jit(jax.value_and_grad(_dense_soft_cross_entropy))(
                state.params, teacher
            )
            np.testing.assert_allclose(metrics["train/distill_soft_ce"], reference_soft, rtol=1e-5)
            np.testing.assert_allclose(loss, hard_loss + alpha * reference_soft, rtol=1e-5)
            for actual, hard_grad, soft_grad in zip(
                jax.tree.leaves(grads), jax.tree.leaves(hard_grads), jax.tree.leaves(reference_soft_grads), strict=True
            ):
                np.testing.assert_allclose(actual, hard_grad + alpha * soft_grad, rtol=2e-4, atol=1e-6)

            # Arbitrary targets in the masked row leave loss and gradients unchanged.
            ids = np.array(teacher.teacher.ids)
            logprobs = np.array(teacher.teacher.logprobs)
            ids[1] = 7
            logprobs[1] = -3.0
            scrambled = TeacherTopK(
                ids=jax.device_put(ids, teacher.teacher.ids.sharding),
                logprobs=jax.device_put(logprobs, teacher.teacher.logprobs.sharding),
                mask=teacher.teacher.mask,
            )
            other = TeacherLmExample(
                tokens=teacher.tokens, loss_weight=teacher.loss_weight, attn_mask=teacher.attn_mask, teacher=scrambled
            )
            (other_loss, _), other_grads = run(other, alpha)
            np.testing.assert_allclose(other_loss, loss, rtol=1e-6)
            for actual, expected in zip(jax.tree.leaves(other_grads), jax.tree.leaves(grads), strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.timeout(300)
def test_blocked_student_logprobs_and_teacher_kl_match_dense():
    with use_cpu_devices(4):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            state = initial_state(CONFIG, optimizer=optax.adam(1e-3), mp=MP, key=jax.random.key(1), ema_beta=None)
            _, batch = _batches(mesh, np.ones((4, 8)))
            rng = np.random.default_rng(1)
            spec = P(("replica_dcn", "data", "expert"), "context", None)
            hidden = jax.device_put(rng.normal(size=(4, 8, 16)).astype(np.float32), NamedSharding(mesh, spec))
            lm_head = state.params.output_proj
            cotangent = jnp.asarray(rng.normal(size=(4, 8, K)).astype(np.float32))

            # A block of 3 does not divide each shard's 8 positions, so the padded path runs.
            def blocked(h, w):
                return jnp.sum(student_topk_logprobs(h, w, batch.teacher.ids, token_block_size=3) * cotangent)

            def dense(h, w):
                logp = jax.nn.log_softmax(_replicated(h) @ _replicated(w), axis=-1)
                return jnp.sum(_dense_gather(logp, batch.teacher.ids) * cotangent)

            value, grads = jax.jit(jax.value_and_grad(blocked, argnums=(0, 1)))(hidden, lm_head)
            expected_value, expected_grads = jax.jit(jax.value_and_grad(dense, argnums=(0, 1)))(hidden, lm_head)
            np.testing.assert_allclose(value, expected_value, rtol=1e-5)
            for actual, expected in zip(grads, expected_grads, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)

            kl = jax.jit(lambda p, b: p.teacher_kl(b.tokens, b.teacher, mask=b.attn_mask))(state.params, batch)
            logits = jax.jit(lambda p, b: p.logits(b.tokens, mask=b.attn_mask))(state.params, batch)
            student = _dense_gather(jax.nn.log_softmax(logits, axis=-1), batch.teacher.ids)
            log_q = _replicated(jax.nn.log_softmax(batch.teacher.logprobs, axis=-1))
            np.testing.assert_allclose(kl, jnp.sum(jnp.exp(log_q) * (log_q - student), axis=-1), rtol=1e-4, atol=1e-6)
