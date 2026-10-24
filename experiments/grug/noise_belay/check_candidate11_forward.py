# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.testing.cpu_devices import run_on_cpu_devices
from scipy.special import logsumexp

from experiments.grug.noise_belay.candidate11_boost import (
    measure_boost_hubs,
    reconstruct_shared_experts,
)
from experiments.grug.noise_belay.candidate11_forward import capture_forward
from experiments.grug.noise_belay.candidate11_validation import capture_screen
from experiments.grug.noise_belay.forward_credit import direction_frame
from experiments.grug.noise_belay.validation_cases import assert_routed_reconstruction, make_small_model

# These cases require the pinned runtime prepared by prepare.py.
pytestmark = pytest.mark.manual


@pytest.fixture
def model_and_mesh():
    return make_small_model()


@pytest.mark.parametrize("precision", ["f32", "bf16"])
def test_captured_forward_matches_native_with_nonuniform_depth_gates(model_and_mesh, precision):
    model, mesh = model_and_mesh
    with jax.set_mesh(mesh):
        model = jmp.get_policy(precision).cast_to_compute(model)
        tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 128), 0, 128)
        segments = jnp.broadcast_to(jnp.repeat(jnp.arange(2), 64)[None], tokens.shape)
        mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
        expected = eqx.filter_jit(lambda model, tokens, mask: model(tokens, mask)[0])(model, tokens, mask)
        actual, _ = eqx.filter_jit(capture_forward)(model, tokens, mask)
        zero_moves = tuple(jnp.zeros((2, 128, 64), jnp.float32) for _ in range(12))
        unchanged, _ = eqx.filter_jit(capture_forward)(model, tokens, mask, interventions=zero_moves)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(unchanged, expected, atol=1e-5, rtol=1e-5)


def test_intervention_respects_document_boundaries(model_and_mesh):
    model, mesh = model_and_mesh
    with jax.set_mesh(mesh):
        tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 128), 0, 128)
        segments = jnp.broadcast_to(jnp.repeat(jnp.arange(2), 64)[None], tokens.shape)
        mask = AttentionMask(is_causal=True, segment_ids=(segments, segments))
        forward = eqx.filter_jit(capture_forward)
        clean, _ = forward(model, tokens, mask)
        moves = [jnp.zeros((2, 128, 64)) for _ in range(12)]
        moves[0] = moves[0].at[:, :64, 0].set(0.1)
        moved, _ = forward(model, tokens, mask, interventions=tuple(moves))
    np.testing.assert_allclose(moved[:, 64:], clean[:, 64:], atol=1e-5, rtol=1e-5)
    assert float(jnp.max(jnp.abs(moved[:, :64] - clean[:, :64]))) > 1e-3


def test_bf16_capture_screen_accepts_sparse_rounding_but_rejects_corruption(
    model_and_mesh,
):
    model, mesh = model_and_mesh
    with jax.set_mesh(mesh):
        model = jmp.get_policy("bf16").cast_to_compute(model)
        tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 128), 0, 128)
        weights = jnp.ones(tokens.shape).at[:, -1].set(0)
        hidden, capture = eqx.filter_jit(capture_forward)(model, tokens, AttentionMask.causal())
        rounded = hidden.at[0, 0, 0].add(jnp.asarray(0.015625, jnp.bfloat16), out_sharding=hidden.sharding)
        accepted = capture_screen(
            model,
            rounded,
            hidden,
            tokens,
            weights,
            capture.routing_selected,
            capture.routing_selected,
        )
        corrupted = capture_screen(
            model,
            hidden + jnp.asarray(1.0, jnp.bfloat16),
            hidden,
            tokens,
            weights,
            capture.routing_selected,
            capture.routing_selected,
        )
    assert accepted["old_fraction_outside_1e5"] > 0
    assert accepted["passed"]
    assert not corrupted["passed"]


def test_boost_final_hub_matches_native_readout_probes(model_and_mesh):
    model, mesh = model_and_mesh
    with jax.set_mesh(mesh):
        tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 128), 0, 128)
        mask = AttentionMask.causal()
        _, capture = eqx.filter_jit(capture_forward)(model, tokens, mask)
        weights = jnp.ones(tokens.shape).at[:, -1].set(0).at[0, 3].set(0.5)
        key = jax.random.PRNGKey(4)
        actual = eqx.filter_jit(measure_boost_hubs)(
            capture.stages,
            tokens,
            weights,
            key,
            head=model.output_proj,
            norm_gain=model.final_norm.weight,
            query=model.attn_res_query_final,
            epsilon=model.config.layer_norm_eps,
            soft_cap=model.config.logit_soft_cap,
            relative_step=0.005,
            token_chunk=37,
            direction_chunk=7,
            z_loss_weight=0.0001,
        )[-1]
        labels = jnp.pad(tokens[:, 1:], ((0, 0), (0, 1)))
        labels = jnp.where(weights > 0, labels, -1)
        stage = capture.stages[-1]

        frame = np.asarray(direction_frame(64, jax.random.fold_in(key, len(capture.stages) - 1)))
    # Full float64 forward evaluations are an independent oracle. Subtracting
    # float32 losses loses precision for the model's small residual writes.
    point = np.asarray(stage.partial, dtype=np.float64).reshape(-1, 64)
    fixed = np.stack([np.asarray(source, dtype=np.float64).reshape(-1, 64) for source in stage.completed_sources])
    query = np.asarray(model.attn_res_query_final, dtype=np.float64)
    gain = np.asarray(model.final_norm.weight, dtype=np.float64)
    head = np.asarray(model.output_proj, dtype=np.float64)
    target = np.asarray(labels).reshape(-1)

    def full_readout(moved):
        sources = np.concatenate([fixed, moved[None]], axis=0)
        normalized = sources / np.sqrt(np.mean(sources**2, axis=-1, keepdims=True) + 1e-5)
        logits = normalized @ query
        source_weights = np.exp(logits - logsumexp(logits, axis=0, keepdims=True))
        mixed = np.sum(source_weights[..., None] * sources, axis=0)
        hidden = mixed / np.sqrt(np.mean(mixed**2, axis=-1, keepdims=True) + 1e-5) * gain
        capped = 10 * np.tanh((hidden @ head) / 10)
        partition = logsumexp(capped, axis=-1)
        loss = partition - capped[np.arange(len(target)), np.maximum(target, 0)] + 0.0001 * partition**2
        return np.where(target >= 0, loss, 0)

    step = 0.005 * np.maximum(np.sqrt(np.mean(point**2, axis=-1)), 1e-6)
    expected = np.zeros_like(point)
    for row in frame:
        move = step[:, None] * row
        slope = (full_readout(point + move) - full_readout(point - move)) / (2 * step)
        expected += slope[:, None] * row / len(frame)
    expected = expected.reshape(stage.partial.shape) * np.asarray(weights)[..., None]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_boost_hubs_two_device_partition_matches_single_device():
    run_on_cpu_devices(
        device_count=2,
        script="""
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
        from experiments.grug.noise_belay.candidate11_boost import measure_boost_hubs, measure_readout_parameters
        from experiments.grug.noise_belay.candidate11_forward import SublayerCapture

        rng = np.random.default_rng(52)
        partial = rng.normal(size=(4, 8, 24)).astype(np.float32)
        fixed = rng.normal(size=(4, 8, 24)).astype(np.float32)
        head = rng.normal(0, 0.1, size=(24, 32)).astype(np.float32)
        gain = rng.uniform(0.5, 1.5, size=24).astype(np.float32)
        query = rng.normal(0, 0.1, size=24).astype(np.float32)
        tokens = rng.integers(0, 32, size=(4, 8), dtype=np.int32)
        weights = np.ones((4, 8), dtype=np.float32)
        weights[:, -1] = 0
        weights[0, 3] = 0.25
        results = []
        parameter_results = []
        for count in (1, 2):
            mesh = Mesh(
                np.asarray(jax.devices()[:count]).reshape(1, 1, count, 1),
                ('replica_dcn', 'data', 'expert', 'model'),
                axis_types=(AxisType.Explicit,) * 4,
            )
            with jax.set_mesh(mesh):
                replicated = NamedSharding(mesh, P())
                p, f, h, g, q, t, w = [
                    jax.device_put(a, replicated)
                    for a in (partial, fixed, head, gain, query, tokens, weights)
                ]
                stage = SublayerCapture((f,), None, p, p, p)
                result = eqx.filter_jit(measure_boost_hubs)(
                    (stage,), t, w, jax.random.PRNGKey(2),
                    head=h, norm_gain=g, query=q, epsilon=1e-5, soft_cap=10.0,
                    relative_step=0.005, token_chunk=7, direction_chunk=5,
                    z_loss_weight=0.0001,
                )[0]
                results.append(np.asarray(result))
                parameter_credit = eqx.filter_jit(measure_readout_parameters)(
                    (stage,), t, w, jax.random.PRNGKey(2),
                    head=h, norm_gain=g, query=q, epsilon=1e-5, soft_cap=10.0,
                    head_step=0.1, query_step=0.005, token_chunk=7, direction_chunk=5,
                    z_loss_weight=0.0001,
                )
                parameter_results.append(jax.tree.map(np.asarray, parameter_credit))
        np.testing.assert_allclose(results[1], results[0], atol=1e-5, rtol=1e-4)
        for a, b in zip(parameter_results[1], parameter_results[0], strict=True):
            np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-4)
        assert np.max(np.abs(results[1][:, :-1])) > 1e-3
        np.testing.assert_array_equal(results[1][:, -1], 0)
        """,
    )


@pytest.mark.parametrize("precision", ["f32", "bf16"])
def test_shared_expert_reconstruction_matches_independent_forward_responses(model_and_mesh, precision):
    model, mesh = model_and_mesh
    rng = np.random.default_rng(43)
    with jax.set_mesh(mesh):
        model = jmp.get_policy(precision).cast_to_compute(model)
        tokens = jax.random.randint(jax.random.PRNGKey(7), (2, 128), 0, 128)
        mask = AttentionMask(is_causal=True)
        _, capture = eqx.filter_jit(capture_forward)(model, tokens, mask)
        hub_credits = []
        for stage in capture.stages:
            value = np.zeros(stage.output.shape, np.float32)
            value[0, :5] = rng.normal(size=(5, 64))
            hub_credits.append(jnp.asarray(value))
        actual = eqx.filter_jit(reconstruct_shared_experts)(model, capture, tuple(hub_credits), relative_step=0.05)
        for index in (0, 5):
            layer = model.layers()[index]
            inputs = layer.mlp_gated_norm(layer.rms_mlp(capture.stages[2 * index + 1].input))
            inputs = np.asarray(inputs, np.float64).reshape(-1, 64)[:5]
            pre = capture.shared_preactivations[index][:5]
            point = np.asarray(pre, np.float64)
            clean_hidden = np.asarray(jnp.square(jax.nn.relu(pre)), np.float64)
            weight = np.asarray(layer.shared[0].w_down, np.float64)
            measured = np.asarray(hub_credits[2 * index + 1], np.float64).reshape(-1, 64)[:5]
            frame = np.asarray(direction_frame(point.shape[-1], jax.random.PRNGKey(4)), np.float64)
            steps = 0.05 * np.maximum(np.sqrt(np.mean(point**2, axis=-1)), 1e-6)
            local_credit = np.zeros_like(point)
            for row in frame:
                upper = np.maximum(point + steps[:, None] * row, 0) ** 2 @ weight
                lower = np.maximum(point - steps[:, None] * row, 0) ** 2 @ weight
                score = np.sum((upper - lower) * measured, axis=-1) / (2 * steps)
                local_credit += score[:, None] * row / len(frame)
            np.testing.assert_allclose(actual[index].up, inputs.T @ local_credit, atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(actual[index].down, clean_hidden.T @ measured, atol=1e-5, rtol=1e-4)


def test_routed_expert_reconstruction_matches_native_mlp_with_convolution(model_and_mesh):
    assert_routed_reconstruction(model_and_mesh)


def test_routed_reconstruction_with_expert_parallelism():
    run_on_cpu_devices(
        device_count=2,
        script="""
        import equinox as eqx
        import jax
        import numpy as np
        from jax.sharding import AxisType, Mesh
        from experiments.grug.noise_belay.validation_cases import make_small_model, assert_routed_reconstruction
        from experiments.grug.fast_track.model import Transformer

        small_model, _ = make_small_model()
        mesh = Mesh(np.asarray(jax.devices()).reshape(1,1,2,1),
                    ('replica_dcn','data','expert','model'), axis_types=(AxisType.Explicit,)*4)
        with jax.set_mesh(mesh):
            model = eqx.filter_jit(Transformer.init)(small_model.config,key=jax.random.PRNGKey(0))
        assert_routed_reconstruction((model,mesh))
        """,
    )
