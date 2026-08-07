# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.activation import ActivationFunctionEnum

from experiments.grug.moe.expert_merge import (
    AssignmentMode,
    ExpertCostMatrix,
    ExpertProbeSet,
    ExpertReservoirCollection,
    MoeLayerTrace,
    ReservoirSample,
    SpectralProbeConfig,
    WeightedReservoir,
    add_moe_trace_to_reservoirs,
    build_spectral_probe_set,
    convert_one_expert_pair,
    estimate_input_manifold,
    eval_all_experts,
    eval_expert,
    expert_costs,
    finalize_spectral_probe_set,
    forward_with_moe_traces,
    functional_cost_matrix,
    permute_pending_qb_beta,
    permute_router,
    prepare_spectral_probe_set,
    solve_expert_assignment,
)
from experiments.grug.moe.model import GrugModelConfig, MoEMLP, Transformer


def _bank(*, num_experts: int, hidden_dim: int, intermediate_dim: int, seed: int) -> MoEExpertMlp:
    key_gate, key_up, key_down = jax.random.split(jax.random.key(seed), 3)
    return MoEExpertMlp(
        w_gate=jax.random.normal(key_gate, (num_experts, hidden_dim, intermediate_dim)),
        w_up=jax.random.normal(key_up, (num_experts, hidden_dim, intermediate_dim)),
        w_down=jax.random.normal(key_down, (num_experts, intermediate_dim, hidden_dim)),
        implementation="scatter",
        activation=ActivationFunctionEnum.silu,
        capacity_factor=1.0,
    )


def _permute_bank_source_to_shared(bank: MoEExpertMlp, source_to_shared: np.ndarray) -> MoEExpertMlp:
    shared_to_source = np.argsort(source_to_shared)
    return dataclasses.replace(
        bank,
        w_gate=bank.w_gate[shared_to_source],
        w_up=bank.w_up[shared_to_source],
        w_down=bank.w_down[shared_to_source],
    )


def _tiny_config(*, mapping: tuple[int, ...]) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=32,
        hidden_dim=8,
        intermediate_dim=3,
        shared_expert_intermediate_dim=4,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=len(mapping),
        expert_bank_for_layer=mapping,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=8,
        sliding_window=4,
        moe_implementation="scatter",
    )


def _empty_spectral_probes(inputs: np.ndarray) -> ExpertProbeSet:
    return ExpertProbeSet(
        ordinary_inputs=inputs,
        ordinary_weights=np.linspace(1.0, 2.0, inputs.shape[0], dtype=np.float32),
        centers=np.empty((0, inputs.shape[1]), dtype=np.float32),
        spectral_pairs=np.empty((0, 2, inputs.shape[1]), dtype=np.float32),
        input_directions=np.empty((inputs.shape[1], 0), dtype=np.float32),
        sensitivity_eigenvalues=np.empty((0,), dtype=np.float32),
    )


def test_eval_all_experts_matches_independent_swiglu_reference_across_chunks():
    bank = _bank(num_experts=3, hidden_dim=3, intermediate_dim=2, seed=0)
    inputs = jnp.arange(15, dtype=jnp.float32).reshape(5, 3) / 7.0 - 1.0

    actual = eval_all_experts(bank, inputs, probe_chunk_size=2, expert_chunk_size=2)
    expected = np.empty((5, 3, 3), dtype=np.float32)
    for probe_index in range(5):
        for expert_index in range(3):
            gate = inputs[probe_index] @ bank.w_gate[expert_index]
            up = inputs[probe_index] @ bank.w_up[expert_index]
            expected[probe_index, expert_index] = (jax.nn.silu(gate) * up) @ bank.w_down[expert_index]

    assert actual.shape == (5, 3, 3)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual, eval_all_experts(bank, inputs), rtol=0, atol=0)


def test_eval_all_experts_supports_model_axis_sharding():
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1, model_axis_size=jax.device_count())
    with jax.set_mesh(mesh):
        bank = _bank(num_experts=4, hidden_dim=8, intermediate_dim=8, seed=17)
        inputs = jnp.arange(128 * 8, dtype=jnp.float32).reshape(128, 8) / 100.0
        actual = eval_all_experts(bank, inputs, expert_chunk_size=2)
        gate = jnp.einsum("pd,edi->pei", inputs, bank.w_gate)
        up = jnp.einsum("pd,edi->pei", inputs, bank.w_up)
        expected = jnp.einsum("pei,eid->ped", jax.nn.silu(gate) * up, bank.w_down)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_router_and_pending_qb_permutation_preserve_routed_function():
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    config = _tiny_config(mapping=(0,))
    router_weights = jnp.array(
        [
            [1.7, -0.2, 0.5, -1.1],
            [-0.3, 1.2, 0.8, 0.1],
            [0.4, -0.7, 1.5, 0.9],
            [0.6, 0.2, -1.3, 1.1],
        ],
        dtype=jnp.float32,
    )
    source_router = MoEMLP(
        router=router_weights,
        router_bias=jnp.array([0.13, -0.27, 0.09, 0.31], dtype=jnp.float32),
        cfg=config,
    )
    source_bank = _bank(num_experts=4, hidden_dim=4, intermediate_dim=3, seed=1)
    shared_bank = _permute_bank_source_to_shared(source_bank, permutation)
    shared_router = permute_router(source_router, permutation)
    inputs = jnp.array(
        [[[0.2, -0.4, 0.7, 1.1], [1.3, 0.1, -0.6, 0.9], [-0.2, 0.8, 1.4, -0.5]]],
        dtype=jnp.float32,
    )

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        source_routing = source_router.route(inputs)
        shared_routing = shared_router.route(inputs)
        source_output, source_stats = source_router(inputs, source_bank)
        shared_output, shared_stats = shared_router(inputs, shared_bank)

    np.testing.assert_array_equal(
        shared_routing.selected_experts,
        permutation[np.asarray(source_routing.selected_experts)],
    )
    np.testing.assert_allclose(shared_routing.combine_weights, source_routing.combine_weights, rtol=0, atol=0)
    np.testing.assert_allclose(shared_output, source_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(shared_stats["routing_counts"][permutation], source_stats["routing_counts"])
    np.testing.assert_allclose(shared_stats["qb_beta"][permutation], source_stats["qb_beta"], rtol=0, atol=0)

    pending = jnp.array([-1.0, 0.5, 2.0, -0.25], dtype=jnp.float32)
    permuted_pending = permute_pending_qb_beta(pending, permutation)
    np.testing.assert_array_equal(permuted_pending[permutation], pending)


def test_assignment_modes_preserve_source_to_shared_convention():
    native = np.array([[0.0, 1.0], [1.0, 0.0]])
    tangent = np.array([[8.0, 0.0], [0.0, 8.0]])
    costs = ExpertCostMatrix(native=native, tangent=tangent, total=native + 0.5 * tangent)

    np.testing.assert_array_equal(solve_expert_assignment(costs, AssignmentMode.IDENTITY), [0, 1])
    np.testing.assert_array_equal(solve_expert_assignment(costs, AssignmentMode.NATIVE), [0, 1])
    np.testing.assert_array_equal(solve_expert_assignment(costs, AssignmentMode.SPECTRAL), [1, 0])

    four_expert_cost = np.full((4, 4), 100.0)
    expected = np.array([2, 0, 3, 1], dtype=np.int32)
    four_expert_cost[np.arange(4), expected] = np.arange(4)
    four_expert = ExpertCostMatrix(
        native=four_expert_cost,
        tangent=np.zeros_like(four_expert_cost),
        total=four_expert_cost,
    )
    np.testing.assert_array_equal(solve_expert_assignment(four_expert, AssignmentMode.NATIVE), expected)


def test_functional_native_cost_recovers_exact_expert_permutation():
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    source = _bank(num_experts=4, hidden_dim=3, intermediate_dim=2, seed=2)
    candidate = _permute_bank_source_to_shared(source, permutation)
    probes = tuple(
        _empty_spectral_probes(np.asarray(jax.random.normal(jax.random.fold_in(jax.random.key(3), expert), (5, 3))))
        for expert in range(4)
    )

    costs = functional_cost_matrix(source, candidate, probes, expert_chunk_size=2)
    assignment = solve_expert_assignment(costs, AssignmentMode.NATIVE)

    np.testing.assert_array_equal(assignment, permutation)
    np.testing.assert_allclose(costs.native[np.arange(4), permutation], 0.0, atol=1e-12)


@pytest.mark.parametrize("num_spectral_pairs", [0, 3])
def test_compiled_expert_costs_match_exploded_reference_for_partial_final_chunk(num_spectral_pairs):
    source = _bank(num_experts=3, hidden_dim=4, intermediate_dim=3, seed=20)
    candidate = _bank(num_experts=3, hidden_dim=4, intermediate_dim=3, seed=21)
    ordinary_inputs = np.asarray(jax.random.normal(jax.random.key(22), (5, 4)))
    ordinary_weights = np.linspace(0.25, 1.25, 5, dtype=np.float32)
    spectral_pairs = np.asarray(jax.random.normal(jax.random.key(23), (num_spectral_pairs, 2, 4)))
    probes = ExpertProbeSet(
        ordinary_inputs=ordinary_inputs,
        ordinary_weights=ordinary_weights,
        centers=np.empty((0, 4), dtype=np.float32),
        spectral_pairs=spectral_pairs,
        input_directions=np.empty((4, 0), dtype=np.float32),
        sensitivity_eigenvalues=np.empty((0,), dtype=np.float32),
    )
    eta = 0.7
    epsilon = 1e-6

    actual = expert_costs(
        source,
        1,
        candidate,
        probes,
        eta=eta,
        epsilon=epsilon,
        expert_chunk_size=2,
    )

    source_native = np.asarray(eval_expert(source, 1, ordinary_inputs), dtype=np.float32)
    candidate_native = np.asarray(
        eval_all_experts(candidate, ordinary_inputs, expert_chunk_size=2),
        dtype=np.float32,
    )
    native = np.sum(
        ordinary_weights[:, None, None] * np.square(candidate_native - source_native[:, None, :]),
        axis=(0, 2),
    ) / (np.sum(ordinary_weights[:, None] * np.square(source_native)) + epsilon)
    if num_spectral_pairs == 0:
        tangent = np.zeros_like(native)
    else:
        flattened_pairs = spectral_pairs.reshape(-1, 4)
        source_spectral = np.asarray(eval_expert(source, 1, flattened_pairs), dtype=np.float32).reshape(
            num_spectral_pairs, 2, 4
        )
        candidate_spectral = np.asarray(
            eval_all_experts(candidate, flattened_pairs, expert_chunk_size=2),
            dtype=np.float32,
        ).reshape(num_spectral_pairs, 2, 3, 4)
        source_delta = source_spectral[:, 1] - source_spectral[:, 0]
        candidate_delta = candidate_spectral[:, 1] - candidate_spectral[:, 0]
        tangent = np.sum(np.square(candidate_delta - source_delta[:, None, :]), axis=(0, 2)) / (
            np.sum(np.square(source_delta)) + epsilon
        )

    np.testing.assert_allclose(actual.native, native, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.tangent, tangent, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.total, native + eta * tangent, rtol=1e-6, atol=1e-6)


def test_weighted_reservoir_uses_squared_route_weights():
    second_state_wins = 0
    trials = 2048
    states = np.array([[0.0], [1.0]], dtype=np.float32)
    for seed in range(trials):
        reservoir = WeightedReservoir(capacity=1, state_dim=1, seed=seed)
        reservoir.add(states, np.array([1.0, 9.0]))
        second_state_wins += int(reservoir.sample().states[0, 0] == 1.0)

    win_rate = second_state_wins / trials
    assert 0.87 < win_rate < 0.93

    under_capacity = WeightedReservoir(capacity=3, state_dim=1, seed=0)
    under_capacity.add(states, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(under_capacity.sample().states, [[0.0]])


def test_route_collection_keeps_expert_membership_and_squared_weights():
    collection = ExpertReservoirCollection(
        num_experts=3,
        state_dim=2,
        capacity_per_expert=16,
        heldout_fraction=0.5,
        seed=7,
    )
    inputs = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]], dtype=np.float32)
    selected = np.array([[0, 2], [1, 0], [2, 1]], dtype=np.int32)
    combine = np.array([[0.5, 1.0], [1.5, 0.25], [2.0, 0.75]], dtype=np.float32)

    collection.add_routes(inputs, selected, combine)

    for expert in range(3):
        calibration = collection.calibration(expert)
        states = np.concatenate([calibration.train.states, calibration.heldout.states])
        weights = np.concatenate([calibration.train.weights, calibration.heldout.weights])
        order = np.argsort(states[:, 0])
        route_locations = np.argwhere(selected == expert)
        expected_states = inputs[route_locations[:, 0]]
        expected_weights = np.square(combine[route_locations[:, 0], route_locations[:, 1]])
        expected_order = np.argsort(expected_states[:, 0])
        np.testing.assert_array_equal(states[order], expected_states[expected_order])
        np.testing.assert_allclose(weights[order], expected_weights[expected_order])


def test_trace_collection_transfers_and_flattens_model_states():
    collection = ExpertReservoirCollection(
        num_experts=2,
        state_dim=2,
        capacity_per_expert=8,
        heldout_fraction=0.5,
        seed=9,
    )
    trace = MoeLayerTrace(
        mlp_input=jnp.array([[[0.0, 1.0], [2.0, 3.0]]]),
        selected_experts=jnp.array([[0], [1]], dtype=jnp.int32),
        combine_weights=jnp.array([[0.5], [0.75]]),
        routed_output=jnp.zeros((1, 2, 2)),
    )

    add_moe_trace_to_reservoirs(collection, trace)

    expert_zero = collection.calibration(0)
    expert_one = collection.calibration(1)
    zero_states = np.concatenate([expert_zero.train.states, expert_zero.heldout.states])
    one_states = np.concatenate([expert_one.train.states, expert_one.heldout.states])
    np.testing.assert_array_equal(zero_states, [[0.0, 1.0]])
    np.testing.assert_array_equal(one_states, [[2.0, 3.0]])


def test_weighted_input_manifold_recovers_rank_two_support():
    states = np.array(
        [[-2.0, -1.0, 0.0], [-2.0, 1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    manifold = estimate_input_manifold(states, np.ones(4), rank=2)

    np.testing.assert_allclose(manifold.mean, 0.0, atol=1e-7)
    np.testing.assert_allclose(manifold.eigenvalues, [4.0, 1.0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        manifold.scaled_basis @ manifold.scaled_basis.T,
        np.diag([4.0, 1.0, 0.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_randomized_input_manifold_recovers_weighted_high_dimensional_subspace():
    rng = np.random.default_rng(12)
    num_states = 192
    state_dim = 384
    rank = 5
    basis, _ = np.linalg.qr(rng.normal(size=(state_dim, rank)))
    latent = rng.normal(size=(num_states, rank)) * np.array([9.0, 5.0, 2.5, 1.2, 0.6])
    states = latent @ basis.T + 0.01 * rng.normal(size=(num_states, state_dim))
    weights = rng.lognormal(mean=0.0, sigma=0.7, size=num_states)
    normalized_weights = weights / np.sum(weights)
    centered = states - np.sum(states * normalized_weights[:, None], axis=0)
    weighted_centered = centered * np.sqrt(normalized_weights[:, None])
    _, expected_singular_values, expected_right_vectors = np.linalg.svd(weighted_centered, full_matrices=False)

    manifold = estimate_input_manifold(states, weights, rank=rank, seed=9)
    repeated = estimate_input_manifold(states, weights, rank=rank, seed=9)

    np.testing.assert_allclose(manifold.eigenvalues, np.square(expected_singular_values[:rank]), rtol=2e-4, atol=1e-6)
    actual_projector = manifold.eigenvectors @ manifold.eigenvectors.T
    expected_basis = expected_right_vectors[:rank].T
    expected_projector = expected_basis @ expected_basis.T
    np.testing.assert_allclose(actual_projector, expected_projector, rtol=2e-3, atol=2e-3)
    np.testing.assert_array_equal(manifold.eigenvectors, repeated.eigenvectors)
    np.testing.assert_array_equal(manifold.eigenvalues, repeated.eigenvalues)


def test_spectral_probes_are_centered_and_bounded_on_native_support():
    bank = _bank(num_experts=1, hidden_dim=3, intermediate_dim=2, seed=4)
    states = np.array(
        [
            [-2.0, -1.0, 0.0],
            [-2.0, 0.0, 0.0],
            [-2.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sample = ReservoirSample(states=states, weights=np.ones(states.shape[0]))
    config = SpectralProbeConfig(
        covariance_rank=2,
        num_centers=2,
        num_sensitive_directions=2,
        directions_per_center=1,
        radii=(0.15, 0.35),
        ordinary_samples=3,
    )

    preparation = prepare_spectral_probe_set(sample, sample, config=config, seed=5)
    probes = finalize_spectral_probe_set(bank, 0, preparation)
    composed_probes = build_spectral_probe_set(bank, 0, sample, sample, config=config, seed=5)
    manifold = estimate_input_manifold(states, np.ones(states.shape[0]), rank=2)

    for probe_field in dataclasses.fields(ExpertProbeSet):
        np.testing.assert_array_equal(getattr(probes, probe_field.name), getattr(composed_probes, probe_field.name))
    assert probes.spectral_pairs.shape == (4, 2, 3)
    assert np.all(np.diff(probes.sensitivity_eigenvalues) <= 0)
    for pair_index, pair in enumerate(probes.spectral_pairs):
        center = probes.centers[pair_index // 2]
        np.testing.assert_allclose(np.mean(pair, axis=0), center, rtol=1e-6, atol=1e-6)
        radii = np.linalg.norm(manifold.whiten(pair), axis=-1)
        assert np.all(radii <= manifold.mahalanobis_radius + 1e-5)


def test_degenerate_support_falls_back_to_finite_native_probes():
    bank = _bank(num_experts=1, hidden_dim=3, intermediate_dim=2, seed=6)
    states = np.ones((4, 3), dtype=np.float32)
    sample = ReservoirSample(states=states, weights=np.ones(4))
    config = SpectralProbeConfig(
        covariance_rank=2,
        num_centers=2,
        num_sensitive_directions=2,
        directions_per_center=1,
        ordinary_samples=2,
    )

    probes = build_spectral_probe_set(bank, 0, sample, sample, config=config, seed=0)

    assert probes.spectral_pairs.shape == (0, 2, 3)
    assert probes.input_directions.shape == (3, 0)
    assert np.all(np.isfinite(probes.all_inputs()))


def test_one_pair_conversion_preserves_full_model_for_permutation_equivalent_banks():
    permutation = np.array([2, 0, 3, 1], dtype=np.int32)
    config = _tiny_config(mapping=(0, 1, 2, 3))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(config, key=jax.random.key(8))
        representative_bank = model.expert_banks[1]
        source_bank = dataclasses.replace(
            representative_bank,
            w_gate=jax.device_put(
                np.asarray(jax.device_get(representative_bank.w_gate))[permutation],
                representative_bank.w_gate.sharding,
            ),
            w_up=jax.device_put(
                np.asarray(jax.device_get(representative_bank.w_up))[permutation],
                representative_bank.w_up.sharding,
            ),
            w_down=jax.device_put(
                np.asarray(jax.device_get(representative_bank.w_down))[permutation],
                representative_bank.w_down.sharding,
            ),
        )
        model = eqx.tree_at(
            lambda current: current.expert_banks,
            model,
            (model.expert_banks[0], representative_bank, source_bank, model.expert_banks[3]),
        )
        converted = convert_one_expert_pair(
            model,
            representative_layer=1,
            source_layer=2,
            source_to_shared=permutation,
        )
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        teacher_logits = model.logits(tokens)
        student_logits = converted.logits(tokens)

    assert converted.config.resolved_expert_bank_for_layer == (0, 1, 1, 2)
    assert tuple(block.expert_bank_index for block in converted.blocks) == (0, 1, 1, 2)
    assert len(converted.expert_banks) == 3
    assert all(block.mlp.cfg == converted.config for block in converted.blocks)
    assert all(block.attn.cfg == converted.config for block in converted.blocks)
    np.testing.assert_allclose(student_logits, teacher_logits, rtol=2e-5, atol=2e-5)
    for expected, actual in zip(
        (model.expert_banks[0], model.expert_banks[1], model.expert_banks[3]),
        converted.expert_banks,
        strict=True,
    ):
        for expected_leaf, actual_leaf in zip(
            jax.tree_util.tree_leaves(expected),
            jax.tree_util.tree_leaves(actual),
            strict=True,
        ):
            np.testing.assert_array_equal(actual_leaf, expected_leaf)


def test_forward_with_moe_traces_matches_normal_forward_and_excludes_shared_dense_output():
    config = _tiny_config(mapping=(0, 1))
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(config, key=jax.random.key(9))
        tokens = jnp.arange(8, dtype=jnp.int32).reshape(1, 8)
        expected_hidden, _ = model(tokens)
        actual_hidden, traces = forward_with_moe_traces(model, tokens, target_layers=(0, 1))

        initial_hidden = model.embed_inputs(tokens)
        block = model.blocks[0]
        options = model.block_call_options(AttentionMask.causal(), 0)
        attention_input = block.attn_gated_norm(block.rms_attn(initial_hidden))
        post_attention = initial_hidden + block.attn(
            attention_input,
            options.mask,
            use_pko=options.use_pko,
            disable_rope=options.disable_rope,
        )
        mlp_input = block.mlp_gated_norm(block.rms_mlp(post_attention))
        routed_output, _ = block.mlp(mlp_input, model.expert_banks[block.expert_bank_index])
        assert block.shared is not None
        shared_output = block.shared(mlp_input, activation=ActivationFunctionEnum.silu)
        block_trace = block.forward_with_moe_trace(
            initial_hidden,
            model.expert_banks[block.expert_bank_index],
            options,
        )

    np.testing.assert_allclose(actual_hidden, expected_hidden, rtol=1e-6, atol=1e-6)
    assert set(traces) == {0, 1}
    assert traces[0].mlp_input.shape == (1, 8, config.hidden_dim)
    assert traces[0].selected_experts.shape == (8, config.num_experts_per_token)
    assert traces[0].combine_weights.shape == traces[0].selected_experts.shape
    assert traces[0].routed_output.shape == traces[0].mlp_input.shape
    np.testing.assert_allclose(traces[0].mlp_input, mlp_input, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(traces[0].routed_output, routed_output, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(block_trace.hidden - post_attention, routed_output + shared_output, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("bad_permutation", [[0, 0, 2, 3], [0, 1, 2], [0, 1, 2, 4]])
def test_router_permutation_rejects_non_bijections(bad_permutation):
    config = _tiny_config(mapping=(0,))
    router = MoEMLP(router=jnp.ones((4, 4)), router_bias=jnp.zeros(4), cfg=config)

    with pytest.raises(ValueError):
        permute_router(router, np.asarray(bad_permutation))
