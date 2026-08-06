# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline functional matching and one-pair expert-bank conversion for Grug MoE."""

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from levanter.grug.attention import AttentionMask
from levanter.grug.grug_moe import MoEExpertMlp
from levanter.utils.activation import ActivationFunctionEnum
from scipy.optimize import linear_sum_assignment

from experiments.grug.moe.model import MoEMLP, Transformer

_DEFAULT_MAHALANOBIS_QUANTILE = 0.995
_DEFAULT_COVARIANCE_EPSILON = 1e-6
_DEFAULT_COST_ETA = 0.5
_DEFAULT_NORMALIZATION_EPSILON = 1e-8


@dataclass(frozen=True)
class ReservoirSample:
    states: np.ndarray
    weights: np.ndarray


@dataclass
class WeightedReservoir:
    """Fixed-size weighted reservoir using exponential priorities."""

    capacity: int
    state_dim: int
    seed: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    _rng: np.random.Generator = field(init=False, repr=False)
    _states: np.ndarray = field(init=False, repr=False)
    _weights: np.ndarray = field(init=False, repr=False)
    _priorities: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        if self.state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {self.state_dim}")
        self._rng = np.random.default_rng(self.seed)
        self._states = np.empty((0, self.state_dim), dtype=self.dtype)
        self._weights = np.empty((0,), dtype=np.float64)
        self._priorities = np.empty((0,), dtype=np.float64)

    def add(self, states: np.ndarray | jax.Array, weights: np.ndarray | jax.Array) -> None:
        states_array = np.asarray(states, dtype=self.dtype)
        weights_array = np.asarray(weights, dtype=np.float64)
        if states_array.ndim != 2 or states_array.shape[1] != self.state_dim:
            raise ValueError(f"states must have shape [N, {self.state_dim}], got {states_array.shape}")
        if weights_array.shape != (states_array.shape[0],):
            raise ValueError(f"weights must have shape [{states_array.shape[0]}], got {weights_array.shape}")
        if np.any(weights_array < 0):
            raise ValueError("reservoir weights must be non-negative")

        keep = weights_array > 0
        if not np.any(keep):
            return
        states_array = states_array[keep]
        weights_array = weights_array[keep]
        uniforms = np.maximum(self._rng.random(weights_array.shape[0]), np.finfo(np.float64).tiny)
        priorities = np.log(uniforms) / weights_array

        all_states = np.concatenate([self._states, states_array], axis=0)
        all_weights = np.concatenate([self._weights, weights_array], axis=0)
        all_priorities = np.concatenate([self._priorities, priorities], axis=0)
        if all_priorities.shape[0] > self.capacity:
            selected = np.argpartition(all_priorities, -self.capacity)[-self.capacity :]
            selected = selected[np.argsort(all_priorities[selected])[::-1]]
            all_states = all_states[selected]
            all_weights = all_weights[selected]
            all_priorities = all_priorities[selected]

        self._states = all_states
        self._weights = all_weights
        self._priorities = all_priorities

    def sample(self) -> ReservoirSample:
        return ReservoirSample(states=self._states.copy(), weights=self._weights.copy())


@dataclass(frozen=True)
class ExpertCalibration:
    train: ReservoirSample
    heldout: ReservoirSample


@dataclass(frozen=True)
class MoeLayerTrace:
    mlp_input: jax.Array
    selected_experts: jax.Array
    combine_weights: jax.Array
    routed_output: jax.Array


@dataclass
class ExpertReservoirCollection:
    """Per-expert train and held-out routed-state reservoirs for one layer."""

    num_experts: int
    state_dim: int
    capacity_per_expert: int
    heldout_fraction: float = 0.2
    seed: int = 0
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    _rng: np.random.Generator = field(init=False, repr=False)
    _train: tuple[WeightedReservoir, ...] = field(init=False, repr=False)
    _heldout: tuple[WeightedReservoir, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {self.num_experts}")
        if self.capacity_per_expert < 2:
            raise ValueError(f"capacity_per_expert must be at least two, got {self.capacity_per_expert}")
        if not 0.0 < self.heldout_fraction < 1.0:
            raise ValueError(f"heldout_fraction must be between zero and one, got {self.heldout_fraction}")
        heldout_capacity = max(1, round(self.capacity_per_expert * self.heldout_fraction))
        train_capacity = self.capacity_per_expert - heldout_capacity
        seed_sequence = np.random.SeedSequence(self.seed)
        child_seeds = seed_sequence.spawn(2 * self.num_experts + 1)
        self._rng = np.random.default_rng(child_seeds[0])
        self._train = tuple(
            WeightedReservoir(
                capacity=train_capacity,
                state_dim=self.state_dim,
                seed=int(child_seeds[1 + expert].generate_state(1)[0]),
                dtype=self.dtype,
            )
            for expert in range(self.num_experts)
        )
        self._heldout = tuple(
            WeightedReservoir(
                capacity=heldout_capacity,
                state_dim=self.state_dim,
                seed=int(child_seeds[1 + self.num_experts + expert].generate_state(1)[0]),
                dtype=self.dtype,
            )
            for expert in range(self.num_experts)
        )

    def add_routes(
        self,
        mlp_inputs: np.ndarray | jax.Array,
        selected_experts: np.ndarray | jax.Array,
        combine_weights: np.ndarray | jax.Array,
    ) -> None:
        inputs = np.asarray(mlp_inputs)
        selected = np.asarray(selected_experts)
        combine = np.asarray(combine_weights)
        if inputs.ndim != 2 or inputs.shape[1] != self.state_dim:
            raise ValueError(f"mlp_inputs must have shape [T, {self.state_dim}], got {inputs.shape}")
        if selected.shape != combine.shape or selected.ndim != 2:
            raise ValueError(
                f"selected_experts and combine_weights must have matching [T, K] shapes, got "
                f"{selected.shape} and {combine.shape}"
            )
        if selected.shape[0] != inputs.shape[0]:
            raise ValueError("routing token count must match mlp_inputs")
        if np.any(selected < 0) or np.any(selected >= self.num_experts):
            raise ValueError("selected_experts contains an out-of-range expert ID")

        route_inputs = np.repeat(inputs, selected.shape[1], axis=0)
        route_experts = selected.reshape(-1)
        route_weights = np.square(combine.astype(np.float64)).reshape(-1)
        heldout = self._rng.random(route_experts.shape[0]) < self.heldout_fraction
        for expert in range(self.num_experts):
            expert_routes = route_experts == expert
            self._train[expert].add(
                route_inputs[expert_routes & ~heldout],
                route_weights[expert_routes & ~heldout],
            )
            self._heldout[expert].add(
                route_inputs[expert_routes & heldout],
                route_weights[expert_routes & heldout],
            )

    def calibration(self, expert_index: int) -> ExpertCalibration:
        if not 0 <= expert_index < self.num_experts:
            raise IndexError(f"expert_index must be in [0, {self.num_experts}), got {expert_index}")
        return ExpertCalibration(
            train=self._train[expert_index].sample(),
            heldout=self._heldout[expert_index].sample(),
        )


def add_moe_trace_to_reservoirs(
    reservoirs: ExpertReservoirCollection,
    trace: MoeLayerTrace,
) -> None:
    """Transfer one traced layer batch to host and update its expert reservoirs."""
    mlp_inputs, selected_experts, combine_weights = jax.device_get(
        (trace.mlp_input, trace.selected_experts, trace.combine_weights)
    )
    reservoirs.add_routes(
        np.asarray(mlp_inputs).reshape(-1, reservoirs.state_dim),
        np.asarray(selected_experts),
        np.asarray(combine_weights),
    )


@dataclass(frozen=True)
class InputManifold:
    mean: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    mahalanobis_radius: float

    @property
    def scaled_basis(self) -> np.ndarray:
        return self.eigenvectors * np.sqrt(self.eigenvalues)[None, :]

    def whiten(self, states: np.ndarray) -> np.ndarray:
        if self.eigenvalues.shape[0] == 0:
            return np.zeros((states.shape[0], 0), dtype=np.float32)
        return (states - self.mean) @ self.eigenvectors / np.sqrt(self.eigenvalues)[None, :]


@dataclass(frozen=True)
class SpectralProbeConfig:
    covariance_rank: int = 32
    num_centers: int = 16
    num_sensitive_directions: int = 8
    directions_per_center: int = 4
    radii: tuple[float, ...] = (0.15, 0.35)
    ordinary_samples: int = 128
    mahalanobis_quantile: float = _DEFAULT_MAHALANOBIS_QUANTILE
    covariance_epsilon: float = _DEFAULT_COVARIANCE_EPSILON


@dataclass(frozen=True)
class SpectralDirections:
    input_directions: np.ndarray
    covariance_directions: np.ndarray
    sensitivity_eigenvalues: np.ndarray


@dataclass(frozen=True)
class ExpertProbeSet:
    ordinary_inputs: np.ndarray
    ordinary_weights: np.ndarray
    centers: np.ndarray
    spectral_pairs: np.ndarray
    input_directions: np.ndarray
    sensitivity_eigenvalues: np.ndarray

    def all_inputs(self) -> np.ndarray:
        return np.concatenate(
            [self.ordinary_inputs, self.centers, self.spectral_pairs.reshape(-1, self.centers.shape[-1])],
            axis=0,
        )


@dataclass(frozen=True)
class ExpertCostMatrix:
    native: np.ndarray
    tangent: np.ndarray
    total: np.ndarray


@dataclass(frozen=True)
class ExpertCostRow:
    native: np.ndarray
    tangent: np.ndarray
    total: np.ndarray


class AssignmentMode(StrEnum):
    IDENTITY = "identity"
    NATIVE = "native"
    SPECTRAL = "spectral"


def forward_with_moe_traces(
    model: Transformer,
    token_ids: jax.Array,
    *,
    target_layers: tuple[int, ...],
    mask: AttentionMask | jax.Array | None = None,
) -> tuple[jax.Array, dict[int, MoeLayerTrace]]:
    """Run the model while retaining routed-MoE boundaries for selected layers."""
    if mask is None:
        mask = AttentionMask.causal()
    if len(set(target_layers)) != len(target_layers):
        raise ValueError(f"target_layers contains duplicates: {target_layers}")
    if any(layer < 0 or layer >= len(model.blocks) for layer in target_layers):
        raise IndexError(f"target_layers must lie in [0, {len(model.blocks)}), got {target_layers}")

    target_layer_set = set(target_layers)
    hidden = model.embed_inputs(token_ids)
    traces: dict[int, MoeLayerTrace] = {}
    for layer_index, block in enumerate(model.blocks):
        options = model.block_call_options(mask, layer_index)
        expert_bank = model.expert_banks[block.expert_bank_index]
        if layer_index in target_layer_set:
            block_trace = block.forward_with_moe_trace(
                hidden,
                expert_bank,
                options,
            )
            hidden = block_trace.hidden
            traces[layer_index] = MoeLayerTrace(
                mlp_input=block_trace.mlp_input,
                selected_experts=block_trace.selected_experts,
                combine_weights=block_trace.combine_weights,
                routed_output=block_trace.routed_output,
            )
        else:
            hidden, _ = block(
                hidden,
                expert_bank,
                options,
            )
    return model.final_gated_norm(model.final_norm(hidden)), traces


def estimate_input_manifold(
    states: np.ndarray,
    weights: np.ndarray,
    *,
    rank: int,
    mahalanobis_quantile: float = _DEFAULT_MAHALANOBIS_QUANTILE,
    epsilon: float = _DEFAULT_COVARIANCE_EPSILON,
) -> InputManifold:
    states = np.asarray(states, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"states must have shape [N, D], got {states.shape}")
    if weights.shape != (states.shape[0],):
        raise ValueError(f"weights must have shape [{states.shape[0]}], got {weights.shape}")
    if states.shape[0] == 0:
        raise ValueError("at least one state is required to estimate an input manifold")
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not 0.0 < mahalanobis_quantile <= 1.0:
        raise ValueError(f"mahalanobis_quantile must be in (0, 1], got {mahalanobis_quantile}")
    if np.any(weights < 0) or float(np.sum(weights)) <= 0:
        raise ValueError("weights must be non-negative with a positive sum")

    normalized_weights = weights / np.sum(weights)
    mean = np.sum(states * normalized_weights[:, None], axis=0)
    centered = states - mean
    covariance = centered.T @ (centered * normalized_weights[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    retained = min(rank, states.shape[1], int(np.sum(eigenvalues > epsilon)))
    if retained == 0:
        return InputManifold(
            mean=mean.astype(np.float32),
            eigenvectors=np.empty((states.shape[1], 0), dtype=np.float32),
            eigenvalues=np.empty((0,), dtype=np.float32),
            mahalanobis_radius=0.0,
        )
    eigenvalues = np.maximum(eigenvalues[:retained], epsilon)
    eigenvectors = eigenvectors[:, :retained]
    whitened = centered @ eigenvectors / np.sqrt(eigenvalues)[None, :]
    radii = np.linalg.norm(whitened, axis=-1)
    return InputManifold(
        mean=mean.astype(np.float32),
        eigenvectors=eigenvectors.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float32),
        mahalanobis_radius=float(np.quantile(radii, mahalanobis_quantile)),
    )


def eval_all_experts(
    bank: MoEExpertMlp,
    x: np.ndarray | jax.Array,
    *,
    probe_chunk_size: int | None = None,
    expert_chunk_size: int | None = None,
) -> jax.Array:
    """Evaluate every expert in ``bank`` on every input without routing."""
    inputs = jnp.asarray(x)
    if inputs.ndim != 2 or inputs.shape[1] != bank.w_gate.shape[1]:
        raise ValueError(f"x must have shape [P, {bank.w_gate.shape[1]}], got {inputs.shape}")
    num_experts = int(bank.w_gate.shape[0])
    expert_chunk = num_experts if expert_chunk_size is None else expert_chunk_size
    probe_chunk = int(inputs.shape[0]) if probe_chunk_size is None else probe_chunk_size
    if expert_chunk <= 0:
        raise ValueError(f"expert_chunk_size must be positive, got {expert_chunk}")
    if probe_chunk <= 0:
        raise ValueError(f"probe_chunk_size must be positive, got {probe_chunk}")
    probe_outputs = []
    for probe_start in range(0, int(inputs.shape[0]), probe_chunk):
        probe_stop = min(probe_start + probe_chunk, int(inputs.shape[0]))
        probe_inputs = inputs[probe_start:probe_stop]
        expert_outputs = []
        for expert_start in range(0, num_experts, expert_chunk):
            expert_stop = min(expert_start + expert_chunk, num_experts)
            chunk_experts = expert_stop - expert_start
            expanded_inputs = jnp.broadcast_to(
                probe_inputs[:, None, :],
                (probe_inputs.shape[0], chunk_experts, probe_inputs.shape[1]),
            ).reshape(-1, probe_inputs.shape[1])
            selected_experts = jnp.broadcast_to(
                jnp.arange(expert_start, expert_stop, dtype=jnp.int32)[None, :],
                (probe_inputs.shape[0], chunk_experts),
            ).reshape(-1, 1)
            evaluation_bank = dataclasses.replace(
                bank,
                capacity_factor=max(bank.capacity_factor, num_experts / chunk_experts),
            )
            chunk_output = evaluation_bank(
                expanded_inputs,
                selected_experts,
                jnp.ones((expanded_inputs.shape[0], 1), dtype=expanded_inputs.dtype),
            )
            if isinstance(chunk_output, tuple):
                chunk_output = chunk_output[0]
            expert_outputs.append(chunk_output.reshape(probe_inputs.shape[0], chunk_experts, -1))
        probe_outputs.append(jnp.concatenate(expert_outputs, axis=1))
    return jnp.concatenate(probe_outputs, axis=0)


def eval_expert(bank: MoEExpertMlp, expert_index: int, x: np.ndarray | jax.Array) -> jax.Array:
    if not 0 <= expert_index < bank.w_gate.shape[0]:
        raise IndexError(f"expert_index must be in [0, {bank.w_gate.shape[0]}), got {expert_index}")
    inputs = jnp.asarray(x)
    evaluation_bank = dataclasses.replace(
        bank,
        capacity_factor=max(bank.capacity_factor, float(bank.w_gate.shape[0])),
    )
    output = evaluation_bank(
        inputs,
        jnp.full((inputs.shape[0], 1), expert_index, dtype=jnp.int32),
        jnp.ones((inputs.shape[0], 1), dtype=inputs.dtype),
    )
    if isinstance(output, tuple):
        return output[0]
    return output


def _farthest_centers(
    states: np.ndarray,
    weights: np.ndarray,
    manifold: InputManifold,
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    whitened = manifold.whiten(states)
    eligible = np.linalg.norm(whitened, axis=-1) <= manifold.mahalanobis_radius
    if not np.any(eligible):
        raise ValueError("no routed state lies within the configured Mahalanobis radius")
    states = states[eligible]
    whitened = whitened[eligible]
    weights = weights[eligible]
    count = min(count, states.shape[0])
    rng = np.random.default_rng(seed)
    probabilities = weights / np.sum(weights)
    selected = [int(rng.choice(states.shape[0], p=probabilities))]
    min_distance = np.sum(np.square(whitened - whitened[selected[0]]), axis=-1)
    min_distance[selected[0]] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(min_distance))
        selected.append(next_index)
        distance = np.sum(np.square(whitened - whitened[next_index]), axis=-1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[np.asarray(selected)] = -np.inf
    return states[np.asarray(selected)]


def spectral_directions(
    bank: MoEExpertMlp,
    expert_index: int,
    centers: np.ndarray,
    manifold: InputManifold,
    *,
    count: int,
) -> SpectralDirections:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    if manifold.eigenvalues.shape[0] == 0:
        return SpectralDirections(
            input_directions=np.empty((manifold.mean.shape[0], 0), dtype=np.float32),
            covariance_directions=np.empty((0, 0), dtype=np.float32),
            sensitivity_eigenvalues=np.empty((0,), dtype=np.float32),
        )
    scaled_basis = jnp.asarray(manifold.scaled_basis)
    centers_array = jnp.asarray(centers)
    mesh = get_abstract_mesh()
    if mesh.empty:
        w_gate = bank.w_gate[expert_index]
        w_up = bank.w_up[expert_index]
        w_down = bank.w_down[expert_index]
    else:
        w_gate = bank.w_gate.at[expert_index].get(out_sharding=P("data", "model"))
        w_up = bank.w_up.at[expert_index].get(out_sharding=P("data", "model"))
        w_down = bank.w_down.at[expert_index].get(out_sharding=P("model", "data"))
    activation_fn = (
        bank.activation.to_jax_fn() if isinstance(bank.activation, ActivationFunctionEnum) else bank.activation
    )
    model_sharding = None if mesh.empty else P("model")
    data_sharding = None if mesh.empty else P("data")
    replicated_matrix_sharding = None if mesh.empty else P(None, None)

    def center_gram(center: jax.Array) -> jax.Array:
        def expert_fn(value: jax.Array) -> jax.Array:
            gate = jnp.einsum("d,di->i", value, w_gate, out_sharding=model_sharding)
            up = jnp.einsum("d,di->i", value, w_up, out_sharding=model_sharding)
            return jnp.einsum(
                "i,id->d",
                activation_fn(gate) * up,
                w_down,
                out_sharding=data_sharding,
            )

        tangents = jax.vmap(lambda direction: jax.jvp(expert_fn, (center,), (direction,))[1])(scaled_basis.T)
        return jnp.einsum("kd,ld->kl", tangents, tangents, out_sharding=replicated_matrix_sharding)

    sensitivity_gram = jnp.mean(jax.vmap(center_gram)(centers_array), axis=0)
    eigenvalues, covariance_directions = jnp.linalg.eigh(sensitivity_gram)
    order = jnp.argsort(eigenvalues)[::-1]
    retained = min(count, manifold.eigenvalues.shape[0])
    covariance_directions = covariance_directions[:, order[:retained]]
    eigenvalues = eigenvalues[order[:retained]]
    input_directions = scaled_basis @ covariance_directions
    return SpectralDirections(
        input_directions=np.asarray(jax.device_get(input_directions), dtype=np.float32),
        covariance_directions=np.asarray(jax.device_get(covariance_directions), dtype=np.float32),
        sensitivity_eigenvalues=np.asarray(jax.device_get(eigenvalues), dtype=np.float32),
    )


def _bounded_radius(
    center_coordinates: np.ndarray,
    direction_coordinates: np.ndarray,
    requested_radius: float,
    mahalanobis_radius: float,
) -> float:
    a = float(np.dot(direction_coordinates, direction_coordinates))
    b = 2.0 * float(np.dot(center_coordinates, direction_coordinates))
    c = float(np.dot(center_coordinates, center_coordinates) - mahalanobis_radius**2)
    discriminant = max(0.0, b * b - 4.0 * a * c)
    boundary = (-b + np.sqrt(discriminant)) / (2.0 * a)
    return max(0.0, min(requested_radius, boundary))


def build_spectral_probe_set(
    bank: MoEExpertMlp,
    expert_index: int,
    manifold_sample: ReservoirSample,
    heldout_sample: ReservoirSample,
    *,
    config: SpectralProbeConfig = SpectralProbeConfig(),
    seed: int = 0,
) -> ExpertProbeSet:
    manifold = estimate_input_manifold(
        manifold_sample.states,
        manifold_sample.weights,
        rank=config.covariance_rank,
        mahalanobis_quantile=config.mahalanobis_quantile,
        epsilon=config.covariance_epsilon,
    )
    centers = _farthest_centers(
        manifold_sample.states,
        manifold_sample.weights,
        manifold,
        count=config.num_centers,
        seed=seed,
    )
    directions = spectral_directions(
        bank,
        expert_index,
        centers,
        manifold,
        count=config.num_sensitive_directions,
    )
    if directions.input_directions.shape[1] > 0 and config.directions_per_center > directions.input_directions.shape[1]:
        raise ValueError(
            "directions_per_center cannot exceed the retained sensitivity directions; "
            f"got {config.directions_per_center} and {directions.input_directions.shape[1]}"
        )

    center_coordinates = manifold.whiten(centers)
    pairs = []
    if directions.input_directions.shape[1] > 0:
        for center_index, center in enumerate(centers):
            start = (center_index * config.directions_per_center) % directions.input_directions.shape[1]
            direction_indices = [
                (start + offset) % directions.input_directions.shape[1] for offset in range(config.directions_per_center)
            ]
            for direction_index in direction_indices:
                input_direction = directions.input_directions[:, direction_index]
                covariance_direction = directions.covariance_directions[:, direction_index]
                for radius in config.radii:
                    positive_radius = _bounded_radius(
                        center_coordinates[center_index],
                        covariance_direction,
                        radius,
                        manifold.mahalanobis_radius,
                    )
                    negative_radius = _bounded_radius(
                        center_coordinates[center_index],
                        -covariance_direction,
                        radius,
                        manifold.mahalanobis_radius,
                    )
                    bounded_radius = min(positive_radius, negative_radius)
                    pairs.append(
                        np.stack([center - bounded_radius * input_direction, center + bounded_radius * input_direction])
                    )

    rng = np.random.default_rng(seed + 1)
    ordinary_count = min(config.ordinary_samples, heldout_sample.states.shape[0])
    if ordinary_count == 0:
        raise ValueError("heldout_sample must contain at least one routed state")
    probabilities = heldout_sample.weights / np.sum(heldout_sample.weights)
    ordinary_indices = rng.choice(
        heldout_sample.states.shape[0],
        size=ordinary_count,
        replace=False,
        p=probabilities,
    )
    return ExpertProbeSet(
        ordinary_inputs=np.asarray(heldout_sample.states[ordinary_indices], dtype=np.float32),
        ordinary_weights=np.asarray(heldout_sample.weights[ordinary_indices], dtype=np.float32),
        centers=np.asarray(centers, dtype=np.float32),
        spectral_pairs=np.asarray(pairs, dtype=np.float32).reshape(-1, 2, manifold.mean.shape[0]),
        input_directions=directions.input_directions,
        sensitivity_eigenvalues=directions.sensitivity_eigenvalues,
    )


def expert_costs(
    source_bank: MoEExpertMlp,
    source_expert: int,
    candidate_bank: MoEExpertMlp,
    probes: ExpertProbeSet,
    *,
    eta: float = _DEFAULT_COST_ETA,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
    expert_chunk_size: int | None = None,
) -> ExpertCostRow:
    native_inputs = jnp.asarray(probes.ordinary_inputs)
    native_weights = jnp.asarray(probes.ordinary_weights, dtype=jnp.float32)
    source_native = eval_expert(source_bank, source_expert, native_inputs).astype(jnp.float32)
    candidate_native = eval_all_experts(
        candidate_bank,
        native_inputs,
        expert_chunk_size=expert_chunk_size,
    ).astype(jnp.float32)
    native_numerator = jnp.sum(
        native_weights[:, None, None] * jnp.square(candidate_native - source_native[:, None, :]),
        axis=(0, 2),
    )
    native_denominator = jnp.sum(native_weights[:, None] * jnp.square(source_native)) + epsilon
    native_cost = native_numerator / native_denominator

    if probes.spectral_pairs.shape[0] == 0:
        tangent_cost = jnp.zeros_like(native_cost)
    else:
        spectral_inputs = jnp.asarray(probes.spectral_pairs).reshape(-1, probes.spectral_pairs.shape[-1])
        source_spectral = eval_expert(source_bank, source_expert, spectral_inputs).astype(jnp.float32)
        source_spectral = source_spectral.reshape(probes.spectral_pairs.shape[0], 2, -1)
        source_delta = source_spectral[:, 1] - source_spectral[:, 0]
        candidate_spectral = eval_all_experts(
            candidate_bank,
            spectral_inputs,
            expert_chunk_size=expert_chunk_size,
        ).astype(jnp.float32)
        candidate_spectral = candidate_spectral.reshape(
            probes.spectral_pairs.shape[0],
            2,
            candidate_bank.w_gate.shape[0],
            -1,
        )
        candidate_delta = candidate_spectral[:, 1] - candidate_spectral[:, 0]
        tangent_numerator = jnp.sum(jnp.square(candidate_delta - source_delta[:, None, :]), axis=(0, 2))
        tangent_denominator = jnp.sum(jnp.square(source_delta)) + epsilon
        tangent_cost = tangent_numerator / tangent_denominator

    total_cost = native_cost + eta * tangent_cost
    return ExpertCostRow(
        native=np.asarray(jax.device_get(native_cost), dtype=np.float64),
        tangent=np.asarray(jax.device_get(tangent_cost), dtype=np.float64),
        total=np.asarray(jax.device_get(total_cost), dtype=np.float64),
    )


def functional_cost_matrix(
    source_bank: MoEExpertMlp,
    candidate_bank: MoEExpertMlp,
    probes_by_source_expert: tuple[ExpertProbeSet, ...],
    *,
    eta: float = _DEFAULT_COST_ETA,
    epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
    expert_chunk_size: int | None = None,
) -> ExpertCostMatrix:
    num_source_experts = int(source_bank.w_gate.shape[0])
    if len(probes_by_source_expert) != num_source_experts:
        raise ValueError(
            f"expected one probe set per source expert ({num_source_experts}), got {len(probes_by_source_expert)}"
        )
    if candidate_bank.w_gate.shape[0] != num_source_experts:
        raise ValueError("the initial conversion requires equal-size source and candidate banks")
    rows = [
        expert_costs(
            source_bank,
            source_expert,
            candidate_bank,
            probes_by_source_expert[source_expert],
            eta=eta,
            epsilon=epsilon,
            expert_chunk_size=expert_chunk_size,
        )
        for source_expert in range(num_source_experts)
    ]
    return ExpertCostMatrix(
        native=np.stack([row.native for row in rows]),
        tangent=np.stack([row.tangent for row in rows]),
        total=np.stack([row.total for row in rows]),
    )


def solve_expert_assignment(costs: ExpertCostMatrix, mode: AssignmentMode) -> np.ndarray:
    """Return a source-expert to shared-slot bijection."""
    num_experts = costs.native.shape[0]
    expected_shape = (num_experts, num_experts)
    if (
        costs.native.shape != expected_shape
        or costs.tangent.shape != expected_shape
        or costs.total.shape != expected_shape
    ):
        raise ValueError("all expert cost matrices must be square and have identical shapes")
    if mode is AssignmentMode.IDENTITY:
        return np.arange(num_experts, dtype=np.int32)
    matrix = costs.native if mode is AssignmentMode.NATIVE else costs.total
    source_indices, shared_indices = linear_sum_assignment(matrix)
    assignment = np.empty((num_experts,), dtype=np.int32)
    assignment[source_indices] = shared_indices
    return assignment


def validate_bijection(source_to_shared: np.ndarray | jax.Array, num_experts: int) -> np.ndarray:
    permutation = np.asarray(source_to_shared, dtype=np.int32)
    if permutation.shape != (num_experts,):
        raise ValueError(f"source_to_shared must have shape [{num_experts}], got {permutation.shape}")
    if not np.array_equal(np.sort(permutation), np.arange(num_experts)):
        raise ValueError("source_to_shared must be a bijection over expert IDs")
    return permutation


def _host_permute_with_sharding(value: jax.Array, indices: np.ndarray, *, axis: int) -> jax.Array:
    permuted = np.take(np.asarray(jax.device_get(value)), indices, axis=axis)
    return jax.device_put(permuted, value.sharding)


def permute_router(router: MoEMLP, source_to_shared: np.ndarray | jax.Array) -> MoEMLP:
    permutation = validate_bijection(source_to_shared, int(router.router.shape[1]))
    shared_to_source = np.argsort(permutation)
    return eqx.tree_at(
        lambda current: (current.router, current.router_bias),
        router,
        (
            _host_permute_with_sharding(router.router, shared_to_source, axis=1),
            _host_permute_with_sharding(router.router_bias, shared_to_source, axis=0),
        ),
    )


def permute_pending_qb_beta(
    pending_qb_beta: np.ndarray | jax.Array,
    source_to_shared: np.ndarray | jax.Array,
) -> jax.Array:
    permutation = validate_bijection(source_to_shared, int(pending_qb_beta.shape[0]))
    value = jnp.asarray(pending_qb_beta)
    return _host_permute_with_sharding(value, np.argsort(permutation), axis=0)


def convert_one_expert_pair(
    model: Transformer,
    *,
    representative_layer: int,
    source_layer: int,
    source_to_shared: np.ndarray | jax.Array,
    shared_bank: MoEExpertMlp | None = None,
) -> Transformer:
    """Merge one source layer into a representative bank and preserve source routing by ID renaming."""
    if representative_layer == source_layer:
        raise ValueError("representative_layer and source_layer must be different")
    num_layers = len(model.blocks)
    if not 0 <= representative_layer < num_layers or not 0 <= source_layer < num_layers:
        raise IndexError(f"layer indices must be in [0, {num_layers})")
    permutation = validate_bijection(source_to_shared, model.config.num_experts)
    representative_bank = model.blocks[representative_layer].expert_bank_index
    source_bank = model.blocks[source_layer].expert_bank_index
    if representative_bank == source_bank:
        raise ValueError("the selected layers already share an expert bank")
    bank_mapping = model.config.resolved_expert_bank_for_layer
    if bank_mapping.count(representative_bank) != 1 or bank_mapping.count(source_bank) != 1:
        raise ValueError("the initial one-pair conversion requires both source banks to be used by one layer")
    representative_experts = model.expert_banks[representative_bank]
    if shared_bank is not None:
        if jax.tree.structure(shared_bank) != jax.tree.structure(representative_experts):
            raise ValueError("prefitted shared bank must have the representative bank's pytree structure")
        for candidate, reference in zip(
            jax.tree.leaves(shared_bank),
            jax.tree.leaves(representative_experts),
            strict=True,
        ):
            if candidate.shape != reference.shape:
                raise ValueError(f"prefitted shared-bank leaf shape {candidate.shape} does not match {reference.shape}")

    retained_old_banks = [bank for bank in range(len(model.expert_banks)) if bank != source_bank]
    old_to_new = {old_bank: new_bank for new_bank, old_bank in enumerate(retained_old_banks)}
    representative_new_bank = old_to_new[representative_bank]
    converted_mapping = tuple(
        representative_new_bank if layer_index == source_layer else old_to_new[block.expert_bank_index]
        for layer_index, block in enumerate(model.blocks)
    )
    converted_config = dataclasses.replace(model.config, expert_bank_for_layer=converted_mapping)
    converted_blocks = []
    for layer_index, block in enumerate(model.blocks):
        mlp = permute_router(block.mlp, permutation) if layer_index == source_layer else block.mlp
        mlp = dataclasses.replace(mlp, cfg=converted_config)
        attention = dataclasses.replace(block.attn, cfg=converted_config)
        converted_blocks.append(
            dataclasses.replace(
                block,
                attn=attention,
                mlp=mlp,
                expert_bank_index=converted_mapping[layer_index],
            )
        )

    converted_model = dataclasses.replace(
        model,
        blocks=tuple(converted_blocks),
        expert_banks=tuple(
            shared_bank if bank == representative_bank and shared_bank is not None else model.expert_banks[bank]
            for bank in retained_old_banks
        ),
        config=converted_config,
    )
    return converted_model


__all__ = [
    "AssignmentMode",
    "ExpertCalibration",
    "ExpertCostMatrix",
    "ExpertCostRow",
    "ExpertProbeSet",
    "ExpertReservoirCollection",
    "InputManifold",
    "MoeLayerTrace",
    "ReservoirSample",
    "SpectralProbeConfig",
    "WeightedReservoir",
    "add_moe_trace_to_reservoirs",
    "build_spectral_probe_set",
    "convert_one_expert_pair",
    "estimate_input_manifold",
    "eval_all_experts",
    "eval_expert",
    "expert_costs",
    "forward_with_moe_traces",
    "functional_cost_matrix",
    "permute_pending_qb_beta",
    "permute_router",
    "solve_expert_assignment",
    "spectral_directions",
    "validate_bijection",
]
