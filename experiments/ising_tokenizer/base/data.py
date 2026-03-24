# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CRITICAL_TEMPERATURE_2D = 2.26918531421


@dataclass(frozen=True)
class BklIsingConfig:
    """Synthetic 2D Ising trajectory settings."""

    lattice_size: int = 8
    num_events: int = 48
    burn_in_events: int = 96
    coupling: float = 1.0
    pad_to_multiple: int | None = 128

    @property
    def num_sites(self) -> int:
        return self.lattice_size * self.lattice_size

    @property
    def initial_state_token_count(self) -> int:
        return 2 * self.num_sites

    @property
    def unpadded_seq_len(self) -> int:
        return self.initial_state_token_count + 2 * self.num_events + 1

    @property
    def seq_len(self) -> int:
        if self.pad_to_multiple is None:
            return self.unpadded_seq_len
        remainder = self.unpadded_seq_len % self.pad_to_multiple
        if remainder == 0:
            return self.unpadded_seq_len
        return self.unpadded_seq_len + (self.pad_to_multiple - remainder)


@dataclass(frozen=True)
class TrajectoryTokenizerConfig:
    """Discrete tokenization for BKL trajectory streams."""

    num_dt_bins: int = 64
    log_wait_time_min: float = -5.0
    log_wait_time_max: float = 1.0

    def dt_token_offset(self, lattice_size: int) -> int:
        return lattice_size * lattice_size + 2

    def eos_token_id(self, lattice_size: int) -> int:
        return self.dt_token_offset(lattice_size) + self.num_dt_bins

    def pad_token_id(self, lattice_size: int) -> int:
        return self.eos_token_id(lattice_size) + 1

    def vocab_size(self, lattice_size: int) -> int:
        return self.pad_token_id(lattice_size) + 1

    def dt_token(self, wait_time: float, lattice_size: int) -> int:
        clipped = max(wait_time, 10**self.log_wait_time_min)
        log_wait_time = float(np.log10(clipped))
        normalized = (log_wait_time - self.log_wait_time_min) / (self.log_wait_time_max - self.log_wait_time_min)
        normalized = min(max(normalized, 0.0), np.nextafter(1.0, 0.0))
        bin_index = int(normalized * self.num_dt_bins)
        return self.dt_token_offset(lattice_size) + bin_index

    def decode_dt_token(self, token_id: int, lattice_size: int) -> float:
        dt_offset = self.dt_token_offset(lattice_size)
        bin_index = int(token_id) - dt_offset
        if bin_index < 0 or bin_index >= self.num_dt_bins:
            raise ValueError(f"Token {token_id} is not a dt token for lattice_size={lattice_size}")
        log_span = self.log_wait_time_max - self.log_wait_time_min
        log_center = self.log_wait_time_min + ((bin_index + 0.5) / self.num_dt_bins) * log_span
        return float(10**log_center)


@dataclass(frozen=True)
class SyntheticSplitConfig:
    """One synthetic dataset split."""

    name: str
    temperatures: tuple[float, ...]
    num_examples: int
    seed: int


@dataclass(frozen=True)
class IsingTrajectory:
    """One fixed-length Ising trajectory sample."""

    temperature: float
    initial_spins: np.ndarray
    event_positions: np.ndarray
    waiting_times: np.ndarray


@dataclass(frozen=True)
class TemperatureConditionedDataset:
    """Tokenized fixed-length sequences paired with scalar temperatures."""

    name: str
    tokens: np.ndarray
    temperatures: np.ndarray
    normalized_temperatures: np.ndarray
    initial_abs_magnetization: np.ndarray
    mean_wait_time: np.ndarray
    lattice_size: int
    num_events: int
    seq_len: int
    vocab_size: int
    initial_state_token_count: int
    valid_token_count: int

    def summary(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "num_examples": int(self.tokens.shape[0]),
            "seq_len": self.seq_len,
            "valid_token_count": self.valid_token_count,
            "vocab_size": self.vocab_size,
            "mean_abs_magnetization": float(self.initial_abs_magnetization.mean()),
            "mean_wait_time": float(self.mean_wait_time.mean()),
            "temperature_min": float(self.temperatures.min()),
            "temperature_max": float(self.temperatures.max()),
        }


def _neighbor_sum(spins: np.ndarray) -> np.ndarray:
    north = np.roll(spins, 1, axis=0)
    south = np.roll(spins, -1, axis=0)
    west = np.roll(spins, 1, axis=1)
    east = np.roll(spins, -1, axis=1)
    return north + south + west + east


def _glauber_flip_rates(spins: np.ndarray, temperature: float, coupling: float) -> np.ndarray:
    delta_energy = 2.0 * coupling * spins * _neighbor_sum(spins)
    beta_delta = np.clip(delta_energy / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(beta_delta))


def _sample_bkl_event(
    spins: np.ndarray,
    *,
    temperature: float,
    coupling: float,
    rng: np.random.Generator,
) -> tuple[int, float]:
    rates = _glauber_flip_rates(spins, temperature, coupling)
    flat_rates = rates.reshape(-1)
    total_rate = float(flat_rates.sum(dtype=np.float64))
    if total_rate <= 0.0:
        raise ValueError("Total flip rate must be positive.")

    wait_time = float(rng.exponential(scale=1.0 / total_rate))
    flat_index = int(rng.choice(flat_rates.size, p=flat_rates / total_rate))
    row, col = divmod(flat_index, spins.shape[1])
    spins[row, col] *= -1
    return flat_index, wait_time


def simulate_bkl_trajectory(
    *,
    config: BklIsingConfig,
    temperature: float,
    rng: np.random.Generator,
) -> IsingTrajectory:
    """Generate one fixed-length continuous-time Ising trajectory."""

    spins = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=(config.lattice_size, config.lattice_size))

    for _ in range(config.burn_in_events):
        _sample_bkl_event(spins, temperature=temperature, coupling=config.coupling, rng=rng)

    initial_spins = spins.copy()
    event_positions = np.empty(config.num_events, dtype=np.int32)
    waiting_times = np.empty(config.num_events, dtype=np.float32)
    for event_index in range(config.num_events):
        position, wait_time = _sample_bkl_event(spins, temperature=temperature, coupling=config.coupling, rng=rng)
        event_positions[event_index] = position
        waiting_times[event_index] = wait_time

    return IsingTrajectory(
        temperature=temperature,
        initial_spins=initial_spins,
        event_positions=event_positions,
        waiting_times=waiting_times,
    )


def encode_trajectory_tokens(
    trajectory: IsingTrajectory,
    *,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
) -> np.ndarray:
    """Encode one trajectory as `[pos][spin]...[pos][dt_bin]...` tokens."""

    tokens = np.full(
        dynamics_config.seq_len,
        tokenizer_config.pad_token_id(dynamics_config.lattice_size),
        dtype=np.int32,
    )
    spin_up_token = dynamics_config.num_sites
    spin_down_token = dynamics_config.num_sites + 1

    cursor = 0
    for position, spin in enumerate(trajectory.initial_spins.reshape(-1)):
        tokens[cursor] = position
        tokens[cursor + 1] = spin_up_token if spin > 0 else spin_down_token
        cursor += 2

    for position, wait_time in zip(trajectory.event_positions, trajectory.waiting_times, strict=True):
        tokens[cursor] = int(position)
        tokens[cursor + 1] = tokenizer_config.dt_token(float(wait_time), dynamics_config.lattice_size)
        cursor += 2

    tokens[cursor] = tokenizer_config.eos_token_id(dynamics_config.lattice_size)
    cursor += 1
    if cursor != dynamics_config.unpadded_seq_len:
        raise ValueError(f"Encoded {cursor} tokens but expected {dynamics_config.unpadded_seq_len}")

    return tokens


def decode_initial_spins(
    tokens: np.ndarray,
    *,
    dynamics_config: BklIsingConfig,
) -> np.ndarray:
    """Recover the lattice spins from the initial `[pos][spin]...` prompt."""

    flat_spins = np.empty(dynamics_config.num_sites, dtype=np.int8)
    spin_up_token = dynamics_config.num_sites
    spin_down_token = dynamics_config.num_sites + 1
    for position in range(dynamics_config.num_sites):
        token_position = int(tokens[2 * position])
        spin_token = int(tokens[2 * position + 1])
        if token_position != position:
            raise ValueError(f"Expected position token {position}, got {token_position}")
        if spin_token == spin_up_token:
            flat_spins[position] = 1
            continue
        if spin_token == spin_down_token:
            flat_spins[position] = -1
            continue
        raise ValueError(f"Expected spin token {spin_up_token} or {spin_down_token}, got {spin_token}")
    return flat_spins.reshape(dynamics_config.lattice_size, dynamics_config.lattice_size)


def decode_event_positions(
    tokens: np.ndarray,
    *,
    dynamics_config: BklIsingConfig,
) -> np.ndarray:
    """Recover the event positions from the dynamics suffix."""

    start = dynamics_config.initial_state_token_count
    stop = start + 2 * dynamics_config.num_events
    positions = np.asarray(tokens[start:stop:2], dtype=np.int32)
    if positions.shape[0] != dynamics_config.num_events:
        raise ValueError(f"Decoded {positions.shape[0]} events but expected {dynamics_config.num_events}")
    return positions


def decode_wait_times(
    tokens: np.ndarray,
    *,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
) -> np.ndarray:
    """Recover representative wait times from dt-bin tokens."""

    start = dynamics_config.initial_state_token_count + 1
    stop = dynamics_config.initial_state_token_count + 2 * dynamics_config.num_events
    dt_tokens = np.asarray(tokens[start:stop:2], dtype=np.int32)
    if dt_tokens.shape[0] != dynamics_config.num_events:
        raise ValueError(f"Decoded {dt_tokens.shape[0]} wait times but expected {dynamics_config.num_events}")
    return np.asarray(
        [tokenizer_config.decode_dt_token(int(token), dynamics_config.lattice_size) for token in dt_tokens],
        dtype=np.float32,
    )


def temperature_normalization_stats(temperatures: tuple[float, ...]) -> tuple[float, float]:
    """Return mean and standard deviation for scalar temperature conditioning."""

    temps = np.asarray(temperatures, dtype=np.float32)
    if temps.size == 0:
        raise ValueError("At least one temperature is required.")
    mean = float(temps.mean())
    std = float(temps.std())
    if std == 0.0:
        raise ValueError("Temperature standard deviation must be positive.")
    return mean, std


def build_synthetic_split(
    *,
    split_config: SyntheticSplitConfig,
    dynamics_config: BklIsingConfig,
    tokenizer_config: TrajectoryTokenizerConfig,
    temperature_mean: float,
    temperature_std: float,
) -> TemperatureConditionedDataset:
    """Build one deterministic tokenized split."""

    if split_config.num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if not split_config.temperatures:
        raise ValueError("temperatures must be non-empty")

    master_rng = np.random.default_rng(split_config.seed)
    scheduled_temperatures = np.asarray(
        [
            split_config.temperatures[index % len(split_config.temperatures)]
            for index in range(split_config.num_examples)
        ],
        dtype=np.float32,
    )
    master_rng.shuffle(scheduled_temperatures)

    tokens = np.empty((split_config.num_examples, dynamics_config.seq_len), dtype=np.int32)
    initial_abs_magnetization = np.empty(split_config.num_examples, dtype=np.float32)
    mean_wait_time = np.empty(split_config.num_examples, dtype=np.float32)

    for example_index, temperature in enumerate(scheduled_temperatures):
        trajectory_rng = np.random.default_rng(int(master_rng.integers(0, 2**31 - 1)))
        trajectory = simulate_bkl_trajectory(
            config=dynamics_config,
            temperature=float(temperature),
            rng=trajectory_rng,
        )
        tokens[example_index] = encode_trajectory_tokens(
            trajectory,
            dynamics_config=dynamics_config,
            tokenizer_config=tokenizer_config,
        )
        initial_abs_magnetization[example_index] = abs(float(trajectory.initial_spins.mean()))
        mean_wait_time[example_index] = float(trajectory.waiting_times.mean())

    normalized_temperatures = (scheduled_temperatures - temperature_mean) / temperature_std
    return TemperatureConditionedDataset(
        name=split_config.name,
        tokens=tokens,
        temperatures=scheduled_temperatures,
        normalized_temperatures=normalized_temperatures.astype(np.float32),
        initial_abs_magnetization=initial_abs_magnetization,
        mean_wait_time=mean_wait_time,
        lattice_size=dynamics_config.lattice_size,
        num_events=dynamics_config.num_events,
        seq_len=dynamics_config.seq_len,
        vocab_size=tokenizer_config.vocab_size(dynamics_config.lattice_size),
        initial_state_token_count=dynamics_config.initial_state_token_count,
        valid_token_count=dynamics_config.unpadded_seq_len,
    )


__all__ = [
    "CRITICAL_TEMPERATURE_2D",
    "BklIsingConfig",
    "IsingTrajectory",
    "SyntheticSplitConfig",
    "TemperatureConditionedDataset",
    "TrajectoryTokenizerConfig",
    "build_synthetic_split",
    "decode_event_positions",
    "decode_initial_spins",
    "decode_wait_times",
    "encode_trajectory_tokens",
    "simulate_bkl_trajectory",
    "temperature_normalization_stats",
]
