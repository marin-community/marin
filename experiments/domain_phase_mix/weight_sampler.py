# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight sampling for n-domain, n-phase data mixture experiments.

Adapted from RegMix (arXiv:2407.01492) synthesize_mixture.py to support
arbitrary domain partitions and phase counts.

Supports multiple sampling strategies:
- "dirichlet": Temperature-scaled Dirichlet (original RegMix approach)
- "uniform": Uniform sampling on the simplex
- "vertex_biased": Biased toward simplex vertices (extreme weights)
- "mixed": Combination of uniform and vertex-biased for maximum diversity
"""

import hashlib
from dataclasses import dataclass
from enum import Enum

import numpy as np

from experiments.domain_phase_mix.config import WeightConfig, ExperimentConfig


class SamplingStrategy(str, Enum):
    """Sampling strategy for mixture weights."""

    DIRICHLET = "dirichlet"  # Original RegMix approach
    UNIFORM = "uniform"  # Uniform on simplex
    VERTEX_BIASED = "vertex_biased"  # Biased toward extreme weights
    MIXED = "mixed"  # Combination for maximum diversity


@dataclass
class DirichletSamplingParams:
    """Parameters for Dirichlet-based weight sampling.

    Attributes:
        temp: Temperature for the prior distribution (0.5 smooths skewed distributions).
        min_strength: Minimum Dirichlet concentration parameter.
        max_strength: Maximum Dirichlet concentration parameter.
        min_weight: Minimum weight threshold for statistical significance.
        max_ratio: Maximum ratio vs natural proportion to prevent extreme weights.
        strategy: Sampling strategy to use.
        vertex_prob: Probability of sampling near a vertex (for vertex_biased/mixed).
        min_dominant_weight: Minimum weight for dominant domain in vertex sampling.
        min_phase_change: Minimum L1/2 distance between consecutive phases (0 to disable).
    """

    temp: float = 0.5
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_weight: float = 2e-4
    max_ratio: float = 15.0
    # New parameters for diverse sampling
    strategy: SamplingStrategy = SamplingStrategy.MIXED
    vertex_prob: float = 0.3  # Probability of vertex-biased sample in mixed strategy
    min_dominant_weight: float = 0.7  # Min weight for dominant domain in vertex sampling
    min_phase_change: float = 0.15  # Minimum change between phases (L1/2 distance)


class WeightSampler:
    """Samples mixture weights for n-domain, n-phase training.

    Supports multiple sampling strategies:
    - DIRICHLET: Temperature-scaled Dirichlet (original RegMix approach)
    - UNIFORM: Uniform sampling on the simplex
    - VERTEX_BIASED: Biased toward simplex vertices (extreme weights)
    - MIXED: Combination of uniform and vertex-biased for maximum diversity
    """

    SAMPLE_MULTIPLIER = 100  # Oversample factor for reject sampling

    def __init__(
        self,
        domain_names: list[str],
        phase_names: list[str],
        natural_proportions: dict[str, float] | None = None,
        seed: int = 42,
        params: DirichletSamplingParams | None = None,
    ):
        """Initialize the weight sampler.

        Args:
            domain_names: List of domain names to sample weights for.
            phase_names: List of phase names (for labeling output).
            natural_proportions: Prior proportions for each domain. If None, uses uniform.
            seed: Random seed for reproducibility.
            params: Dirichlet sampling parameters. If None, uses defaults.
        """
        self.domain_names = list(domain_names)
        self.phase_names = list(phase_names)
        self.n_domains = len(self.domain_names)
        self.n_phases = len(self.phase_names)

        # Set natural proportions
        if natural_proportions is None:
            # Uniform proportions
            self.natural_proportions = {d: 1.0 / self.n_domains for d in self.domain_names}
        else:
            # Normalize provided proportions
            total = sum(natural_proportions.get(d, 1.0) for d in self.domain_names)
            self.natural_proportions = {d: natural_proportions.get(d, 1.0) / total for d in self.domain_names}

        self.rng = np.random.default_rng(seed)
        self.params = params or DirichletSamplingParams()

        # Compute upper bounds for reject sampling (only used for DIRICHLET strategy)
        self.upper_bounds = {
            name: min(prop * self.params.max_ratio, 1.0) for name, prop in self.natural_proportions.items()
        }

    @classmethod
    def from_experiment_config(
        cls,
        config: ExperimentConfig,
        seed: int = 42,
        params: DirichletSamplingParams | None = None,
    ) -> "WeightSampler":
        """Create a sampler from an experiment configuration.

        Args:
            config: The experiment configuration.
            seed: Random seed.
            params: Optional sampling parameters.

        Returns:
            WeightSampler configured for the experiment.
        """
        return cls(
            domain_names=config.domain_names,
            phase_names=config.phase_schedule.phase_names,
            natural_proportions=config.get_natural_proportions(),
            seed=seed,
            params=params,
        )

    def _sample_uniform_simplex(self) -> np.ndarray:
        """Sample uniformly on the simplex using the stick-breaking method.

        Returns:
            Array of weights summing to 1.0.
        """
        # Use exponential distribution for uniform simplex sampling
        # This is equivalent to Dirichlet(1, 1, ..., 1)
        x = self.rng.exponential(1.0, self.n_domains)
        return x / x.sum()

    def _sample_vertex_biased(self) -> np.ndarray:
        """Sample with bias toward simplex vertices (extreme weights).

        With some probability, one domain gets 70-100% of the weight,
        simulating "dominant domain" scenarios.

        Returns:
            Array of weights summing to 1.0.
        """
        # Pick a dominant domain
        dominant = self.rng.integers(self.n_domains)

        # Give dominant domain a high weight
        dominant_weight = self.rng.uniform(self.params.min_dominant_weight, 1.0)
        remaining = 1 - dominant_weight

        # Distribute remaining weight among other domains
        weights = np.zeros(self.n_domains)
        weights[dominant] = dominant_weight

        if self.n_domains > 1 and remaining > 0:
            # Use Dirichlet for remaining domains
            other_weights = self.rng.dirichlet(np.ones(self.n_domains - 1))
            other_idx = 0
            for i in range(self.n_domains):
                if i != dominant:
                    weights[i] = remaining * other_weights[other_idx]
                    other_idx += 1

        return weights

    def _sample_mixed(self) -> np.ndarray:
        """Sample using mixed strategy for maximum diversity.

        Combines uniform and vertex-biased sampling.

        Returns:
            Array of weights summing to 1.0.
        """
        if self.rng.random() < self.params.vertex_prob:
            return self._sample_vertex_biased()
        else:
            return self._sample_uniform_simplex()

    def _sample_dirichlet(self) -> np.ndarray:
        """Sample using original RegMix Dirichlet approach.

        Returns:
            Array of weights summing to 1.0.
        """
        # Sample concentration parameter from log-uniform distribution
        log_min = np.log10(self.params.min_strength)
        log_max = np.log10(self.params.max_strength)
        strength = 10 ** self.rng.uniform(log_min, log_max)

        # Create temperature-scaled base distribution (normalize after scaling)
        base_probs = np.array([self.natural_proportions[d] ** self.params.temp for d in self.domain_names])
        base_probs = base_probs / base_probs.sum()

        # Calculate Dirichlet alphas
        alphas = strength * base_probs

        # Rejection sampling loop
        max_attempts = 1000
        for _ in range(max_attempts):
            weights = self.rng.dirichlet(alphas)

            # Check upper bound constraints
            if all(w <= self.upper_bounds[d] for d, w in zip(self.domain_names, weights, strict=True)):
                return weights

        # Fallback to base proportions
        return base_probs

    def _sample_weights_array(self) -> np.ndarray:
        """Sample weights based on the configured strategy.

        Returns:
            Array of weights summing to 1.0.
        """
        strategy = self.params.strategy

        if strategy == SamplingStrategy.DIRICHLET:
            return self._sample_dirichlet()
        elif strategy == SamplingStrategy.UNIFORM:
            return self._sample_uniform_simplex()
        elif strategy == SamplingStrategy.VERTEX_BIASED:
            return self._sample_vertex_biased()
        elif strategy == SamplingStrategy.MIXED:
            return self._sample_mixed()
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    def sample_phase_weights(self) -> dict[str, float]:
        """Sample weights for a single phase.

        Returns:
            Dictionary mapping domain names to weights (sums to 1.0).
        """
        weights = self._sample_weights_array()
        weights_dict = dict(zip(self.domain_names, weights, strict=True))
        return self._normalize(weights_dict)

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights and apply minimum threshold."""
        # Zero out very small weights
        weights = {k: v if v >= self.params.min_weight else 0.0 for k, v in weights.items()}

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}

        # Fallback to uniform
        return {k: 1.0 / self.n_domains for k in weights}

    @staticmethod
    def _phase_change_distance(weights1: dict[str, float], weights2: dict[str, float]) -> float:
        """Compute L1/2 distance (total variation) between two weight dicts.

        Returns a value in [0, 1] where 0 means identical and 1 means maximally different.
        """
        total_diff = sum(abs(weights1.get(k, 0) - weights2.get(k, 0)) for k in weights1)
        return total_diff / 2  # Normalize to [0, 1]

    def sample_config(self, run_id: int) -> WeightConfig:
        """Sample a complete weight configuration for all phases.

        If min_phase_change > 0, ensures consecutive phases have sufficient
        weight differences to produce visible changes in training.

        Args:
            run_id: Identifier for this configuration.

        Returns:
            WeightConfig with sampled weights for all phases.
        """
        min_change = self.params.min_phase_change
        max_attempts = 100

        for _ in range(max_attempts):
            phase_weights: dict[str, dict[str, float]] = {}
            prev_weights: dict[str, float] | None = None
            valid = True

            for phase_name in self.phase_names:
                # Sample weights for this phase
                for _ in range(50):  # Inner retry loop for phase change constraint
                    weights = self.sample_phase_weights()

                    # Check minimum phase change if not first phase
                    if prev_weights is None or min_change <= 0:
                        break
                    if self._phase_change_distance(prev_weights, weights) >= min_change:
                        break
                else:
                    # Could not satisfy constraint for this phase
                    valid = False
                    break

                phase_weights[phase_name] = weights
                prev_weights = weights

            if valid:
                return WeightConfig(run_id=run_id, phase_weights=phase_weights)

        # Fallback: return without phase change constraint
        phase_weights = {}
        for phase_name in self.phase_names:
            phase_weights[phase_name] = self.sample_phase_weights()
        return WeightConfig(run_id=run_id, phase_weights=phase_weights)

    def sample_n_configs(
        self,
        n: int,
        deduplicate: bool = True,
        precision: int = 2,
    ) -> list[WeightConfig]:
        """Sample n unique configurations with optional deduplication.

        Args:
            n: Number of configurations to sample.
            deduplicate: Whether to remove near-duplicate configurations.
            precision: Decimal places for deduplication comparison.

        Returns:
            List of n WeightConfig instances.
        """
        configs: list[WeightConfig] = []
        seen_hashes: set[str] = set()

        attempts = 0
        max_attempts = n * self.SAMPLE_MULTIPLIER

        while len(configs) < n and attempts < max_attempts:
            config = self.sample_config(len(configs))

            if deduplicate:
                config_hash = self._config_hash(config, precision)
                if config_hash in seen_hashes:
                    attempts += 1
                    continue
                seen_hashes.add(config_hash)

            config.run_id = len(configs)
            configs.append(config)
            attempts += 1

        if len(configs) < n:
            raise ValueError(
                f"Could only generate {len(configs)} unique configs after {max_attempts} attempts. "
                f"Try reducing deduplication precision or adjusting sampling parameters."
            )

        return configs

    def _config_hash(self, config: WeightConfig, precision: int = 2) -> str:
        """Create a hash for deduplication."""

        def quantize(weights: dict[str, float]) -> tuple:
            return tuple(sorted((k, round(v, precision)) for k, v in weights.items()))

        key = tuple((phase, quantize(weights)) for phase, weights in sorted(config.phase_weights.items()))
        return hashlib.md5(str(key).encode()).hexdigest()

    def summarize_configs(self, configs: list[WeightConfig]) -> dict:
        """Generate summary statistics for a set of configurations.

        Args:
            configs: List of configurations to summarize.

        Returns:
            Dictionary with summary statistics per phase and domain.
        """
        result: dict = {"n_configs": len(configs)}

        for phase_name in self.phase_names:
            phase_data = {}
            for domain_name in self.domain_names:
                weights = [c.phase_weights[phase_name][domain_name] for c in configs]
                phase_data[domain_name] = {
                    "mean": float(np.mean(weights)),
                    "std": float(np.std(weights)),
                    "min": float(np.min(weights)),
                    "max": float(np.max(weights)),
                }
            result[phase_name] = phase_data

        return result
