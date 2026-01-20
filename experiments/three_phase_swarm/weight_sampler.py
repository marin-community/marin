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
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from experiments.three_phase_swarm.config import WeightConfig, ExperimentConfig


@dataclass
class DirichletSamplingParams:
    """Parameters for Dirichlet-based weight sampling.

    Attributes:
        temp: Temperature for the prior distribution (0.5 smooths skewed distributions).
        min_strength: Minimum Dirichlet concentration parameter.
        max_strength: Maximum Dirichlet concentration parameter.
        min_weight: Minimum weight threshold for statistical significance.
        max_ratio: Maximum ratio vs natural proportion to prevent extreme weights.
    """

    temp: float = 0.5
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_weight: float = 2e-4
    max_ratio: float = 15.0


class WeightSampler:
    """Samples mixture weights for n-domain, n-phase training.

    Uses temperature-scaled Dirichlet sampling with reject sampling for constraints,
    following the RegMix methodology.
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

        # Compute upper bounds for reject sampling
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

    def sample_phase_weights(self) -> dict[str, float]:
        """Sample weights for a single phase using Dirichlet distribution.

        Returns:
            Dictionary mapping domain names to weights (sums to 1.0).
        """
        # Sample concentration parameter from log-uniform distribution
        log_min = np.log10(self.params.min_strength)
        log_max = np.log10(self.params.max_strength)
        strength = 10 ** self.rng.uniform(log_min, log_max)

        # Create temperature-scaled alpha vector
        alphas = np.array([strength * (self.natural_proportions[d] ** self.params.temp) for d in self.domain_names])

        # Rejection sampling loop
        max_attempts = 1000
        for _ in range(max_attempts):
            weights = self.rng.dirichlet(alphas)
            weights_dict = dict(zip(self.domain_names, weights, strict=True))

            if self._check_constraints(weights_dict):
                return self._normalize(weights_dict)

        # Fallback to natural proportions
        return self._normalize(self.natural_proportions.copy())

    def _check_constraints(self, weights: dict[str, float]) -> bool:
        """Check if weights satisfy the max ratio constraint."""
        for name, w in weights.items():
            if w > self.upper_bounds[name]:
                return False
        return True

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights and apply minimum threshold."""
        # Zero out very small weights
        weights = {k: v if v >= self.params.min_weight else 0.0 for k, v in weights.items()}

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}

        # Fallback to uniform
        return {k: 1.0 / self.n_domains for k in weights}

    def sample_config(self, run_id: int) -> WeightConfig:
        """Sample a complete weight configuration for all phases.

        Args:
            run_id: Identifier for this configuration.

        Returns:
            WeightConfig with sampled weights for all phases.
        """
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
        result = {"n_configs": len(configs)}

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
