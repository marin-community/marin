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

"""Weight sampling for three-phase data mixture experiments.

Adapted from RegMix (arXiv:2407.01492) synthesize_mixture.py to support
three-partition (pretrain, midtrain, SFT) sampling across three training phases.
"""

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class ThreePartitionWeightConfig:
    """Configuration for a single three-phase training run.

    Each phase has its own weight distribution over the three data partitions.
    Weights sum to 1.0 within each phase.
    """

    run_id: int
    phase1_weights: dict[str, float]  # {"pretrain": 0.8, "midtrain": 0.15, "sft": 0.05}
    phase2_weights: dict[str, float]  # {"pretrain": 0.3, "midtrain": 0.6, "sft": 0.1}
    phase3_weights: dict[str, float]  # {"pretrain": 0.1, "midtrain": 0.2, "sft": 0.7}

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "phase1_weights": self.phase1_weights,
            "phase2_weights": self.phase2_weights,
            "phase3_weights": self.phase3_weights,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThreePartitionWeightConfig":
        """Create from dictionary."""
        return cls(
            run_id=d["run_id"],
            phase1_weights=d["phase1_weights"],
            phase2_weights=d["phase2_weights"],
            phase3_weights=d["phase3_weights"],
        )


class ThreePartitionWeightSampler:
    """Samples mixture weights for three-partition, three-phase training.

    Uses temperature-scaled Dirichlet sampling with reject sampling for constraints,
    following the RegMix methodology.

    Parameters adapted from RegMix synthesize_mixture.py:
    - TEMP: Temperature for the prior distribution (0.5 smooths skewed distributions)
    - MIN_STRENGTH/MAX_STRENGTH: Range for Dirichlet concentration parameter
    - MIN_WEIGHT: Minimum weight threshold for statistical significance
    - MAX_RATIO: Maximum ratio vs natural proportion to prevent extreme weights
    """

    PARTITIONS = ["pretrain", "midtrain", "sft"]

    # RegMix hyperparameters
    TEMP = 0.5  # Temperature for Dirichlet sampling
    MIN_STRENGTH = 0.1  # Minimum concentration parameter
    MAX_STRENGTH = 5.0  # Maximum concentration parameter
    MIN_WEIGHT = 2e-4  # Minimum weight threshold
    MAX_RATIO = 15.0  # Maximum ratio vs natural proportion
    SAMPLE_MULTIPLIER = 100  # Oversample for reject sampling

    # Natural proportions reflecting typical training data distribution
    # These are used as the prior for Dirichlet sampling
    DEFAULT_NATURAL_PROPORTIONS = {
        "pretrain": 0.70,  # Most data is pretraining
        "midtrain": 0.25,  # Some midtraining
        "sft": 0.05,  # Small amount of SFT
    }

    def __init__(
        self,
        natural_proportions: dict[str, float] | None = None,
        seed: int = 42,
        temp: float | None = None,
        min_strength: float | None = None,
        max_strength: float | None = None,
        min_weight: float | None = None,
        max_ratio: float | None = None,
    ):
        """Initialize the weight sampler.

        Args:
            natural_proportions: Prior proportions for each partition. If None, uses defaults.
            seed: Random seed for reproducibility.
            temp: Temperature for Dirichlet sampling. If None, uses TEMP.
            min_strength: Minimum concentration parameter. If None, uses MIN_STRENGTH.
            max_strength: Maximum concentration parameter. If None, uses MAX_STRENGTH.
            min_weight: Minimum weight threshold. If None, uses MIN_WEIGHT.
            max_ratio: Maximum ratio vs natural proportion. If None, uses MAX_RATIO.
        """
        self.natural_proportions = natural_proportions or self.DEFAULT_NATURAL_PROPORTIONS.copy()
        self.rng = np.random.default_rng(seed)

        # Allow overriding hyperparameters
        self.temp = temp if temp is not None else self.TEMP
        self.min_strength = min_strength if min_strength is not None else self.MIN_STRENGTH
        self.max_strength = max_strength if max_strength is not None else self.MAX_STRENGTH
        self.min_weight = min_weight if min_weight is not None else self.MIN_WEIGHT
        self.max_ratio = max_ratio if max_ratio is not None else self.MAX_RATIO

        # Normalize natural proportions
        total = sum(self.natural_proportions.values())
        self.natural_proportions = {k: v / total for k, v in self.natural_proportions.items()}

        # Compute upper bounds for reject sampling (max ratio constraint)
        self.upper_bounds = {
            name: min(prop * self.max_ratio, 1.0) for name, prop in self.natural_proportions.items()
        }

    def sample_phase_weights(self) -> dict[str, float]:
        """Sample weights for a single phase using Dirichlet distribution.

        Uses temperature-scaled alphas and reject sampling to enforce constraints.

        Returns:
            Dictionary mapping partition names to weights (sums to 1.0).
        """
        # Sample concentration parameter (strength) from log-uniform distribution
        log_min = np.log10(self.min_strength)
        log_max = np.log10(self.max_strength)
        strength = 10 ** self.rng.uniform(log_min, log_max)

        # Create temperature-scaled alpha vector
        alphas = np.array(
            [strength * (self.natural_proportions[p] ** self.temp) for p in self.PARTITIONS]
        )

        # Rejection sampling loop
        max_attempts = 1000
        for _ in range(max_attempts):
            weights = self.rng.dirichlet(alphas)
            weights_dict = dict(zip(self.PARTITIONS, weights))

            if self._check_constraints(weights_dict):
                return self._normalize(weights_dict)

        # Fallback to natural proportions if rejection sampling fails
        return self._normalize(self.natural_proportions.copy())

    def _check_constraints(self, weights: dict[str, float]) -> bool:
        """Check if weights satisfy all constraints.

        Constraints:
        1. Maximum ratio constraint: weight <= natural_prop * MAX_RATIO
        2. Minimum weight threshold is applied during normalization, not rejection

        Args:
            weights: Dictionary of partition weights.

        Returns:
            True if weights satisfy constraints, False otherwise.
        """
        for name, w in weights.items():
            upper_bound = self.upper_bounds[name]
            if w > upper_bound:
                return False
        return True

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum to 1.0 and apply minimum threshold.

        Weights below the minimum threshold are set to zero, then remaining
        weights are renormalized.

        Args:
            weights: Dictionary of partition weights.

        Returns:
            Normalized weights summing to 1.0.
        """
        # Zero out very small weights
        weights = {k: (v if v >= self.min_weight else 0.0) for k, v in weights.items()}

        total = sum(weights.values())
        if total > 0:
            return {k: v / total for k, v in weights.items()}

        # Fallback to uniform if all weights are zero
        n = len(weights)
        return {k: 1.0 / n for k in weights}

    def sample_config(self, run_id: int) -> ThreePartitionWeightConfig:
        """Sample a complete three-phase weight configuration.

        Each phase's weights are sampled independently.

        Args:
            run_id: Identifier for this configuration.

        Returns:
            ThreePartitionWeightConfig with sampled weights for all phases.
        """
        return ThreePartitionWeightConfig(
            run_id=run_id,
            phase1_weights=self.sample_phase_weights(),
            phase2_weights=self.sample_phase_weights(),
            phase3_weights=self.sample_phase_weights(),
        )

    def sample_n_configs(
        self,
        n: int,
        deduplicate: bool = True,
        precision: int = 2,
    ) -> list[ThreePartitionWeightConfig]:
        """Sample n unique configurations with optional deduplication.

        Args:
            n: Number of configurations to sample.
            deduplicate: Whether to remove near-duplicate configurations.
            precision: Decimal places for deduplication comparison.

        Returns:
            List of n ThreePartitionWeightConfig instances.
        """
        configs: list[ThreePartitionWeightConfig] = []
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

    def _config_hash(self, config: ThreePartitionWeightConfig, precision: int = 2) -> str:
        """Create a hash for deduplication based on quantized weights.

        Args:
            config: Configuration to hash.
            precision: Decimal places for quantization.

        Returns:
            Hash string for the configuration.
        """

        def quantize(weights: dict[str, float]) -> tuple:
            return tuple(sorted((k, round(v, precision)) for k, v in weights.items()))

        key = (
            quantize(config.phase1_weights),
            quantize(config.phase2_weights),
            quantize(config.phase3_weights),
        )
        return hashlib.md5(str(key).encode()).hexdigest()

    def summarize_configs(self, configs: list[ThreePartitionWeightConfig]) -> dict:
        """Generate summary statistics for a set of configurations.

        Args:
            configs: List of configurations to summarize.

        Returns:
            Dictionary with summary statistics.
        """
        phase1_weights = np.array([[c.phase1_weights[p] for p in self.PARTITIONS] for c in configs])
        phase2_weights = np.array([[c.phase2_weights[p] for p in self.PARTITIONS] for c in configs])
        phase3_weights = np.array([[c.phase3_weights[p] for p in self.PARTITIONS] for c in configs])

        def stats(arr: np.ndarray, partition_idx: int) -> dict:
            col = arr[:, partition_idx]
            return {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }

        return {
            "n_configs": len(configs),
            "phase1": {p: stats(phase1_weights, i) for i, p in enumerate(self.PARTITIONS)},
            "phase2": {p: stats(phase2_weights, i) for i, p in enumerate(self.PARTITIONS)},
            "phase3": {p: stats(phase3_weights, i) for i, p in enumerate(self.PARTITIONS)},
        }
