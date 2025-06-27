from typing import Any, NamedTuple

import numpy as np


class EnvStep(NamedTuple):
    """Container for an environment step."""

    examples: list[dict[str, Any]]  # The problems sampled from dataset
    samples: list[list[dict[str, np.ndarray]]]  # samples[i][j]: The j-th sample for the i-th problem in a batch
    rewards: np.ndarray  # Shape: (number of examples, number of generations per example)
    metrics: dict[str, float]  # Additional metrics


class MarinEnv:
    def __init__(self, **kwargs):
        """Initialize environment. Environment-specific setup (e.g., databases, file loads)."""
        pass

    def step(self, **kwargs) -> EnvStep:
        """Calls environment specific step."""
        pass
