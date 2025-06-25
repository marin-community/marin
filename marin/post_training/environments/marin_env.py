from typing import NamedTuple, List, Dict, Any
import numpy as np


class EnvStep(NamedTuple):
    """Container for an environment step."""
    examples: List[Dict[str, Any]]  # The problems sampled from dataset
    samples: List[List[Dict[str, np.ndarray]]]  # Generated samples from model
    rewards: np.ndarray  # Computed rewards for each sample
    metrics: Dict[str, float]  # Additional metrics


class MarinEnv:
    def __init__(self, **kwargs):
        """Initialize environment. Environment-specific setup (e.g., databases, file loads)."""
        pass
    
    def step(self, **kwargs) -> EnvStep:
        """Calls environment specific step."""
        pass
