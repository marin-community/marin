from typing import NamedTuple

class EnvStep(NamedTuple):
    """Container for an environment step."""
    llm_in: str
    llm_out: str
    reward: float

class MarinEnv:

    def __init__(self, endpoint: str, **kwargs):
        """Initialize endpoint. Environment-specific setup (e.g., databases, file loads)."""
        pass

    def step(self, **kwargs) -> EnvStep:
        """Calls environment specific setup."""
        pass
