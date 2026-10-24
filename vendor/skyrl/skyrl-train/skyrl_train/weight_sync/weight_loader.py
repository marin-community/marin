"""Weight loader interface for inference engines."""

from abc import ABC, abstractmethod

from skyrl_train.inference_engines.base import NamedWeightsUpdateRequest


class WeightLoader(ABC):
    """Loads received weights into inference engine.

    Implementations coordinate vLLM weight transfer and apply received weights.
    """

    @abstractmethod
    async def load_weights(self, request: NamedWeightsUpdateRequest) -> None:
        """Load weights into the inference engine.

        Coordinates with the receiver to fetch weights and applies them
        to the model. Handles RPC coordination for distributed engines.

        Args:
            request: Weight update request containing names, dtypes, shapes,
                    and optionally IPC handles or other transfer metadata.
        """
        ...
