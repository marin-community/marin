from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleTrainConfig:
    """Simplified configuration for training (the things that matter)."""

    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    weight_decay: float
