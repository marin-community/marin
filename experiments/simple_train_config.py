from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleTrainConfig:
    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    weight_decay: float | None = None
    min_lr_ratio: float | None = None
    warmup: float | None = None
    decay: float | None = None
    lr_schedule: str | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None
    steps_per_eval: int | None = None
    node_count: int = 1
