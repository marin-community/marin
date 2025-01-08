from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleTrainConfig:
    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    data_seed: int | None = None
    weight_decay: float | None = None
    min_lr_ratio: float | None = None
    warmup: float | None = None
    decay: float | None = None
    lr_schedule: str | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None

    steps_per_eval: int | None = None
    """how often to run validation losses"""
    steps_per_export: int = 10000
    steps_per_task_eval: int | None = None
    """how often to run task evaluations"""

    node_count: int = 1
