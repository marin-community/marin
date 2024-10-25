from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleTrainConfig:
    tpu_type: str | None = None
    train_batch_size: int | None = None
    num_train_steps: int | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    min_lr_ratio: float | None = None
    warmup: int | None = None
    cooldown: float | None = None
    z_loss_weight: float | None = None
