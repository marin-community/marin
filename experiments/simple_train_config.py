from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleTrainConfig:
    tpu_type: str
    train_batch_size: int
    num_train_steps: int
    learning_rate: float
    data_seed: int | None = None
    weight_decay: float | None = None
    beta1: float | None = None
    beta2: float | None = None
    epsilon: float | None = None
    max_grad_norm: float | None = None
    warmup: float | None = None
    decay: float | None = None
    lr_schedule: str | None = None
    min_lr_ratio: float | None = None
    cycle_length: int | list[int] | None = None
    z_loss_weight: float | None = None
    ema_beta: float | None = None
    """exponential moving average beta"""

    steps_per_eval: int | None = None
    """how often to run validation losses"""
    steps_per_export: int = 10000
    steps_per_task_eval: int | None = None
    """how often to run task evaluations"""
    steps_per_hf_export: int | None = None
    """None means match steps_per_export, -1 disables"""

    node_count: int = 1

    allow_out_of_region_reads: bool = False
    """Allow us to read data from other regions. On GCS, intra-continent bandwidth is roughly 1 month of storage,
    so sometimes it makes more sense to just read across regions."""
    allow_out_of_region_writes: bool = False
    """This makes less sense than reading across regions, but for completeness."""
