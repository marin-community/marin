import dataclasses
from dataclasses import dataclass

from levanter.callbacks.watch import WatchConfig
from levanter.optim import OptimizerConfig
from levanter.schedule import IntSchedule

from marin.resources import ResourceConfig, TpuPodConfig


@dataclass(frozen=True)
class SimpleTrainConfig:
    resources: ResourceConfig
    train_batch_size: int | IntSchedule
    """
    The batch size for training. If an IntSchedule is provided, the batch size will be
    varied according to the schedule.
    """
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
    rewarmup: float | None = None
    """
    The rewarmup parameter is used to re-warmup the learning rate after a decay cycles
    """
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
    per_device_eval_parallelism: int | None = None
    """Number of examples to evaluate in parallel on each device"""
    max_eval_batches: int | None = None
    """Maximum number of batches to evaluate on. None means all batches"""

    initialize_from_checkpoint_path: str | None = None
    """If set, the training will resume from the checkpoint at this path. Otherwise, training will start from scratch."""
    initialize_from_hf: str | None = None
    """If set, the training will start from the hf model at this path. Otherwise, training will start from scratch."""
    reset_data_loader_on_init: bool = True
    """Pairs with initialize_from_checkpoint_path. If True, initialize_from_checkpoint_path will reset the data loader
    so that it starts from step 0. Otherwise, it will resume from the step in the checkpoint."""

    allow_partial_checkpoint: bool = False
    """
    Allow loading partial checkpoints. This is useful for converting training to EMA, e.g.
    """

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    optimizer_config: OptimizerConfig | None = None
    """Optimizer configuration to use. If not set, Adam will be used."""

    reinit_tokens: list[str] | bool = False
    """
    if set, will reinitialize the embeddings for the given tokens. If True, will reinitialize the default tokens
    for llama3's tokenizer
    """

    watch: WatchConfig = dataclasses.field(default_factory=WatchConfig)
    """Config for watching gradients, parameters, etc. Default is to log norms of gradients and parameters."""

    @property
    def tpu_type(self) -> str | None:
        """For backward compatibility."""
        if isinstance(self.resources, TpuPodConfig):
            return self.resources.tpu_type
        return None

    @property
    def node_count(self) -> int:
        """For backward compatibility."""
        if isinstance(self.resources, TpuPodConfig):
            return self.resources.node_count
        return 1
