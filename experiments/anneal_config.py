from dataclasses import dataclass

from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig
from marin.resources import ResourceConfig, TpuPodConfig


@dataclass(frozen=True)
class AnnealConfig:
    # 198468 steps is roughly 198468 steps * 1024 batch size * 4096 seq len = 0.832T tokens
    # Numbers were taken from exp600_tootsie.py. We start with this 8B model because it would take a long time to train
    # another one from scratch.
    DEFAULT_CHECKPOINT_PATH = "gs://marin-us-east1/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-660000"
    LLAMA_MAX_SEQ_LEN = 4096

    # Annealing dataset and proportions
    dataset_config: LMMixtureDatasetConfig

    # Model Checkpoint related
    initialize_from_checkpoint_path: str = DEFAULT_CHECKPOINT_PATH

    # Training schedule related
    learning_rate: float = 1e-3
    min_lr_ratio: float = 0.0
    weight_decay: float = 0.05
    lr_schedule: str = "linear"
    train_batch_size: int = 1024
    num_anneal_training_tokens: int = 50_000_000_000  # 50B tokens

    # Hardware related
    resources: ResourceConfig = TpuPodConfig(tpu_type="v4-128", node_count=2)  # noqa: RUF009

    # Checkpoint related
    steps_per_export: int = 10000

    # Validation related
    use_default_validation: bool = True

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
