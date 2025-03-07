from dataclasses import dataclass

from marin.processing.tokenize.data_configs import LMMixtureDatasetConfig


@dataclass
class AnnealConfig:
    # 198468 steps is roughly 198468 steps * 1024 batch size * 4096 seq len = 0.832T tokens
    # Numbers were taken from exp600_tootsie.py. We start with this 8B model because it would take a long time to train
    # another one from scratch.
    DEFAULT_CHECKPOINT_PATH = "gs://marin-eu-west4/checkpoints/llama-8b-tootsie-0.001-19ad63/checkpoints/step-660000"
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
    tpu_type: str = "v4-128"
    node_count: int = 2

    # Checkpoint related
    steps_per_export: int = 10000
