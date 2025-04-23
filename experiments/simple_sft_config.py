from dataclasses import dataclass

from levanter.schedule import IntSchedule


@dataclass(frozen=True)
class SimpleSFTConfig:
    """
    A simplified configuration for Supervised Fine-Tuning (SFT) that works for both
    single dataset and mixture training approaches.
    """

    # Core training parameters
    train_batch_size: int | IntSchedule = 128
    """
    The batch size for training. If an IntSchedule is provided, the batch size will be
    varied according to the schedule.
    """
    num_train_steps: int = 10000
    """Number of training steps."""

    learning_rate: float = 5e-6
    """Learning rate for the optimizer."""

    # Hardware configuration
    tpu_type: str | None = None
    """Type of TPU to use for training. None for local training."""

    # Model configuration
    tokenizer: str | None = None
    """Tokenizer to use for training."""

    model_name_or_path: str | None = None
    """Path to the pretrained HF model checkpoint to initialize from"""

    initialize_from_checkpoint_path: str | None = None
    """Path to a levanter checkpoint to initialize from."""

    max_seq_len: int = 4096
    """Maximum sequence length for training."""

    # Optimizer parameters
    weight_decay: float = 0.0
    """Weight decay for the optimizer."""

    warmup: float = 0.03
    """Fraction of training steps to use for learning rate warmup."""

    cooldown: float = 0.0
    """Fraction of training steps to use for learning rate cooldown."""

    lr_schedule: str = "linear"
    """Learning rate schedule to use: 'linear', 'cosine', etc."""

    min_lr_ratio: float = 0.0
    """Minimum learning rate as a ratio of the base learning rate."""

    max_grad_norm: float | None = None
    """Maximum gradient norm for gradient clipping."""

    # Checkpointing and evaluation
    steps_per_eval: int = 1000
    """How often to run validation losses."""

    steps_per_checkpoint: int = 1000
    """How often to save checkpoints."""

    steps_per_hf_export: int = 500
    """How often to save HuggingFace checkpoints."""

    # Mixture-specific parameters
    mixture_block_size: int = 2048
    """Block size for dataset mixing (only used with mixture training)."""

    stop_strategy: str = "restart"
    """
    Strategy for handling dataset completion (only used with mixture training).
    Options: 'restart' or 'exit'.
    """

    # Other parameters
    seed: int = 0
    """Random seed for training."""

    initialize_from_hf: bool = True
    """Whether to initialize from HuggingFace model. If false, we will load a levanter checkpoint."""

    node_count: int = 1
    """Number of TPU slices for training."""

    int8: bool = False
    """Int8 (quantized) training in Levanter."""

    z_loss_weight: float = 0.0

    reinit_tokens: list[str] | bool = False
    """
    if set, will reinitialize the embeddings for the given tokens. If True, will reinitialize the default tokens
    for llama3's tokenizer
    """
