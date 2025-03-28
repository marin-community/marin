from typing import Optional
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
    tpu_type: Optional[str] = None
    """Type of TPU to use for training. None for local training."""
    
    # Model configuration
    tokenizer: Optional[str] = None
    """Tokenizer to use for training."""
    
    model_name_or_path: Optional[str] = None
    """Path to the pretrained model."""
    
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
    
    max_grad_norm: Optional[float] = None
    """Maximum gradient norm for gradient clipping."""
    
    # Checkpointing and evaluation
    steps_per_eval: int = 1000
    """How often to run validation losses."""
    
    steps_per_checkpoint: int = 1000
    """How often to save checkpoints."""
    
    steps_per_hf_export: int = 500
    """How often to save HuggingFace checkpoints."""
    
    input_role: str = "user"
    """Role for input in chat format."""
    
    output_role: str = "assistant"
    """Role for output in chat format."""
    
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
    """Whether to initialize from HuggingFace model."""
    
    bypass_path_checks: bool = False
    """Whether to bypass path checks."""
    
    node_count: int = 1
    """Number of TPU slices for training."""
    
    int8: bool = False
    """Int8 (quantized) training in Levanter."""
    
    allow_out_of_region_reads: bool = False
    """
    Allow us to read data from other regions. On GCS, intra-continent bandwidth is roughly 1 month of storage,
    so sometimes it makes more sense to just read across regions.
    """
    
    allow_out_of_region_writes: bool = False
    """This makes less sense than reading across regions, but for completeness."""