from marin.execution import ResourceConfig
from fray.cluster import CpuConfig, TpuConfig, GpuConfig

@dataclass(frozen=True)
class QwenSFTConfig:
    """
    Configuration for Qwen-4B Supervised Fine-Tuning.
    """
    # Hardware resources (CPU fallback enabled)
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(
        device=CpuConfig(),  # Changed to CPU for local testing
        cpu=16,
        ram="64g"
    ))

    # Optimization
    # Optimization (Reduced for CPU testing)
    train_batch_size: int = 1
    num_train_steps: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    
    # Model
    model_name_or_path: str = "Qwen/Qwen1.5-4B-Chat"
    max_seq_len: int = 4096
    
    # Scheduler
    warmup: float = 0.02
    cooldown: float = 0.0
    lr_schedule: str = "linear"
    min_lr_ratio: float = 0.1
    
    # Checkpointing & Evaluation
    steps_per_eval: int = 500
    steps_per_checkpoint: int = 1000
    steps_per_hf_export: int = 1000
    
    # Misc
    reinit_tokens: bool = True
    seed: int = 0
