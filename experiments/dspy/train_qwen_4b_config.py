import os
from marin.execution import ResourceConfig
from fray.cluster import CpuConfig, TpuConfig, GpuConfig

# Check environment for CPU mode (useful for local testing)
USE_CPU = os.environ.get("MARIN_USE_CPU", "0") == "1"

@dataclass(frozen=True)
class QwenSFTConfig:
    """
    Configuration for Qwen-4B Supervised Fine-Tuning.
    """
    # Hardware resources
    resources: ResourceConfig = field(default_factory=lambda: ResourceConfig(
        device=(
            CpuConfig() if USE_CPU 
            else TpuConfig(type="v5p-8", count=1)
        ),
        cpu=16,
        ram="64g"
    ))

    # Optimization
    # Reduce load significantly if running on CPU
    train_batch_size: int = 1 if USE_CPU else 64
    num_train_steps: int = 10 if USE_CPU else 5000
    
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
    # If using CPU test, checkpoint less frequently (effectively never for a 10 step run)
    steps_per_eval: int = 500
    steps_per_checkpoint: int = 10 if USE_CPU else 1000
    steps_per_hf_export: int = 10 if USE_CPU else 1000
    
    # Misc
    reinit_tokens: bool = True
    seed: int = 0
