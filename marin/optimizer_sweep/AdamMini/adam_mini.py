from levanter.optim import MiniConfig
from typing import Optional

def adam_mini_config(
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    epsilon: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    stable_lr_schedule: Optional[str] = None,
    min_lr_ratio: Optional[float] = None,
    cycle_length: Optional[int] = None,
) -> MiniConfig:
    optimizer=MiniConfig(
            learning_rate=learning_rate,
            weight_decay=(
                weight_decay if weight_decay is not None else MiniConfig().weight_decay
            ),
            beta1=(beta1 if beta1 is not None else MiniConfig().beta1),
            beta2=(beta2 if beta2 is not None else MiniConfig().beta2),
            epsilon=(epsilon if epsilon is not None else MiniConfig().epsilon),
            max_grad_norm=(
                max_grad_norm if max_grad_norm is not None else MiniConfig().max_grad_norm
            ),
            warmup=(warmup if warmup is not None else MiniConfig().warmup),
            decay=(decay if decay is not None else MiniConfig().decay),
            lr_schedule=(lr_schedule if lr_schedule is not None else MiniConfig().lr_schedule),
            stable_lr_schedule=(
                stable_lr_schedule
                if stable_lr_schedule is not None
                else MiniConfig().stable_lr_schedule
            ),
            cycle_length=cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                min_lr_ratio if min_lr_ratio is not None else MiniConfig().min_lr_ratio
            ),
        )
    return optimizer
