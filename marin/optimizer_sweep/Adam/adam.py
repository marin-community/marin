from levanter.optim import AdamConfig
from typing import Optional

def adam_config(
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    epsilon: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    warmup: Optional[float] = None,
    rewarmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    cycle_length: Optional[int] = None,
    min_lr_ratio: Optional[float] = None,
    nesterov: Optional[bool] = None,
) -> AdamConfig:
    optimizer = AdamConfig(
        learning_rate=learning_rate,
        weight_decay=(
            weight_decay if weight_decay is not None else AdamConfig().weight_decay
        ),
        beta1=(beta1 if beta1 is not None else AdamConfig().beta1),
        beta2=(beta2 if beta2 is not None else AdamConfig().beta2),
        epsilon=(epsilon if epsilon is not None else AdamConfig().epsilon),
        max_grad_norm=(
            max_grad_norm if max_grad_norm is not None else AdamConfig().max_grad_norm
        ),
        warmup=(warmup if warmup is not None else AdamConfig().warmup),
        rewarmup=(rewarmup if rewarmup is not None else AdamConfig().rewarmup),
        decay=(decay if decay is not None else AdamConfig().decay),
        lr_schedule=(
            lr_schedule if lr_schedule is not None else AdamConfig().lr_schedule
        ),
        cycle_length=cycle_length,  # can be int, list[int], or None
        min_lr_ratio=(
            min_lr_ratio if min_lr_ratio is not None else AdamConfig().min_lr_ratio
        ),
        nesterov = False
    )
    return optimizer