from levanter.optim import CautiousConfig
from typing import Optional


def cautious_config(
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
) -> CautiousConfig:
    optimizer = CautiousConfig(
        learning_rate=learning_rate,
        weight_decay=(
            weight_decay if weight_decay is not None else CautiousConfig().weight_decay
        ),
        beta1=(beta1 if beta1 is not None else CautiousConfig().beta1),
        beta2=(beta2 if beta2 is not None else CautiousConfig().beta2),
        epsilon=(epsilon if epsilon is not None else CautiousConfig().epsilon),
        max_grad_norm=(
            max_grad_norm if max_grad_norm is not None else CautiousConfig().max_grad_norm
        ),
        warmup=(warmup if warmup is not None else CautiousConfig().warmup),
        decay=(decay if decay is not None else CautiousConfig().decay),
        lr_schedule=(lr_schedule if lr_schedule is not None else CautiousConfig().lr_schedule),
        stable_lr_schedule=(
            stable_lr_schedule
            if stable_lr_schedule is not None
            else CautiousConfig().stable_lr_schedule
        ),
        cycle_length=cycle_length,  # can be int, list[int], or None
        min_lr_ratio=(
            min_lr_ratio if min_lr_ratio is not None else CautiousConfig().min_lr_ratio
        ),
    )
    return optimizer