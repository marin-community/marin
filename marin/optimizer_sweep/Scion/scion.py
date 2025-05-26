from levanter.optim import ScionConfig
from typing import Optional

def scion_config(
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    momentum: Optional[float] = None,
    beta1: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    scion_epsilon: Optional[float] = None,
    scion_to_signum_lr: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    stable_lr_schedule: Optional[str] = None,
    min_lr_ratio: Optional[float] = None,
    cycle_length: Optional[int] = None,
) -> ScionConfig:
    optimizer=ScionConfig(
        learning_rate=learning_rate,
            weight_decay=(
                weight_decay if weight_decay is not None else ScionConfig().weight_decay
            ),
            momentum=(momentum if momentum is not None else ScionConfig().momentum),
            beta1=(beta1 if beta1 is not None else ScionConfig().beta1),
            max_grad_norm=(
                max_grad_norm if max_grad_norm is not None else ScionConfig().max_grad_norm
            ),
            scion_epsilon=(
                scion_epsilon if scion_epsilon is not None else ScionConfig().scion_epsilon
            ),
            scion_to_signum_lr=(
                scion_to_signum_lr
                if scion_to_signum_lr is not None
                else ScionConfig().scion_to_signum_lr
            ),
            warmup=(warmup if warmup is not None else ScionConfig().warmup),
            decay=(decay if decay is not None else ScionConfig().decay),
            lr_schedule=(
                lr_schedule if lr_schedule is not None else ScionConfig().lr_schedule
            ),
            stable_lr_schedule=(
                stable_lr_schedule
                if stable_lr_schedule is not None
                else ScionConfig().stable_lr_schedule
            ),
            cycle_length=cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                min_lr_ratio if min_lr_ratio is not None else ScionConfig().min_lr_ratio
            ),
        )
    return optimizer