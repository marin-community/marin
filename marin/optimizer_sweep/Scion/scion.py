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
                train_config.weight_decay if train_config.weight_decay is not None else ScionConfig().weight_decay
            ),
            momentum=(train_config.momentum if train_config.momentum is not None else ScionConfig().momentum),
            beta1=(train_config.beta1 if train_config.beta1 is not None else ScionConfig().beta1),
            max_grad_norm=(
                train_config.max_grad_norm if train_config.max_grad_norm is not None else ScionConfig().max_grad_norm
            ),
            scion_epsilon=(
                train_config.scion_epsilon if train_config.scion_epsilon is not None else ScionConfig().scion_epsilon
            ),
            scion_to_signum_lr=(
                train_config.scion_to_signum_lr
                if train_config.scion_to_signum_lr is not None
                else ScionConfig().scion_to_signum_lr
            ),
            warmup=(train_config.warmup if train_config.warmup is not None else ScionConfig().warmup),
            decay=(train_config.decay if train_config.decay is not None else ScionConfig().decay),
            lr_schedule=(
                train_config.lr_schedule if train_config.lr_schedule is not None else ScionConfig().lr_schedule
            ),
            stable_lr_schedule=(
                train_config.stable_lr_schedule
                if train_config.stable_lr_schedule is not None
                else ScionConfig().stable_lr_schedule
            ),
            cycle_length=train_config.cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                train_config.min_lr_ratio if train_config.min_lr_ratio is not None else ScionConfig().min_lr_ratio
            ),
        )
    return optimizer