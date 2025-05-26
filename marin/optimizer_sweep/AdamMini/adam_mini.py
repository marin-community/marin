from levanter.optim import MiniConfig


def adam_mini_config(
    learning_rate: float | None = None,
    weight_decay: float | None = None,
    beta1: float | None = None,
    beta2: float | None = None,
    epsilon: float | None = None,
    max_grad_norm: float | None = None,
    warmup: float | None = None,
    decay: float | None = None,
    lr_schedule: str | None = None,
    stable_lr_schedule: str | None = None,
    min_lr_ratio: float | None = None,
    cycle_length: int | None = None,
    nesterov: bool | None = None,
) -> MiniConfig:
    optimizer = MiniConfig(
        learning_rate=learning_rate,
        weight_decay=(weight_decay if weight_decay is not None else MiniConfig().weight_decay),
        beta1=(beta1 if beta1 is not None else MiniConfig().beta1),
        beta2=(beta2 if beta2 is not None else MiniConfig().beta2),
        epsilon=(epsilon if epsilon is not None else MiniConfig().epsilon),
        max_grad_norm=(max_grad_norm if max_grad_norm is not None else MiniConfig().max_grad_norm),
        warmup=(warmup if warmup is not None else MiniConfig().warmup),
        decay=(decay if decay is not None else MiniConfig().decay),
        lr_schedule=(lr_schedule if lr_schedule is not None else MiniConfig().lr_schedule),
        stable_lr_schedule=(stable_lr_schedule if stable_lr_schedule is not None else MiniConfig().stable_lr_schedule),
        cycle_length=cycle_length,  # can be int, list[int], or None
        min_lr_ratio=(min_lr_ratio if min_lr_ratio is not None else MiniConfig().min_lr_ratio),
        nesterov=nesterov,
    )
    return optimizer
