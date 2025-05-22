from levanter.optim import KronConfig
from typing import Optional

def kron_config(
    learning_rate: float,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    preconditioner_lr: Optional[float] = None,
    preconditioner_init_scale: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    normalize_grads: Optional[bool] = None,
    partition_grads_into_blocks: Optional[bool] = None,
    block_size: Optional[int] = None,
    preconditioner_update_probability: Optional[float] = None,
    update_prob_flat_start: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    stable_lr_schedule: Optional[bool] = None,
    cycle_length: Optional[int | list[int]] = None,
    min_lr_ratio: Optional[float] = None,
) -> KronConfig:
    default_config = KronConfig()
    return KronConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay if weight_decay is not None else default_config.weight_decay,
        beta1=beta1 if beta1 is not None else default_config.beta1,
        preconditioner_lr=preconditioner_lr if preconditioner_lr is not None else default_config.preconditioner_lr,
        preconditioner_init_scale=preconditioner_init_scale if preconditioner_init_scale is not None else default_config.preconditioner_init_scale,
        max_grad_norm=max_grad_norm if max_grad_norm is not None else default_config.max_grad_norm,
        normalize_grads=normalize_grads if normalize_grads is not None else default_config.normalize_grads,
        partition_grads_into_blocks=partition_grads_into_blocks if partition_grads_into_blocks is not None else default_config.partition_grads_into_blocks,
        block_size=block_size if block_size is not None else default_config.block_size,
        preconditioner_update_probability=preconditioner_update_probability if preconditioner_update_probability is not None else default_config.preconditioner_update_probability,
        update_prob_flat_start=update_prob_flat_start if update_prob_flat_start is not None else default_config.update_prob_flat_start,
        warmup=warmup if warmup is not None else default_config.warmup,
        decay=decay if decay is not None else default_config.decay,
        lr_schedule=lr_schedule if lr_schedule is not None else default_config.lr_schedule,
        stable_lr_schedule=stable_lr_schedule if stable_lr_schedule is not None else default_config.stable_lr_schedule,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else default_config.min_lr_ratio,
    )
