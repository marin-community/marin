from levanter.optim import KronConfig


def kron_config(
    learning_rate: float,
    weight_decay: float | None = None,
    beta1: float | None = None,
    preconditioner_lr: float | None = None,
    preconditioner_init_scale: float | None = None,
    max_grad_norm: float | None = None,
    normalize_grads: bool | None = None,
    partition_grads_into_blocks: bool | None = None,
    block_size: int | None = None,
    preconditioner_update_probability: float | None = None,
    update_prob_flat_start: float | None = None,
    warmup: float | None = None,
    decay: float | None = None,
    lr_schedule: str | None = None,
    cycle_length: int | list[int] | None = None,
    min_lr_ratio: float | None = None,
) -> KronConfig:
    default_config = KronConfig()
    return KronConfig(
        learning_rate=learning_rate,
        weight_decay=weight_decay if weight_decay is not None else default_config.weight_decay,
        beta1=beta1 if beta1 is not None else default_config.beta1,
        preconditioner_lr=preconditioner_lr if preconditioner_lr is not None else default_config.preconditioner_lr,
        preconditioner_init_scale=(
            preconditioner_init_scale
            if preconditioner_init_scale is not None
            else default_config.preconditioner_init_scale
        ),
        max_grad_norm=max_grad_norm if max_grad_norm is not None else default_config.max_grad_norm,
        normalize_grads=normalize_grads if normalize_grads is not None else default_config.normalize_grads,
        partition_grads_into_blocks=(
            partition_grads_into_blocks
            if partition_grads_into_blocks is not None
            else default_config.partition_grads_into_blocks
        ),
        block_size=block_size if block_size is not None else default_config.block_size,
        preconditioner_update_probability=(
            preconditioner_update_probability
            if preconditioner_update_probability is not None
            else default_config.preconditioner_update_probability
        ),
        update_prob_flat_start=(
            update_prob_flat_start if update_prob_flat_start is not None else default_config.update_prob_flat_start
        ),
        warmup=warmup if warmup is not None else default_config.warmup,
        decay=decay if decay is not None else default_config.decay,
        lr_schedule=lr_schedule if lr_schedule is not None else default_config.lr_schedule,
        cycle_length=cycle_length,
        min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else default_config.min_lr_ratio,
    )
